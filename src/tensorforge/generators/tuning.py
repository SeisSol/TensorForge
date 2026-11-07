# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The configurations a kernel can be generated in, and how to choose one.

`lanes.search` chooses the lane count and nothing else, and the measurements
say that is not the decision that stands alone.  `local_flux` on GB200
(package 4) was fastest at eight lanes up to b = 56 -- but with the four faces
merged, the operators prepared and, at 56, the reduction rolled by 28 -- and at
16 lanes of width 2 from b = 80 on.  Eight lanes without the rest is another
kernel, and the width is not in the lane candidates at all.  So the space is
the product: the lane geometry and the options that change what a geometry
compiles to.

Three parts, kept apart because they answer to different things:

* `space` says which configurations exist for a descriptor list -- every knob
  with the values that can make a difference here, and none that cannot (no
  merging where nothing repeats, no preparing where no operand is
  batch-constant, no matrix path off NVIDIA).
* A *scorer* says how good one built configuration is.  `static_score` reads
  the build alone; `CompiledScore` asks a compiler for registers and spills;
  `MeasuredScore` hands the build to a caller who times it.  Lower is better,
  and `None` is "could not score" rather than "scored zero".
* A *strategy* walks the space: `exhaustive` for a small one, `coordinate`
  (one knob at a time from the default, until nothing improves) otherwise.

The compiler is the caller's to name.  Nothing here assumes one is installed:
`toolchain.Toolchain` takes paths, falls back on the environment (`TF_NVCC`,
`TF_HIPCC`, `TF_ICPX`) and then on `PATH`, and a scorer that finds none says
so instead of guessing.

So is the build.  The generator tunes a kernel through this module
(`Options.autotune`), so nothing here names it: whatever builds a
configuration takes `generate`, which builds a descriptor list in a context
at a lane geometry and returns the generator that built it, or raises --
`Generator.trial`.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge import toolchain
from tensorforge.toolchain import Resources, Toolchain
from tensorforge.generators import lanes as lane_config
from tensorforge.generators import preferences
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 contraction_lengths)
from tensorforge.generators.lanes import LaneConfig


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    """One configuration: a lane geometry and the options it is built with.

    `lanes` of None is the generator's own deduction.  The options are the
    ones this candidate *sets*, on top of whatever the base context asked for.
    Hashable and plain, so that a tuner can keep a table of them and a result
    can be written down and rebuilt later.
    """
    lanes: Optional[LaneConfig] = None
    options: Tuple[Tuple[str, Any], ...] = ()

    def set(self, name: str, value: Any) -> 'Candidate':
        if name == 'lanes':
            return replace(self, lanes=value)
        opts = dict(self.options)
        opts[name] = value
        return replace(self, options=tuple(sorted(opts.items())))

    def get(self, name: str, default: Any = None) -> Any:
        if name == 'lanes':
            return self.lanes
        return dict(self.options).get(name, default)

    def context(self, base: Context, **override) -> Context:
        """A context for this candidate: the base's target and options, with
        this candidate's on top, and `override` over everything."""
        hw = base.target.hw
        asked = dict(base._asked_options.asked())
        asked.update(self.options)
        asked.update(override)
        return Context(arch=hw.model, backend=base.target.backend,
                       fp_type=base.fp_type, options=Options(**asked))

    def label(self) -> str:
        geo = ('deduced' if self.lanes is None
               else f'{self.lanes.num_threads}w{self.lanes.lead_width}')
        opts = ','.join(f'{n}={_spell(v)}' for n, v in self.options)
        return f'{geo}' + (f' {opts}' if opts else '')

    def to_dict(self) -> Dict[str, Any]:
        return {'lanes': None if self.lanes is None else
                [self.lanes.num_threads, self.lanes.num_active_threads,
                 self.lanes.lead_width],
                'options': dict(self.options)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candidate':
        lanes = data.get('lanes')
        return cls(lanes=None if lanes is None else LaneConfig(*lanes),
                   options=tuple(sorted(data.get('options', {}).items())))


def _spell(value: Any) -> str:
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


# --------------------------------------------------------------------------- #
# The space
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Knob:
    """One axis of the space.  `values(candidate)` may depend on the others:
    the prefetch depth of the matrix path is only a question where that path
    is taken."""
    name: str
    values: Callable[[Candidate], Sequence[Any]]


def _flat(descrs):
    return [op for d in descrs for op in d.operations()]


def _tensors(descrs):
    for d in _flat(descrs):
        for op in list(getattr(d, 'ops', [])) + [getattr(d, 'dest', None)]:
            tensor = getattr(op, 'tensor', None)
            if tensor is not None:
                yield tensor


def _roll_values(descrs) -> List[int]:
    """0, the largest divisors of the longest reduction at up to 32 and up to
    8 steps, and 2 -- the ends of what rolling buys and the smallest body it
    can leave.

    A divisor rolls without a remainder, which is why the two ends are chosen
    from them, but the option does not require one and the shortest body is
    what a kernel at the register limit wants.  `elastic-o6s:derivative`
    contracts over 55, so the divisors offer 11 and 5; on a GH200 it runs at
    21.2 ns/el rolled by two and 25.7 rolled by four, and rolled by eleven it
    is slower than the 23.2 it reaches unrolled.  Without two in the space no
    ranking could find it.
    """
    ks = contraction_lengths(descrs)
    if not ks:
        return [0]
    k = max(ks)
    out = [0]
    for cap in (32, 8):
        divisors = [d for d in range(2, min(cap, k - 1) + 1) if k % d == 0]
        if divisors and divisors[-1] not in out:
            out.append(divisors[-1])
    if k > 2 and 2 not in out:
        out.append(2)
    return out


def _mergeable(descrs, context) -> bool:
    from tensorforge.generators.rolling import ForDescr
    from tensorforge.generators.rolling import roll
    options = context.get_user_options()
    try:
        rolled = roll(list(descrs), min_count=options.merge_min_count,
                      max_arity=options.merge_max_arity)
    except Exception:
        return False
    return any(isinstance(d, ForDescr) for d in rolled)


def _geometries(descrs, context) -> List[LaneConfig]:
    """The lane candidates, and each power of two of them at width 2 where the
    backend spells a widened lead (CUDA and HIP) and the pair still fits the
    rows -- width 2 at 16 lanes was GB200's fastest from b = 80 on, and no
    lane candidate has it."""
    flat = _flat(descrs)
    out = list(lane_config.candidates(flat, context))
    if not context.target.lead_vectors():
        return out
    base = lane_config.deduce(flat, context)
    rows = base.num_active_threads or base.num_threads
    for config in list(out):
        t = config.num_threads
        if config.lead_width != 1 or t & (t - 1) or t < lane_config.MIN_LANES:
            continue
        if 2 * t > 2 * max(rows, t) or t >= rows:
            continue
        out.append(LaneConfig(num_threads=t,
                              num_active_threads=base.num_active_threads,
                              lead_width=2))
    return out


def space(descrs, context: Context) -> List[Knob]:
    """The knobs worth turning for this descriptor list on this target.

    Each with the values that can change the kernel here and only those, so
    that a strategy does not spend builds on a switch that does nothing.
    """
    geometries = _geometries(descrs, context)
    knobs = [Knob('lanes', lambda c, g=tuple(geometries): g)]
    if _mergeable(descrs, context):
        knobs.append(Knob('merge_variants', lambda c: (False, True)))
    if any(t.addressing == Addressing.NONE for t in _tensors(descrs)):
        knobs.append(Knob('prepare_operands', lambda c: (False, True)))
        knobs.append(Knob('preload_globals', lambda c: (False, True)))
    rolls = _roll_values(descrs)
    if len(rolls) > 1:
        knobs.append(Knob('k_roll', lambda c, r=tuple(rolls): r))
    # The literal limit at exactly the counts that change a kernel: 0, and each
    # batch-constant operand's non-zero entries, where the description carries
    # its numbers.  Which of those the generator inlines also depends on the
    # role and the unrolling (`Generator._embed_constants`); a value that
    # changes nothing here costs a duplicate build, not a wrong one.
    counts = sorted({sum(1 for v in values if v != 0)
                     for values in (t.storage_values() for t in _tensors(descrs)
                                    if t.addressing == Addressing.NONE)
                     if values is not None})
    if counts:
        knobs.append(Knob('inline_constants',
                          lambda c, v=tuple([0] + counts): v))
    if context.target.prefs.tune_matrix_path and contraction_lengths(descrs):
        knobs.append(Knob('tensor_cores', lambda c: (False, True)))
        knobs.append(Knob('mma_prefetch',
                          lambda c: ((1, 2) if c.get('tensor_cores') else (None,))))
        knobs.append(Knob('mma_min_depth',
                          lambda c: ((None, 8) if c.get('tensor_cores') else (None,))))
        if context.fp_type is Datatype.F32:
            knobs.append(Knob('mma_presplit',
                              lambda c: ((False, True) if c.get('tensor_cores') else (False,))))
    if _stackable(descrs):
        knobs.append(Knob('stack_elements', lambda c: (1, 2)))
    widths = _matrix_widths(descrs, context)
    if any(widths(Candidate(lanes=g)) for g in geometries):
        knobs.append(Knob('matrix_columns',
                          lambda c: (0,) + widths(c) if widths(c) else ()))
    return knobs


def _stackable(descrs) -> bool:
    """Whether the list builds in stacks of two at all: what `stacking.stack`
    refuses, it refuses by name.  What it cannot see -- a traversal that does
    not take a stack, `launch_control`'s -- fails the build, and a build that
    fails has no score."""
    from tensorforge.common.exceptions import GenerationError
    from tensorforge.generators import stacking
    try:
        stacking.stack(list(descrs), 2)
    except GenerationError:
        return False
    return True


def _matrix_widths(descrs, context: Context):
    """`widths(candidate)`: the output widths the target's matrix instruction
    may cover at the candidate's lanes (`MatrixPaths.widths`) where there are
    two or more to choose from, and none otherwise -- also where nothing is a
    contraction."""
    from tensorforge.backend.instructions.compute.paths import matrix_paths
    module = matrix_paths(context.target)
    if module is None or not contraction_lengths(descrs):
        return lambda c: ()
    deduced = lane_config.deduce(_flat(descrs), context)

    def widths(candidate):
        lanes = candidate.lanes or deduced
        found = tuple(module.widths(lanes.num_threads, context.fp_type,
                                    context))
        return found if len(found) > 1 else ()
    return widths


def _wider(base, rows: int, hw) -> List[LaneConfig]:
    """Lane counts *above* the deduced one, up to the next power of two at or
    above the extent.

    The deduced count is the widest the lead dimension is worth spreading
    over in one slot; a multiplication may still be given more lanes than
    that, and where the block is capped by threads rather than by rows this
    is the only way to make it hold fewer multiplications.  On NVIDIA with
    nothing preloaded the cap is 128 threads, so the block holds
    `128 // lanes` multiplications -- and at 128 lanes that is exactly one,
    whose shared memory is one multiplication's worth instead of four.

    `poroelastic-stp` is what this is for.  Its operand tile is what caps the
    block, and from order 6 up only one block fits on an SM: order 8 in
    double precision runs one multiplication of 32 lanes, which is one warp
    per SM.  At 128 lanes the same kernel keeps its 140672 B of shared memory
    and gets four times the threads, at 2632 B of register pressure instead
    of 6739 and 433261 code units instead of 755906.

    The ceiling is the next power of two at or above the extent, and not one
    step further.  Lanes past the extent hold nothing: order 8 spreads 120
    rows, so 128 lanes waste eight, while order 6 spreads 56 and 128 lanes
    would waste 72 -- and the register pressure that buys is a figure *per
    thread*, which is exactly how idle lanes hide.  64 is the honest widening
    there, and it is in this list; 128 is not.

    Offered to the compiled scorer alone, for the reason the axis downward is
    (`CompiledScore`): the modelled figures do not separate lane widths, and
    a candidate nothing can rank is a candidate taken at random.  Measured
    over the twenty elastic kernels on sm_90, the widening costs 0.969x under
    `static` -- `o6s:derivativeTaylorExpansion` 3.35 -> 4.88 ns an element,
    `o6d:derivativeTaylorExpansion` 5.78 -> 7.29 -- and 1.0008x under
    `compiled`, which is the noise.
    """
    if base.num_threads & (base.num_threads - 1):
        return []                      # not a power of two: nothing to double
    ceiling = min(1 << max(0, (rows - 1).bit_length()),
                  getattr(hw, 'max_threads_per_block', 1024))
    out = []
    t = base.num_threads * 2
    while t <= ceiling:
        out.append(LaneConfig(t, base.num_active_threads, base.lead_width))
        t *= 2
    return out


def _whole_wave(descrs, context: Context, origin: Candidate,
                have: Iterable[Candidate],
                fixed: Optional[Mapping[str, Any]] = None
                ) -> List[Candidate]:
    """The candidates of one multiplication a wave -- as many lanes as the
    vector unit, merging both ways where `_mergeable` -- where the wave is
    wider than every lane count in `have`; none where `fixed` names the
    lanes.  What `lanes_rule` asks about where nothing up to the extent's
    ceiling (`_wider`) spills nothing (`CLEAN_WAVE`)."""
    fixed = fixed or {}
    wave = getattr(context.target.hw, 'vec_unit_length', None)
    if not wave or 'lanes' in fixed:
        return []
    widest = max((c.lanes.num_threads for c in have if c.lanes is not None),
                 default=0)
    if wave <= widest:
        return []
    base = lane_config.deduce(_flat(descrs), context)
    cand = origin.set('lanes', LaneConfig(wave, base.num_active_threads,
                                          base.lead_width))
    if 'merge_variants' in fixed or not _mergeable(descrs, context):
        return [cand]
    return [cand.set('merge_variants', False),
            cand.set('merge_variants', True)]


def simple_space(descrs, context: Context,
                 mode: Optional[str] = None) -> List[Knob]:
    """The knobs `Options.autotune` turns: only those whose every value is
    safe to ship without the caller knowing.  `mode` is the scorer that will
    rank them -- `Options.autotune`, unless `compiled` found no compiler and
    the ranking is static after all (`autotune`).

    Lane counts are the powers of two from the deduced one down to
    `lanes.MIN_LANES` -- the geometries measured on every target; the
    divisors of the extent (5 and 7 lanes for 20 and 35 rows) rank well on
    both scorers and have never been timed.  A width of two only where one
    instruction does two FMAs (`Target.packed_fma_width`) and the extent is even:
    elsewhere it is two scalar FMAs, and at 35 rows on GB200 it was 70 %
    slower.  Merging where something repeats, and rolling by the largest
    divisor up to 32.  Not `prepare_operands`: the host packs for it.  Not
    the matrix path, whose default is per vendor and not yet measured.  On
    AMD and under the compiled scorer, a stack of two elements
    (`Options.stack_elements`) and the width of the matrix instruction
    (`Options.matrix_columns`): the caller passes the same arguments to
    either, and gets the same products.

    `preload_globals` on Intel, where it is measured and where the answer is
    per kernel rather than per vendor.  Under the explicit-vector lowering
    it is the difference between 0.31x and 0.74x of the SPMD default over the
    order-6 elastic kernels -- the operators are one shared copy there instead
    of one per work-item -- and up to 1.22x once the block holds the
    work-items to amortise it.  Under SPMD it is 0.86x on average and ranges
    from 0.35x to 1.41x, which is exactly a knob and not a default.  Elsewhere
    the vendor default stands: nobody has taken the measurement.
    """
    from tensorforge.common.basic_types import Datatype
    mode = context.get_user_options().autotune if mode is None else mode
    flat = _flat(descrs)
    if any(isinstance(d, ElementwiseDescr) for d in flat):
        return []
    hw = context.target.hw
    base = lane_config.deduce(flat, context)
    rows = base.num_active_threads or base.num_threads
    # A multiplication narrower than the vector unit leaves lanes of it idle
    # under SPMD -- there is one work-item per lane and nothing else to put in
    # the rest of the vector, where a wave that holds several multiplications
    # fills itself.  On a 16-wide PVC that is measured and steep: with the
    # geometry free to move, both scorers took `elastic-o6s:derivative` to
    # eight lanes and 66.1 ns an element, against 29.9 at the deduced 32 and
    # 12.7 at sixteen.  So the floor is the vector unit there
    # (`Preferences.lanes_down_to_wave`), and `MIN_LANES` elsewhere, where a
    # narrow multiplication shares its wave.
    floor = lane_config.MIN_LANES
    if context.target.prefs.lanes_down_to_wave:
        floor = max(floor, getattr(hw, 'vec_unit_length', 1))
    # Under the explicit vector, the deduced width alone.  The scorers do not
    # rank these: over the twenty elastic kernels on pvc, against the fastest
    # of 8, 16 and 32, the implemented order picks the best 3 times at a
    # geomean loss of 1.404 -- exactly what "always take the narrowest" does,
    # and so does every variant tried (issue alone, issue without the spill
    # price, the cliff or the register footprint in front of it).  Taking the
    # widest every time, which knows nothing, is 11 of 20 at 1.127.
    #
    # A floor at the vector unit looks better on that evaluation (7 of 20 at
    # 1.097) and is worse on the clock: tuned, it runs at 0.99x of the SPMD
    # default where pinning the deduced width is 1.14x, against 1.09x untuned
    # and 3 % of run-to-run noise.  The evaluation weighs one axis with
    # everything else held still; the walk turns `k_roll` and `k_width`
    # around whatever width it took first, and a bad first step is not a bad
    # step alone.  So the axis goes, not its lower end.
    #
    # The widths differ by spilling and the modelled footprint cannot see it:
    # `o6d:localFluxAll` is 79396 B at sixteen lanes and 79760 at 32, half a
    # percent apart, while the build spills 0 times and 1376.  What would
    # replace this is the compiled scorer's own figure.  What would earn the
    # narrow end back is packing: a `simd` of eight on a sixteen-wide unit uses
    # half of it, and four such multiplications side by side in one 32-wide
    # register would be the same work at full occupancy.  Nothing does either
    # today.
    explicit = context.target.explicit_simd
    if explicit and mode != 'compiled':
        # Nothing else can rank these.  The modelled footprint counts the
        # bytes of the tile and those barely move -- half a percent between
        # sixteen lanes and 32 on `o6d:localFluxAll`, where the clock is
        # threefold apart -- because what separates them is not how much the
        # tile holds but how it is cut: a `simd<double,32>` occupies four
        # register banks in a row where a `simd<double,16>` occupies two, and
        # the allocator gives up on the first where it places the second.
        # Only a build sees that, so only the compiled scorer gets the axis,
        # and there only as an escape from spilling (`CompiledScore`).
        geometries = [LaneConfig(base.num_threads, base.num_active_threads,
                                 base.lead_width)]
    else:
        geometries = []
        t = base.num_threads
        while t >= floor:
            geometries.append(LaneConfig(t, base.num_active_threads,
                                         base.lead_width))
            if t & (t - 1):
                break
            t //= 2
        if mode == 'compiled':
            geometries += _wider(base, rows, hw)
    if (context.fp_type == Datatype.F32
            and context.target.packed_fma_width(Datatype.F32) == 2
            and context.target.lead_vectors() and rows % 2 == 0
            and base.lead_width == 1):
        # Only where a lane holds one pair per column, where the target's
        # compiler spills the rest (`Preferences.one_pair_per_lane`).  ptxas
        # does not: 16 lanes at width two was GB200's fastest at 80 and 120
        # rows.
        one_pair = context.target.prefs.one_pair_per_lane
        geometries += [LaneConfig(g.num_threads, base.num_active_threads, 2)
                       for g in list(geometries)
                       if g.num_threads <= rows
                       and (not one_pair or 2 * g.num_threads >= rows)]
    knobs = [Knob('lanes', lambda c, g=tuple(geometries): g)] if len(geometries) > 1 else []
    # Two elements of the batch in one multiplication, where a single element
    # leaves the target's matrix instruction half empty
    # (`Preferences.tune_stack`) and the list stacks at all; and the width of
    # the matrix instruction, where the target has several for a product.  The
    # count of passes takes the narrowest unless the columns fill the wider
    # one, and the wider one reads the lead operand a quarter as often --
    # ahead where the loads bind and behind where the arithmetic does, which
    # the kernel's busiest pipe decides and no count of one product can
    # (`Options.matrix_columns`).  The bound is per element (`pipeline.of`), so
    # a stack is weighed against a plain build in the same unit; each width is
    # a choice of its own, the count's mix of them being the origin, and the
    # stack only its other side.
    #
    # Under the compiled scorer alone, which sees what they cost in
    # registers.  A stack holds twice an element's operands in a lane and the
    # wide tile gathers its accumulator back, and the fitted estimate
    # (`register_estimate`) follows neither: ranked statically over SeisSol's
    # kernels on gfx942, the walk took stacks at order 8 that clang then
    # spilled to scratch -- `localFluxAll` from nothing to 1728 bytes a lane,
    # `volume` from nothing to 624.  The compiled scorer ranks residency ahead
    # of the bound, which on gfx942 often takes the walk to eight lanes and,
    # past the margin (`_worth_it`), back to the default: a stack it keeps has
    # won on both.
    if mode == 'compiled':
        if context.target.prefs.tune_stack and _stackable(descrs):
            knobs.append(Knob('stack_elements', lambda c: (2,)))
        widths = _matrix_widths(descrs, context)
        if any(widths(Candidate(lanes=g)) for g in geometries or [base]):
            knobs.append(Knob('matrix_columns', widths))
    if _mergeable(descrs, context):
        knobs.append(Knob('merge_variants', lambda c: (False, True)))
    # Unrolled, rolled by the largest divisor, and rolled by two.  The first
    # two are the ends `_roll_values` picks; the third is the shortest body
    # the option can leave, which no divisor offers where the contraction is
    # prime to it.  `elastic-o6s:derivative` contracts over 55, so the
    # divisors are 11 and 5: on a GH200 it runs at 21.2 ns/el rolled by two,
    # 23.2 unrolled and 25.7 rolled by four, and the value that wins is not a
    # divisor at all.
    rolls = _roll_values(descrs)
    rolls = rolls[:2] + [r for r in rolls[2:] if r == 2]
    if len(rolls) > 1:
        knobs.append(Knob('k_roll', lambda c, r=tuple(rolls): r))
    # And unrolled whole where `k_unroll_max` would roll.  Rolled, a reduction
    # reads its operands from memory at every step; SeisSol's elastic time
    # derivative at order 8 (K = 119, rolled by 17 under the cap) ran 30 %
    # faster whole on sm_120, 60k lines against 25k.  The cap stays the
    # default, which is the origin the walk starts from -- `coordinate` sets a
    # value as it is, so the knob offers only the other side.
    cap = context.get_user_options().k_unroll_max
    lengths = contraction_lengths(descrs)
    if cap and lengths and max(lengths) > cap:
        knobs.append(Knob('k_unroll_max', lambda c: (0,)))
    # Staging the batch-constant operators once per block, where a measurement
    # says both answers are live.  Only with something to stage: with no
    # `Addressing.NONE` operand the option changes nothing and the trial is a
    # duplicate build.
    #
    # Not under the explicit vector, where it is the default and worth 1.71x.
    # A tuned build is *worse* there than the untuned one -- 0.85x of the SPMD
    # default against 1.14x -- so the walk gives away more than it finds, and
    # what it gives away is not this: taking the knob out of the space moved
    # the tuned figure from 0.88x to 0.85x and the untuned one not at all.
    # The damage is in the other knobs under this lowering, the lane count
    # above all: the walk loses `o6d:derivative` 59.8 -> 112.0 ns an element
    # and `o4s:derivative` 4.5 -> 10.5, which is the size and the direction of
    # the vector-width difference measured on the same kernels.
    #
    # So this is a stay of execution and not a verdict on the option: with
    # nothing to gain here it is one build per kernel spent on a decision the
    # default already makes correctly.  It belongs in the space once a tuned
    # ESIMD build beats its own default, which is a measurement and not an
    # opinion.
    if (context.target.prefs.tune_preload_globals
            and any(t.addressing == Addressing.NONE for t in _tensors(descrs))):
        knobs.append(Knob('preload_globals', lambda c: (False, True)))
    # Two reduction steps per body, so an operand contiguous along the
    # reduction is read once for both.  What it removes is mostly *address*
    # arithmetic rather than loads -- SeisSol's damage step goes from 40 % of
    # its instruction mix in `int` to 19 %, and runs 17 % faster -- and what it
    # costs is a larger body, which is why it is measured per kernel rather
    # than turned on: over 78 corpus cases it is a geomean of 0.991, 14 faster,
    # 62 within 2 %, and two slower (a lead window spanning two blocks by 13 %).
    #
    # Only where a reduction is long enough to have whole groups to pack, and
    # only the one value: 1 is the default the walk starts from, and
    # `coordinate` sets a knob's value as it is given.
    if lengths and max(lengths) >= 4:
        knobs.append(Knob('k_width', lambda c: (2,)))
    return knobs


def start(descrs, context: Context) -> Candidate:
    """The configuration the generator would pick unasked: the lanes it
    builds with -- the target's count where it has one
    (`Preferences.lanes_per_mult`, sixteen under DPC++ on Intel), the
    deduced ones otherwise -- and every option at the base context's value.

    The deduction alone is not the default everywhere: on the Max 1550 it is
    64 lanes for order-6 `volume`, which the generator builds at sixteen, and
    a walk that starts there measures its margin against a build nobody
    gets."""
    flat = _flat(descrs)
    base = (lane_config.requested(flat, context)
            or lane_config.deduce(flat, context))
    return Candidate(lanes=base)


def preferred(pref: preferences.Preference, descrs,
              context: Context) -> Candidate:
    """The configuration a measured preference asks for, on this kernel."""
    base = lane_config.deduce(_flat(descrs), context)
    asked = dict(pref.prefer)
    lanes = LaneConfig(
        num_threads=int(asked.pop('lanes', base.num_threads)),
        num_active_threads=base.num_active_threads,
        lead_width=int(asked.pop('width', base.lead_width)))
    if asked.get('k_roll') == 'auto':
        depths = contraction_lengths(descrs)
        k = max(depths) if depths else 0
        divisors = [d for d in range(2, min(32, k - 1) + 1) if k and k % d == 0]
        asked['k_roll'] = divisors[-1] if divisors else 0
    return Candidate(lanes=lanes, options=tuple(sorted(asked.items())))


def enumerate_space(knobs: Sequence[Knob], origin: Candidate) -> Iterable[Candidate]:
    """Every point, conditional knobs included.  For a small space.  A knob
    with no values at a point leaves it as it is, as None does."""
    def expand(i, cand):
        if i == len(knobs):
            yield cand
            return
        for v in knobs[i].values(cand) or (None,):
            yield from expand(i + 1, cand if v is None else cand.set(knobs[i].name, v))
    yield from expand(0, origin)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

@dataclass
class Build:
    """One candidate, generated: the generator on success, the exception
    otherwise."""
    candidate: Candidate
    generator: Any = None
    context: Optional[Context] = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build(descr_factory, base: Context, candidate: Candidate,
          generate) -> Build:
    """Generate `candidate` from a fresh descriptor list, with `generate`.

    Fresh every time, and not as a convenience: preparing an operand stores
    its order on the `Tensor` itself, so a build at eight lanes leaves an
    eight-lane interleave behind for the next build to inherit.
    """
    # Tuning off for the trial: it asks what *this* candidate costs, so it
    # builds this one and does not go looking for another.  With a knob pinned
    # the tuner does not stand aside when a geometry is given, so a trial that
    # kept `autotune` would open a walk of its own, once per trial, all the
    # way down.
    ctx = candidate.context(base, autotune='off')
    try:
        gen = generate(descr_factory(), ctx, candidate.lanes)
    except Exception as exc:
        return Build(candidate, context=ctx, error=exc)
    return Build(candidate, generator=gen, context=ctx)


def kernel_source(result: Build) -> str:
    """The kernel as a translation unit a compiler can take on its own."""
    gen, ctx = result.generator, result.context
    headers = ctx.target.headers() + gen.get_helper_headers()
    lines = [f'#include {h}' if h.startswith('<') else f'#include "{h}"'
             for h in headers]
    return '\n'.join(lines) + '\n' + gen.get_kernel()


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #

def _geometry(result: Build) -> Tuple[int, int, int]:
    """`(lanes, wave, resident multiplications per SM from what is exact)`."""
    gen = result.generator
    hw = result.context.target.hw
    mults = gen.launch_config().mults_per_block
    return (gen.lanes.num_threads, hw.vec_unit_length,
            (gen.resident_blocks or 0) * mults)


def static_score(result: Build):
    """What the build alone says, per multiplication rather than per block.

    Past the register file first (`_over_budget`) where a spill is a cliff
    (`pipeline.spills_are_a_cliff`; not on CDNA), but only as *whether* and
    not by how much: a build that spills is worse than one that does not, and
    past that the size of the overshoot is not a better predictor than the
    clocks it costs.  So the overshoot is also priced into the issue estimate
    (`_least_cycles(spill_bytes=...)`, which counts a spill as the store, the
    load back and the bytes of both) and the two keys say different things --
    the first that it spills at all, the second what that is worth against
    everything else.

    Measured against ranking the overshoot by size ahead of the issue estimate
    (2026-09-20): over twenty elastic kernels on pvc with the lane count free
    it is a wash -- 15 of 20 picks against 14, geomean 1.079 against 1.089 --
    and over the ten where `preload_globals` is the question it is not: 3 of 10
    against 8, geomean 1.822 against 1.035.  Staging the operators into shared
    memory moves bytes out of the register file, which is exactly the quantity
    that first key would sort on, so it would decide that axis by itself and
    always the same way.  Taken together, 1.285 against 1.071.  The same shape
    wins on the compiled scorer's own corpus (`~/tf/beast/spillrank.py`, 72
    workloads on sm_90 and sm_80): the cliff *and* the penalty, at 9.6 % mean
    against 10.1 % for the cliff alone and 9.7 % for the penalty alone.  A
    cliff that keeps its size but ranks behind the issue estimate picks the
    same on either axis (15 of 20, 3 of 10), so what decides is the precedence
    and not the granularity.

    Then multiplications resident per SM -- blocks times the multiplications a
    block holds, since eight lanes put four times as many in a block as 32 --
    counted in elements, which is what a stack holds two of
    (`Options.stack_elements`).

    Only then how far the body is past the instruction cache
    (`_icache_over`), which is where measuring puts it (2026-09-17,
    ~/tf-probe/tune_order.py, 100 items on sm_120, geomean of the pick against
    the default: 0.911 here against 0.973 with the cache ahead of everything).
    The case for ranking it second, as a cliff, is that a body that does not
    fit is fetched again on every iteration of the batch loop.  The cliff is
    real where one candidate fits and another does not -- which is what
    merging local_flux decides -- but ahead of the issue estimate it also
    decides between two candidates that both overflow, and there the smaller
    one is not the faster one.  SeisSol's elastic time derivative at order 8:
    rolled by 17 it is 230 kB past the cache and whole 625 kB, and whole is
    28 % faster.  The same key, ranked second, would pick `k_roll=11` for the
    order 6 derivative, which is 75 % slower than the default.  Ranking it as
    a cliff only (fits or not) fixes the rolling but keeps three of four such
    picks unmade -- 0.920 -- so the excess ranks after the issue estimate and
    nothing ranks before it.

    Then the modeled register footprint, in granules of sixteen registers
    (`_GRANULE`).  Last the length of the source: where nothing else differs,
    the smaller kernel -- merged, on every measurement taken (GB200's winners,
    sm_120 by 5 to 24 %, and fewer spills from hipcc).
    """
    if not result.ok:
        return None
    gen = result.generator
    lanes, wave, resident = _geometry(result)
    blocks = _register_blocks(result)
    if blocks is not None:
        mults = gen.launch_config().mults_per_block
        resident = min(resident, blocks * mults)
    over = _over_budget(result)
    hw = result.context.target.hw
    issue = _least_cycles(gen, lanes, wave, spill_bytes=float(over),
                          resident=resident, hw=hw)
    from tensorforge.analysis import pipeline
    cliff = over > 0 and pipeline.spills_are_a_cliff(hw)
    return (cliff, issue, -resident * _stack_of(gen),
            _icache_over(result), _granule(gen.metrics.peak_pressure or 0),
            len(gen.get_kernel() or ''))


def _least_cycles(gen, lanes: int, wave: int, spill_bytes: float = 0.0,
                  resident: int = 0, hw=None) -> float:
    """The busiest pipe's clocks per multiplication (`analysis.pipeline`), or
    the arithmetic's warp issue slots where the build counted no mix --
    stretched by however far the candidate falls short of keeping the issue
    ports fed.

    `spill_bytes` is what a compiler reported for this build, written into
    the same tables as every other access (`pipeline.spilled`): a spill costs
    a store, a load back and the bytes of both, and counting it here is what
    lets it be weighed against the issue count rather than ranked before it.

    The stretch is what makes the bound rankable.  `pipeline.of` is a lower
    bound on one multiplication given the whole SM, and it is an honest one;
    what it does not say is how close the machine can come to it, and that
    differs between candidates by more than the bound itself does.  A
    multiplication spread over more lanes is more warps, and its pipe
    occupancy rises with them -- so the bound grows with the width while the
    clock falls, and ranking by the bound alone takes the narrowest every
    time.

    `poroelastic-stp` order 8 in double precision is the case that shows it.
    One multiplication of 32 lanes is **one warp** on an SM whose shared
    memory admits one block, and a single warp cannot saturate an FP64 pipe
    that takes a warp instruction per clock, because its instructions depend
    on one another.  Measured against the bound: 13.8 % of it at 32 lanes,
    6.1 % at 64 and 2.6 % at 128, so the bound gets looser exactly as the
    kernel gets faster.

      lanes   bound   x shortfall   stretched      clock
         32  146006            x4      584024   20161 ns
         64  207662            x2      415324   12617
        128  339960            x1      339960    8927

    and `poroelastic-stp` order 4 in single precision, where 20 to 64 warps
    are already resident, keeps every factor at one and so keeps its order:
    9693, 19405, 38797 against 72, 122 and 238 ns an element.

    The shortfall is measured against the issue ports themselves -- the rate
    of the `issue` resource, which is the SM's warp schedulers (four on
    NVIDIA since Volta) -- rather than against a fresh constant.  Where the
    caller knows neither the residency nor the hardware, nothing is
    stretched and this is the bound itself.
    """
    from tensorforge.analysis import pipeline
    b = pipeline.of(gen, spill_bytes=spill_bytes)
    cycles = (b.cycles if b is not None
              else (gen.metrics.emitted_work or 0) * lanes / wave
              / _stack_of(gen))
    return cycles * _shortfall(lanes, wave, resident, hw)


def _stack_of(gen) -> int:
    """The elements one multiplication of `gen` computes
    (`Options.stack_elements`); the figures per multiplication are divided by
    it, as `pipeline.of` divides its own."""
    return max(1, int(getattr(gen, 'stack', 1) or 1))


def _shortfall(lanes: int, wave: int, resident: int, hw) -> float:
    """How much of the bound the machine cannot reach for want of warps.

    One where the resident warps meet what the target needs to keep its
    pipes fed (`pipeline.latency_waves`), and the ratio short of it
    otherwise.  `resident` of zero means the caller does not know, and an
    unknown residency stretches nothing: guessing here would rank by a number
    nobody measured.

    A multiplication narrower than the wave is that share of a wave, not a
    whole one: four of sixteen lanes fill one wave of 64.  Counting each as a
    wave took a CU holding four 64-lane multiplications of SeisSol's order-6
    `localFluxAll` (one wave a SIMD) for as fed as one holding 24 of 32 lanes,
    and ranked it ahead at 79 % slower (gfx942).
    """
    if not resident or hw is None:
        return 1.0
    from tensorforge.analysis import pipeline
    target = pipeline.latency_waves(hw)
    if not target:
        return 1.0
    waves = resident * lanes / max(1, wave)
    return max(1.0, target / waves)


def _icache_over(result: Build) -> int:
    """Kilobytes of code past the instruction cache (`analysis.icache`).

    Right after the register file and before occupancy: a body that does not
    fit is fetched again on every iteration of the batch loop, which no number
    of resident multiplications makes up for.  A cliff and not a slope, so 0
    below the capacity whatever the size, and every kilobyte past it counts.
    """
    from tensorforge.analysis.icache import icache_excess
    hw = result.context.target.hw
    return icache_excess(result.generator.metrics.code_units, hw) // 1024


#: Bytes below which two modeled footprints are the same footprint: sixteen
#: registers.  The model is within about forty registers of what a compiler
#: allocates, so a difference of eleven bytes -- merged and written-out
#: `local_flux` at b = 80, 1359 against 1348 -- is not one the compiler makes;
#: hipcc spilled 6 KB for the written-out one and nothing for the merged.
_GRANULE = 64


def _granule(value) -> int:
    return int(value // _GRANULE)


def register_estimate(result: Build) -> Optional[float]:
    """Registers per lane the target compiler is expected to allocate
    (`Preferences.register_fit`)."""
    fit = result.context.target.prefs.register_fit
    peak = result.generator.metrics.peak_pressure
    if fit is None or not peak:
        return None
    return fit[0] + fit[1] * peak / 4


def _register_blocks(result: Build) -> Optional[int]:
    """Blocks per CU the register file admits, where that is what decides
    (`Preferences.rank_by_register_waves`).

    AMD: a CDNA lane has 512 registers, VGPRs and AGPRs together, and a SIMD
    holds as many waves as fit -- eight at 64 registers, one above 256.  On
    gfx942 over `local_flux` a third of the geometries sat at one wave per
    SIMD without spilling a byte, which blocks per CU from shared memory and
    threads alone never shows.  NVIDIA keeps what is exact: its fit has an
    intercept that the occupancy would inherit, and the eight lanes that GB200
    ran fastest sit right at the limit.
    """
    if not result.context.target.prefs.rank_by_register_waves:
        return None
    regs = register_estimate(result)
    if regs is None:
        return None
    gen = result.generator
    threads = gen.lanes.num_threads * gen.launch_config().mults_per_block
    return register_blocks(result.context, regs, threads)


def register_blocks(context: Context, regs: float, threads: int) -> int:
    """Blocks of `threads` a CU holds at `regs` registers a lane: the waves
    a SIMD's file takes at that allocation, four SIMDs, whole blocks
    (`_register_blocks`)."""
    hw = context.target.hw
    file = (hw.max_reg_per_thread or 1024) // 4
    granule = 8
    per_lane = max(granule, -(-int(regs) // granule) * granule)
    waves_per_simd = max(0, min(8, file // per_lane))
    waves_per_block = max(1, -(-threads // hw.vec_unit_length))
    return (4 * waves_per_simd) // waves_per_block


def _over_budget(result: Build) -> float:
    """How far the modeled footprint is past the register file a thread
    has, or 0 where it fits -- an amount and not a verdict, because where
    every candidate is past it (`local_flux` at b = 120 on gfx942, all of
    them spilling) the one past it least is the one that spills least: 880 B
    of scratch at 32 lanes against 11 KB at eight, which a yes/no would leave
    to the next key to decide the wrong way.

    First, before anything is ranked: a build that spills is slower than any
    difference the other keys can see.  Calibrated against ptxas on sm_100a
    (`local_flux`, 79 builds), registers come out at about 51 + 1.05 times the
    modeled bytes over four, and every build above the 1020 B a thread has
    there spilled -- eight lanes at b = 80 and 120, which GB200 then ran 75 %
    and 148 % behind the default.  Not a spill predictor in the other
    direction: ptxas also spills at 168 registers where it chooses occupancy,
    and nothing in the model says when.  Where the target states no budget,
    there is no guard.
    """
    target = result.context.target
    budget = getattr(target.hw, 'max_reg_per_thread', None)
    peak = result.generator.metrics.peak_pressure
    if not (budget and peak):
        return _over_scalar_budget(result)
    if target.prefs.budget_by_fit:
        # hipcc allocates about 1.26 registers per modeled four bytes, so the
        # byte budget alone would let eight lanes at b = 56 through (2449 B
        # against 2048) that gfx942 spills 2 KB for.  In bytes, like the rest.
        #
        # Against the *whole* figure and not the lane-varying part of it,
        # although the vector file is what this budget is: the fit was taken
        # over totals, and feeding it a smaller number without refitting moves
        # every estimate down by however much of a kernel is uniform.  The
        # scalar file gets its own term instead.
        return (max(0.0, 4 * register_estimate(result) - budget)
                + _over_scalar_budget(result))
    if target.lanes_share_register_file():
        # The file is a *thread's*, and under SPMD one thread holds the whole
        # sub-group: the budget a lane may spend is the file divided by the
        # lanes that share it, or -- the same statement the other way up --
        # what has to fit is the lane's footprint times the sub-group.
        #
        # Compared against the whole file, as every other target is, the guard
        # would never fire on Intel at all: `elastic-o6s:derivative` models
        # 1716 B a lane against 8192, and IGC spills it hard.  Per thread it is
        # 54.9 KB at 32 lanes against 31.6 at sixteen -- 6.7 and 3.9 times the
        # file -- and the IGC dump agrees about which of the two that hurts:
        # 1108 scratch messages against 9, which is the whole difference in
        # sends between the two builds.
        lanes = result.generator.lanes
        share = lanes.num_threads if lanes else 1
        return max(0, peak * share - budget) + _over_scalar_budget(result)
    return max(0, peak - budget) + _over_scalar_budget(result)


def _over_scalar_budget(result: Build) -> float:
    """How far the wave-uniform values are past the scalar register file.

    A second file and a second budget, on the targets that have one: AMD keeps
    what a whole wave agrees on in SGPRs, about a hundred of them per wave, and
    runs it on a pipe of its own.  A model with one number cannot see that file
    fill -- SeisSol's damage step at order 4 on gfx1150 allocates 107 SGPRs and
    spills 117 more while its VGPRs are also full.

    A floor rather than an estimate, and the difference is stated because it is
    large: the compiler also keeps addresses, loop counters and the kernel's
    own arguments there, and `pir.pressure` counts none of them (an address
    offset is an immediate -- see `_affine_costs`).  The same damage step
    models 336 B of 424 B here.  So it catches a body whose *values* alone
    overflow the file, which is what a scorer can act on, and stays quiet where
    the overflow is the compiler's own bookkeeping.
    """
    if not result.ok:
        return 0.0
    hw = result.context.target.hw
    budget = getattr(hw, 'max_scalar_reg_per_wave', None)
    peak = result.generator.metrics.peak_uniform_pressure
    if not (budget and peak):
        return 0.0
    return max(0.0, peak - budget)


class CompiledScore:
    """Registers and spills from the target's own compiler.

    The static model cannot say whether a configuration spills: on GB200 it
    gave eight lanes at b = 56 its highest figure and ptxas no spill, and at
    b = 80 a low one and ptxas 328 bytes.  The compiler can, so where one is
    available its figure is what the spilling is taken from -- and as
    *traffic*, written into the same tables as every other access, so that
    how much is spilled is weighed against what it buys: whether anything
    spills, then code past the instruction cache, then multiplications
    resident per SM under shared memory, threads *and* registers (in
    elements, two to a stack), then the busiest pipe's clocks with the
    spilling in them.

    Which of those the clock agrees with is measured, not argued.  Over the
    72 (device, kernel) groups of the benchmark corpus that carry three or
    more configurations, as loss against the measured best
    (`tools/../beast/spillrank.py`):

        rule                                   mean   median   picked best
        the default build                     20.3 %    9.7 %      31 %
        spilled bytes as a tier of their own  10.1 %    0.8 %      49 %
        the bound with the spilling in it      9.7 %    1.4 %      43 %
        the bound alone, spilling and all     15.2 %    2.9 %      39 %
        the bound alone, no spilling in it    25.3 %    3.3 %      38 %
        this one                               9.6 %    0.8 %      49 %

    The two halves are doing different work: "does it spill" is a cliff and
    belongs in front, the bytes are a slope and belong in the figure they
    compete with.
    """

    def __init__(self, toolchain: Optional[Toolchain] = None,
                 flags: Sequence[str] = (), timeout: float = 600):
        self.toolchain = toolchain or Toolchain()
        self.flags = tuple(flags)
        self.timeout = timeout
        self.reports: Dict[Candidate, Resources] = {}

    def available(self, context: Context) -> bool:
        return self.toolchain.compiler(context.target.hw.vendor) is not None

    def resources(self, result: Build) -> Optional[Resources]:
        hw = result.context.target.hw
        compiler = self.toolchain.compiler(hw.vendor)
        if compiler is None:
            raise RuntimeError(
                f'no compiler for {hw.vendor}: pass one in `Toolchain`, or set '
                f'TF_NVCC / TF_HIPCC')
        entry = toolchain.compiler_for_vendor(hw.vendor)
        with tempfile.TemporaryDirectory(prefix='tf-tune-') as tmp:
            src = os.path.join(tmp, 'kernel' + entry.suffix)
            with open(src, 'w') as f:
                f.write(kernel_source(result))
            cmd, product = entry.report_build(compiler, src, tmp, hw.model,
                                              self.toolchain.include_dir())
            run = subprocess.run(cmd + list(self.flags), capture_output=True,
                                 text=True, timeout=self.timeout)
            spilled = entry.reported_spill(product)
            code = entry.code_bytes(product)
        report = toolchain.resources(entry.backend, run.stdout + run.stderr)
        if report is None or run.returncode:
            return None
        if code is not None:
            report = replace(report, code_bytes=code)
        if spilled is not None:
            # The object's own figure, and it decides: the console heuristic
            # reads a recompilation as "spilled, amount unknown", and IGC
            # recompiles for other reasons too -- the 32-lane
            # `neighboringFlux` build retries and spills nothing, which that
            # heuristic would report as a spill for the ranking to act on.
            report = replace(report, spill_bytes=spilled)
        if report.registers:
            gen = result.generator
            threads = (gen.lanes.num_threads
                       * gen.launch_config().mults_per_block)
            blocks = entry.register_blocks(report.registers, threads, hw)
            if blocks is not None:
                report = replace(report, register_blocks=blocks)
        self.reports[result.candidate] = report
        return report

    def __call__(self, result: Build):
        if not result.ok:
            return None
        report = self.resources(result)
        if report is None:
            return None
        lanes, wave, resident = _geometry(result)
        hw = result.context.target.hw
        mults = result.generator.launch_config().mults_per_block
        if (report.register_blocks is not None
                and toolchain.compiler_for_vendor(hw.vendor).blocks_from_registers):
            blocks = min(result.generator.resident_blocks or 0, report.register_blocks)
            resident = blocks * mults
        # The *amount* spilled goes into the issue figure rather than in
        # front of it (`pipeline.spilled`): it is a store, a load back and
        # the bytes of both, and ranking bytes before every issue count
        # cannot trade 32 of them against a reduction rolled by two.
        # Whether anything spills at all stays in front, because over 72
        # measured (device, kernel) groups that is what the clock agrees
        # with -- see the table in the class docstring.
        issue = _least_cycles(result.generator, lanes, wave,
                              spill_bytes=report.spill_bytes,
                              resident=resident, hw=hw)
        if result.context.target.explicit_simd:
            # Widest unless it spills.  Under the explicit vector the lane
            # count is the vector's width, and a narrower one does strictly
            # less work per instruction; the only thing it buys is a value
            # the allocator can place.  Where nothing spills there is nothing
            # to buy, and letting the rest of the tuple decide loses 16 %
            # over the twenty elastic kernels on pvc --
            # `o6s:derivative` 15.92 -> 35.10 ns an element, `o6s:localFluxAll`
            # 11.02 -> 24.34, neither of which spills at either width, while
            # `o6d:localFluxAll` spills 10688 B at 32 lanes and none at
            # sixteen and wants the move (99.58 -> 46.08).
            return (report.spill_bytes > 0, -lanes, _icache_over(result),
                    -resident * _stack_of(result.generator), issue)
        # Residency in elements: a stack is one multiplication of two.
        return (report.spill_bytes > 0, _icache_over(result),
                -resident * _stack_of(result.generator), issue)


class MeasuredScore:
    """A caller's measurement: `run(result)` returns the seconds the whole
    batch took, or None where it could not take one.  What a real autotuner
    plugs in -- this module does not launch anything.

    Told the device -- `sms`, `clock_ghz` -- and the `batch` it measures at,
    it prunes: a candidate whose least time (`analysis.pipeline`, the
    conservative bound, stretched by the persistent grid's last round) is
    longer than the best measured so far cannot be better, and is not run.
    That is a proof and not a guess, which is what makes it safe to skip the
    measurement; it scores `inf` and is listed in `pruned`.
    """

    def __init__(self, run: Callable[[Build], Optional[float]],
                 sms: Optional[int] = None, clock_ghz: Optional[float] = None,
                 batch: Optional[int] = None):
        self.run = run
        self.sms, self.clock_ghz, self.batch = sms, clock_ghz, batch
        self.best: Optional[float] = None
        self.pruned: List[Candidate] = []

    def least(self, result: Build) -> Optional[float]:
        """The least seconds `result` can take, or None where unknown."""
        if not (self.sms and self.clock_ghz and self.batch):
            return None
        from tensorforge.analysis import pipeline
        gen = result.generator
        b = pipeline.of(gen, conservative=True)
        if b is None:
            return None
        efficiency = 1.0
        if gen.resident_blocks:
            # The rounds are counted in multiplications, and one of a stacked
            # build takes a stack of the batch.
            stacks = -(-self.batch // _stack_of(gen))
            efficiency = pipeline.persistent_efficiency(
                stacks, gen.launch_config().mults_per_block,
                gen.resident_blocks, self.sms)
        return pipeline.least_seconds(b, self.batch, self.sms,
                                      self.clock_ghz, efficiency)

    def __call__(self, result: Build):
        if not result.ok:
            return None
        from tensorforge.analysis import pipeline
        least = self.least(result)
        if (least is not None and self.best is not None
                and pipeline.prunable(least, self.best)):
            self.pruned.append(result.candidate)
            return float('inf')
        measured = self.run(result)
        if measured is not None and (self.best is None or measured < self.best):
            self.best = measured
        return measured


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

@dataclass
class Trial:
    candidate: Candidate
    score: Any
    error: Optional[BaseException] = None


@dataclass
class Outcome:
    best: Candidate
    score: Any
    trials: List[Trial] = field(default_factory=list)


def _evaluate(descr_factory, context, scorer, candidate, cache, generate):
    if candidate in cache:
        return cache[candidate]
    result = build(descr_factory, context, candidate, generate)
    if not result.ok:
        # Not the scorer's to judge: a build that failed has no figure, and a
        # scorer written for successes should not have to know that.
        trial = Trial(candidate, None, result.error)
    else:
        try:
            trial = Trial(candidate, scorer(result))
        except Exception as exc:
            trial = Trial(candidate, None, exc)
    cache[candidate] = trial
    return trial


def _better(a, b) -> bool:
    return b is None or (a is not None and a < b)


def exhaustive(descr_factory, context: Context, scorer,
               knobs: Optional[Sequence[Knob]] = None,
               origin: Optional[Candidate] = None, *, generate) -> Outcome:
    """Every point.  Only for a space small enough to build whole."""
    descrs = descr_factory()
    knobs = space(descrs, context) if knobs is None else knobs
    origin = start(descrs, context) if origin is None else origin
    cache: Dict[Candidate, Trial] = {}
    best = None
    for cand in enumerate_space(knobs, origin):
        trial = _evaluate(descr_factory, context, scorer, cand, cache,
                          generate)
        if _better(trial.score, None if best is None else best.score):
            best = trial
    return _outcome(best, cache, origin)


def coordinate(descr_factory, context: Context, scorer,
               knobs: Optional[Sequence[Knob]] = None,
               origin: Optional[Candidate] = None,
               rounds: int = 3, budget: Optional[int] = None,
               seed: Optional[Dict[Candidate, Trial]] = None, *,
               generate) -> Outcome:
    """One knob at a time, from the default, keeping whatever improves.

    A round costs the sum of the knobs' value counts rather than their
    product -- tens of builds instead of thousands -- and it misses a pair that
    only pays together.  Eight lanes without `k_roll` at b = 56 spills and with
    it does not, so the round is repeated until nothing moves.
    """
    descrs = descr_factory()
    knobs = space(descrs, context) if knobs is None else knobs
    origin = start(descrs, context) if origin is None else origin
    cache: Dict[Candidate, Trial] = dict(seed or {})
    best = _evaluate(descr_factory, context, scorer, origin, cache, generate)
    for _ in range(rounds):
        moved = False
        for knob in knobs:
            for value in knob.values(best.candidate):
                cand = best.candidate.set(knob.name, value)
                if (budget is not None and cand not in cache
                        and len(cache) >= budget):
                    continue
                trial = _evaluate(descr_factory, context, scorer, cand, cache,
                                  generate)
                if _better(trial.score, best.score):
                    best, moved = trial, True
        if not moved:
            break
    return _outcome(best, cache, origin)


def _outcome(best: Optional[Trial], cache, origin) -> Outcome:
    trials = list(cache.values())
    if best is None or best.score is None:
        failed = [t.error for t in trials if t.error is not None]
        if failed and len(failed) == len(trials):
            raise failed[0]
        return Outcome(origin, None, trials)
    return Outcome(best.candidate, best.score, trials)


def tune(descr_factory, context: Context, scorer=static_score,
         strategy: Callable = coordinate, *, generate, **kwargs) -> Outcome:
    """Choose a configuration for `descr_factory()`'s kernel on `context`'s
    target.  `descr_factory` returns a fresh descriptor list per call."""
    return strategy(descr_factory, context, scorer, generate=generate,
                    **kwargs)


# --------------------------------------------------------------------------- #
# Autotuning a kernel as it is generated
# --------------------------------------------------------------------------- #

#: Picks made in this process, by key (`_key`).  The file behind
#: `Options.autotune_cache` is read into it and written from it.
_PICKS: Dict[str, Dict[str, Any]] = {}
_LOADED: set = set()


def _key(result: Build, mode: str) -> str:
    """The default build's source and the target: what the pick depends on.

    The source rather than the descriptors, because it is the one thing that
    says everything -- shapes, addressing, the options already asked -- and
    it has been built anyway, as the walk's first point.
    """
    hw = result.context.target.hw
    sha = hashlib.sha256()
    for part in (mode, hw.model, result.context.target.backend, str(result.context.fp_type),
                 kernel_source(result)):
        sha.update(part.encode())
        sha.update(b'\0')
    return sha.hexdigest()


def _load(path: Optional[str]) -> None:
    if not path or path in _LOADED:
        return
    _LOADED.add(path)
    try:
        with open(path) as f:
            _PICKS.update(json.load(f))
    except (OSError, ValueError):
        pass


def _store(path: Optional[str], key: str, pick: Candidate) -> None:
    _PICKS[key] = pick.to_dict()
    if not path:
        return
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(_PICKS, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        warnings.warn(f'autotune: could not write {path}: {exc}')


def autotune(descr_factory, context: Context, mode: str = 'static',
             budget: Optional[int] = 24,
             cache: Optional[str] = None,
             fixed: Optional[Mapping[str, Any]] = None, *,
             generate) -> Optional[Candidate]:
    """The configuration `Options.autotune` builds a kernel with.

    One build of the default first: its source is the cache key, and a kernel
    already tuned costs nothing more.  Otherwise a coordinate walk over
    `simple_space`, the default seeded in, within `budget` builds.  None where
    the default does not build -- the generator then fails the way it would
    have, instead of this failing differently.

    `fixed` names what the caller has already decided: each one is set in the
    origin and its knob leaves the space, so the walk turns the rest around
    it.  A caller who states the geometry is stating that and not "do not
    tune" -- `elastic-o6s:derivative` is built at a geometry the benchmark
    harness passes explicitly, and with the whole space abandoned for it
    nothing would be tuned there.
    """
    descrs = descr_factory()
    # A measurement first: where one says what is best for this device and
    # this shape, it is taken as it is, whatever the scorers would say --
    # they rank what a build reports about itself, and the preference is what
    # the machine did.  One build, to check that it builds at all.
    pref = preferences.lookup(descrs, context)
    if pref is not None:
        chosen = preferred(pref, descrs, context)
        if build(descr_factory, context, chosen, generate).ok:
            return chosen
        warnings.warn(f'autotune: the preference from {pref.source} '
                      f'({pref.evidence}) does not build here; ranking instead')
    if mode == 'prefer':
        return None
    if mode == 'lanes':
        return _lanes_mode(descr_factory, context, cache, fixed, generate)
    # The scorer first: the space depends on which one ranks it.
    if mode == 'compiled':
        scorer = CompiledScore()
        if not scorer.available(context):
            warnings.warn('autotune=compiled: no compiler for this target '
                          '(Toolchain, TF_NVCC, TF_HIPCC); ranking statically')
            scorer, mode = static_score, 'static'
    elif mode == 'static':
        scorer = static_score
    else:
        raise ValueError(f'autotune: unknown mode {mode!r}; off, prefer, '
                         f'static, compiled or lanes')
    knobs = simple_space(descrs, context, mode=mode)
    origin = start(descrs, context)
    for name, value in (fixed or {}).items():
        origin = origin.set(name, value)
    knobs = [knob for knob in knobs if knob.name not in (fixed or {})]
    first = build(descr_factory, context, origin, generate)
    if not first.ok:
        return None
    if not knobs:
        return origin
    _load(cache)
    key = _key(first, mode)
    if key in _PICKS:
        return Candidate.from_dict(_PICKS[key])
    try:
        seed = {origin: Trial(origin, scorer(first))}
    except Exception as exc:
        seed = {origin: Trial(origin, None, exc)}
    # One build held back for the margin below, so that `autotune_budget`
    # stays what it says: the most builds this spends on one kernel.
    out = coordinate(descr_factory, context, scorer, knobs=knobs, origin=origin,
                     budget=max(1, budget - 1) if budget else budget, seed=seed,
                     generate=generate)
    best = _worth_it(descr_factory, context, out.best, origin, first, generate)
    _store(cache, key, best)
    return best


#: How much better the winner's bound has to be than the default's before its
#: configuration is taken instead.  A scorer ranks; it does not say by how
#: much, and a rank won on a tie-breaker is how a pick goes badly wrong.
DEVIATION_MARGIN = 0.05


def _bound_cycles(result: Optional['Build']) -> Optional[float]:
    """The busiest pipe's clocks per multiplication for a build, or None --
    stretched the way the ranking stretches it (`_least_cycles`).

    The same yardstick on purpose.  `_worth_it` asks whether the move the
    ranking chose is worth its margin, and a margin measured on a different
    figure does not check the decision, it overrules it with another one.
    `poroelastic-stp` order 8 in double precision is where that shows: ranked
    on the stretched bound the walk takes 128 lanes (339960 against 584025)
    and is right by 2.26x on the clock, while the raw bound would read 339960
    against 146006, call the move a threefold loss and put the default back
    -- as it would for every candidate the widening exists for.
    """
    if result is None or not result.ok:
        return None
    from tensorforge.analysis import pipeline
    bound = pipeline.of(result.generator)
    if bound is None:
        return None
    lanes, wave, resident = _geometry(result)
    return bound.cycles * _shortfall(
        lanes, wave, resident, result.context.target.hw)


def _worth_it(descr_factory, context: Context, best: Candidate,
              origin: Candidate, first: 'Build', generate) -> Candidate:
    """`best`, or the default where the bound says the move is not worth it.

    Ranking alone picks the winner of a tie-breaker as readily as a winner:
    `static_score` sorts by the register budget, the instruction cache and
    residency before it reaches the pipe bound, and two configurations that
    differ in none of those are then separated by figures that say nothing
    about time.  Over the measured corpus -- 72 kernels on A100 and GH200, 670
    timings -- following the rank costs 16.6 % against the best configuration
    on average and 210 % in the worst case, where the default alone costs
    20.3 % and 95 %.  Taken only past this margin it costs 10.8 % and 84 %:
    better than the default everywhere, including the tail.

    The margin is read off the bound and not off the score, because the bound
    is the one term in clocks.  It costs one build of the winner, which the
    walk does not keep.
    """
    if best == origin:
        return origin
    least = _bound_cycles(build(descr_factory, context, best, generate))
    was = _bound_cycles(first)
    if least is None or was is None:
        return best
    return best if least < (1.0 - DEVIATION_MARGIN) * was else origin


# --------------------------------------------------------------------------- #
# The lanes rule: the lane count and merging, with one compiler question each
# --------------------------------------------------------------------------- #

#: How much more a candidate may spill than the default build and still be
#: taken: never where the default spills nothing.  Over SeisSol's elastic
#: kernels at orders 6 and 8 on four GPUs (`autotune=lanes`), 1.0, 1.5 and 2
#: pick the same; what the guard keeps out is a narrower count that spills
#: where the default does not, which is where the cost below cannot be trusted
#: -- the same 212 B of spilling cost order-8 `volume` 47 % on A100 at sixteen
#: lanes, and 112 B of it gained 13 % on sm_120.
SPILL_GROWTH = 1.5

#: What a spilled value is worth against one executed instruction, per byte of
#: spill traffic ptxas reports and element of the data type, per warp.  A
#: weight rather than a model: any value from 48 to 512 picks the same over
#: the measured kernels, and 0 does not -- without it the guard alone lets
#: order-8 `localFluxAll` in double precision take sixteen lanes unmerged,
#: which spills within the default's 4.8 KB and runs 131 % behind the 64 lanes
#: unmerged that spill nothing (A100).
SPILL_WEIGHT = 64

#: What an instruction from code past the instruction cache costs on top of
#: one from inside it: this share of what the part's microbenchmark measured
#: beyond the one it costs anyway (`hw_descr.icache_miss_factor` less 1).  It
#: is charged on as many of the executed instructions as the code has past the
#: cache (`analysis.icache.past_share`), with the code as large as the
#: compiler wrote it: the emitter's estimate puts the build below at 176 kB,
#: ptxas wrote 258.
#:
#: A share and not the whole, because the probe's FFMA chains issue every
#: clock, while a kernel also waits on other things, and part of a fetch hides
#: behind them.  Any value from 0.44 to 0.50 picks the same over the measured
#: kernels.  What it moves:
#:
#: - Order-8 `localFluxAll` in single precision on GH200 and GB200 is merged
#:   again.  Written out at 32 lanes, its faces count 28 % fewer instructions
#:   than merged, but they are 258 and 243 kB of code and run 26 % slower on
#:   both.  The same build is 3 and 10 % faster on A100 and sm_120, whose
#:   factors leave it alone.
#: - Order-6 `localFluxAll` moves from sixteen lanes unmerged (204 to 216 kB)
#:   to sixteen merged, which is 3 to 4 % slower on those two.  The unmerged
#:   build runs at 1.07 to 1.11 times the default's time per instruction, not
#:   the 1.3 to 1.4 charged.
#:
#: Below 0.44 GB200 loses the order-8 move.  From 0.51 GH200 merges order-6
#: `localFluxAll` in double precision (7 % slower), and from 0.62 it takes 64
#: lanes for order-8 `volume` (24 % slower).  On the generator before merged
#: runs left out the union, the merged sixteen lanes counted 21 % higher, and
#: the order-6 move went to 32 lanes unmerged instead, 25 % slower on GH200.
ICACHE_SHARE = 0.47


def executed_instructions(result: Build) -> Optional[float]:
    """Warp instructions one element of the batch executes, by the emitter's
    own count (`pipeline.instructions_per_element`, summed): loops counted
    with their trips, which is the difference from the instructions a
    disassembly lists.  A merged run states its body once and runs it once
    per member; a reduction rolled by 28 states one trip.  Counted that way,
    the static listing puts 128 lanes ahead for order-8 `volume`, where they
    run 42 % (A100) to 145 % (sm_120) behind, and cannot say what merging
    does at all.  None where the build counted no mix."""
    from tensorforge.analysis import pipeline
    gen = result.generator
    mix = gen.metrics.issue_mix
    if not mix:
        return None
    per = pipeline.instructions_per_element(mix, result.context.target.hw,
                                            gen.lanes.num_threads)
    return sum(per.values()) / _stack_of(gen)


def pipe_bound(result: Build, spill_bytes: float = 0.0) -> Optional[float]:
    """Clocks one element of the batch keeps the busiest pipe of a CU busy
    (`pipeline.of`), with what the compiler spilled written into the same
    tables and stretched where the build holds too few waves to feed the
    pipes (`_least_cycles`).  None where the build counted no mix.

    What the rule ranks by on AMD, where a count of instructions does not say
    what they cost: an FP32 statement is 3.7 instructions on CDNA and 1.1 on
    RDNA (`pipeline.INSTRUCTIONS_PER_STATEMENT`), and the matrix and FP64 units
    run at rates of their own.  The bound weighs each category at its rate."""
    gen = result.generator
    if not gen.metrics.issue_mix:
        return None
    lanes, wave, resident = _geometry(result)
    return _least_cycles(gen, lanes, wave, spill_bytes=spill_bytes,
                         resident=resident, hw=result.context.target.hw)


#: What the lanes rule ranks by, per vendor: the instructions an element
#: executes (`executed_instructions`) or the pipe bound (`pipe_bound`).
#: Measured as the loss against the fastest candidate (mean / worst) over
#: SeisSol's elastic kernels -- eight to 64 lanes, merged and not, builds of
#: one source pooled; on AMD orders 6 and 8 in double, order 6 in single (12
#: kernels), on the Max 1550 orders 6 and 8 (16), sixteen to 64 lanes:
#:
#:     ranked by                     MI300A     MI210      Max 1550
#:     the default build             6.9 / 35   23.7 / 48   1.3 / 11 %
#:     instructions + spilling       8.4 / 58   96.6 / 232  0.7 / 11
#:     the pipe bound                2.2 /  9   61.4 / 227  3.6 / 32
#:     the bound, waves stretched    2.2 /  9    7.7 / 30
#:
#: With order 8 in single precision on MI300A too (16 kernels) the default
#: loses 8.2 / 37.5 % and the stretched bound 4.1 / 37.5, or 1.7 / 8.9 with
#: a build that spills nothing taken first (`CLEAN_FIRST`).
#:
#: On the NVIDIA grids of `lanes_rule` the bound loses 14 to 20 % on average
#: where the count loses 6 to 7: the bound takes the narrowest count wherever
#: the issue port binds.  On MI300A the count takes eight lanes for order-6
#: `volume` in single precision, 58 % slower than the sixteen the bound takes.
#: On the Max 1550 the sixteen lanes of `Preferences.lanes_per_mult` are
#: already within 1.3 %; the count moves one kernel, order-6
#: `neighboringFlux` in single precision, to 32 lanes (8 % faster: IGC
#: spills 11.8 kB at sixteen and nothing at 32).  The stretch is
#: `LATENCY_WAVES_PER_SIMD`'s.
RANKED_BY = {'nvidia': 'instructions', 'amd': 'bound',
             'intel': 'instructions'}

#: Waves a SIMD has to hold before an AMD part runs at the pipe bound, as the
#: lanes rule measured it: the bound of a build at fewer waves (the
#: compiler's occupancy, `Resources.register_blocks` on AMD) is stretched by
#: the shortfall.  One on MI300A, where any stretch costs (2 waves: 10.6 %
#: mean loss, 84 % worst), eight on MI210, which runs SeisSol's kernels as
#: if latency bound: stretched from three waves on, the rule's mean loss is
#: 10.8 % (7.7 % from six), against 61 % unstretched -- sixteen lanes of
#: order-6 `derivative` in single precision count 22 % fewer clocks than 32,
#: both at one wave, and run twice as long, while 64 lanes at two waves run a
#: third faster.  The code is the same (ROCm 7.2 emits the same instructions
#: for both, to within gfx942's DPP-hazard `s_nop`s); the memory below it is
#: not -- 8 MB of L2 against 256 MB of Infinity Cache -- which is probably
#: the difference.  An AMD part without an entry was never measured, and its
#: default stands.
LATENCY_WAVES_PER_SIMD = {'gfx942': 1, 'gfx90a': 8}

#: Where the default build spills on AMD, take the cheapest candidate that
#: spills nothing, if there is one, whatever the bound says of the spilling
#: ones.  The bound prices a spill as the traffic it adds, and at one wave a
#: SIMD that is not what it costs: order-8 `derivative` in single precision
#: on MI300A spills 1340 B at 32 lanes, which the bound charges 8 %, and runs
#: 38 % behind 64 lanes, which spill nothing and count 18 % more.  Over the
#: 16 kernels of MI300A (orders 6 and 8, single and double) the rule's loss
#: goes from 4.1 / 37.5 % (mean / worst) to 1.7 / 8.9 %; over MI210's 12 it
#: is 7.7 / 30 % either way.
CLEAN_FIRST = True

#: AMD parts, by `hw.model` prefix, whose SIMDs issue one instruction of a
#: wave per turn and send instructions of different kinds out together only
#: from different waves: CDNA.  At one wave a SIMD every instruction --
#: scalar, memory, branch, wait -- then takes the turn a vector instruction
#: would have, and the rule charges a build at least the instructions an
#: element executes over the waves the compiler gives it
#: (`executed_instructions` over `Resources.register_blocks`), where that is
#: above its bound.  Viscoelastic `fluxLocalAll` in double precision holds
#: one wave a SIMD at 16 and 32 lanes on MI300A: at order 4 sixteen lanes
#: are 3.0 % below the default's bound and 13.2 % below its count, and run
#: 20.6 % faster; at order 5 the unmerged 32 lanes are 2.9 % and 6.7 % below
#: and run 19.3 % faster -- both inside `DEVIATION_MARGIN` by the bound
#: alone.  Over SeisSol's elastic kernels at orders 4 to 8 and the
#: viscoelastic ones at orders 4 to 6, both precisions (56 kernels), the
#: rule's loss against the fastest candidate goes from 4.8 / 31.3 % to
#: 3.8 / 30.4 (mean / worst), and on the viscoelastic ones from 5.2 / 31.3
#: to 3.1 / 18.8; those two are the only picks that move, and none of
#: MI210's 12 does.
ISSUE_PER_WAVE = ('gfx9',)

#: Where the default spills on AMD and no candidate up to the extent's
#: ceiling (`_wider`) spills nothing, ask about lanes as wide as the wave
#: too (`_whole_wave`).  One multiplication a wave makes what is uniform over
#: a multiplication uniform over the wave, and the compiler holds it in
#: scalar registers -- what they cannot hold it parks in the lanes of vector
#: registers, 64 values to one, rather than in scratch.  Viscoelastic order-4
#: `derivative` in double precision spreads 20 rows: at the 32 unmerged lanes
#: the rule took, hipcc on MI300A holds 512 registers and spills 952 B a lane
#: to scratch (ROCm 7.2: 307 vector registers and 44 scalar ones spilled);
#: at 64 it holds 404 and spills nothing to scratch (415 scalar registers
#: parked, no vector one).  With twice the instructions an element executes
#: it runs 30 % faster (38.0 ns an element against 54.5), and 51 % faster
#: than the default.  Lanes past the extent hold nothing (`_wider`), which is
#: why the whole wave is asked about only where nothing narrower is clean.
CLEAN_WAVE = True

#: Where `Options.pin_loads` is `'auto'`, the rule has ptxas decide it rather
#: than the estimate (`abstract_instruction.PTXAS_NEED`): a candidate whose
#: spilling -- stores and loads together, as `Resources.spill_bytes` counts
#: it on NVIDIA -- comes to at least this many bytes a lane is built again
#: pinned, and the pinned build stands in for it where it spills at most
#: `PIN_KEEP` of that.  A small spill is a cold value ptxas parks, and a pin
#: there costs the latency the free schedule hid: elastic order-6
#: `derivative` in single precision spills 148 B on GB200 (72 stored, 76
#: loaded), 8 B pinned, and runs 90 % slower pinned; viscoelastic order-5
#: `volumeExt` in double spills 256 B and runs 8 % faster pinned, with
#: nothing left.  The viscoelastic time derivatives at orders 4 to 6 spill
#: 1.3 to 15 kB and run 20 to 74 % faster pinned.  Over 36 SeisSol kernels on
#: GB200 (elastic and viscoelastic, orders 4 to 6, both precisions), as the
#: loss against the faster of the two builds (mean / worst): never pinned
#: 26.0 / 278 %, pinned by the estimate 1.5 / 24 % (it misses the
#: viscoelastic order-4 `derivative` in single precision, 1.3 kB spilled, and
#: pins elastic order-6 `volume` in double for 28 B), and by this rule
#: 0.1 / 3 % anywhere from 160 to 256 B; on sm_120 it picks the faster build
#: of all 13 kernels measured there.
PIN_SPILL = 192

#: The share of the spilling a pinned build may keep and still be taken.
#: Viscoelastic order-6 `volumeExt` in single precision spills 168 B on
#: sm_120 and 196 B pinned, and runs 24 % slower pinned; where the pin pays,
#: it removed 80 to 100 % of the spilling.
PIN_KEEP = 0.5


#: Where the rule ranks by the emitter's count (NVIDIA) and `Options.lanes_band`
#: asks for it, the compiler is asked only about a candidate whose spilling
#: the estimate cannot settle
#: (`_settled`): one whose estimate (`abstract_instruction.PTXAS_NEED`) is
#: within `ASK_BAND` registers of the file, one with a rolled loop -- a merged
#: run, a rolled chain, where the estimate errs both ways -- from `ASK_LOOPS`
#: below the file up, and one whose code may be past the instruction cache
#: (`ASK_ICACHE` of it), whose charge needs the compiler's size.  Elsewhere
#: the estimate's spilling stands: none below the file, the registers past
#: it stored and loaded above.  Over SeisSol's elastic kernels at orders 6
#: and 8 on four GPUs (356 candidates; `~/tf-probe/nv/lanes/eval_band.py`),
#: as the mean loss against the fastest candidate: estimate alone 15.3 %,
#: worst 401 %; the compiler about every candidate 7.8 %; asked so, 7.1 %
#: about 122 of them.  Its pin is the estimate's as well (`PIN_MARGIN`).
ASK_BAND = 30
ASK_LOOPS = 80
ASK_ICACHE = 0.5


def _estimated_need(result: Build) -> Optional[float]:
    """The registers a lane of `result` is expected to need, as
    `pin_loads='auto'` estimates them: `PTXAS_NEED` over the largest body's
    modeled footprint."""
    from tensorforge.backend.instructions.abstract_instruction import (
        PTXAS_NEED)
    peak = result.generator.metrics.peak_pressure
    if not peak:
        return None
    return PTXAS_NEED[0] + PTXAS_NEED[1] * peak / 4


def _rolled(result: Build) -> bool:
    """Whether `result` keeps a loop of its operations: a merged run, a
    rolled chain (what `Generator.kernel_info` lists as `loops`)."""
    return any(hasattr(d, 'decompose') and hasattr(d, 'to_dict')
               for d in getattr(result.generator, '_descr_list', ()))


def _settled(result: Build) -> Optional[Resources]:
    """What the estimate says of `result`'s spilling and code, where it can
    say it (`ASK_BAND`); None where the compiler has to be asked."""
    from tensorforge.analysis.icache import code_bytes
    hw = result.context.target.hw
    budget = getattr(hw, 'max_reg_per_thread', None)
    need = _estimated_need(result)
    if budget is None or need is None:
        return None
    file = budget / 4
    if abs(need - file) <= ASK_BAND:
        return None
    if _rolled(result) and need > file - ASK_LOOPS:
        return None
    code = code_bytes(result.generator.metrics.code_units, hw)
    capacity = getattr(hw, 'icache_size', None)
    if (getattr(hw, 'icache_miss_factor', None) and capacity
            and (code is None or code > ASK_ICACHE * capacity)):
        return None
    spill = 0 if need <= file else int(8 * (need - file))
    return Resources(registers=int(min(need, file)), spill_bytes=spill,
                     code_bytes=code)


def _pins_by_compiler(context: Context) -> bool:
    """Whether the compiler decides `Options.pin_loads` here: `'auto'`, on
    NVIDIA under CUDA, where the fence is (`abstract_instruction._pin_every`).
    """
    target = context.target
    return (getattr(context.get_user_options(), 'pin_loads', 0) == 'auto'
            and str(target.hw.vendor).lower() == 'nvidia'
            and target.backend == 'cuda'
            and not getattr(target, 'explicit_simd', False))


def _ranked_by(context: Context) -> Optional[str]:
    """`RANKED_BY` for the target, or None where it was never measured: on
    AMD the parts of `LATENCY_WAVES_PER_SIMD`; on Intel DPC++'s SPMD lowering
    (`oneapi`), where the lanes are the sub-group -- under the explicit
    vector they are the vector's width, which `CompiledScore` ranks apart,
    and AdaptiveCpp keeps the deduction (`ONEAPI_ONLY`)."""
    target = context.target
    vendor = str(target.hw.vendor).lower()
    if vendor == 'intel' and (target.explicit_simd
                              or target.backend != 'oneapi'):
        return None
    if vendor == 'amd' and target.hw.model not in LATENCY_WAVES_PER_SIMD:
        return None
    return RANKED_BY.get(vendor)


class _Ahead:
    """A candidate built in another process (`_generate_ahead`): what the
    walk asks of a build -- its lanes, its modeled footprint and code, its
    loops, its source for the compiler -- and nothing else of the generator,
    which stays behind in the process that built it."""

    def __init__(self, summary: dict):
        from types import SimpleNamespace
        self.lanes = SimpleNamespace(num_threads=summary['lanes'])
        self.metrics = SimpleNamespace(peak_pressure=summary['peak_pressure'],
                                       code_units=summary['code_units'])
        self._mults = summary['mults']
        self._headers = list(summary['headers'])
        self._kernel = summary['kernel']
        # `_rolled` asks for descriptors with a loop of their own.
        self._descr_list = ([SimpleNamespace(decompose=None, to_dict=None)]
                            if summary['rolled'] else [])

    def launch_config(self):
        from types import SimpleNamespace
        return SimpleNamespace(mults_per_block=self._mults)

    def get_helper_headers(self):
        return list(self._headers)

    def get_kernel(self):
        return self._kernel


def _rewritten(result: Build, alt: Candidate, pin: int, figure, context
               ) -> Tuple[Build, Optional[float]]:
    """`alt` -- `result`'s candidate pinned every `pin` loads -- as the
    same build written again (`Generator.emitted_with_pin`), and its figure:
    the fence is the emitter's alone, so the bodies are `result`'s.  A build
    made for `alt` writes the same body under another kernel name."""
    from types import SimpleNamespace
    gen = result.generator
    text, metrics = gen.emitted_with_pin(pin)
    ctx = alt.context(context, autotune='off')
    view = SimpleNamespace(candidate=alt, context=ctx, generator=SimpleNamespace(
        metrics=metrics, lanes=gen.lanes, stack=getattr(gen, 'stack', 1)))
    count = figure(view)
    stub = _Ahead(dict(lanes=gen.lanes.num_threads,
                       peak_pressure=metrics.peak_pressure,
                       code_units=metrics.code_units, rolled=_rolled(result),
                       mults=gen.launch_config().mults_per_block,
                       headers=list(gen.get_helper_headers()), kernel=text))
    return Build(alt, generator=stub, context=ctx), count


#: What the forked workers of `_generate_ahead` build: set before the fork,
#: read by index in the child, so that nothing but the index and the summary
#: crosses between the processes.
_FORKED: Optional[Tuple] = None


def _summary_of(index: int) -> dict:
    descr_factory, context, cands, generate, figure, pin = _FORKED
    cand = cands[index]
    result = build(descr_factory, context, cand, generate)
    if not result.ok:
        return dict(ok=False, error=repr(result.error))
    gen = result.generator
    out = dict(ok=True, count=figure(result), lanes=gen.lanes.num_threads,
               peak_pressure=gen.metrics.peak_pressure,
               code_units=gen.metrics.code_units, rolled=_rolled(result),
               mults=gen.launch_config().mults_per_block,
               headers=list(gen.get_helper_headers()),
               kernel=gen.get_kernel())
    wave = max(1, getattr(context.target.hw, 'vec_unit_length', 32))
    if pin and not cand.get('pin_loads') and not gen.lanes.num_threads % wave:
        # The pinned build the walk may ask about: the same build written
        # again, a tenth of a second where a build takes a minute.
        alt = cand.set('pin_loads', pin)
        twin, count = _rewritten(result, alt, pin, figure, context)
        out['pinned'] = dict(count=count, peak_pressure=twin.generator.metrics.peak_pressure,
                             code_units=twin.generator.metrics.code_units,
                             kernel=twin.generator.get_kernel())
    return out


def _generate_ahead(descr_factory, context: Context, cands: List[Candidate],
                    generate, figure, jobs: int, pin: int = 0
                    ) -> List[Tuple[Candidate, Build, Optional[float]]]:
    """`cands` built `jobs` at a time in forked processes, each as
    `(candidate, build, figure)`: the build an `_Ahead` in place of its
    generator, or one with the error a build in this process would have.
    With `pin`, each whole-warp candidate's pinned twin follows it, written
    again by the process that built it (`_rewritten`)."""
    global _FORKED
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    if not cands:
        return []
    _FORKED = (descr_factory, context, list(cands), generate, figure, pin)
    try:
        with ProcessPoolExecutor(
                max_workers=min(jobs, len(cands)),
                mp_context=multiprocessing.get_context('fork')) as pool:
            summaries = list(pool.map(_summary_of, range(len(cands))))
    finally:
        _FORKED = None
    out = []
    for cand, summary in zip(cands, summaries):
        ctx = cand.context(context, autotune='off')
        if not summary['ok']:
            out.append((cand, Build(cand, context=ctx,
                                    error=RuntimeError(summary['error'])),
                        None))
            continue
        out.append((cand, Build(cand, generator=_Ahead(summary), context=ctx),
                    summary['count']))
        twin = summary.get('pinned')
        if twin is not None:
            alt = cand.set('pin_loads', pin)
            out.append((alt, Build(alt, generator=_Ahead(dict(summary, **twin)),
                                   context=alt.context(context,
                                                       autotune='off')),
                         twin['count']))
    return out


def _compile_ahead(scorer: 'CompiledScore', builds: Mapping[Candidate, Build],
                   cands: Iterable[Candidate], jobs: int
                   ) -> Dict[Candidate, Any]:
    """What the compiler says of `cands`, `jobs` at a time: the report, or
    the exception asking raised, which the walk raises again where it asks
    (`lanes_rule`)."""
    from concurrent.futures import ThreadPoolExecutor
    cands = list(cands)

    def ask(cand):
        try:
            return scorer.resources(builds[cand])
        except Exception as exc:              # noqa: BLE001
            return exc
    if not cands:
        return {}
    with ThreadPoolExecutor(max_workers=min(jobs, len(cands))) as pool:
        return dict(zip(cands, pool.map(ask, cands)))


def lane_candidates(descrs, context: Context, origin: Candidate,
                    fixed: Optional[Mapping[str, Any]] = None
                    ) -> List[Candidate]:
    """Every lane count the rule weighs, times merging where something
    repeats.

    The powers of two from the deduced count down to `lanes.MIN_LANES` (the
    vector unit under SPMD, `Preferences.lanes_down_to_wave`), and the wider
    ones up to the next power of two at or above the extent (`_wider`):
    wider is what order-8 `localFluxAll` in double precision wants, 64 lanes
    unmerged at -80 % on A100 and -83 % on GH200, and it loses everywhere in
    single precision, which the count below sees by itself.  Merging both
    ways where `_mergeable`, since which way wins depends on the count -- off
    at 16 and 32 lanes on A100, GH200 and sm_120, on at eight.  What `fixed`
    names stays as the origin has it.
    """
    flat = _flat(descrs)
    hw = context.target.hw
    base = lane_config.deduce(flat, context)
    rows = base.num_active_threads or base.num_threads
    fixed = fixed or {}
    if 'lanes' in fixed:
        geometries = [origin.lanes]
    else:
        floor = lane_config.MIN_LANES
        if context.target.prefs.lanes_down_to_wave:
            floor = max(floor, getattr(hw, 'vec_unit_length', 1))
        geometries = []
        t = base.num_threads
        while t >= floor:
            geometries.append(LaneConfig(t, base.num_active_threads,
                                         base.lead_width))
            if t & (t - 1):
                break
            t //= 2
        geometries += _wider(base, rows, hw)
    merging = [None]
    if 'merge_variants' not in fixed and _mergeable(descrs, context):
        merging = [False, True]
    out = []
    for geometry in geometries:
        for merge in merging:
            cand = origin.set('lanes', geometry)
            if merge is not None:
                cand = cand.set('merge_variants', merge)
            if cand not in out:
                out.append(cand)
    return out


def lanes_rule(descr_factory, context: Context, scorer: 'CompiledScore',
               origin: Candidate, first: Build,
               fixed: Optional[Mapping[str, Any]] = None, *,
               generate) -> Candidate:
    """The configuration `autotune=lanes` builds: the fewest instructions an
    element executes, among the candidates that spill no more than the
    default allows, with the spilling (`SPILL_WEIGHT`) and the code past the
    instruction cache (`ICACHE_SHARE`) priced in.

    Every candidate is generated -- the emitter's count needs nothing else --
    and the compiler is asked about as few as the bound admits: the default
    first, which sets the guard (`SPILL_GROWTH`), then the rest in order of
    their count, until a count alone is past the best cost found, which
    spilling and code can only raise.  Over SeisSol's elastic kernels at
    orders 6 and 8 that compiles three to five of five to eight candidates.

    Measured over the same kernels on four GPUs -- 48 of them, eight to 64
    lanes with merging on and off, builds of one source pooled -- as the loss
    against the fastest candidate (mean / worst):

        GPU            the default    spilling only      this rule
        A100 sm_80     13.4 / 41 %     6.0 / 41 %        6.0 / 41 %
        GH200 sm_90    15.9 / 77       7.3 / 29          5.9 / 29
        GB200 sm_100   15.0 / 52       7.6 / 29          4.7 / 14
        sm_120 laptop  14.8 / 38       5.8 / 15          5.8 / 15

    The loss that is left is mostly the time derivative at sixteen lanes,
    which spills about twice what 32 do and is 22 to 29 % faster for it on
    A100 and GH200 in double precision and 43 % slower on GH200 in single:
    nothing a build says about itself separates those.  Then 64 lanes for
    order-8 `neighboringFlux` in double precision (16 to 27 %), which count
    17 % more instructions than 32, with 32 spilling 28 bytes or nothing.

    The default unless the winner's cost is `DEVIATION_MARGIN` below the
    default's.

    On AMD the same walk ranks by the pipe bound instead (`RANKED_BY`,
    `pipe_bound`), with the spilling written into it rather than priced
    beside it, and stretched where the compiler's occupancy falls short of
    the waves the part needs (`LATENCY_WAVES_PER_SIMD`); the guard, the
    margin and the order of the compiler's questions are the same, since the
    bound without either is as much a floor of the cost as the count is of
    the count's -- except where the default spills: then the cheapest
    candidate that spills nothing is taken first (`CLEAN_FIRST`), the whole
    wave asked about where nothing up to the extent is (`CLEAN_WAVE`).  On CDNA
    the instructions an element executes, over the waves the compiler gives
    a build, are a floor of its cost too (`ISSUE_PER_WAVE`).  On Intel, under DPC++, the count as on NVIDIA, starting
    from the sixteen lanes the target builds (`start`).

    Where `Options.pin_loads` is `'auto'` on NVIDIA, the compiler decides the
    pin too: a whole-warp candidate it spills `PIN_SPILL` bytes or more for
    is built again pinned, and that build stands in for it -- the default's
    included, guard and margin then the pinned one's -- where it spills at
    most `PIN_KEEP` of that.  The origin comes unpinned (`_lanes_mode`).
    """
    hw = context.target.hw
    wave = max(1, getattr(hw, 'vec_unit_length', 32))
    elem = context.fp_type.size()
    by_bound = _ranked_by(context) == 'bound'
    latency = LATENCY_WAVES_PER_SIMD.get(hw.model, 1)
    issue = by_bound and str(hw.model).startswith(ISSUE_PER_WAVE)
    figure = pipe_bound if by_bound else executed_instructions
    builds: Dict[Candidate, Build] = {origin: first}
    counts: Dict[Candidate, float] = {}
    # Several at once where the count ranks (`Options.autotune_jobs`): the
    # candidates built in forked processes, the compiler asked about every
    # one the estimate cannot settle, and the pinned builds that may be
    # wanted, ahead of the walk -- which then takes what it asks from there
    # and decides as it would one at a time.
    jobs = max(1, int(getattr(context.get_user_options(), 'autotune_jobs', 1)
                      or 1))
    ahead = jobs > 1 and not by_bound
    pins = _pins_by_compiler(context)
    from tensorforge.backend.instructions.abstract_instruction import (
        PIN_EVERY)
    # The pinned builds the compiler's word may call for (`pinned`): each
    # whole-warp candidate written again by the process that built it.
    pinned_ahead: Dict[Candidate, Tuple[Build, Optional[float]]] = {}
    cands = [origin] + lane_candidates(descr_factory(), context, origin,
                                       fixed)
    if ahead:
        if first.ok:
            count = figure(first)
            if count is not None:
                counts[origin] = count
            if pins and not first.generator.lanes.num_threads % wave:
                alt = origin.set('pin_loads', PIN_EVERY)
                pinned_ahead[alt] = _rewritten(first, alt, PIN_EVERY, figure,
                                               context)
        rest: List[Candidate] = []
        for cand in cands:
            if cand not in builds and cand not in rest:
                rest.append(cand)
        for cand, result, count in _generate_ahead(
                descr_factory, context, rest, generate, figure, jobs,
                PIN_EVERY if pins else 0):
            if cand.get('pin_loads') == PIN_EVERY and cand not in rest:
                pinned_ahead[cand] = (result, count)
                continue
            builds[cand] = result
            if count is not None:
                counts[cand] = count
    else:
        for cand in cands:
            result = builds.get(cand) or build(descr_factory, context, cand,
                                               generate)
            builds[cand] = result
            if result.ok:
                count = figure(result)
                if count is not None:
                    counts[cand] = count
    if origin not in counts:
        return origin
    # Where the count ranks, the estimate settles what it can (`ASK_BAND`).
    banded = (getattr(context.get_user_options(), 'lanes_band', False)
              and not by_bound and str(hw.vendor).lower() == 'nvidia')
    settled: set = set()

    # Asked ahead: the reports, or what asking raised.
    reports: Dict[Candidate, Any] = {}

    def resources(cand: Candidate) -> Optional[Resources]:
        if banded:
            known = _settled(builds[cand])
            if known is not None:
                settled.add(cand)
                return known
        if cand in reports:
            got = reports[cand]
            if isinstance(got, Exception):
                raise got
            return got
        return scorer.resources(builds[cand])

    if ahead:
        reports.update(_compile_ahead(
            scorer, builds, [c for c in counts if not banded
                             or _settled(builds[c]) is None], jobs))

    try:
        report = resources(origin)
    except Exception:                         # noqa: BLE001
        report = None
    if report is None:
        return origin
    if ahead and pins:
        # Asked about ahead: the twins of what the compiler saw spill
        # `PIN_SPILL` -- the ones `pinned` goes on to ask about.
        wanted = [c.set('pin_loads', PIN_EVERY) for c, got in reports.items()
                  if isinstance(got, Resources)
                  and got.spill_bytes >= PIN_SPILL and not c.get('pin_loads')]
        reports.update(_compile_ahead(
            scorer, {a: b for a, (b, _) in pinned_ahead.items()},
            [a for a in wanted if a in pinned_ahead
             and pinned_ahead[a][0].ok], jobs))

    def pinned(cand: Candidate, seen: Optional[Resources]
               ) -> Tuple[Candidate, Optional[Resources]]:
        """`cand` and what the compiler says of it, or its pinned build and
        what the compiler says of that, where the pin pays (`PIN_SPILL`).
        A candidate the estimate settled is pinned by the estimate as well:
        built with `pin_loads='auto'` (`abstract_instruction.PIN_MARGIN`)."""
        if pins and cand in settled:
            alt = cand.set('pin_loads', 'auto')
            builds[alt], counts[alt] = builds[cand], counts[cand]
            settled.add(alt)
            return alt, seen
        if (not pins or seen is None or seen.spill_bytes < PIN_SPILL
                or cand.get('pin_loads')
                or builds[cand].generator.lanes.num_threads % wave):
            return cand, seen
        alt = cand.set('pin_loads', PIN_EVERY)
        if alt in pinned_ahead:
            result, count = pinned_ahead[alt]
        elif hasattr(builds[cand].generator, 'emitted_with_pin'):
            # The same build written again: the fence is the emitter's.
            result, count = _rewritten(builds[cand], alt, PIN_EVERY, figure,
                                       context)
            pinned_ahead[alt] = (result, count)
        else:
            result, count = build(descr_factory, context, alt, generate), None
        if not result.ok:
            return cand, seen
        try:
            if alt in reports:
                other = reports[alt]
                if isinstance(other, Exception):
                    raise other
            else:
                other = scorer.resources(result)
        except Exception:                     # noqa: BLE001
            other = None
        if other is None or other.spill_bytes > PIN_KEEP * seen.spill_bytes:
            return cand, seen
        builds[alt] = result
        if alt not in pinned_ahead:
            count = figure(result)
        counts[alt] = counts[cand] if count is None else count
        return alt, other

    base, report = pinned(origin, report)
    from tensorforge.analysis.icache import past_share
    limit = SPILL_GROWTH * report.spill_bytes
    miss = getattr(hw, 'icache_miss_factor', None)
    refetch = ICACHE_SHARE * max(0.0, miss - 1.0) if miss else 0.0

    def cost(cand: Candidate, seen: Resources) -> float:
        lanes = builds[cand].generator.lanes.num_threads
        past = past_share(seen.code_bytes, hw)
        if by_bound:
            spilled = (pipe_bound(builds[cand], seen.spill_bytes)
                       if seen.spill_bytes else counts[cand])
            waves = max(1, seen.register_blocks or 1)
            if issue and seen.register_blocks:
                issued = executed_instructions(builds[cand])
                if issued is not None:
                    spilled = max(spilled, issued / seen.register_blocks)
            return (spilled * max(1.0, latency / waves)
                    * (1.0 + refetch * past))
        return (counts[cand] * (1.0 + refetch * past)
                + SPILL_WEIGHT * seen.spill_bytes / elem * lanes / wave)

    best, least = base, cost(base, report)
    default = least
    # Where the default spills on AMD, a build that spills nothing wins over
    # every build that does (`CLEAN_FIRST`): until one is found the compiler
    # is asked about every candidate, after that only about those whose
    # bound alone is still below the best clean one's cost.
    clean_first = CLEAN_FIRST and by_bound and report.spill_bytes > 0
    clean: Optional[Tuple[Candidate, float]] = None
    for cand in sorted((c for c in counts if c not in (origin, base)),
                       key=counts.get):
        if clean is not None:
            if counts[cand] >= clean[1]:
                break
        elif not clean_first and counts[cand] >= least:
            break
        try:
            seen = resources(cand)
        except Exception:                     # noqa: BLE001
            seen = None
        cand, seen = pinned(cand, seen)
        if seen is None or seen.spill_bytes > limit:
            continue
        price = cost(cand, seen)
        if clean_first and not seen.spill_bytes:
            if clean is None or price < clean[1]:
                clean = (cand, price)
            continue
        if price < least:
            best, least = cand, price
    if clean_first and clean is None and CLEAN_WAVE:
        # Nothing up to the extent's ceiling spills nothing: the whole wave.
        for cand in _whole_wave(descr_factory(), context, origin, counts,
                                fixed):
            result = build(descr_factory, context, cand, generate)
            count = figure(result) if result.ok else None
            if count is None:
                continue
            builds[cand], counts[cand] = result, count
            try:
                seen = scorer.resources(result)
            except Exception:                 # noqa: BLE001
                seen = None
            if seen is None or seen.spill_bytes:
                continue
            price = cost(cand, seen)
            if clean is None or price < clean[1]:
                clean = (cand, price)
    if clean is not None:
        return clean[0]
    return best if least < (1.0 - DEVIATION_MARGIN) * default else base


def _lanes_mode(descr_factory, context: Context, cache: Optional[str],
                fixed: Optional[Mapping[str, Any]], generate
                ) -> Optional[Candidate]:
    """`autotune=lanes`: `lanes_rule` where a compiler can be asked, the
    default where none can or the vendor was never measured (`RANKED_BY`).
    Remembered by the default build's source, as the walk's picks are
    (`Options.autotune_cache`)."""
    hw = context.target.hw
    if _ranked_by(context) is None:
        warnings.warn(f'autotune=lanes: not measured for {hw.model} under '
                      f'{context.target.backend}; the default stands')
        return None
    scorer = CompiledScore()
    if not scorer.available(context):
        warnings.warn('autotune=lanes: no compiler for this target '
                      '(Toolchain, TF_NVCC, TF_HIPCC, TF_ICPX); the default '
                      'stands')
        return None
    descrs = descr_factory()
    origin = start(descrs, context)
    for name, value in (fixed or {}).items():
        origin = origin.set(name, value)
    pins = _pins_by_compiler(context)
    if pins:
        # Unpinned to begin with, and every candidate with it: the compiler
        # says where a pin pays (`PIN_SPILL`), not the estimate.
        origin = origin.set('pin_loads', 0)
    first = build(descr_factory, context, origin, generate)
    if not first.ok:
        return None
    _load(cache)
    # Remembered apart where the pin is the compiler's: the default's source
    # is the same unpinned either way, and the pick of a walk without the
    # pin in it is not this one's.
    mode = 'lanes+pin' if pins else 'lanes'
    if (getattr(context.get_user_options(), 'lanes_band', False)
            and _ranked_by(context) == 'instructions'
            and str(hw.vendor).lower() == 'nvidia'):
        # Picks of the walk that asks the compiler about every candidate are
        # not this one's (`ASK_BAND`).
        mode += '+band'
    key = _key(first, mode)
    if key in _PICKS:
        return Candidate.from_dict(_PICKS[key])
    pick = lanes_rule(descr_factory, context, scorer, origin, first, fixed,
                      generate=generate)
    _store(cache, key, pick)
    return pick
