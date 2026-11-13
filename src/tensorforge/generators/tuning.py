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

from tensorforge.common.basic_types import Addressing, Datatype, work_type
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


def _product_types(descrs, context: Context) -> set:
    """What the list's products are computed in: for every multilinear of
    two operands or more that contracts an axis -- the ones
    `contraction_lengths` counts -- the widest floating type of its operands
    and its destination (`work_type`), the kernel's where none states one.

    The matrix path is chosen per product for that type
    (`MultilinearInstruction._plan`), and so is what the tuner weighs it for:
    asked of the kernel's type, the twin took SeisSol's `toCanonical` in
    single precision onto the TF32 path on sm_100, a double product split
    into halves of eleven bits."""
    out = set()
    for d in _flat(descrs):
        target = getattr(d, 'target', None)
        ops = list(getattr(d, 'ops', ()))
        if target is None or len(ops) < 2:
            continue
        if not any(t == -1 for axes in target for t in axes):
            continue
        out.add(work_type(
            [getattr(getattr(view, 'tensor', None), 'datatype', None)
             for view in ops + [getattr(d, 'dest', None)]],
            context.fp_type))
    return out


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
        if Datatype.F32 in _product_types(descrs, context):
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
    # The widths of the instruction for the type the products are computed
    # in, where they all agree on one; a knob over several would set one
    # width for instructions of different types.
    works = _product_types(descrs, context)
    if len(works) != 1:
        return lambda c: ()
    work = next(iter(works))
    deduced = lane_config.deduce(_flat(descrs), context)

    def widths(candidate):
        lanes = candidate.lanes or deduced
        found = tuple(module.widths(lanes.num_threads, work, context))
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
    if base.num_active_threads < CLEAN_WAVE_SHARE * wave:
        # Mostly idle lanes (`CLEAN_WAVE_SHARE`).
        return []
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
    floor = lane_floor(context, rule=False)
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
        # The sub-group, where it is wider than one multiplication: eight
        # lanes on PVC are two multiplications in one 16-wide sub-group
        # (`Options.lanes_split_floor`), whose thread holds both footprints.
        # Counted as eight, a split candidate would be measured against
        # twice the file it has.
        width = getattr(target, 'sub_group_width', lambda n: None)(share)
        if width and width > share:
            share = width
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
        self.references: Dict[Candidate, Optional[int]] = {}
        #: What the same listing counts of instructions (`listed_instructions`).
        self.instructions: Dict[Candidate, Optional[int]] = {}

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

    def spill_references(self, result: Build) -> Optional[int]:
        """How often the compiler expects `result`'s kernel to touch what it
        spilled (`Compiler.spill_references`, IGC's listing), or None where
        the compiler writes no such listing or the build fails.  A build of
        its own, with the listing on: it writes megabytes, so `resources`
        does not ask for it, and the lanes rule asks only where it decides
        (`SPLIT_REF_GROWTH`)."""
        if result.candidate in self.references:
            return self.references[result.candidate]
        hw = result.context.target.hw
        compiler = self.toolchain.compiler(hw.vendor)
        entry = toolchain.compiler_for_vendor(hw.vendor)
        refs = code = None
        if compiler is not None and entry is not None:
            with tempfile.TemporaryDirectory(prefix='tf-tune-') as tmp:
                env = entry.listing_env(tmp)
                if env is not None:
                    src = os.path.join(tmp, 'kernel' + entry.suffix)
                    with open(src, 'w') as f:
                        f.write(kernel_source(result))
                    cmd, _ = entry.report_build(
                        compiler, src, tmp, hw.model,
                        self.toolchain.include_dir())
                    run = subprocess.run(cmd + list(self.flags),
                                         capture_output=True, text=True,
                                         timeout=self.timeout,
                                         env={**os.environ, **env})
                    if not run.returncode:
                        refs = entry.spill_references(tmp)
                        code = entry.listed_instructions(tmp)
        self.references[result.candidate] = refs
        self.instructions[result.candidate] = code
        return refs

    def listed_instructions(self, result: Build) -> Optional[int]:
        """How many instructions the compiler's listing of `result` counts,
        or None where it writes none: the size of the code where the build
        reports none (`Icpx`).  From the build `spill_references` makes."""
        if result.candidate not in self.instructions:
            self.spill_references(result)
        return self.instructions.get(result.candidate)

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


def _store(path: Optional[str], key: str, pick) -> None:
    _PICKS[key] = pick.to_dict() if isinstance(pick, Candidate) else pick
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
             generate,
             dispatch: Optional[Dict[str, Any]] = None) -> Optional[Candidate]:
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

    With `dispatch`, a dict, `autotune=lanes` also says which candidate runs
    small batches (`Options.batch_dispatch`): `dispatch['small']`, None where
    the pick's own chain is as short (`small_variant`).
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
        return _lanes_mode(descr_factory, context, cache, fixed, generate,
                           dispatch)
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

#: How much more often, per element, a split sub-group may reference what
#: IGC spilled than the default does, where both spill (`lanes_rule`;
#: `Options.lanes_split_floor`: two multiplications of eight lanes in one
#: sub-group of 16).  Its scratch alone does not show it.  The order-6
#: `neighboringFlux` in double precision at eight lanes holds 20160 B of
#: scratch against the default's 14592, 1.38 times and within
#: `SPILL_GROWTH`.  IGC's listing counts 5086 references to it against 672,
#: 3.8 times an element, and the build ran 2.3 times as long (sng2 559986,
#: IGC 2.41.5).  The eight-lane builds that won where both spill reference
#: 0.6 to 1.5 times the default's: in double precision, s8 order-4
#: `localFluxAll` -48 %; in single, s8 order-4 `localFluxAll` -34 %, order-3
#: `neighboringFlux` -47 % and `localFluxAll` -7 %, order-4
#: `neighboringFlux` -5 %.
SPLIT_REF_GROWTH = 2.0

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

#: Whether the lanes rule offers a candidate whose code the compiler wrote
#: past `generator.FISSION_PAST` instruction caches cut into pieces as well
#: (`Options.fission`), as many as `generator.fission_pieces` says for that
#: size.  On GH200 the order-8 derivative in double ran 192 ns whole, 142 on
#: the matrix path and 124 on it cut into three; in single 95, 89 and 71.
#: The volume and the Taylor expansion stay whole (cut, 4 to 96 % slower): the
#: one fits on the matrix path, the other is far below the threshold.
FISSION_TWIN = True

#: On Intel, where IGC states no code size to weigh a cut by, the twin is
#: the build IGC spills cut in two, taken where IGC's listing references
#: what it spilled at most `FISSION_REF_SHARE` as often as the whole build
#: does, for at most `FISSION_COUNT_GROWTH` as much to execute -- or where
#: the cut saves `FISSION_REF_SAVING` references or more for every
#: instruction an element executes whole, for at most `FISSION_DENSE_GROWTH`
#: as much.  The cut changes how IGC allocates, which neither the count nor
#: the scratch says by itself, and the references alone weigh it wrongly
#: both ways: no one price of a reference in instructions orders the
#: order-6 and order-7 `derivative` in double.  Measured on a Max 1550 stack
#: at 262144 elements (sng2 560632, 560633, 560645, 560665; IGC 2.41.5),
#: the cut in two against the whole build:
#:
#: - taken: order-6 double `derivative` -31 % (14017 references against
#:   8450, 1.9 % more executed), `localFluxAll` -10 % (4567 against 2162),
#:   `volume` -11 % (453 against 130); order-7 single `derivative` -38 %
#:   (19192 against 8426 for 25652 instructions, 7.4 % more); order-7
#:   double `volume` +4 % (2946 against 1811, 1.4 % more);
#: - not taken, slower: order-7 double `derivative` +7 % (43633 against
#:   36120); order-4 double `derivative` +16 % (282 against 95, 3.4 % more),
#:   its s8 build +27 % (12827 against 6610, 11 % more), single +20 % and
#:   +7 %, where the cut executes 19 and 27 % more; order-4 double
#:   `localFluxAll` +83 %, which the cut makes spill;
#: - not taken, faster: order-4 double `volume` -29 % (20 references against
#:   none), the s8 `localFluxAll` -13 and -18 % (0.61 and 0.86 of the
#:   references, 11 and 1.6 % more executed), order-7 double `localFluxAll`
#:   -8 % (0.85).
#:
#: And where IGC wrote `FISSION_LISTED` instructions or more for the whole
#: build, for at most `FISSION_LARGE_GROWTH` as much to execute: there the
#: references barely move, and the cut gains anyway, as on NVIDIA past its
#: instruction cache (`FISSION_TWIN`).  Order 8, elastic, cut in two: in
#: double `derivative` -35 % (95188 instructions, 0.92 of the references),
#: `localFluxAll` -15 %, `volume` -25 %; in single `derivative` -23 %
#: (65518, 6.4 % more executed), `localFluxAll` -8 %.  Order-7 double
#: `localFluxAll` -8 % (32021), viscoelastic `fluxLocalAll` -26 % (47053);
#: the s8 order-4 `localFluxAll` -18 % (41703).  Taken as well, and lost:
#: order-7 double `derivative` +7 % (44363), viscoelastic +2 % (202826).
FISSION_REF_SHARE = 0.65
FISSION_COUNT_GROWTH = 1.02
FISSION_REF_SAVING = 0.3
FISSION_DENSE_GROWTH = 1.10
FISSION_LISTED = 30000
FISSION_LARGE_GROWTH = 1.08

#: Parts whose fetch past the instruction cache costs a kernel more than the
#: share of its code past the cache says: the charge on every executed
#: instruction grows from nothing at the capacity to this factor less one at
#: twice the capacity, and stays there.  MI210, CDNA 2's 32 KB, measured on
#: SeisSol's elastic derivatives of paper-1106 against MI300A, whose 64 KB
#: hold them (time over MI300A's, scaled by clocks and compute units): 1.5 at
#: 37.6 kB of code, 2.6 at 41.6, 3.3 at 47.1, 4.7 at 65.7 and 4.0 at 90.4 --
#: where the probe's loop, which streams, levels off at 2.8 from 56 kB
#: (`hw_descr.icache_miss_factor`).  The share past the cache charges 41.6 kB
#: a fifth of what the ramp does, and took 16 lanes for the order-6
#: derivative in single, which runs 94 % slower there.  With the ramp the
#: rule picks the measured best of 16, 32 and 64 lanes in all eight of those
#: kernels on both parts: 16 lanes at orders 4 and 6 in double, which fit or
#: are past twice the cache either way, 32 at order 5 and at order 6 in
#: single, where 16 lanes take the code across it.
ICACHE_RAMP = {'gfx90a': 5.0}


def icache_charge(code: Optional[int], hw, refetch: float) -> float:
    """What an executed instruction costs on top of one from inside the
    instruction cache in a body of `code` bytes: `refetch` times the share of
    the code past the cache, or on the parts of `ICACHE_RAMP` the ramp up to
    twice the capacity.  Nothing where the size or the capacity is not
    known."""
    from tensorforge.analysis.icache import past_share
    ramp = ICACHE_RAMP.get(str(hw.model))
    capacity = getattr(hw, 'icache_size', None)
    if ramp is None:
        return refetch * past_share(code, hw)
    if not code or not capacity:
        return 0.0
    return (ramp - 1.0) * min(1.0, max(0.0, (code - capacity) / capacity))


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

#: The share of the wave's lanes the extent has to fill for `CLEAN_WAVE` to
#: ask about it.  Order 3 has ten rows: at 64 lanes 54 of them hold nothing,
#: and what keeps such a build out of scratch is the scalar registers it
#: parks in vector ones -- 1019 for SeisSol's `localFluxAll` with eight
#: simulations in single precision, 378 for viscoelastic `derivative` in
#: double -- which on MI300A ran 72 and 20 % slower than the sixteen lanes of
#: the default, spilling 1112 and 644 B to scratch.  Viscoelastic order-4
#: `derivative` (20 rows, 31 % of the wave) is the case the step is for.
CLEAN_WAVE_SHARE = 0.25

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
#: of all 13 kernels measured there.  On four GPUs (A100, GH200, GB200,
#: RTX 5070; 145 kernels) the pin cost 2 to 90 % at 128 to 600 B spilled in
#: all but one (256 B, 8 % faster): elastic order-6 `derivative` in single
#: precision spills 212 B on GH200 and ran 74 % slower pinned under the
#: tuner (paper-1106).  What pays is kilobytes: the viscoelastic derivatives.
PIN_SPILL = 512

#: The share of the spilling a pinned build may keep and still be taken.
#: Viscoelastic order-6 `volumeExt` in single precision spills 168 B on
#: sm_120 and 196 B pinned, and runs 24 % slower pinned; where the pin pays,
#: it removed 80 to 100 % of the spilling.
PIN_KEEP = 0.5

#: Whether a pinned candidate also has a twin whose fences count the loads
#: from global memory only (`Options.pin_shared`), and the cheaper of the two
#: is taken.  A fence between two shared loads keeps the second from reusing
#: what the first read, so pinned on every load, SeisSol's order-8 elastic
#: kernels in double on GH200 ran 2.2 (`volume`), 1.4 (`localFluxAll`) and
#: 1.2 times (`derivative`) as long as fenced on global loads only -- and the
#: viscoelastic order-4 derivative in double 0.6 times, whose shared loads
#: the compiler, unfenced, hoists into 2924 B of spill where fenced it
#: spills 192.  Over 98 kernels each way is a geomean of -0.8 % (GH200) and
#: -2.4 % (A100) with the worst at +68 and +146 %.  Priced as every candidate
#: is, the fully fenced twin's count scaled by each twin's code -- the
#: emitter counts them alike, the compiler does not -- the choice is -1.6 %
#: and -2.9 % (the better of the two each time: -2.7 and -3.6 %), worst +59
#: and +30 %: order-8 `neighboringFlux` and viscoelastic order-6 `neighbor`
#: in double, faster fenced on every load at no less code and spilling.
PIN_GLOBAL_TWIN = True

#: The reduction unrolls (`Options.k_unroll_max`) the walk builds beside the
#: default's where the default's code is past the instruction cache: rolled
#: by up to 16 steps, and written out whole.  At order 8 in double on GH200,
#: fenced on global loads, `volume` ran 20 % faster rolled by 16 and 76 %
#: slower written out, `localFluxAll` 9 % faster written out and 7 % slower
#: rolled, the derivative 4 % faster written out: each kernel its own, which
#: the count with the code past the cache priced in tells apart.
#: Whether `lanes_rule` builds a candidate IGC spills again with the large
#: register file (`Options.sycl_grf`, 256 general registers a thread), on
#: Intel under DPC++, and takes that build where it spills less and costs
#: less.  Measured on a Max 1550 stack (oneAPI 2025.2, the wall clock of a
#: launch at 262144 elements): every SeisSol kernel of elastic order 6 and
#: viscoelastic order 6 that IGC spills at 128 ran 2 to 57 % faster at 256 --
#: but the volume kernels, `volume` in single precision +38 % and `volumeExt`
#: +45 %, whose spilling shrinks as much as the others' (471 to 316 bytes);
#: nothing a build says about itself, the count, the bound or the spill, tells
#: them apart.  Over the suites the large file is still 18 to 26 % faster;
#: at 16 elements every kernel is (11 to 61 %).
GRF_TWIN = True

ROLL_UNROLLS = (16, 0)


def _roll_candidates(first: Build, context: Context, origin: Candidate,
                     fixed: Optional[Mapping[str, Any]]) -> List[Candidate]:
    """`origin` at each of `ROLL_UNROLLS` that differs from its own, where
    its code is past the instruction cache by the emitter's estimate; none
    where the caller fixed the unroll, or rolls by `k_roll`."""
    from tensorforge.analysis.icache import code_bytes
    opts = context.get_user_options()
    if (fixed and 'k_unroll_max' in fixed) or getattr(opts, 'k_roll', 0):
        return []
    if 'k_unroll_max' in dict(context.get_asked_options().asked()):
        return []
    hw = context.target.hw
    capacity = getattr(hw, 'icache_size', None)
    if not first.ok or not capacity:
        return []
    code = code_bytes(first.generator.metrics.code_units, hw)
    if code is None or code <= capacity:
        return []
    own = getattr(opts, 'k_unroll_max', None)
    return [origin.set('k_unroll_max', value) for value in ROLL_UNROLLS
            if value != own]


#: Where the lanes rule offers the matrix path (`Options.tensor_cores`) as a
#: twin of the default (`_matrix_candidates`): by compute capability, the
#: precisions whose matrix units outrun the vector units of the part.  A100 and
#: H100/GH200 in double (DMMA at twice the FP64 FMA rate) and in single (three
#: TF32 products at several times the FP32 rate); B200/GB200 in single only, its
#: FP64 matrix rate being its vector rate (`mma.sync` f64 is eight DMMA.8x8x4
#: there).  The consumer parts are not in it: sm_120's HMMA runs at the FFMA
#: rate, and none of them has a fast FP64 matrix unit.  Which kernels take the
#: twin is the rule's count, as for every other twin: over SeisSol's elastic
#: order-6 kernels on GH200 the count ranks the faster of the two in five of six
#: (the double-precision derivative, 3 % to the matrix path, is the miss), and
#: declines the single-precision derivative, 22 % slower on the matrix path.
#: The shortest contraction the matrix twin puts on the matrix path
#: (`Options.mma_min_depth`), where the path's own default takes 16.  On GH200
#: the products of 16 to 23 steps -- `localFluxAll`'s 21, the derivative's
#: third order -- ran slower on it than beside it: order-6 `localFluxAll` 29 %
#: (double) and 26 % (single) under SIMT with them on the nest against 21 and
#: 4 % on the path, the derivative 28 against 24 %; order 5 `volume` in single
#: no longer 19 % slower, in double 5 % of a gain given up.  The count ranks the
#: two the other way round, so the twin is built at the one depth.
MATRIX_TWIN_DEPTH = 24

#: The reduction unrolls (`Options.k_unroll_max`) the matrix twin is built
#: at beside the default's, as `ROLL_UNROLLS` for the default: on GH200 the
#: order-8 `volume` on the matrix path ran 27 % (double) and 30 % (single)
#: under SIMT written out, against 23 and 3 % at the default unroll -- the
#: path's code is short enough to take it, where SIMT's written out ran 83 %
#: slower -- and the order-6 derivative 28 % and 7 % rolled by 16, against
#: 24 and 4 %.
MATRIX_ROLLS = (16, 0)

#:
#: Not A100 in single precision any more: over SeisSol's elastic and
#: viscoelastic kernels (orders 4, 6 and 8) the twin never beat the rule's
#: best SIMT build there -- `localFluxAll` of order 6 13 % under the default on
#: the path, 24 % at sixteen lanes beside it -- and the order-6 `volume` it was
#: taken for ran 28 % slower than without it (3xTF32's chains, `wait`).
MATRIX_TWIN = {80: (Datatype.F64,),
               90: (Datatype.F64, Datatype.F32),
               100: (Datatype.F32,)}


def _matrix_candidates(descrs, first: Build, context: Context,
                       origin: Candidate,
                       fixed: Optional[Mapping[str, Any]]) -> List[Candidate]:
    """`origin` on the matrix path, where the part's matrix units are faster
    than its vector units in this precision (`MATRIX_TWIN`) and the kernel
    contracts -- with each reader of an operator that others read too given
    a copy in its own fragment order (`Options.mma_operator_copies`), and in
    double precision with the operator laid out in 16-byte chunks
    (`Options.mma_chunked_order`) and the temporaries for the B reads
    (`Options.mma_fragment_ld`).  That is what made the path pay on GH200:
    the derivative 24 % (order 6) and 26 % (order 8) under SIMT in double, 5
    and 7 % in single, where without the copies it lost up to 21 %; volume
    and `localFluxAll` 17 to 25 %.  Not the B layout in single precision,
    which took order-6 `volume`'s 19 % back.  None where the caller fixed the
    path either way, or the default takes it already.  Products shorter than
    `MATRIX_TWIN_DEPTH` stay beside the path."""
    if fixed and 'tensor_cores' in fixed:
        return []
    if 'tensor_cores' in dict(context.get_asked_options().asked()):
        return []
    if (not first.ok or not context.target.prefs.tune_matrix_path
            or getattr(context.get_user_options(), 'tensor_cores', None)):
        return []
    from tensorforge.backend.instructions.compute.primitives.nvidia import (
        sm_of)
    # Every product in a type the part's matrix units are faster in, which
    # is what each one is computed in and not the kernel's type.
    works = _product_types(descrs, context)
    if not works or not works <= set(MATRIX_TWIN.get(sm_of(context), ())):
        return []
    lengths = contraction_lengths(_flat(descrs))
    if not lengths:
        return []
    twin = origin.set('tensor_cores', True).set('mma_operator_copies', True)
    if works == {Datatype.F64}:
        twin = twin.set('mma_chunked_order', True).set('mma_fragment_ld', True)
    if ('mma_min_depth' not in dict(context.get_asked_options().asked())
            and not (fixed and 'mma_min_depth' in fixed)):
        twin = twin.set('mma_min_depth', MATRIX_TWIN_DEPTH)
    twins = [twin]
    # Rolled and written out as well (`MATRIX_ROLLS`), where a reduction is
    # long enough for the unroll to change it and the caller left it free.
    opts = context.get_user_options()
    if not ((fixed and 'k_unroll_max' in fixed) or getattr(opts, 'k_roll', 0)
            or 'k_unroll_max' in dict(context.get_asked_options().asked())):
        own = getattr(opts, 'k_unroll_max', None)
        longest = max(lengths)
        for value in MATRIX_ROLLS:
            # Written out changes something only past the default's roll,
            # rolled by `value` only past `value`.
            reach = own if value == 0 else value
            if value != own and reach and longest > reach:
                twins.append(twin.set('k_unroll_max', value))
    return twins


#: Spilled bytes past which the default's spilling lets the rule take a
#: wider candidate than the default; at or below it no wider one is taken.
#: Over paper-1106 (88 SeisSol kernels a GPU, the rule's picks against the
#: default): where the default spilled the wider picks won (MI300A -50 %, MI210
#: -28 %, GH200 -12 % geomean), where it did not they won nothing and lost
#: big -- the Taylor expansions at 64 lanes 15 to 58 % on A100, GH200 and
#: GB200, viscoelastic `fluxLocalAll` in double 74 and 371 % on MI210, 25
#: picks there at -0.1 % together.
WIDE_SPILL = 0


def _resident_warps(result: Build, seen: Optional[Resources], hw
                    ) -> Optional[int]:
    """Warps an SM holds of `result` (`Options.lanes_occupancy_keep`): the
    blocks the compiler's registers admit (`Resources.register_blocks`, on
    NVIDIA a count of blocks), no more than shared memory and threads admit,
    times the warps a block holds.  None where the compiler did not say."""
    if seen is None or not seen.register_blocks:
        return None
    gen = result.generator
    wave = max(1, getattr(hw, 'vec_unit_length', 32))
    threads = gen.lanes.num_threads * gen.launch_config().mults_per_block
    per_block = -(-threads // wave)
    blocks = seen.register_blocks
    resident = getattr(gen, 'resident_blocks', None)
    if resident:
        blocks = min(blocks, resident)
    cap = (getattr(hw, 'max_threads_per_sm', None) or 2048) // wave
    return min(cap, blocks * per_block)

#: Parts where a narrower candidate is taken only where the default spills.
#: None any more: MI210's narrower picks lost 8 % together over paper-1106,
#: the time derivatives at 16 lanes 57, 84 and 45 % (elastic order 5,
#: viscoelastic orders 4 and 3 in single), and each of the three took its
#: code across CDNA 2's 32 KB of instruction cache, which `ICACHE_RAMP` now
#: charges; the narrower picks that won, up to 28 % at order 6, stayed inside
#: it or were past twice it either way.
NARROW_WHERE_SPILLING = ()

#: Lanes AMD does not take in single precision: eight.  Over the MI300A grid
#: (56 kernels) they were the rule's worst picks -- elastic orders 3 to 5 lost
#: 16 to 61 % to the default in paper-1106 -- and without them, with the
#: wider ones held to where the default spills (`WIDE_SPILL`), the loss
#: against the fastest candidate falls from 4.8 to 2.1 % (elastic orders 4
#: and 5 from 7.0 to 0.2 %).
AMD_SINGLE_NARROWEST = 16


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
        #: `chain_cycles` and `storage_key` of the build, taken where it was
        #: built.
        self.chain = summary.get('chain')
        self.storage = summary.get('storage')
        #: Blocks an SM holds by shared memory and threads (`Generator.
        #: resident_blocks`), for `_resident_warps`.
        self.resident_blocks = summary.get('resident')
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


def _rewritten(result: Build, alt: Candidate, pin: int, figure, context,
               shared: Optional[bool] = None
               ) -> Tuple[Build, Optional[float]]:
    """`alt` -- `result`'s candidate pinned every `pin` loads -- as the
    same build written again (`Generator.emitted_with_pin`), and its figure:
    the fence is the emitter's alone, so the bodies are `result`'s.  A build
    made for `alt` writes the same body under another kernel name."""
    from types import SimpleNamespace
    gen = result.generator
    text, metrics = gen.emitted_with_pin(pin, shared)
    ctx = alt.context(context, autotune='off')
    view = SimpleNamespace(candidate=alt, context=ctx, generator=SimpleNamespace(
        metrics=metrics, lanes=gen.lanes, stack=getattr(gen, 'stack', 1)))
    count = figure(view)
    stub = _Ahead(dict(lanes=gen.lanes.num_threads,
                       peak_pressure=metrics.peak_pressure,
                       code_units=metrics.code_units, rolled=_rolled(result),
                       mults=gen.launch_config().mults_per_block,
                       headers=list(gen.get_helper_headers()), kernel=text,
                       resident=getattr(gen, 'resident_blocks', None),
                       **_dispatch_figures(result)))
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
               kernel=gen.get_kernel(), resident=gen.resident_blocks,
               **_dispatch_figures(result))
    wave = max(1, getattr(context.target.hw, 'vec_unit_length', 32))
    if pin and not cand.get('pin_loads') and not gen.lanes.num_threads % wave:
        # The pinned build the walk may ask about: the same build written
        # again, a tenth of a second where a build takes a minute.
        alt = cand.set('pin_loads', pin)
        twin, count = _rewritten(result, alt, pin, figure, context)
        out['pinned'] = dict(count=count, peak_pressure=twin.generator.metrics.peak_pressure,
                             code_units=twin.generator.metrics.code_units,
                             kernel=twin.generator.get_kernel())
        if _global_twin(context, alt):
            alt = alt.set('pin_shared', False)
            twin, count = _rewritten(result, alt, pin, figure, context,
                                     shared=False)
            out['pinned_global'] = dict(
                count=count, peak_pressure=twin.generator.metrics.peak_pressure,
                code_units=twin.generator.metrics.code_units,
                kernel=twin.generator.get_kernel())
    return out


def _global_twin(context: Context, alt: Candidate) -> bool:
    """Whether pinned `alt` has a twin fenced on global loads only
    (`PIN_GLOBAL_TWIN`): not where shared loads count for nothing already."""
    if not PIN_GLOBAL_TWIN or alt.get('pin_shared') is False:
        return False
    return bool(getattr(context.get_user_options(), 'pin_shared', False))


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
            twin = summary.get('pinned_global')
            if twin is not None:
                alt = alt.set('pin_shared', False)
                out.append((alt, Build(
                    alt, generator=_Ahead(dict(summary, **twin)),
                    context=alt.context(context, autotune='off')),
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


def lane_floor(context: Context, rule: bool = True) -> int:
    """The narrowest lane count the scorers weigh.

    `lanes.MIN_LANES`, or the vector unit where narrower multiplications
    leave it idle (`Preferences.lanes_down_to_wave`, PVC under SPMD) -- unless
    the caller lets a sub-group hold several, as many as their broadcasts are
    split over (`Options.lanes_split_floor`, `Options.sycl_split_broadcast`):
    split, a narrow multiplication reads its lanes by register regions instead
    of indirect register accesses, and eight lanes of the order-4 kernels of
    eight fused simulations ran up to 62 % faster than sixteen on the Max
    1550, where the order-6 derivative still lost three-fold.
    """
    floor = lane_config.MIN_LANES
    if not context.target.prefs.lanes_down_to_wave:
        return floor
    wave = getattr(context.target.hw, 'vec_unit_length', 1)
    opts = context.get_user_options()
    blocks = 1
    # The lanes rule only (`rule`): its guards on the compiler's spill
    # (`SPILL_GROWTH`, `SPLIT_REF_GROWTH`) are what keep eight lanes out
    # where they lose.  The walks of `autotune=static` and `compiled` have
    # neither, and took order-6 `derivative` in single precision to eight
    # lanes at 66.1 ns an element against 12.7 at sixteen.
    if (rule and getattr(opts, 'lanes_split_floor', False)
            and context.target.pinned_sub_groups is not None):
        blocks = max(1, int(getattr(opts, 'sycl_split_broadcast', 0) or 0))
    return max(floor, wave // blocks)


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
        floor = lane_floor(context)
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
               generate,
               builds: Optional[Dict[Candidate, Build]] = None) -> Candidate:
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
    # The caller's, where it asks for them (`small_variant`).
    builds = {} if builds is None else builds
    builds[origin] = first
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
    cands += [c for c in _roll_candidates(first, context, origin, fixed)
              if c not in cands]
    cands += [c for c in _matrix_candidates(descr_factory(), first, context,
                                            origin, fixed)
              if c not in cands]
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
    # A roll the default's reductions are too short for builds the default.
    own = getattr(builds[origin].generator, 'metrics', None)
    for cand in cands:
        if (cand.get('k_unroll_max') is not None and cand != origin
                and cand.lanes == origin.lanes and cand in counts
                and counts[cand] == counts[origin]
                and getattr(getattr(builds[cand].generator, 'metrics', None),
                            'code_units', None)
                == getattr(own, 'code_units', None)):
            del counts[cand]
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
        wanted += [a.set('pin_shared', False) for a in wanted
                   if _global_twin(context, a)]
        reports.update(_compile_ahead(
            scorer, {a: b for a, (b, _) in pinned_ahead.items()},
            [a for a in wanted if a in pinned_ahead
             and pinned_ahead[a][0].ok], jobs))

    miss = getattr(hw, 'icache_miss_factor', None)
    refetch = ICACHE_SHARE * max(0.0, miss - 1.0) if miss else 0.0

    def twin(cand: Candidate, alt: Candidate
             ) -> Optional[Tuple[Build, Optional[float], Resources]]:
        """Pinned `alt` of `cand`: its build, its figure where known, and
        what the compiler says of it; None where either fails."""
        shared = False if alt.get('pin_shared') is False else None
        if alt in pinned_ahead:
            result, count = pinned_ahead[alt]
        elif hasattr(builds[cand].generator, 'emitted_with_pin'):
            # The same build written again: the fence is the emitter's.
            result, count = _rewritten(builds[cand], alt, PIN_EVERY, figure,
                                       context, shared=shared)
            pinned_ahead[alt] = (result, count)
        else:
            result, count = build(descr_factory, context, alt, generate), None
        if not result.ok:
            return None
        try:
            if alt in reports:
                other = reports[alt]
                if isinstance(other, Exception):
                    raise other
            else:
                other = scorer.resources(result)
        except Exception:                     # noqa: BLE001
            other = None
        if other is None:
            return None
        if count is None:
            count = figure(result)
        return result, count, other

    def pinned(cand: Candidate, seen: Optional[Resources]
               ) -> Tuple[Candidate, Optional[Resources]]:
        """`cand` and what the compiler says of it, or its pinned build and
        what the compiler says of that, where the pin pays (`PIN_SPILL`):
        fenced on every load, or on global loads only (`PIN_GLOBAL_TWIN`),
        whichever costs less.  A candidate the estimate settled is pinned by
        the estimate as well: built with `pin_loads='auto'`
        (`abstract_instruction.PIN_MARGIN`)."""
        if pins and cand in settled:
            alt = cand.set('pin_loads', 'auto')
            builds[alt], counts[alt] = builds[cand], counts[cand]
            settled.add(alt)
            return alt, seen
        if (not pins or seen is None or seen.spill_bytes < PIN_SPILL
                or cand.get('pin_loads')
                or builds[cand].generator.lanes.num_threads % wave):
            return cand, seen
        alts = [cand.set('pin_loads', PIN_EVERY)]
        if _global_twin(context, alts[0]):
            alts.append(alts[0].set('pin_shared', False))
        lanes = builds[cand].generator.lanes.num_threads
        taken = None
        got = [(alt, twin(cand, alt)) for alt in alts]
        # The twins are one build fenced two ways, and the emitter counts
        # them alike; what the fences cost is the loads the compiler may not
        # reuse across them, which its code says: the count of the fully
        # fenced twin, scaled by the code of each (`PIN_GLOBAL_TWIN`).
        ref = got[0][1][2].code_bytes if got[0][1] is not None else None
        for alt, one in got:
            if one is None or one[2].spill_bytes > PIN_KEEP * seen.spill_bytes:
                continue
            result, count, other = one
            count = counts[cand] if count is None else count
            scale = (other.code_bytes / ref if ref and other.code_bytes
                     else 1.0)
            price = (count * scale * (1.0 + icache_charge(other.code_bytes, hw,
                                                          refetch))
                     + SPILL_WEIGHT * other.spill_bytes / elem * lanes / wave)
            if taken is None or price < taken[0]:
                taken = (price, alt, result, count, other)
        if taken is None:
            return cand, seen
        _, alt, result, count, other = taken
        builds[alt] = result
        counts[alt] = count
        return alt, other

    # What the compiler said of the default as built, before any pin: the
    # guards ask how much it spills and how many registers it holds.
    unpinned = report

    def spill_cut(cand: Candidate, seen: Resources
                  ) -> Tuple[Candidate, Optional[Resources]]:
        """`cand`, or `cand` cut in two where IGC spills it and the cut
        references what IGC spilled far less often for little more to execute
        (`FISSION_REF_SHARE` and the rest): Intel, where the compiler states
        no code size for `fissioned` to weigh."""
        ask = getattr(scorer, 'spill_references', None)
        if (not ask or not seen.spill_bytes
                or str(hw.vendor).lower() != 'intel'):
            return cand, seen
        from .generator import fission_cuts
        if not fission_cuts(descr_factory()):
            return cand, seen
        mine = ask(builds[cand])
        if not mine:
            return cand, seen
        alt = cand.set('fission', 2)
        result = builds.get(alt) or build(descr_factory, context, alt, generate)
        if not result.ok:
            return cand, seen
        count = figure(result)
        whole = counts[cand]
        if count is None or count > FISSION_DENSE_GROWTH * whole:
            return cand, seen
        try:
            other = scorer.resources(result)
        except Exception:                     # noqa: BLE001
            other = None
        theirs = ask(result) if other is not None else None
        if theirs is None:
            return cand, seen
        # References are the thread's, the count an element's: a sub-group
        # wider than the multiplication holds several elements.
        lanes = builds[cand].generator.lanes.num_threads
        share = lanes / max(lanes, context.target.sub_group_width(lanes)
                            or lanes)
        fewer = (theirs <= FISSION_REF_SHARE * mine
                 and count <= FISSION_COUNT_GROWTH * whole)
        denser = (mine - theirs) * share >= FISSION_REF_SAVING * whole
        listed = getattr(scorer, 'listed_instructions', None)
        code = listed(builds[cand]) if listed else None
        bulky = (code is not None and code >= FISSION_LISTED
                 and count <= FISSION_LARGE_GROWTH * whole)
        if not (fewer or denser or bulky):
            return cand, seen
        builds[alt], counts[alt] = result, count
        return alt, other

    def fissioned(cand: Candidate, seen: Optional[Resources]
                  ) -> Tuple[Candidate, Optional[Resources]]:
        """`cand`, or `cand` cut into as many pieces as its code takes past
        the instruction cache (`fission_pieces`, from the size the compiler
        wrote) where that costs less (`FISSION_TWIN`).  The cut build counts
        more -- every piece reads its operands again -- and each piece streams
        only its own code, which `cost` charges."""
        if (not FISSION_TWIN or seen is None or cand.get('fission')
                or 'fission' in dict(context.get_asked_options().asked())
                or (fixed and 'fission' in fixed)):
            # A cut the caller asked for either way stands.
            return cand, seen
        if not getattr(seen, 'code_bytes', None):
            return spill_cut(cand, seen)
        from .generator import fission_cuts, fission_pieces
        pieces = fission_pieces(seen.code_bytes, hw)
        if pieces <= 1 or not fission_cuts(descr_factory()):
            return cand, seen
        alt = cand.set('fission', pieces)
        result = builds.get(alt) or build(descr_factory, context, alt, generate)
        if not result.ok:
            return cand, seen
        count = figure(result)
        if count is None:
            return cand, seen
        try:
            other = scorer.resources(result)
        except Exception:                     # noqa: BLE001
            other = None
        if other is None or other.spill_bytes > max(limit, seen.spill_bytes):
            return cand, seen
        builds[alt], counts[alt] = result, count
        if cost(alt, other) < cost(cand, seen):
            return alt, other
        return cand, seen

    base, report = pinned(origin, report)
    # Where the count ranks on NVIDIA, the caller may let a few hundred bytes
    # of spilling count as none (`Options.lanes_spill_floor`) and keep the
    # warps a multiprocessor holds (`Options.lanes_occupancy_keep`).
    opts = context.get_user_options()
    counted = not by_bound and str(hw.vendor).lower() == 'nvidia'
    floor = int(getattr(opts, 'lanes_spill_floor', 0) or 0) if counted else 0
    keep = (float(getattr(opts, 'lanes_occupancy_keep', 0.0) or 0.0)
            if counted else 0.0)
    limit = max(SPILL_GROWTH * report.spill_bytes, floor)
    held = _resident_warps(builds[origin], unpinned, hw) if keep else None
    large_grf = _grf_twin(context, fixed)

    def grf(cand: Candidate, seen: Optional[Resources]
            ) -> Tuple[Candidate, Optional[Resources]]:
        """`cand`, or `cand` with the large register file where IGC spills
        it and spills the large build less, at a lower cost (`GRF_TWIN`).
        The code is the same and so is its count: what changes is the
        spilling the compiler reports, which `cost` charges."""
        if (not large_grf or seen is None or not seen.spill_bytes
                or str(cand.get('sycl_grf', 'default')) != 'default'):
            return cand, seen
        alt = cand.set('sycl_grf', '256')
        result = builds.get(alt) or build(descr_factory, context, alt, generate)
        if not result.ok:
            return cand, seen
        try:
            other = scorer.resources(result)
        except Exception:                     # noqa: BLE001
            other = None
        if other is None or other.spill_bytes >= seen.spill_bytes:
            return cand, seen
        builds[alt] = result
        counts[alt] = counts[cand] if cand in counts else figure(result)
        if counts[alt] is None:
            return cand, seen
        if cost(alt, other) < cost(cand, seen):
            return alt, other
        return cand, seen

    def cost(cand: Candidate, seen: Resources) -> float:
        lanes = builds[cand].generator.lanes.num_threads
        # A cut kernel runs its pieces one after another, each over its own
        # code (`Options.fission`).
        pieces = cand.get('fission')
        code = seen.code_bytes
        if code and isinstance(pieces, int) and pieces > 1:
            code = code / pieces
        charge = icache_charge(code, hw, refetch)
        if by_bound:
            spilled = (pipe_bound(builds[cand], seen.spill_bytes)
                       if seen.spill_bytes else counts[cand])
            waves = max(1, seen.register_blocks or 1)
            if issue and seen.register_blocks:
                issued = executed_instructions(builds[cand])
                if issued is not None:
                    spilled = max(spilled, issued / seen.register_blocks)
            return spilled * max(1.0, latency / waves) * (1.0 + charge)
        return (counts[cand] * (1.0 + charge)
                + SPILL_WEIGHT * seen.spill_bytes / elem * lanes / wave)

    # The default as built, whole: what anything taken has to beat by the
    # margin, its own cut twin included.
    default, whole = cost(base, report), base
    base, report = fissioned(base, report)
    base, report = grf(base, report)
    best, least = base, cost(base, report)
    # Where the default spills on AMD, a build that spills nothing wins over
    # every build that does (`CLEAN_FIRST`): until one is found the compiler
    # is asked about every candidate, after that only about those whose
    # bound alone is still below the best clean one's cost.
    clean_first = CLEAN_FIRST and by_bound and report.spill_bytes > 0
    clean: Optional[Tuple[Candidate, float]] = None
    vendor = str(hw.vendor).lower()
    default_lanes = builds[origin].generator.lanes.num_threads

    def per_element(refs: int, lanes: int) -> float:
        """`refs` of a build at `lanes` per element: a sub-group wider than
        the multiplication holds several elements, and its thread spills for
        all of them."""
        width = context.target.sub_group_width(lanes) or lanes
        return refs * lanes / max(lanes, width)

    def split_traffic(cand: Candidate, seen: Resources) -> bool:
        """Whether `cand` is a split sub-group that references what IGC
        spilled more than `SPLIT_REF_GROWTH` times as often per element as
        the default, where both spill.  Asked of the compiler's listing only
        then: two builds more, the default's once."""
        lanes = builds[cand].generator.lanes.num_threads
        width = context.target.sub_group_width(lanes)
        ask = getattr(scorer, 'spill_references', None)
        if not (ask and width and lanes < width and seen.spill_bytes
                and unpinned.spill_bytes):
            return False
        mine = ask(builds[cand])
        theirs = ask(builds[origin])
        if mine is None or not theirs:
            return False
        return (per_element(mine, lanes)
                > SPLIT_REF_GROWTH * per_element(theirs, default_lanes))

    def guarded(cand: Candidate) -> bool:
        """Whether the lanes of `cand` are out before any compiler is asked:
        wider than the default where the default spills no more than
        `WIDE_SPILL`, narrower where it spills nothing on the parts of
        `NARROW_WHERE_SPILLING`, eight in single precision on AMD
        (`AMD_SINGLE_NARROWEST`)."""
        lanes = builds[cand].generator.lanes.num_threads
        spills = unpinned.spill_bytes > max(WIDE_SPILL, floor)
        if lanes > default_lanes and not spills:
            return True
        if (lanes < default_lanes and not unpinned.spill_bytes
                and str(hw.model) in NARROW_WHERE_SPILLING):
            return True
        return (vendor == 'amd' and context.fp_type is Datatype.F32
                and lanes < AMD_SINGLE_NARROWEST)

    for cand in sorted((c for c in counts if c not in (origin, base)),
                       key=counts.get):
        if clean is not None:
            if counts[cand] >= clean[1]:
                break
        elif not clean_first and counts[cand] >= least:
            break
        if guarded(cand):
            continue
        try:
            seen = resources(cand)
        except Exception:                     # noqa: BLE001
            seen = None
        cand, seen = pinned(cand, seen)
        if seen is None or seen.spill_bytes > limit:
            continue
        if split_traffic(cand, seen):
            continue
        if held and (_resident_warps(builds[cand], seen, hw) or 0) < keep * held:
            continue
        cand, seen = fissioned(cand, seen)
        cand, seen = grf(cand, seen)
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
    return best if least < (1.0 - DEVIATION_MARGIN) * default else whole


def _grf_twin(context: Context, fixed: Optional[Mapping[str, Any]]) -> bool:
    """Whether `lanes_rule` offers the large register file (`GRF_TWIN`):
    Intel under DPC++'s SPMD lowering -- the explicit vector takes it already
    -- and a register file the caller did not state."""
    hw = context.target.hw
    asked = dict(context.get_asked_options().asked())
    return bool(GRF_TWIN and str(hw.vendor).lower() == 'intel'
                and context.target.backend == 'oneapi'
                and not context.target.explicit_simd
                and 'sycl_grf' not in asked
                and not (fixed and 'sycl_grf' in fixed))


#: Where VC -- IGC's vector backend, which builds every explicit-SIMD kernel
#: -- spills a kernel at every width, the width that spills least: in double
#: precision, and not in single (`esimd_width_rule`).  Measured on the Max
#: 1550 at a batch of 262144 (sng2 jobs 560300 and 560306, IGC 2.41.5),
#: against the deduced 32 lanes, which VC spills 13.8 to 33.8 kB at:
#:
#: - double precision, eight lanes spilling 4.3 to 22 kB: order-5
#:   `derivative` -48 %; order-6 `derivative` -37 %, `localFluxAll` -75 %,
#:   `volume` -38 %;
#: - single precision, order-6 `volume`: sixteen lanes spill 3.3 kB against
#:   5.8 and run 17 % longer.
ESIMD_LEAST_SPILL = frozenset({Datatype.F64})


def _register_lanes(context: Context) -> int:
    """How many elements of the kernel's type one vector register holds: the
    narrowest width `esimd_width_rule` weighs.  Narrower, a vector leaves part
    of every register and every instruction it occupies empty.  64 bytes on
    PVC: sixteen floats, eight doubles."""
    hw = context.target.hw
    width = (getattr(hw, 'vec_unit_length', 16)
             * getattr(hw, 'hw_fp_word_size', 4))
    return max(1, width // context.fp_type.size())


def esimd_width_rule(descr_factory, context: Context, scorer: 'CompiledScore',
                     origin: Candidate, first: Build, *, generate
                     ) -> Candidate:
    """The vector width under the explicit vector (`autotune=lanes` there).

    The widest width VC builds without spilling, from the deduced count down
    to one register's worth (`_register_lanes`).  Where every width spills,
    the one that spills least in double precision (`ESIMD_LEAST_SPILL`), and
    the default otherwise.

    Spilling and nothing else, because nothing else ranks these widths.
    Ranked by the instructions an element executes, eight lanes come out
    ahead of 32 for order-5 `neighboringFlux` in double precision.  That
    kernel spills at neither width and runs 37 % longer at eight (sng2
    560300).  The walk of `autotune=compiled` (`CompiledScore`) asks VC about
    the same builds: the widest unless it spills, and where all spill, the
    widest.  It kept 32 lanes for the order-6 kernels in double precision and
    missed the -37 to -75 % above.  Its rolled twins (`k_roll`) that spill
    nothing at 32 lanes ran no faster than the default (order-5 `derivative`
    +3 %), so this rule turns the width alone.

    Asked from the widest down, stopping at the first build that spills
    nothing: a kernel whose default spills nothing costs one compile.
    """
    base = origin.lanes
    floor = _register_lanes(context)
    spilled: List[Tuple[int, int, Candidate]] = []
    t = base.num_threads
    while t >= floor:
        cand = origin if t == base.num_threads else origin.set(
            'lanes', LaneConfig(t, base.num_active_threads, base.lead_width))
        result = (first if cand == origin
                  else build(descr_factory, context, cand, generate))
        report = None
        if result.ok:
            try:
                report = scorer.resources(result)
            except Exception:                 # noqa: BLE001
                report = None
        if report is not None:
            if not report.spill_bytes:
                return cand
            spilled.append((report.spill_bytes, -t, cand))
        if t & (t - 1):
            break
        t //= 2
    if spilled and context.fp_type in ESIMD_LEAST_SPILL:
        return min(spilled, key=lambda s: s[:2])[2]
    return origin


def _esimd_mode(descr_factory, context: Context, cache: Optional[str],
                fixed: Optional[Mapping[str, Any]], generate
                ) -> Optional[Candidate]:
    """`autotune=lanes` under the explicit vector: `esimd_width_rule` where
    `icpx` can be asked and the caller left the width open, the default
    otherwise.  Remembered by the default build's source, as the lanes rule's
    picks are (`Options.autotune_cache`)."""
    if fixed and 'lanes' in fixed:
        return None
    scorer = CompiledScore()
    if not scorer.available(context):
        warnings.warn('autotune=lanes: no compiler for this target '
                      '(Toolchain, TF_ICPX); the default stands')
        return None
    origin = start(descr_factory(), context)
    for name, value in (fixed or {}).items():
        origin = origin.set(name, value)
    first = build(descr_factory, context, origin, generate)
    if not first.ok:
        return None
    _load(cache)
    key = _key(first, 'esimd-width')
    if key in _PICKS:
        return Candidate.from_dict(_PICKS[key])
    pick = esimd_width_rule(descr_factory, context, scorer, origin, first,
                            generate=generate)
    _store(cache, key, pick)
    return pick


def _lanes_mode(descr_factory, context: Context, cache: Optional[str],
                fixed: Optional[Mapping[str, Any]], generate,
                dispatch: Optional[Dict[str, Any]] = None
                ) -> Optional[Candidate]:
    """`autotune=lanes`: `lanes_rule` where a compiler can be asked, the
    default where none can or the vendor was never measured (`RANKED_BY`).
    Under the explicit vector the width alone (`esimd_width_rule`).
    Remembered by the default build's source, as the walk's picks are
    (`Options.autotune_cache`)."""
    hw = context.target.hw
    if context.target.explicit_simd and str(hw.vendor).lower() == 'intel':
        return _esimd_mode(descr_factory, context, cache, fixed, generate)
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
    opts = context.get_user_options()
    for name, tag in (('lanes_occupancy_keep', 'occ'),
                      ('lanes_spill_floor', 'floor'),
                      ('lanes_split_floor', 'split')):
        value = getattr(opts, name, 0)
        if value:
            # Picks of a walk with another guard are not this one's.
            mode += f'+{tag}{value}'
    key = _key(first, mode)
    # The small batches' candidate apart, so that the picks of a walk without
    # it are still this one's.
    small_key = None if dispatch is None else _key(first, mode + '+small')
    if key in _PICKS and (small_key is None or small_key in _PICKS):
        if small_key is not None:
            small = _PICKS[small_key].get('small')
            dispatch['small'] = (None if small is None
                                 else Candidate.from_dict(small))
        return Candidate.from_dict(_PICKS[key])
    builds: Dict[Candidate, Build] = {}
    pick = lanes_rule(descr_factory, context, scorer, origin, first, fixed,
                      generate=generate, builds=builds)
    _store(cache, key, pick)
    if small_key is not None:
        small = small_variant(descr_factory, context, builds, pick, generate)
        _store(cache, small_key,
               {'small': None if small is None else small.to_dict()})
        dispatch['small'] = small
    return pick


#: How much shorter the small batches' candidate's chain has to be than the
#: pick's before a second kernel is built for them (`small_variant`): the
#: lanes rule's own margin.  It hardly matters for the time: over the
#: paper-1106 sweeps none, 5 and 10 % leave the same against the faster of
#: the two to within half a percent on every GPU; it saves the kernels that
#: would win by less than it.
DISPATCH_MARGIN = 0.05


#: What the small batches' kernel (`small_variant`) is built with on top of
#: its candidate, where the caller did not ask otherwise: no transfer issued
#: ahead within the element (`Options.enable_move_loads`) or for the next one
#: (`Options.enable_wrap_loads`).  Below the switch a lane group has one
#: element at most, so there is no next one to fetch for, and what the moved
#: transfers hold costs registers in a launch that is one wave's chain long.
#: Measured as the small-batch sweep's `nomw` against the default, geomean
#: over 20 kernels (elastic orders 4 and 6, viscoelastic order 4) at 16 to
#: 1024 elements: MI300A 0.92-0.94 (AAC7 15837; Taylor -10..-19 %), GB200
#: 0.94-0.97 (Horizon 88742); worst +3 and +13 %.
SMALL_BATCH_OPTIONS = {'enable_move_loads': False, 'enable_wrap_loads': False}


#: Waves one SIMD (on NVIDIA, one scheduler) runs side by side before they
#: lengthen each other's chain: what a wave's launch takes per clock it issues
#: at up to 64 elements, over the paper-1106 sweeps (a fit of the launch time
#: over `chain_cycles`, times the clock: 5.5 on GH200, 10 on A100, 6.1 on
#: GB200, 1.4 on MI300A, 3.8 on MI210), in whole waves.  Up to that many waves
#: a SIMD the small batches' kernel (`small_variant`) launches as fast as with
#: one; past it every wave more is a turn more, and the throughput pick takes
#: over (`Generator._generate_dispatch`).  On MI300A, the small kernels the
#: tuner takes for SeisSol's elastic order 4 and 6 and viscoelastic order 4
#: against the lanes pick, switching at one wave a SIMD left 1.1 % against the
#: faster of the two at every batch, at four 1.5 %, and the lanes pick alone
#: 9.7 % (AAC7 15837); at the pick's own resident capacity, where the runtime
#: put the switch first, 64-lane kernels ran 4096 elements up to 53 % slower.
#: Over the paper-1106 pairs both switches leave the same to within 0.6 %.
#: A part without an entry takes its vendor's narrowest measured.
DISPATCH_OVERLAP = {'sm_80': 10, 'sm_90': 5, 'sm_100': 6, 'gfx942': 1,
                    'gfx90a': 3}


def dispatch_overlap(hw) -> int:
    """`DISPATCH_OVERLAP` for `hw`."""
    model = str(getattr(hw, 'model', ''))
    if model in DISPATCH_OVERLAP:
        return DISPATCH_OVERLAP[model]
    return 5 if str(getattr(hw, 'vendor', '')).lower() == 'nvidia' else 1


def chain_cycles(result: Build) -> Optional[float]:
    """Clocks one wave of `result` keeps the busiest pipe busy: the pipe
    bound of an element (`pipeline.of`, not stretched for waves or spills)
    times the elements a wave holds.  None where the build counted no mix.

    What a launch the device holds at once lasts, give or take its fixed
    cost: every wave runs its chain once, side by side with the others, and
    the longest decides.  Over the paper-1106 sweeps (88 kernels, the default
    and the lanes pick, 16 to 64 elements) it ranks the pair the way the
    launches do in 73 of 88 kernels on GH200, 84 on A100, 72 on GB200, 77 on
    MI300A and 70 on MI210; a fit of the launch time over it and the loads and
    code a wave issues ranks no better (73/84/72/75/68), and the same with the
    code left out (73/84/72/77/69).
    """
    gen = result.generator
    if isinstance(gen, _Ahead):
        return gen.chain
    from tensorforge.analysis import pipeline
    if not getattr(getattr(gen, 'metrics', None), 'issue_mix', None):
        return None
    bound = pipeline.of(gen)
    if bound is None:
        return None
    wave = max(1, getattr(result.context.target.hw, 'vec_unit_length', 32))
    return bound.cycles * wave / max(1, gen.lanes.num_threads) * _stack_of(gen)


def _dispatch_figures(result: Build) -> Dict[str, Any]:
    """What `small_variant` asks of a build made in another process
    (`_Ahead`), where a second kernel was asked for at all
    (`Options.batch_dispatch`).  A figure that cannot be had is missing, and
    the build is then no small batches' candidate."""
    if not getattr(result.context.get_user_options(), 'batch_dispatch', 0):
        return {}
    out = {}
    for name, figure in (('chain', chain_cycles), ('storage', storage_key)):
        try:
            out[name] = figure(result)
        except Exception:                     # noqa: BLE001
            pass
    return out


def storage_key(result: Build) -> Optional[tuple]:
    """How the build stores the operands the host lays out, by name
    (`Generator._stored_operands`): what two kernels launched on one set of
    buffers have to agree on.  None where the build does not say."""
    gen = result.generator
    if isinstance(gen, _Ahead):
        return gen.storage
    built = getattr(gen, 'built', None)
    stored = getattr(built, '_stored_operands', None)
    if stored is None:
        return None
    return tuple(sorted(((tensor.alias,) + tuple(state)
                         for tensor, state in stored().values()),
                        key=lambda entry: str(entry[0])))


def small_variant(descr_factory, context: Context,
                  builds: Mapping[Candidate, Build], pick: Candidate,
                  generate) -> Optional[Candidate]:
    """The candidate that runs a batch the device holds at once
    (`Options.batch_dispatch`): of everything the lanes rule built that
    stores the operands as `pick` does (`storage_key`), the one with the
    shortest chain (`chain_cycles`), where it is `DISPATCH_MARGIN` shorter
    than `pick`'s; None otherwise.

    Ranked by the chain alone, without the compiler's word: the measured
    pairs ranked as well without the spilling as with it, and the waves a
    build holds do not matter where the batch fills fewer.  The storage,
    because the host lays the operands out once for both kernels: an
    operator interleaved for sixteen lanes is not one 64 can read.  `pick`
    itself is built again where the walk took it as a twin it never built on
    its own (pinned, fissioned).
    """
    own_build = builds.get(pick)
    if own_build is None:
        own_build = build(descr_factory, context, pick, generate)
    if not own_build.ok:
        return None
    own, stored = chain_cycles(own_build), storage_key(own_build)
    if not own or stored is None:
        return None
    chains: Dict[Candidate, float] = {}
    for cand, result in builds.items():
        if result.ok and storage_key(result) == stored:
            chain = chain_cycles(result)
            if chain is not None:
                chains[cand] = chain
    if not chains:
        return None
    best = min(chains, key=chains.get)
    if best == pick or chains[best] >= (1.0 - DISPATCH_MARGIN) * own:
        return None
    return best
