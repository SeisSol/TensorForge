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

from tensorforge.common.basic_types import Addressing
from tensorforge.common.context import Context, Options
from tensorforge import toolchain
from tensorforge.toolchain import Resources, Toolchain
from tensorforge.generators import lanes as lane_config
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


def contraction_lengths(descrs) -> List[int]:
    """The extent of every contracted axis, one per multilinear operand axis
    that the destination does not have (`target` -1)."""
    out = []
    for d in _flat(descrs):
        target = getattr(d, 'target', None)
        if target is None or len(getattr(d, 'ops', ())) < 2:
            continue
        for op, axes in zip(d.ops, target):
            box = getattr(op, 'bbox', None)
            if box is None:
                continue
            for axis, t in enumerate(axes):
                if t == -1 and axis < box.rank():
                    out.append(int(box.size(axis)))
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
    from tensorforge.generators.descriptions import ForDescr
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
    hw = context.target.hw
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
    if hw.vendor == 'nvidia' and contraction_lengths(descrs):
        knobs.append(Knob('tensor_cores', lambda c: (False, True)))
        knobs.append(Knob('mma_prefetch',
                          lambda c: ((1, 2) if c.get('tensor_cores') else (None,))))
    return knobs


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


def simple_space(descrs, context: Context) -> List[Knob]:
    """The knobs `Options.autotune` turns: only those whose every value is
    safe to ship without the caller knowing.

    Lane counts are the powers of two from the deduced one down to
    `lanes.MIN_LANES` -- the geometries measured on every target; the
    divisors of the extent (5 and 7 lanes for 20 and 35 rows) rank well on
    both scorers and have never been timed.  A width of two only where one
    instruction does two FMAs (`Target.packed_fma_width`) and the extent is even:
    elsewhere it is two scalar FMAs, and at 35 rows on GB200 it was 70 %
    slower.  Merging where something repeats, and rolling by the largest
    divisor up to 32.  Not `prepare_operands`: the host packs for it.  Not
    the matrix path, whose default is per vendor and not yet measured.

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
    from tensorforge.generators.descriptions import ElementwiseDescr
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
    # 12.7 at sixteen.  So the floor is the vector unit there, and
    # `MIN_LANES` elsewhere, where a narrow multiplication shares its wave.
    floor = lane_config.MIN_LANES
    if hw.vendor == 'intel' and not context.target.explicit_simd:
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
    # everything else held still; the walk turns `k_roll`, `k_width` and
    # `register_temporaries` around whatever width it took first, and a bad
    # first step is not a bad step alone.  So the axis goes, not its lower end.
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
    if explicit and context.get_user_options().autotune != 'compiled':
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
        if context.get_user_options().autotune == 'compiled':
            geometries += _wider(base, rows, hw)
    if (context.fp_type == Datatype.F32
            and context.target.packed_fma_width(Datatype.F32) == 2
            and context.target.lead_vectors() and rows % 2 == 0
            and base.lead_width == 1):
        # On AMD only where a lane holds one pair per column.  hipcc spilled
        # every gfx942 build that needed two -- 16 lanes at 35 and 56 rows,
        # 32 at 80 and 120: 512 registers and 2 to 8 KB of scratch, where the
        # model saw 200 -- and none that needed one.  ptxas does not: 16 lanes
        # at width two was GB200's fastest at 80 and 120 rows.
        one_pair = hw.vendor == 'amd'
        geometries += [LaneConfig(g.num_threads, base.num_active_threads, 2)
                       for g in list(geometries)
                       if g.num_threads <= rows
                       and (not one_pair or 2 * g.num_threads >= rows)]
    knobs = [Knob('lanes', lambda c, g=tuple(geometries): g)] if len(geometries) > 1 else []
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
    if (hw.vendor == 'intel'
            and not context.target.explicit_simd
            and any(t.addressing == Addressing.NONE for t in _tensors(descrs))):
        knobs.append(Knob('preload_globals', lambda c: (False, True)))
    # Reading an array temporary out of its producer's register image trades
    # a shared buffer for registers that stay live to the last reader, and
    # which of the two a kernel wants is measured rather than argued: over 70
    # corpus cases on sm_120 it is a geomean of 1 % for `all`, with
    # `mixed/ew_then_ew` at +72 % and five cases 4-6 % the other way;
    # SeisSol's damage step gains 6 %, and 11 % once its material part is a
    # kernel of its own.  Only where something is written that a later
    # operation reads: with no intermediate there is no image to keep, and the
    # knob would spend a build to arrive back where it started.
    written = {id(d.writes().tensor) for d in flat
               if getattr(d, 'writes', None) is not None
               and d.writes() is not None}
    if written & {id(v.tensor) for d in flat for v in d.reads()}:
        knobs.append(Knob('register_temporaries', lambda c: ('all', 'scalars')))
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
    """The configuration the generator would pick unasked: the deduced lanes
    and every option at the base context's value."""
    base = lane_config.deduce(_flat(descrs), context)
    return Candidate(lanes=base)


def enumerate_space(knobs: Sequence[Knob], origin: Candidate) -> Iterable[Candidate]:
    """Every point, conditional knobs included.  For a small space."""
    def expand(i, cand):
        if i == len(knobs):
            yield cand
            return
        for v in knobs[i].values(cand):
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


def build(descr_factory, base: Context, candidate: Candidate) -> Build:
    """Generate `candidate` from a fresh descriptor list.

    Fresh every time, and not as a convenience: preparing an operand stores
    its order on the `Tensor` itself, so a build at eight lanes leaves an
    eight-lane interleave behind for the next build to inherit.
    """
    from tensorforge.generators.generator import Generator
    # Tuning off for the trial: it asks what *this* candidate costs, so it
    # builds this one and does not go looking for another.  With a knob pinned
    # the tuner does not stand aside when a geometry is given, so a trial that
    # kept `autotune` would open a walk of its own, once per trial, all the
    # way down.
    ctx = candidate.context(base, autotune='off')
    ctx.measure_pressure = True
    try:
        gen = Generator(descr_factory(), ctx, lanes=candidate.lanes)
        # asked a question, not emitting: nothing it builds reaches a file
        gen._announce_identity = False
        gen.generate()
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
    return gen._num_threads, hw.vec_unit_length, (gen.resident_blocks or 0) * mults


def static_score(result: Build):
    """What the build alone says, per multiplication rather than per block.

    Past the register file first (`_over_budget`), but only as *whether* and
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
    block holds, since eight lanes put four times as many in a block as 32.

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
    issue = _least_cycles(gen, lanes, wave, spill_bytes=float(over),
                          resident=resident,
                          hw=result.context.target.hw)
    return (over > 0, issue, -resident,
            _icache_over(result), _granule(gen.peak_pressure or 0),
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
    cycles = b.cycles if b is not None else (gen.emitted_work or 0) * lanes / wave
    return cycles * _shortfall(lanes, wave, resident, hw)


def _shortfall(lanes: int, wave: int, resident: int, hw) -> float:
    """How much of the bound the machine cannot reach for want of warps.

    One where the resident warps meet the issue ports, and the ratio short of
    them otherwise.  `resident` of zero means the caller does not know, and
    an unknown residency stretches nothing: guessing here would rank by a
    number nobody measured.
    """
    if not resident or hw is None:
        return 1.0
    from tensorforge.analysis import pipeline
    ports = pipeline.resources(hw).get('issue')
    if ports is None or ports.rate <= 0:
        return 1.0
    warps = max(1, -(-lanes // max(1, wave)))
    return max(1.0, ports.rate / max(1, resident * warps))


def _icache_over(result: Build) -> int:
    """Kilobytes of code past the instruction cache (`analysis.icache`).

    Right after the register file and before occupancy: a body that does not
    fit is fetched again on every iteration of the batch loop, which no number
    of resident multiplications makes up for.  A cliff and not a slope, so 0
    below the capacity whatever the size, and every kilobyte past it counts.
    """
    from tensorforge.analysis.icache import icache_excess
    hw = result.context.target.hw
    return icache_excess(getattr(result.generator, 'code_units', None),
                         hw) // 1024


#: Bytes below which two modeled footprints are the same footprint: sixteen
#: registers.  The model is within about forty registers of what a compiler
#: allocates, so a difference of eleven bytes -- merged and written-out
#: `local_flux` at b = 80, 1359 against 1348 -- is not one the compiler makes;
#: hipcc spilled 6 KB for the written-out one and nothing for the merged.
_GRANULE = 64


def _granule(value) -> int:
    return int(value // _GRANULE)


#: Registers per lane as a function of the modeled bytes, per vendor, fitted
#: over builds that did not spill: `(intercept, slope)` on bytes / 4.
#: NVIDIA: ptxas sm_100a, 52 builds, residuals within 40.  AMD: hipcc gfx942,
#: 29 builds, residuals within 38 -- no intercept worth the name.
_REGISTER_FIT = {'nvidia': (51.0, 1.05), 'amd': (0.0, 1.26)}


def register_estimate(result: Build) -> Optional[float]:
    """Registers per lane the target compiler is expected to allocate."""
    hw = result.context.target.hw
    fit = _REGISTER_FIT.get(hw.vendor)
    peak = result.generator.peak_pressure
    if fit is None or not peak:
        return None
    return fit[0] + fit[1] * peak / 4


def _register_blocks(result: Build) -> Optional[int]:
    """Blocks per CU the register file admits, where that is what decides.

    AMD only.  A CDNA lane has 512 registers, VGPRs and AGPRs together, and a
    SIMD holds as many waves as fit -- eight at 64 registers, one above 256.
    On gfx942 over `local_flux` a third of the geometries sat at one wave per
    SIMD without spilling a byte, which blocks per CU from shared memory and
    threads alone never shows.  NVIDIA keeps what is exact: its fit has an
    intercept that the occupancy would inherit, and the eight lanes that GB200
    ran fastest sit right at the limit.
    """
    hw = result.context.target.hw
    if hw.vendor != 'amd':
        return None
    regs = register_estimate(result)
    if regs is None:
        return None
    file = (hw.max_reg_per_thread or 1024) // 4
    granule = 8
    per_lane = max(granule, -(-int(regs) // granule) * granule)
    waves_per_simd = max(0, min(8, file // per_lane))
    gen = result.generator
    threads = gen._num_threads * gen.launch_config().mults_per_block
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
    hw = result.context.target.hw
    budget = getattr(hw, 'max_reg_per_thread', None)
    peak = result.generator.peak_pressure
    if not (budget and peak):
        return _over_scalar_budget(result)
    if hw.vendor == 'amd':
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
    if hw.vendor == 'intel' and not result.context.target.explicit_simd:
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
    peak = getattr(result.generator, 'peak_uniform_pressure', None)
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
    resident per SM under shared memory, threads *and* registers, then the
    busiest pipe's clocks with the spilling in them.

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
        with tempfile.TemporaryDirectory(prefix='tf-tune-') as tmp:
            src = os.path.join(tmp, 'kernel.cpp' if hw.vendor == 'intel'
                               else 'kernel.cu')
            with open(src, 'w') as f:
                f.write(kernel_source(result))
            inc = self.toolchain.include_dir()
            entry = toolchain.compiler_for_vendor(hw.vendor)
            backend = entry.backend
            flags = entry.language_flags() + entry.report_flags()
            if hw.vendor == 'nvidia':
                arch = hw.model if hw.model.endswith('a') else (
                    hw.model + 'a' if hw.model in ('sm_90', 'sm_100', 'sm_101', 'sm_120')
                    else hw.model)
                cmd = [compiler, '-cubin', *flags, *entry.target_flags(arch),
                       '-I', inc, '-o', os.path.join(tmp, 'k.cubin'), src]
            elif hw.vendor == 'intel':
                # Linked, as a shared object: the ahead-of-time device build
                # runs at link time, and `-c` leaves `-device` unused and IGC
                # silent.
                cmd = [compiler, *flags, *entry.target_flags(hw.model), '-O3',
                       '-shared', '-fPIC', '-I', inc,
                       '-o', os.path.join(tmp, 'k.so'), src]
            else:
                cmd = [compiler, '-x', 'hip', '-c', *flags,
                       *entry.target_flags(hw.model), '--offload-device-only',
                       '-O3', '-I', inc, '-o', os.path.join(tmp, 'k.o'), src]
            run = subprocess.run(cmd + list(self.flags), capture_output=True,
                                 text=True, timeout=self.timeout)
            spilled = (toolchain.zeinfo_spill(os.path.join(tmp, 'k.so'))
                       if hw.vendor == 'intel' else None)
        report = toolchain.resources(backend, run.stdout + run.stderr)
        if report is None or run.returncode:
            return None
        if spilled is not None:
            # The object's own figure, and it decides: the console heuristic
            # reads a recompilation as "spilled, amount unknown", and IGC
            # recompiles for other reasons too -- the 32-lane
            # `neighboringFlux` build retries and spills nothing, which that
            # heuristic would report as a spill for the ranking to act on.
            report = replace(report, spill_bytes=spilled)
        if hw.vendor == 'nvidia' and report.registers:
            gen = result.generator
            threads = gen._num_threads * gen.launch_config().mults_per_block
            per_thread = -(-report.registers // 8) * 8
            report = replace(report, register_blocks=hw.max_reg_per_block
                             // max(1, per_thread * threads))
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
        if report.register_blocks is not None and hw.vendor == 'nvidia':
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
                    -resident, issue)
        return (report.spill_bytes > 0, _icache_over(result), -resident, issue)


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
            efficiency = pipeline.persistent_efficiency(
                self.batch, gen.launch_config().mults_per_block,
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


def _evaluate(descr_factory, context, scorer, candidate, cache):
    if candidate in cache:
        return cache[candidate]
    result = build(descr_factory, context, candidate)
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
               origin: Optional[Candidate] = None) -> Outcome:
    """Every point.  Only for a space small enough to build whole."""
    descrs = descr_factory()
    knobs = space(descrs, context) if knobs is None else knobs
    origin = start(descrs, context) if origin is None else origin
    cache: Dict[Candidate, Trial] = {}
    best = None
    for cand in enumerate_space(knobs, origin):
        trial = _evaluate(descr_factory, context, scorer, cand, cache)
        if _better(trial.score, None if best is None else best.score):
            best = trial
    return _outcome(best, cache, origin)


def coordinate(descr_factory, context: Context, scorer,
               knobs: Optional[Sequence[Knob]] = None,
               origin: Optional[Candidate] = None,
               rounds: int = 3, budget: Optional[int] = None,
               seed: Optional[Dict[Candidate, Trial]] = None) -> Outcome:
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
    best = _evaluate(descr_factory, context, scorer, origin, cache)
    for _ in range(rounds):
        moved = False
        for knob in knobs:
            for value in knob.values(best.candidate):
                cand = best.candidate.set(knob.name, value)
                if (budget is not None and cand not in cache
                        and len(cache) >= budget):
                    continue
                trial = _evaluate(descr_factory, context, scorer, cand, cache)
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
         strategy: Callable = coordinate, **kwargs) -> Outcome:
    """Choose a configuration for `descr_factory()`'s kernel on `context`'s
    target.  `descr_factory` returns a fresh descriptor list per call."""
    return strategy(descr_factory, context, scorer, **kwargs)


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
             fixed: Optional[Mapping[str, Any]] = None) -> Optional[Candidate]:
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
    from tensorforge.generators import preferences
    pref = preferences.lookup(descrs, context)
    if pref is not None:
        chosen = preferences.candidate(pref, descrs, context)
        if build(descr_factory, context, chosen).ok:
            return chosen
        warnings.warn(f'autotune: the preference from {pref.source} '
                      f'({pref.evidence}) does not build here; ranking instead')
    if mode == 'prefer':
        return None
    knobs = simple_space(descrs, context)
    origin = start(descrs, context)
    for name, value in (fixed or {}).items():
        origin = origin.set(name, value)
    knobs = [knob for knob in knobs if knob.name not in (fixed or {})]
    first = build(descr_factory, context, origin)
    if not first.ok:
        return None
    if not knobs:
        return origin
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
                         f'static or compiled')
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
                     budget=max(1, budget - 1) if budget else budget, seed=seed)
    best = _worth_it(descr_factory, context, out.best, origin, first)
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
              origin: Candidate, first: 'Build') -> Candidate:
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
    least = _bound_cycles(build(descr_factory, context, best))
    was = _bound_cycles(first)
    if least is None or was is None:
        return best
    return best if least < (1.0 - DEVIATION_MARGIN) * was else origin
