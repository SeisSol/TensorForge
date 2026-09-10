# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How many lanes a section's lead dimension is spread over.

One number with more leverage than it looks like.  A register array is sized
`extent / lanes` along the distributed axis, so the lane count scales *every*
register array in the section at once -- where a placement decision only says
which arrays exist.  Order 6 in double on gfx90a is the case that shows it: the
stiffness operator needs 148 register slots per lane at 32 lanes and 74 at 64,
which at 8 bytes a slot is the difference between 296 and 148 VGPRs.

It is per section and not per operation.  The descriptors of a section share
one register image, so a lane count one of them needs is a lane count all of
them must have -- which is also why changing it means rebuilding the section
rather than revising one operand.

Stated as a value rather than computed in place so that it can be *chosen*.
Nothing here chooses yet: `deduce` returns what the generator has always
produced, and an explicit config overrides it. That is the whole point --
"build this section with 64 lanes instead" was not previously expressible, and
a search over configurations cannot start from a constant.
"""

from dataclasses import dataclass
from typing import List, Optional

from tensorforge.common.context import Context
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 OperationDescription)

#: The default ceiling on lanes per multiplication, and *not* a hardware fact:
#: it bites only where the wave is wider, which today is AMD alone (NVIDIA and
#: Intel report 32 and 16, so the minimum is a no-op there).
#:
#: It is also not arbitrary. Running gfx90a at the full wave halves the
#: per-lane register footprint and has measured *slower* on some kernels, so
#: the ceiling encodes a result rather than an oversight. What it lacked was
#: anywhere to say so, and any way to ask for the other answer.
DEFAULT_LANE_CEILING = 32


@dataclass(frozen=True)
class LaneConfig:
    """The lane geometry one section is built with."""

    #: Lanes the lead dimension is distributed over.
    num_threads: int
    #: Of those, how many do useful work; the rest are masked off at the edges.
    num_active_threads: int
    #: Lead-dimension elements one lane covers where it used to cover one.
    lead_width: int


def deduce(descr_list: List[OperationDescription],
           context: Context,
           ceiling: Optional[int] = DEFAULT_LANE_CEILING) -> LaneConfig:
    """The lane geometry a descriptor list asks for.

    A maximum over the lane counts and a *minimum* over the widths, and the
    asymmetry is not an oversight: one register image is shared across the
    section, so a width one descriptor cannot take is a width none may take,
    while a lane count one descriptor needs is one they must all have.

    `ceiling` caps the result. `None` means the hardware's wave width, which is
    the largest value that is meaningful -- a barrier inside a batch loop is
    only simd-uniform, so a section wider than a wave would need a group
    barrier where it may not have one.

    An elementwise descriptor waives the ceiling. Its iteration space is the
    vector unit's, not a contraction's lead dimension, so compressing it below
    that would leave lanes idle for no gain.
    """
    num_threads = 0
    num_active = 0
    widths = []
    for descr in descr_list:
        threads, active = descr.get_num_threads(context)
        num_threads = max(threads, num_threads)
        num_active = max(active, num_active)
        widths.append(getattr(descr, 'lead_width', lambda _c: 1)(context))

    # Deliberately *not* also clamped to the wave width, which is what the
    # generator has always done.  Clamping would be a change, and a large one:
    # the Intel targets report a 16-wide vector unit, so a ceiling of 16 would
    # take 35 kernels from refusing to generate -- a group barrier inside a
    # simd-uniform loop -- to generating. That is a lead worth following and
    # not a side effect to take while extracting a decision.
    cap = (context.get_vm().get_hw_descr().vec_unit_length
           if ceiling is None else ceiling)
    if not any(isinstance(d, ElementwiseDescr) for d in descr_list):
        num_threads = min(cap, num_threads)

    return LaneConfig(num_threads=num_threads,
                      num_active_threads=num_active,
                      lead_width=min(widths) if widths else 1)


def candidates(descr_list: List[OperationDescription],
               context: Context) -> List[LaneConfig]:
    """The lane geometries worth building this section at, widest first.

    Two today: what the descriptors ask for under the default ceiling, and the
    same at the wave width.  They coincide wherever the ceiling does not bind,
    which is every target whose wave is 32 or narrower -- so on NVIDIA and on
    RDNA there is one candidate and no search to run.

    Deduplicated by lane count rather than by config, since two ceilings that
    land on the same width give the same kernel and building it twice buys a
    tie.
    """
    seen = {}
    for ceiling in (None, DEFAULT_LANE_CEILING):
        config = deduce(descr_list, context, ceiling=ceiling)
        seen.setdefault(config.num_threads, config)
    return [seen[k] for k in sorted(seen, reverse=True)]


#: The narrowest lane count `narrower` offers.  Below it a lane holds so many
#: rows that the register images of every corpus kernel measured spill.
MIN_LANES = 8


def narrower(descr_list: List[OperationDescription],
             context: Context,
             floor: int = MIN_LANES) -> List[LaneConfig]:
    """The lane geometries below the deduced one, widest first.

    Powers of two only, halving from the deduced count down to `floor`, with
    the same row count and width: a narrower section covers the lead
    dimension with more rows per lane and pads the last ones.  Not the
    divisors of the row count -- 56 rows over 7 or 28 lanes would leave lanes
    idle in every warp, while a power of two packs whole multiplications into
    it; 56 over 8 lanes is 7 rows with no padding at all, and 63 over 8 is 8
    rows with one row padded.

    Candidates, not a choice.  Measured on sm_120 at the deduced count, 16 and
    8 lanes, the narrower build ranged from 10 % faster (`local_flux` at 8,
    no padding) to five times slower (`chain_five` at 16, spilling), and
    among those that did not spill from -5 % to +9 % with nothing in padding
    or rows per lane that predicted the sign.  `peak_pressure` separates the
    spilling ones now that it counts register slots rather than whole arrays
    -- on sm_120 every build that spilled more than a few registers was above
    about 85 % of the 255 -- but it says nothing about the sign among the
    rest.  So these are for a search that builds and times, which
    `Options.lanes_per_mult` makes expressible, and the pressure is at most a
    filter in front of it.

    None for a section with an elementwise descriptor: its iteration space is
    the vector unit's, as in `deduce`.
    """
    if any(isinstance(d, ElementwiseDescr) for d in descr_list):
        return []
    base = deduce(descr_list, context)
    out = []
    n = base.num_threads // 2
    while n >= floor:
        out.append(LaneConfig(num_threads=n,
                              num_active_threads=base.num_active_threads,
                              lead_width=base.lead_width))
        n //= 2
    return out


def requested(descr_list: List[OperationDescription],
              context: Context) -> Optional[LaneConfig]:
    """The geometry `Options.lanes_per_mult` asks for, or `None`.

    `None` where it is unset, or where the count is not one of `narrower`'s:
    a wider count than the deduction, or one that is not a power of two, is
    not taken rather than forced, so that one setting can be passed to a whole
    corpus and change only the sections it applies to.
    """
    want = context.get_user_options().lanes_per_mult
    if not want:
        return None
    return next((c for c in narrower(descr_list, context, floor=1)
                 if c.num_threads == want), None)


def search(descr_factory, context: Context,
           options: Optional[List[LaneConfig]] = None):
    """Build the kernel at each candidate geometry and keep the tightest.

    `descr_factory` returns a fresh descriptor list per attempt.  A list would
    also work -- generating twice from one is idempotent -- but a factory says
    that this builds repeatedly, which a caller passing a list it still holds
    a reference to should know.

    Three keys, in order, and the order is the point: what is known exactly
    decides before what is modelled, and neither decides where both are
    silent.

    First, blocks resident per SM under shared memory and threads per block.
    Those are not estimates -- one is what was allocated, the other is the
    launch geometry -- so where they differ they are a fact and outrank the
    model.  Over the corpus they differ once, on the kernel that runs out of
    registers, and they agree with the model there.  They also explain why the
    two are worth separating: the lane count moves `mults_per_block` as well,
    so every contested case changes its shared memory per block at the same
    time as its registers, and a single number would have hidden which of the
    two moved.

    Second, peak register footprint per lane: the *maximum* over the kernel's
    bodies, since a budget is per kernel and the widest body has to fit.

    Third, the geometry the descriptors asked for.  A tie means the model sees
    no difference, and changing the configuration for no modelled reason is
    exactly where "the wider one measured slower" would bite -- so a tie keeps
    what the generator would have done anyway.  Six of the corpus's fourteen
    contested cases on gfx90a are ties.

    Ranking only, and deliberately.  Measured against `hipcc` over the corpus
    on gfx90a and gfx942, the model picks the same configuration the compiler
    gives fewer registers to in 7 of 8 decided cases, and in 3 of 3 of the
    cases where the choice changes occupancy at all -- there by a factor of
    two each time. What it cannot do is say whether a configuration *fits*:
    registers per model byte spread over a factor of 72, so there is no
    threshold in it, only an order.

    A candidate that does not build is not a candidate.  It is skipped and its
    entry records the exception, because building at a given width is not a
    given: on Intel the two candidates are 32 and the 16-wide vector unit, and
    the wider one leaves a barrier in a batch loop at group scope, which
    `verify` refuses -- so on that target the configuration that fails is the
    *default* one, and a search that propagated the failure would be unusable
    exactly where it has something to offer.

    If nothing builds, the first exception is re-raised: an empty result and a
    kernel that cannot be generated are different situations, and the caller
    needs the second one to look like one.

    Returns `(config, results)` where `results` maps lane count to the peak
    figure -- or to the exception for a candidate that did not build -- so a
    caller can see how close the decision was and what it passed over.
    """
    options = options or candidates(descr_factory(), context)
    default = deduce(descr_factory(), context)
    if len(options) == 1:
        return options[0], {options[0].num_threads: None}

    from tensorforge.generators.generator import Generator

    was = context.measure_pressure
    context.measure_pressure = True
    scores = {}
    blocks = {}
    failed = []
    try:
        for config in options:
            try:
                gen = Generator(descr_factory(), context, lanes=config)
                gen.generate()
            except Exception as exc:
                scores[config.num_threads] = exc
                failed.append(exc)
                continue
            scores[config.num_threads] = gen.peak_pressure
            blocks[config.num_threads] = gen.resident_blocks
    finally:
        context.measure_pressure = was

    # A build that measured nothing is not a build that measured zero: it has
    # no bodies the flag reached, and picking it for scoring lowest would be
    # picking it for having said nothing.
    scored = {k: v for k, v in scores.items()
              if isinstance(v, int) and v}
    if not scored:
        if len(failed) == len(options):
            raise failed[0]
        built = [c for c in options
                 if not isinstance(scores.get(c.num_threads), Exception)]
        return built[0], scores
    def rank(width):
        return (-(blocks.get(width) or 0), scored[width],
                width != default.num_threads)

    best = min(scored, key=rank)
    return next(c for c in options if c.num_threads == best), scores
