# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which instruction turns one lane distribution into another.

A pass that wants to combine two values has to know whether they are laid out
the same way, and if not, what it would cost to bring them together.  The
first question is `RegisterLayout.__eq__`.  This module is the second.

It is a *table*, not an algebra.  The generator produces a small closed set of
distributions --- a census over the whole corpus and every supported
architecture found single figures, all of rank one --- so every relayout worth
naming can be written down, and nothing has to be solved for.  If fused
operators later bring genuine rank-2 layouts, `RegisterLayout` already carries
several axes and `tiles()` already distinguishes a second dimension from mere
replication; what would have to change is this table, not the vocabulary.

Every row is a claim about hardware, and claims about hardware are exactly
what has already gone wrong here twice --- the `LaneAxis` docstring, and the
broadcast annotation derived from it, both stating the right numbers in the
wrong roles.  So `tests/test_amd_relayout.py` re-derives each row by
simulating the instruction from its own definition in `hip.h` and fails if the
row disagrees.

A primitive whose lane map is not established has no row.  That is the
intended outcome rather than a gap: no row means no relayout is offered, and a
pass that cannot find one stays where it is instead of acting on a guess.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from tensorforge.backend.pir.core import LaneAxis, RegisterLayout

from . import catalog
from .reorder import compose_cost, compose_exchange, emittable
from ... import bitlayout, staging


@dataclass(frozen=True)
class Relayout:
    """One instruction, and the distribution it produces.

    `produces` is a function of the instruction's own parameters rather than a
    constant, because that is what the hardware does: `broadcast<B, S, L>`
    lands on `LaneAxis(S, 1)` whatever `B` and `L` are, and writing the answer
    out once per parameter combination would be the `scale` table again.

    `arity` is how many registers the instruction consumes at once.  It is 1
    for a broadcast and 4 for the quad transpose, and the difference matters:
    a relayout over several registers moves a dimension between slots and
    lanes, which one register cannot express.
    """

    name: str
    callee: str
    arity: int
    #: parameters -> the layout of the result
    produces: Callable[..., RegisterLayout]
    #: parameters -> whether this instruction is applicable at all
    applies: Callable[..., bool] = lambda **kw: True
    #: True when the result holds fewer distinct elements than the input.
    #: A broadcast is not invertible; a transpose is.
    lossy: bool = False
    #: Parameters that pick *which* elements rather than *how* they are
    #: distributed.  `broadcast<B, S, L>` lands on the same layout for every
    #: `L`; which sub-block `L` names is the algorithm's business, not the
    #: layout's.  A search over layouts therefore cannot determine them, and
    #: pretending otherwise would have it return an arbitrary one.
    selects_data: Tuple[str, ...] = ()
    note: str = ''


def _broadcast_result(threads, step, lane):
    # Result lane `l` takes the source value from `lane*step + l % step`, so
    # the result repeats every `step` lanes with neighbours differing.
    return RegisterLayout((LaneAxis(step, 1),))


def _movdpp16_result(threads, row):
    # Row share within 16: one distinct value per 16-lane row, rows sitting 16
    # consecutive threads apart.
    return RegisterLayout((LaneAxis(max(threads // 16, 1), 16),))


BROADCAST = Relayout(
    name='broadcast',
    callee='tensorforge::broadcast<{threads}, {step}, {lane}>',
    arity=1,
    produces=_broadcast_result,
    applies=lambda threads, step, lane: (threads % step == 0
                                         and lane * step < threads),
    lossy=True,
    selects_data=('lane',),
    note='selects one sub-block and repeats it; not invertible',
)

MOVDPP16 = Relayout(
    name='movdpp16',
    callee='tensorforge::movdpp16<{row}>',
    arity=1,
    produces=_movdpp16_result,
    applies=lambda threads, row: 0 <= row < 16,
    lossy=True,
    selects_data=('row',),
    note='row share within 16 lanes',
)

def _transpose4x4_result(threads):
    """Rank 2, and the only rank-2 layout the generator currently produces.

    Before the exchange one dimension sits across the four registers and the
    other is spread over the lanes.  Afterwards output register `r` at lane
    `l` holds `(register l % 4, lane (l & ~3) + r)`, so *both* dimensions
    vary with the lane: the first with period 4, the second in runs of 4.

    At 64 lanes that is `LaneAxis(4, 1)` beside `LaneAxis(16, 4)` --- lane `l`
    holds `(l % 4, l // 4)`, one lane per pair, replication 1.  Describing it
    with a single axis, as this row first did, throws away the half of the
    answer that says which element of the other dimension a lane is holding.
    """
    return RegisterLayout((LaneAxis(4, 1), LaneAxis(max(threads // 4, 1), 4)))


TRANSPOSE4X4 = Relayout(
    name='transpose4x4',
    callee='tensorforge::transpose4x4b32',
    arity=4,
    produces=_transpose4x4_result,
    applies=lambda threads: threads >= 4 and threads % 4 == 0,
    lossy=False,
    note='4x4 transpose of (register, lane % 4); the one relayout here that '
         'moves a dimension between slots and lanes',
)

#: Everything whose lane map has been established.  `transpose16x16b32` is
#: absent on purpose: it is defined in the runtime, but its body uses row and
#: wave DPP controls that the simulator does not model, so no row for it could
#: be checked.
def nest_shared(ext: int, threads: int) -> bitlayout.BitLayout:
    """The shared matrix as the loop nest hands it over.

    One register per column of the tile and the contraction spread across the
    lanes, which is what `A(writer, None, j + jj, ...)` returns: `jj` picks the
    register and the lane picks the contraction value.
    """
    return bitlayout.BitLayout((
        tuple(bitlayout.Bit(bitlayout.Place.SLOT, 1 << b)
              for b in range((ext - 1).bit_length())),
        tuple(bitlayout.Bit(bitlayout.Place.LANE, 1 << b)
              for b in range((threads - 1).bit_length())),
    ))


def transposed(ext: int, threads: int) -> bitlayout.BitLayout:
    """The same elements at the layout the A fragment wants.

    The column on the low lane bits, and the contraction bits it displaced now
    in the registers -- which is why `matmul32` reads its contraction in
    stride-`ext` groups and not in steps of one.
    """
    low = (ext - 1).bit_length()
    return bitlayout.BitLayout((
        tuple(bitlayout.Bit(bitlayout.Place.LANE, 1 << b) for b in range(low)),
        tuple(bitlayout.Bit(bitlayout.Place.SLOT, 1 << b) if b < low
              else bitlayout.Bit(bitlayout.Place.LANE, 1 << b)
              for b in range((threads - 1).bit_length())),
    ))


def has_transpose(ext: int) -> bool:
    """Whether the runtime defines `transpose{ext}x{ext}b32`.

    A copy of a C++ fact, and the reason a gap that *is* the exchange still
    may not be one instruction: `DEFINED_TRANSPOSES` names four widths and the
    catalogue has entries at others.
    """
    return f'tensorforge::transpose{ext}x{ext}b32' in catalog.DEFINED_TRANSPOSES


def extracts(have) -> int:
    """Element reads it takes to unpack `have` before `reach` reads it.

    Separate from the route because it happens before one: an extract names an
    element of a register the caller already holds, and the register it lands
    in is the caller's to choose.  A caller comparing routes adds this to
    whichever it takes.
    """
    return bitlayout.unpacked(have)[1]


def reach(have, want, ext: int, indices, wave: Optional[int] = None):
    """How the operand gets from one distribution to the other.

    Four answers, cheapest first, and each is a different kind of thing: `0`
    is nothing to emit, `1` is the runtime's transpose, a tuple of `Move` is
    that same exchange assembled out of swaps and merges, and a tuple of
    `Transfer` is the trip through memory.  Ordered by what they cost rather
    than by which is convenient, because the order *is* the preference.

    The middle rung is where a width the runtime has no `transpose*` for would
    stay in registers, and today it is not reached: a transpose's regions are
    one lane out of every `ext`, which no `dppUpdate` mask expresses.  It is
    offered only when every region has a mask that exists, and otherwise the
    trip is what is left.  Which of the last two is taken is a comparison of
    their counts, not their order.

    A packed operand is unpacked first and then answered like any other, and
    whether that closes the gap depends on which operand it is.  The shared
    matrix reduces to `nest_shared` and costs the extracts.  The lead operand
    does not: `lead_width` puts its low bits inside the register and the
    fragment wants the leading dimension across the lanes, so unpacking moves
    them the wrong way and what remains is a permutation between lane weights
    -- which no row of `RELAYOUTS` performs, and the trip is what answers it.

    Never `None`.  The staged path closes every gap, so a caller reaching here
    always has an answer -- what it does not always have is one it can
    afford.
    """
    # A vector bit is an element of a packed register and reaching one is a
    # subscript rather than a shuffle, so it is not a gap for the cross-lane
    # machinery -- it closes first, and `extracts` is what it costs.  What is
    # left is a distribution the rungs below already answer: unpacking a
    # packed shared matrix yields exactly `nest_shared`.
    have, _ = bitlayout.unpacked(have)

    direct = transposes_between(have, want, ext)
    if direct == 0:
        return 0
    if direct == 1 and has_transpose(ext):
        return 1
    trip = staging.staged(have, want, indices)
    if wave is None:
        return trip
    composed = compose_exchange(have, want, indices, wave)
    if composed is None or not emittable(composed):
        # Priced but not emittable: a transpose's regions are one lane out of
        # every `ext`, which no `dppUpdate` mask expresses, so the merge would
        # need a ternary on the lane id.  Offering a route nothing can emit
        # would be worse than the trip it is cheaper than.
        return trip
    # The last two rungs are compared rather than ordered.  Registers beat
    # memory at every width in the catalogue -- 224 instructions against 1024
    # accesses at width eight -- but that is a count and not a law, and a gap
    # with one region per element would not.
    stores, loads = staging.accesses(trip)
    return composed if compose_cost(composed) <= stores + loads else trip


def transposes_between(have, want, ext: int) -> Optional[int]:
    """How many `transpose{ext}x{ext}b32` calls close the gap: 0, 1 or `None`.

    Nothing and one instruction are the two answers this family has, and the
    first is the one worth having: an operand that already arrives at the
    layout the instruction wants needs no relayout, and asking is what lets an
    emitter find that out instead of transposing unconditionally.

    `None` is the gap this instruction does not close -- a bit permuted inside
    the lanes, an unpaired move, an element bit of a packed operand.  Not an
    error: it says a different instruction is wanted, or none exists, and the
    caller is what knows whether it has another.
    """
    gap = bitlayout.is_exchange(bitlayout.displacement(have, want))
    if gap is None:
        return None
    if gap == ():
        return 0
    return 1 if gap == transpose_exchange(ext) else None


def transpose_exchange(ext: int):
    """What `transpose{ext}x{ext}b32` does, as `(slot weight, lane weight)`
    pairs.

    The instruction exchanges the register index with the low lane index
    inside each group of `ext` lanes, so slot weight `2**b` and lane weight
    `2**b` trade places for every `b` below `log2(ext)`.  Stated in the shared
    vocabulary because that is where it can be compared with a gap: what
    `bitlayout.is_exchange` reports for two layouts is exactly this, or it is
    not this instruction.

    Not the same statement as `produces`.  That one names the distribution of
    the *result*, one register at a time, which is what a value carries and
    what a pass compares.  This names what the instruction did to the whole
    value, which is what a solver needs -- the index space is the same on both
    sides of it, and `produces` re-factors it.

    The other two rows of `RELAYOUTS` have no entry here on purpose: both are
    lossy, they select a sub-block and repeat it rather than moving bits, and
    a replication is not a permutation.
    """
    return tuple((1 << bit, 1 << bit) for bit in range((ext - 1).bit_length()))


RELAYOUTS = (BROADCAST, MOVDPP16, TRANSPOSE4X4)


def find_relayout(target: RegisterLayout, threads: int
                  ) -> Optional[Tuple[Relayout, dict]]:
    """An instruction whose result has layout `target`, or None.

    Searches rather than solves, which is the whole point of a table: with a
    handful of rows and a handful of parameter values there is nothing to
    solve.  Lossless relayouts come first --- given a choice, an instruction
    that keeps every element is preferable to one that replicates a subset.

    The returned parameters cover only what the *layout* determines; anything
    listed in `selects_data` is left out, because the search genuinely cannot
    know it.  The caller fills those in, which is the honest division: the
    table says how to get the distribution, the algorithm says which elements
    it wants.
    """
    for entry in sorted(RELAYOUTS, key=lambda e: e.lossy):
        for candidate in _candidates(entry, threads):
            if not entry.applies(**candidate):
                continue
            if entry.produces(**candidate) == target:
                return entry, {k: v for k, v in candidate.items()
                               if k not in entry.selects_data}
    return None


def _candidates(entry, threads, **fixed):
    """Parameter combinations worth trying for one row."""
    if entry is TRANSPOSE4X4:
        yield {'threads': threads}
        return
    if entry is MOVDPP16:
        for row in range(16):
            yield {'threads': threads, 'row': row}
        return
    for step in (1, 2, 4, 8, 16, 32, 64):
        if step > threads:
            break
        for lane in range(threads // step):
            yield {'threads': threads, 'step': step, 'lane': lane}


def fmadpp_operand_layout(step: int) -> RegisterLayout:
    """The distribution `fmacdpp{step}` needs for its broadcast operand.

    Stated once, and used twice: `hfma` searches the table for an instruction
    that reaches it, and `fmadpp` checks that what arrives matches.  Splitting
    those --- a hard-coded broadcast on one side, an assumption on the other
    --- is the arrangement that already produced two wrong layout claims here.

    `step` lanes hold `step` distinct elements and the pattern repeats, which
    is exactly what `broadcast<threads, step, L>` leaves behind, and what a
    load already has when `step == threads` and no broadcast is emitted at
    all.
    """
    return RegisterLayout((LaneAxis(step, 1),))
