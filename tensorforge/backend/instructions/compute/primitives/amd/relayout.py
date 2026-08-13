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
