# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Building a matrix fragment out of the registers the loop nest holds.

The nest hands a path one register per contraction value, each holding the
leading dimension across the lanes: register `k`, lane `l` is
``data[lead = l][k]``.  A matrix instruction wants the other arrangement ---
part of the contraction in the lane index, the leading dimension crammed into
whatever lane bits are left.  `layouts` says exactly which; this says how to
get there without leaving the register file.

The move is always the same shape, and it is a shape `swap` and `dppUpdate`
cover between them:

* Every lane of the fragment that reads source register `k` forms a *region*,
  and within a region the lane movement is a single XOR of the lane index ---
  the contraction field replaced by the fragment's own.  A sequence of
  `swap<2**(b+1)>` toggles exactly the bits of that XOR, one instruction per
  set bit.
* The regions are whole 16-lane rows.  Not by luck: a lane bit only carries
  the contraction once the leading dimension and the blocks have used the
  ones below, and that is `n * blocks >= 16` on every instruction in the
  catalogue.  So `row_mask` reaches them and the select is `dppUpdate`,
  which needs no lane id.

So a fragment slot costs one `dppUpdate` per region plus the swaps each region
needs --- for `mfma_f64_16x16x4f64`, four merges and four swaps, once per
k-block and reused across every output column.

Nothing here emits.  It answers what to emit, and `tests/test_amd_reorder.py`
runs the answer through the wave simulator and checks that every lane of the
result holds what `layouts` says it should.  The plan is *derived* by
enumerating the fragment rather than by inverting the bit table in closed
form: the structure it depends on --- constant XOR per region, regions
row-aligned --- is then something the derivation checks rather than something
it assumes.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from . import layouts

#: `quad_perm:[0,1,2,3]`.  The DPP control that moves nothing, so that
#: `dppUpdate` is a masked merge and not also a shuffle.
IDENTITY_DPP = 0xE4

#: Lanes per DPP row.  What `row_mask` selects, on every wave size.
ROW = 16


@dataclass(frozen=True)
class Move:
    """One source register into one region of one fragment slot.

    Emitted as ``acc = dppUpdate<IDENTITY_DPP, row_mask, 0xf, false>(v,
    acc)``, where `v` is the source register put through `swaps` in order.
    """

    #: Contraction index of the source register the nest holds.
    contraction: int
    #: `swap<Block>` sequence, in order.  Empty when the region does not move.
    swaps: Tuple[int, ...]
    #: `dppUpdate`'s row mask: bit `r` enables lanes `16r` to `16r + 15`.
    row_mask: int

    @property
    def cost(self) -> int:
        """Instructions: the swaps, and the merge."""
        return len(self.swaps) + 1


def _swaps_for(mask: int) -> Tuple[int, ...]:
    """`swap` sequence toggling exactly the lane bits in `mask`.

    `swap<Block>` reads lane ``i ^ (Block / 2)``, so bit `b` is `swap<2**(b+1)>`
    and the sequence is one instruction per set bit.  Ascending, which is
    arbitrary --- they commute --- but fixed, so two plans for the same mask
    compare equal.
    """
    return tuple(1 << (bit + 1) for bit in range(6) if mask >> bit & 1)


def fragment_moves(op, which: str, slot: int,
                   group: int = 0) -> Optional[Tuple[Move, ...]]:
    """How to build one slot of an operand fragment, or `None`.

    `group` picks which stretch of the leading dimension the fragment covers.
    A fragment holds `n * blocks` of it and a wave holds `wave`, so there are
    `wave // (n * blocks)` of them and the whole wave takes that many
    fragments --- each with its own accumulator, all sharing these registers.

    `None` when the operand has no layout, when the source is not one scalar
    per lane, or when the fragment does not have the structure above.  The
    last is the interesting one: it means an instruction needs more than
    swaps and a masked merge, and a caller that gets `None` should stay on the
    generic nest rather than emit something close.
    """
    if which.upper() not in ('A', 'B'):
        return None            # the accumulator is written, not built
    frag = op.a if which.upper() == 'A' else op.b
    if not layouts.covers(op, which) or frag.per_lane <= slot:
        return None
    extent = op.m if which.upper() == 'A' else op.n
    span = extent * op.blocks
    if op.wave % span or group >= op.wave // span:
        return None

    regions = {}
    for lane in range(op.wave):
        element = layouts.element_at(op, which, slot, lane)
        block, first, second = element
        contraction, index = (second, first) if which.upper() == 'A' \
            else (first, second)
        source = group * span + block * extent + index
        regions.setdefault(contraction, []).append((lane, source))

    moves = []
    for contraction, pairs in sorted(regions.items()):
        masks = {lane ^ source for lane, source in pairs}
        if len(masks) != 1:
            return None        # not a single XOR: no swap sequence does it
        mask = masks.pop()
        rows = {lane // ROW for lane, _ in pairs}
        if any(sum(1 for lane, _ in pairs if lane // ROW == row) != ROW
               for row in rows):
            return None        # a partial row: `row_mask` cannot select it
        moves.append(Move(contraction, _swaps_for(mask),
                          sum(1 << row for row in rows)))
    return tuple(moves)


def fragment_cost(op, which: str, group: int = 0) -> Optional[int]:
    """Instructions to build a whole fragment of `which`, or `None`.

    The number to weigh against staging the operand through shared memory,
    and against not taking the matrix path at all.  It is paid once per
    k-block and per group, and the fragment is then read by every output
    column, so what it competes with is the column loop rather than one
    instruction.
    """
    total = 0
    frag = op.a if which.upper() == 'A' else op.b
    for slot in range(frag.per_lane):
        moves = fragment_moves(op, which, slot, group)
        if moves is None:
            return None
        total += sum(move.cost for move in moves)
    return total
