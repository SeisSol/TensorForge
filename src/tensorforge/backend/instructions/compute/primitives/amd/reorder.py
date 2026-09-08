# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Building a matrix fragment out of the registers the loop nest holds.

The nest hands the two operands over differently, and which one a plan is for
decides whether the plan applies at all.

``ops.A``, the operand that shares the leading dimension, arrives as one
register per contraction value with the leading dimension across the lanes:
register `k`, lane `l` is ``data[lead = l][k]``.  That is the arrangement
everything below is written against.

``ops.B``, the shared matrix, arrives *transposed* against it --- the column
in the registers and the contraction in the lanes, because `unwindK` hands it
a `LeadIndex` and `unwindI` hands the other one.  A `swap` moves a value
between lanes and never between a register and a lane, so no sequence of them
reaches a fragment from there.  `fragment_moves` therefore covers the
instruction's **B** fragment, which the leading operand feeds, and refuses its
**A** fragment, which the shared matrix feeds.

That is not a gap in the plan; it is where the instruction does the work
itself.  `cbsz` and `abid` broadcast one block's A operand to the others, which is how
`matmul32` feeds a 16-block tile from a single transpose --- and it is
available exactly where `blocks > 1`.  That is *not* the same as `k == 1`:
`mfma_f64_4x4x4f64` has four blocks and a contraction of four, so it can
broadcast its A and still needs the plan below for its B.  What `k == 1` is
the same statement as is `n * blocks == wave`, which is `lane_batched`, and
that is a stronger condition than having blocks at all.

Where an instruction has one block --- `mfma_f64_16x16x4f64`, both XF32
entries, both native gfx125x WMMAs --- there is nothing to broadcast between
and the A operand needs a register-to-lane exchange first.  `a_exchange` says
when one `transpose*` from `hip.h` is the whole answer, and it is a better
answer than it looks: see below.

A matrix instruction wants part of the contraction in the lane index and the
leading dimension crammed into whatever lane bits are left.  `layouts` says
exactly which; this says how to get there without leaving the register file.

The move is always the same shape, and it is a shape `swap` and `dppUpdate`
cover between them:

* Every lane of the fragment that reads source register `k` forms a *region*,
  and within a region the lane movement is a single XOR of the lane index ---
  the contraction field replaced by the fragment's own.  A sequence of
  `swap<2**(b+1)>` toggles exactly the bits of that XOR, one instruction per
  set bit.
* The regions are selected by `dppUpdate`\'s masks, which need no lane id.
  `row_mask` reaches 16-lane rows and `bank_mask` reaches 4-lane banks inside
  them, and the two multiply --- a mask writes `rows x banks`, the same bank
  pattern in every enabled row.  A region that is not a product of those is
  not reachable by a mask at all and needs a `cndmask`, which is a ternary on
  the lane id.

Every region in the catalogue today is whole 16-lane rows, so every select is
a `row_mask`.  That is not luck --- a lane bit only carries the contraction
once the leading dimension and the blocks have used the ones below, and `n *
blocks` is 16 or 32 everywhere --- but it is also not a property to build on:
it holds for *this* assignment of the generator\'s dimensions onto the
instruction\'s, and the accumulator writeback or a sub-wave thread count
assigns them differently.  So `Select` covers what the hardware offers rather
than what the current caller happens to need, and says which mechanism a
region requires instead of assuming one.

So a fragment slot costs one `dppUpdate` per region plus the swaps each region
needs --- for `mfma_f64_16x16x4f64`, four merges and four swaps, once per
k-block and reused across every output column.

The accumulator comes back the same way, in reverse.  The instruction leaves
`D[i][j]` spread over the fragment; the nest wants one register per output
column with the leading dimension across the lanes, which is a set of regions
and a constant XOR each --- so `accumulator_gathers` is `fragment_moves` read
backwards, and shares the swap sequences and the select with it.

Nothing here emits.  It answers what to emit, and `tests/test_amd_reorder.py`
runs the answer through the wave simulator and checks that every lane of the
result holds what `layouts` says it should.  The plan is *derived* by
enumerating the fragment rather than by inverting the bit table in closed
form: the structure it depends on --- constant XOR per region, regions
row-aligned --- is then something the derivation checks rather than something
it assumes.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from . import layouts

from ... import bitlayout

#: `quad_perm:[0,1,2,3]`.  The DPP control that moves nothing, so that
#: `dppUpdate` is a masked merge and not also a shuffle.
IDENTITY_DPP = 0xE4

#: Lanes per DPP row.  What `row_mask` selects, on every wave size.
ROW = 16

#: Lanes per DPP bank.  What `bank_mask` selects, inside each row.
BANK = 4


@dataclass(frozen=True)
class Select:
    """Which lanes a merge writes into.

    `dppUpdate` masks are free --- they are modifiers on the merge, not
    instructions --- but they only express a product: `row_mask` picks
    16-lane rows, `bank_mask` picks 4-lane banks, and a lane is written when
    both bits are set.  So the same bank pattern applies in every enabled row,
    and a region that varies between rows, or that splits a bank, is outside
    them.

    Below a bank there is nothing: the masks stop at four lanes, so a region
    finer than that is a `cndmask` --- a ternary on the lane id in C++.  That
    costs the select plus whatever reading the lane id costs, which is why
    `kind` is worth knowing before an emitter commits to a plan rather than
    after.
    """

    #: `'row'`, `'bank'` or `'cndmask'` -- the cheapest mechanism that
    #: expresses this region.  `'row'` is `'bank'` with every bank enabled,
    #: named separately because it is the case that needs no bank reasoning.
    kind: str
    #: Bit `r` enables lanes `16r` to `16r + 15`.
    row_mask: int
    #: Bit `b` enables lanes `4b` to `4b + 3` of every enabled row.
    bank_mask: int
    #: The region itself.  What a `cndmask` predicate has to test, and what
    #: the masks are checked to reproduce.
    lanes: frozenset

    @property
    def mask(self) -> int:
        """The region as a lane bitmap, which is what `laneMerge` takes."""
        return sum(1 << lane for lane in self.lanes)

    @property
    def mergeable(self) -> bool:
        """Whether a merge can write this region at all.

        Every region: the mask pair where it is a product of rows and banks,
        a lane bitmap otherwise.  Not the same question as `free`, which asks
        whether the select costs an instruction of its own -- a `cndmask` does
        and is still emittable, and conflating the two is what kept the
        assembled exchange out of reach.
        """
        return True

    @classmethod
    def of(cls, lanes, wave: int) -> 'Select':
        """The cheapest select for a set of lanes."""
        lanes = frozenset(lanes)
        rows = {lane // ROW for lane in lanes}
        banks = {(lane % ROW) // BANK for lane in lanes}
        row_mask = sum(1 << row for row in rows)
        bank_mask = sum(1 << bank for bank in banks)

        product = {row * ROW + bank * BANK + off
                   for row in rows for bank in banks for off in range(BANK)}
        if product != lanes:
            return cls('cndmask', 0, 0, lanes)
        full = (1 << (ROW // BANK)) - 1
        return cls('bank' if bank_mask != full else 'row',
                   row_mask, bank_mask, lanes)

    @property
    def free(self) -> bool:
        """Does this cost an instruction of its own?"""
        return self.kind != 'cndmask'


@dataclass(frozen=True)
class Move:
    """One source register into one region of one fragment slot.

    Emitted as ``acc = dppUpdate<IDENTITY_DPP, row_mask, bank_mask,
    false>(v, acc)`` where the select is a mask, and as a ternary on the lane
    id where it is not.  `v` is the source register put through `swaps` in
    order.
    """

    #: Contraction index of the source register the nest holds.
    contraction: int
    #: `swap<Block>` sequence, in order.  Empty when the region does not move.
    swaps: Tuple[int, ...]
    #: Which lanes this move writes.
    select: Select

    @property
    def row_mask(self) -> int:
        return self.select.row_mask

    @property
    def cost(self) -> int:
        """Instructions: the swaps, the merge, and the select if it is not
        a modifier on it."""
        return len(self.swaps) + 1 + (0 if self.select.free else 1)


def _swaps_for(mask: int) -> Tuple[int, ...]:
    """`swap` sequence toggling exactly the lane bits in `mask`.

    `swap<Block>` reads lane ``i ^ (Block / 2)``, so bit `b` is `swap<2**(b+1)>`
    and the sequence is one instruction per set bit.  Ascending, which is
    arbitrary --- they commute --- but fixed, so two plans for the same mask
    compare equal.
    """
    return tuple(1 << (bit + 1) for bit in range(6) if mask >> bit & 1)


@dataclass(frozen=True)
class Exchange:
    """One transpose, and the contraction order that makes it sufficient.

    The shared matrix arrives with the column in the registers and the
    contraction in the lanes.  `transpose{ext}x{ext}b32` swaps those inside
    each group of `ext` lanes, so output register `g` at lane `l` holds
    ``shared[l % ext][ext * (l // ext) + g]``: the column the fragment wants,
    and a contraction value that runs in steps of `ext` down the lane rows.

    The fragment wants its contraction to run in steps of *one*.  It does not
    have to: the contraction is a sum, so which values a given instruction
    issue covers is ours to choose, and choosing the stride-`ext` group
    ``{g, g + ext, g + 2*ext, ...}`` makes the transpose output *be* the
    fragment.  No swaps, no permute, no staging --- and the transpose emits
    all `ext` registers at once, so one of them feeds every issue of a whole
    k-block rather than one issue.

    Which is why it is worth having a name.  The obvious reading --- walk `k`
    in contiguous blocks of four and move the operand to match --- needs a
    gather per source register, because after the transpose the four values a
    contiguous block wants sit in four *different* registers, one per lane
    row, and no lane permutation crosses between registers.  Reordering the
    sum costs nothing and removes the movement entirely.

    It is not free of everything, though, and `covers` is where that shows.
    An issue's contraction set is `{g, g + stride, g + 2*stride, ...}` and the
    stride is forced --- it is the transpose width, because that is where the
    lane rows put `k`.  So the set is never a contiguous run, and anything
    that wants one does not get it:

    * **Block sparsity along `k`.**  A 16x16x4 instruction can skip an issue
      whose four contraction values are all zero, and a 4-deep zero block is
      exactly that shape under a contiguous walk.  Under this one the four
      values sit 16 apart and a 4-deep block never covers an issue, so
      nothing is skippable.  Sparse operands do not reach the matrix path
      today, so this is a collision with a plan rather than with code --- but
      it is a real one, and the two cannot both be had from one walk.
    * **Summation order.**  Same terms, different order.  Benign over 64
      terms and not a layout question, but generated results move.

    What it does *not* disturb: the accumulator, whose layout has no `k` in
    it; the leading operand's accessor, which takes `k` as a plain index and
    does not care in which order it is asked; and the shared operand's, whose
    `k` argument is a slot and where every group here lives inside one slot.
    """

    #: The `hip.h` name, as `DEFINED_TRANSPOSES` spells it.
    transpose: str
    #: Contraction groups the one transpose produces, and registers it emits.
    groups: int
    #: Step between the contraction values inside a group.  Forced: it is the
    #: transpose width, because that is where the lane rows put `k`.
    stride: int

    def covers(self, group: int, k: int) -> Tuple[int, ...]:
        """The contraction values one issue of group `group` sums over."""
        return tuple(group + self.stride * step for step in range(k))

    @property
    def contiguous(self) -> bool:
        """Is an issue's contraction set a contiguous run?

        Never, for stride above one -- and the stride is the transpose width.
        Here so that a caller wanting a run (block sparsity along `k` is the
        one that does) asks rather than assumes, and so that the answer is a
        property of the exchange rather than a remark in a docstring.
        """
        return self.stride == 1


def a_exchange(op) -> Optional[Exchange]:
    """The transpose that feeds this instruction's A fragment, or `None`.

    `None` where the instruction broadcasts its own A, where `hip.h` has no
    transpose of the right width, or where the fragment keeps part of its
    contraction in the register --- with more than one element per lane the
    relabelling has to be injective across slots as well, which is a further
    claim and not one this has checked.
    """
    if broadcast_feeds_a(op) or op.a.per_lane != 1:
        return None
    rows = op.wave // (op.m * op.blocks)
    if op.blocks != 1 or op.k != rows:
        return None
    name = f'tensorforge::transpose{op.m}x{op.m}b32'
    from .catalog import DEFINED_TRANSPOSES
    if name not in DEFINED_TRANSPOSES:
        return None
    return Exchange(name, op.m, op.m)


#: Which accessor feeds which fragment, under the assignment this path uses:
#: the instruction's M takes the output columns and its N takes the leading
#: dimension.  Stated because the two are not interchangeable and the names
#: collide --- the instruction's A is the caller's `ops.B`.
FED_BY = {'A': 'ops.B (shared matrix)', 'B': 'ops.A (leading operand)'}


def broadcast_feeds_a(op) -> bool:
    """Does the instruction fetch its own A operand across the blocks?

    `cbsz` selects a broadcast group and `abid` picks the block to read, so
    one block's worth of A reaches all of them --- which is why `matmul32`
    transposes four registers once and feeds sixteen blocks.  It needs blocks
    to broadcast between, and that is all it needs: `mfma_f64_4x4x4f64` has
    four of them alongside a contraction of four, so a wider contraction does
    not cost an instruction its broadcast.
    """
    return op.broadcast and op.blocks > 1


def compose_exchange(have, want, indices, wave: int, base_have=None,
                     base_want=None) -> Optional[Tuple[Tuple[Move, ...], ...]]:
    """A bit exchange assembled out of swaps and merges, or `None`.

    A `swap` does not move anything between a register and a lane, so no
    sequence of them transposes one register.  Several registers and a merge
    do: the elements going from register `r` to register `s` all move by one
    XOR, and masking the merge to the lanes they land on leaves the rest of
    `s` alone.  `bitlayout.moves` is what finds those regions, so this is the
    same question the operands already ask, put to a different pair of
    layouts.

    Which makes the transposes a convenience rather than a capability: the
    4x4 gap comes out as sixteen regions whose XOR is `source ^ target`, and
    the runtime's `transpose4x4b32` is that arrangement written once.  Where a
    width has no such function -- and `DEFINED_TRANSPOSES` names four -- this
    is what closes the gap in registers, at a cost between the builtin and a
    trip through memory.

    Grouped by destination register, because that is what an emitter walks:
    one output at a time, merging each source into it under the region's mask.
    `Move.contraction` is the source register, which is what it is in a
    fragment plan too -- the register the region is read from.
    """
    kwargs = {}
    if base_have is not None:
        kwargs['base_have'] = base_have
    if base_want is not None:
        kwargs['base_want'] = base_want
    found = bitlayout.moves(have, want, indices, **kwargs)
    if found is None:
        return None
    groups = [[] for _ in range(max(move.target for move in found) + 1)]
    for move in found:
        groups[move.target].append(
            Move(move.source, _swaps_for(move.xor), Select.of(move.lanes, wave)))
    return tuple(tuple(group) for group in groups)


def emittable(groups) -> bool:
    """Whether every region of an assembled exchange has a mask that exists.

    `dppUpdate` carries a region in a row mask and a bank mask, and a
    transpose's regions are one lane out of every `ext` -- one per bank, which
    neither expresses.  `Select` reports that as `cndmask`, a ternary on the
    lane id, and the merge refuses it: the path reads no lane id anywhere else
    and one appearing is a thing to look at rather than to paper over.

    So the assembled form is priced and not emitted.  The cost is right -- the
    ternary is in `Move.cost` -- and what is missing is a merge primitive, not
    a plan.
    """
    return all(move.select.free for group in groups for move in group)


def compose_cost(groups) -> int:
    """Instructions the assembled exchange issues."""
    return sum(move.cost for group in groups for move in group)


def fragment_moves(op, which: str, slot: int,
                   group: int = 0) -> Optional[Tuple[Move, ...]]:
    """How to build one slot of the instruction's B fragment, or `None`.

    `group` picks which stretch of the leading dimension the fragment covers.
    A fragment holds `n * blocks` of it and a wave holds `wave`, so there are
    `wave // (n * blocks)` of them and the whole wave takes that many
    fragments --- each with its own accumulator, all sharing these registers.

    `None` when the operand has no layout, when the source is not one scalar
    per lane, or when a region's lane movement is not a single XOR.  The last
    is the interesting one: it means an instruction needs more than swaps and
    a merge, and a caller that gets `None` should stay on the generic nest
    rather than emit something close.

    A region that no `dppUpdate` mask reaches is *not* a refusal --- `Select`
    reports `cndmask` and the plan carries the extra instruction in its cost.
    Refusing there would decline the cases the masks were never going to
    cover, which is a different thing from the plan not applying.
    """
    if which.upper() != 'B':
        # 'D' is written, not built. 'A' is fed by the shared matrix, which
        # arrives with the contraction in the lanes -- the transpose of what
        # every step below assumes, and no `swap` crosses between a register
        # and a lane. Where `blocks > 1` the instruction fetches its own A
        # through `cbsz`/`abid`; where it does not, a `transpose*` has to run
        # first and that is not this module.
        return None
    frag = op.b
    if not layouts.covers(op, which) or frag.per_lane <= slot:
        return None
    extent = op.n
    span = extent * op.blocks
    if op.wave % span or group >= op.wave // span:
        return None
    fragment = layouts.fragment_layout(op, which)
    if fragment is None or op.replication(which.lower()) != 1:
        # A replicated fragment holds each element in several lanes, and
        # `position` names only the lowest of them -- so a plan read off it
        # would move one copy and leave the others reading whatever was there.
        # The RDNA 3 rows are the ones where that is not 1.
        return None

    # The nest's own registers: one per contraction value, with the leading
    # dimension across the lanes.  The contraction is what picks a register,
    # which is why it is the slot here and the region key below.
    nest = bitlayout.BitLayout((
        tuple(bitlayout.Bit(bitlayout.Place.LANE, extent << bit)
              for bit in range((op.blocks - 1).bit_length())),
        tuple(bitlayout.Bit(bitlayout.Place.SLOT, 1 << bit)
              for bit in range(max(op.k - 1, 0).bit_length())),
        tuple(bitlayout.Bit(bitlayout.Place.LANE, 1 << bit)
              for bit in range((extent - 1).bit_length())),
    ))
    indices = [(block, contraction, index)
               for block in range(op.blocks)
               for contraction in range(op.k)
               for index in range(extent)]

    found = bitlayout.moves(nest, fragment, indices,
                            base_have=bitlayout.Position(lane=group * span))
    if found is None:
        return None
    return tuple(Move(move.source, _swaps_for(move.xor),
                      Select.of(move.lanes, op.wave))
                 for move in found if move.target == slot)


@dataclass(frozen=True)
class Gather:
    """One accumulator register into one region of one output column.

    The mirror of `Move`: that one names a source the nest holds and a region
    of the fragment, this one names a source the fragment holds --- a group's
    accumulator and one of its slots --- and a region of the nest's register.
    """

    #: Which accumulator.  The instruction covers `n * blocks` of the leading
    #: dimension, so a wave takes `wave // (n * blocks)` of them, each with
    #: its own accumulator running over the whole contraction.
    group: int
    #: Which register of that accumulator.
    slot: int
    #: `swap<Block>` sequence, in order.
    swaps: Tuple[int, ...]
    #: Which lanes of the output register this writes.
    select: Select

    @property
    def cost(self) -> int:
        return len(self.swaps) + 1 + (0 if self.select.free else 1)


def accumulator_gathers(op, column: int) -> Optional[Tuple[Gather, ...]]:
    """How to build the nest's register for one output column, or `None`.

    The nest stores through ``C(writer, value, i, j)`` with the leading
    dimension across the lanes, so lane `l` of the result has to hold the
    output for leading-dimension element `l`.  Which group covers that element
    and where inside its accumulator it sits both follow from the layout, and
    what is left is the same constant XOR per region that the operands need.

    Paid once per output tile rather than once per contraction step, which is
    why it can afford to be the more expensive half: `mfma_f64_16x16x4f64`
    spends eight instructions per column against eight for a whole B fragment,
    but the fragment is rebuilt every k-block and this runs in the epilogue.
    """
    if not layouts.covers(op, 'D') or column >= op.m:
        return None
    span = op.n * op.blocks
    if op.wave % span:
        return None
    fragment = layouts.fragment_layout(op, 'D')
    if fragment is None:
        return None

    # The nest's own register: lane `l` holds leading-dimension element `l`,
    # and nothing varies with the column -- each one is a separate store.
    nest = bitlayout.BitLayout((
        tuple(bitlayout.Bit(bitlayout.Place.LANE, op.n << bit)
              for bit in range((op.blocks - 1).bit_length())),
        (),
        tuple(bitlayout.Bit(bitlayout.Place.LANE, 1 << bit)
              for bit in range((op.n - 1).bit_length())),
    ))
    indices = [(block, column, index)
               for block in range(op.blocks) for index in range(op.n)]

    gathers = []
    for group in range(op.wave // span):
        found = bitlayout.moves(
            fragment, nest, indices,
            base_want=bitlayout.Position(lane=group * span))
        if found is None:
            return None
        gathers += [Gather(group, move.source, _swaps_for(move.xor),
                           Select.of(move.lanes, op.wave))
                    for move in found]
    return tuple(gathers)


def accumulator_cost(op) -> Optional[int]:
    """Instructions for the whole writeback, or `None`.

    Every output column, so the epilogue in full.  What it competes with is
    the contraction loop that filled the accumulators, not a single
    instruction.
    """
    total = 0
    for column in range(op.m):
        gathers = accumulator_gathers(op, column)
        if gathers is None:
            return None
        total += sum(gather.cost for gather in gathers)
    return total


def fragment_cost(op, which: str, group: int = 0) -> Optional[int]:
    """Instructions to build a whole fragment of `which`, or `None`.

    The number to weigh against staging the operand through shared memory,
    and against not taking the matrix path at all.  It is paid once per
    k-block and per group, and the fragment is then read by every output
    column, so what it competes with is the column loop rather than one
    instruction.
    """
    total = 0
    frag = op.b
    for slot in range(frag.per_lane):
        moves = fragment_moves(op, which, slot, group)
        if moves is None:
            return None
        total += sum(move.cost for move in moves)
    return total
