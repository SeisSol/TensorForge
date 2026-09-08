# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where each bit of each index of a value lives.

Two layout languages are in the tree and they cannot be compared.  The PIR
says what a value *has*: `RegisterLayout`, a `LaneAxis(block, stride)` per
tensor dimension, with one cut point -- the low bits of a dimension go to the
lanes, the rest to the slots.  `layouts.FRAGMENT_BITS` says what an
instruction *needs*: a weight per index bit, positive for a lane bit and
negative for a slot bit, so an index's bits may be interleaved between the two
however the hardware interleaves them.

The second is strictly the wider language, which is why it is the one both
sides have to speak.  Until they do, no emitter can ask whether the operand it
holds is already in the distribution the instruction wants, and every one of
them bridges the gap by hand: a transpose called outright on one target, an
inline shared-memory address on another, an offset arithmetic and a `select`
on a third.

Three destinations, not two.  A bit can land on a lane, on a register slot, or
on an element of a vector-typed register -- and the third is what `LaneAxis`
excludes on purpose ("a lane holding four consecutive elements is a vector
*type* over the slot dimension, which is a different thing and does not belong
on this axis").  Naming it here is what would let a packed operand be a
starting layout rather than an exclusion: `strategy.is_contraction` refuses
`lead_width > 1` for every matrix arrangement today, and in this language that
is a question about whether a path exists, not a rule.

This module is the vocabulary and the translation into it.  It does not solve
for paths between two layouts; `relayout.find_relayout` searches a table for
an instruction producing one, and turning that into a solver is what the
vocabulary makes possible rather than something it does.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class Place(Enum):
    """Where one bit of an index ends up."""

    #: A bit of the lane id, so the element moves between lanes with it.
    LANE = 'lane'
    #: A bit of the register slot, so the element is a different register of
    #: the same lane.
    SLOT = 'slot'
    #: A bit of the element index inside a vector-typed register.  Distinct
    #: from `SLOT`: a `float4` is one register holding four elements, not four
    #: registers, and a move between two of its elements is neither a shuffle
    #: nor a register rename.
    VECTOR = 'vector'


@dataclass(frozen=True)
class Bit:
    """One bit of one index, and where it lives.

    `weight` is the power of two the bit carries *within its destination*, so
    a lane bit of weight 4 is lane bit 2.  Weights are kept rather than bit
    numbers because that is how `FRAGMENT_BITS` states them and converting on
    the way in would put an arithmetic between the table and its check.
    """

    place: Place
    weight: int


@dataclass(frozen=True)
class Position:
    """Which register of which lane holds an element, and where inside it."""

    slot: int = 0
    lane: int = 0
    element: int = 0

    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.slot + other.slot, self.lane + other.lane,
                        self.element + other.element)


@dataclass(frozen=True)
class BitLayout:
    """One `Bit` per bit of each index, outermost index first.

    An index whose tuple is empty contributes nothing, which is how a fragment
    states that it does not spread that index at all -- the single-block
    instructions carry an empty block tuple for exactly that reason.
    """

    axes: Tuple[Tuple[Bit, ...], ...] = ()

    def locate(self, *indices: int) -> Position:
        """Where the element at these indices sits.

        Where an index has fewer bits listed than its value needs, the extra
        bits contribute nothing -- the same reading `FRAGMENT_BITS` already
        gets from `_place`, which walks the tuple and not the value.
        """
        if len(indices) != len(self.axes):
            raise ValueError(f'{len(indices)} indices for a layout over '
                             f'{len(self.axes)}')
        out = Position()
        for bits, value in zip(self.axes, indices):
            for position, bit in enumerate(bits):
                if value >> position & 1:
                    out = out + _one(bit)
        return out


def _one(bit: Bit) -> Position:
    if bit.place is Place.LANE:
        return Position(lane=bit.weight)
    if bit.place is Place.SLOT:
        return Position(slot=bit.weight)
    return Position(element=bit.weight)


def from_weights(*tuples: Sequence[int]) -> BitLayout:
    """A `BitLayout` from `FRAGMENT_BITS`' own encoding.

    Positive is a lane weight and negative a slot weight, which is what
    `layouts._place` reads.  There is no vector case in that table: a matrix
    fragment is register-resident and the hardware does not put an index bit
    inside a vector element, so the third destination only ever arrives from
    the value side.
    """
    return BitLayout(tuple(
        tuple(Bit(Place.LANE if weight > 0 else Place.SLOT, abs(weight))
              for weight in bits)
        for bits in tuples))


def _bits_of(value: int) -> Optional[int]:
    """`log2(value)` where it is a power of two, else `None`."""
    if value < 1 or value & (value - 1):
        return None
    return value.bit_length() - 1


def from_lane_axis(block: int, stride: int, extent: int, width: int = 1
                   ) -> Optional[Tuple[Bit, ...]]:
    """`LaneAxis(block, stride)` at this packing as bits, or `None`.

    The axis says element `s` lives in slot ``s // block``, held by the threads
    with ``(t // stride) % block == s % block``.  In bits: the low
    ``log2(block)`` bits of `s` are lane bits starting at ``log2(stride)``, and
    everything above them is a slot bit -- one cut point, which is the whole
    of what this language says and a special case of what the fragment tables
    say.

    `width` is the second half of the same reading, and it belongs here rather
    than beside it.  `LeadIndex` maps the packed dimension by

        idx = width * (((tid / stride) % block) + nonlead * block) + c

    with `c` naming a component *inside* the value, so the low ``log2(width)``
    bits of `s` reach neither a lane nor a slot and the cut point moves up by
    exactly that many.  `LaneAxis` is right to leave the width out -- which
    lane holds which share does not change with it, which is why two indices
    differing only in width are the same distribution -- but a consumer that
    indexes *elements* needs both readings at once, and one bit string is what
    holds them.

    `None` where `block`, `stride` or `width` is not a power of two.  That is
    not a gap to fill in later: an axis that wraps at nine elements does not
    decompose into bits at all, and moving between two such distributions is a
    shuffle by lane index rather than a permutation of bits.  Saying so is the
    point -- the fragment side is powers of two throughout because the
    hardware is, and the value side is only sometimes.
    """
    low = _bits_of(block)
    base = _bits_of(stride)
    packed = _bits_of(width)
    if low is None or base is None or packed is None:
        return None
    total = max(1, extent - 1).bit_length()
    out = []
    for position in range(total):
        if position < packed:
            out.append(Bit(Place.VECTOR, 1 << position))
        elif position < packed + low:
            out.append(Bit(Place.LANE, 1 << (base + position - packed)))
        else:
            out.append(Bit(Place.SLOT, 1 << (position - packed - low)))
    return tuple(out)


def from_register_layout(layout, extents: Sequence[int],
                         widths: Optional[Sequence[int]] = None
                         ) -> Optional[BitLayout]:
    """A PIR `RegisterLayout` at this packing in this vocabulary, or `None`.

    One axis per tensor dimension, in the layout's own order, each through
    :func:`from_lane_axis`.  `None` as soon as one of them does not decompose,
    because a layout is only comparable to a fragment if all of it is.

    The extents come from outside: a `LaneAxis` says how a dimension is spread
    and not how far it reaches, and the number of bits an index needs is the
    second of those.  The widths come from outside for the same reason and a
    sharper one -- the packing is on the value's type, not on its layout, and
    reading the two apart is what left a packed operand describable only as a
    number nobody downstream could act on.

    Omitted means unpacked throughout, which is what every layout said before
    a width could be stated and what the corpus still says everywhere.
    """
    if layout is None or len(layout.axes) != len(extents):
        return None
    if widths is None:
        widths = (1,) * len(layout.axes)
    if len(widths) != len(layout.axes):
        return None
    axes = []
    for axis, extent, width in zip(layout.axes, extents, widths):
        bits = from_lane_axis(axis.block, axis.stride, extent, width)
        if bits is None:
            return None
        axes.append(bits)
    return BitLayout(tuple(axes))


def packed(layout) -> bool:
    """Whether any of this layout's index bits land inside a register.

    The question a consumer asks when it has no way to name where an element
    is beyond "which lane" -- a fragment table, a DPP pattern and a lane
    broadcast all address lanes, and a bit in `Place.VECTOR` is an element
    they have no coordinate for.  `False` for a layout that is `None`: an
    unstated distribution is unknown, not packed, and refusing on it would
    turn a missing annotation into an exclusion.
    """
    return layout is not None and any(bit.place is Place.VECTOR
                                      for axis in layout.axes for bit in axis)


def from_value(layout, type_, extents: Sequence[int], axis: int = 0
               ) -> Optional[BitLayout]:
    """What a value holds, read from its layout and its type together.

    The one place the two are joined.  A value states its distribution on
    `layout` and its packing on `type_.length`, and every consumer that indexes
    elements needs both -- a `float4` load and the scalar load it replaces
    carry the same `RegisterLayout`, deliberately, because a pass asking "is
    moving between these a shuffle?" must keep getting "no".  For a matrix
    fragment, which wants the leading dimension across the lanes, the same two
    values are not interchangeable at all, and until they were readable
    together the difference could only be spelled as `lead_width` -- a number
    whose only available use was to switch the arrangement off.

    `axis` names the dimension the packing is on, and defaults to the first
    because that is the emitter's own reading: `lead_width_of` returns the
    width of the *first* lead index of an access, and `layout_of_index` puts
    the lead axes in that same order.  It is stated rather than assumed so
    that a caller whose value is arranged otherwise can say so instead of
    getting a layout that is quietly wrong.

    `None` on everything :func:`from_register_layout` declines, and on an
    `axis` the layout does not have.
    """
    if layout is None:
        return None
    length = getattr(type_, 'length', None)
    width = 1 if length is None else length
    widths = [1] * len(layout.axes)
    if not -len(widths) <= axis < len(widths):
        return None
    widths[axis] = width
    return from_register_layout(layout, extents, widths)


@dataclass(frozen=True)
class Move:
    """One region of a value that reaches its destination by a single XOR.

    `source` and `target` are register slots and `xor` is what the lane id has
    to be toggled by for every element of the region at once.  A region is
    exactly as large as one constant reaches: where two elements sharing a
    pair of slots need different toggles they are not one move, and a plan
    that pretended otherwise would write some of them from the wrong lane.
    """

    source: int
    target: int
    xor: int
    lanes: Tuple[int, ...]


def moves(have: BitLayout, want: BitLayout, indices,
          base_have: Position = Position(),
          base_want: Position = Position()) -> Optional[Tuple[Move, ...]]:
    """How to get from one distribution to the other, or `None`.

    The `swap` family does not permute bits: `swap<B>` reads the lane `B` away,
    so it moves where a copy is read from and leaves the bit structure alone.
    What it can serve is therefore exactly this -- a constant XOR over a
    region -- and what it cannot is a layout that needs a bit moved between a
    lane and a slot, which is a transpose and `relayout.find_relayout`'s
    business.

    The two bases are constants added to each side, which is what a caller
    holding one of several groups of a wave contributes: the group index is an
    offset on the lane and not a bit of any index the two layouts share, so it
    cannot be an axis of either.  Which side carries it depends on which of
    them the wave is divided over, so both are stated rather than one assumed.

    `Move.lanes` are the *destination* lanes -- the region of `want`'s register
    this writes -- because that is what a caller has to mask when it merges.

    `None` where some pair of slots needs more than one toggle.  Not a
    limitation to route around: that is the case where no sequence of reads
    serves the whole region, and the honest answer is that this family does
    not reach it.

    The caller enumerates `indices` because only it knows the extents and,
    for a fragment, which of its indices corresponds to which axis of the
    value -- there is no correspondence that is right for both operands.
    """
    if len(have.axes) != len(want.axes):
        return None
    regions = {}
    for index in indices:
        here = base_have + have.locate(*index)
        there = base_want + want.locate(*index)
        if here.element or there.element:
            # A vector element is neither a lane nor a slot, so no lane toggle
            # reaches it; a packed operand needs the element bits accounted
            # for before this question is even the right one.
            return None
        regions.setdefault((here.slot, there.slot), []).append(
            (here.lane, there.lane))
    out = []
    for (source, target), pairs in sorted(regions.items()):
        toggles = {here ^ there for here, there in pairs}
        if len(toggles) != 1:
            return None
        out.append(Move(source, target, toggles.pop(),
                        tuple(sorted({there for _, there in pairs}))))
    return tuple(out)


def displacement(have: BitLayout, want: BitLayout
                 ) -> Optional[Tuple[Tuple[Bit, Bit], ...]]:
    """Bit by bit, where each one is and where it has to be.

    Only the bits that differ, in axis order then bit order.  `None` where the
    two do not describe the same index space -- a different number of axes, or
    an axis stated to a different number of bits -- because a bit with no
    counterpart is not a bit that has to move.
    """
    if len(have.axes) != len(want.axes):
        return None
    out = []
    for here, there in zip(have.axes, want.axes):
        if len(here) != len(there):
            return None
        out += [(a, b) for a, b in zip(here, there) if a != b]
    return tuple(out)


def is_exchange(pairs) -> Optional[Tuple[Tuple[int, int], ...]]:
    """The `(slot weight, lane weight)` pairs a transpose would have to swap.

    An exchange is the one shape a transpose has: every bit that moves goes
    between a slot and a lane, and the moves pair up -- a slot weight that
    becomes a lane weight has a bit coming back the other way at exactly those
    two weights.  Anything else is a different instruction or none: a bit that
    stays in the lanes and changes weight is a permutation within the lanes,
    which nothing in `RELAYOUTS` does, and a bit that reaches a vector element
    is not a cross-lane question at all.

    `()` where nothing differs, which is the answer that says no relayout is
    called for -- the one an emitter asking "is my operand already right"
    wants most often and cannot get today.
    """
    if pairs is None:
        return None
    forward, backward = {}, {}
    for source, target in pairs:
        if Place.VECTOR in (source.place, target.place):
            return None
        if source.place is target.place:
            return None
        if source.place is Place.SLOT:
            forward[(source.weight, target.weight)] = True
        else:
            backward[(target.weight, source.weight)] = True
    if set(forward) != set(backward):
        return None
    return tuple(sorted(forward))


def unpacked(layout: BitLayout) -> Tuple[BitLayout, int]:
    """The same elements with every vector bit moved into a register slot.

    A vector bit is an element of a packed register -- a `float4` holds four
    consecutive elements of the leading dimension in one -- and reaching one is
    a subscript, not a shuffle.

    Which makes this the right first step only where the fragment wants those
    elements in *registers*.  Where it wants them across the lanes, moving them
    into slots is the wrong direction and closes nothing: the lead operand of a
    matrix instruction is the case, and `test_staging` measures it.  The count
    is what a caller weighs; it is not a promise that the rest then follows.

    Returned with the count, because the extracts are what it costs and a
    caller comparing routes needs the number.  Zero means the layout holds
    nothing packed and this changed nothing.

    The freed bits become the *low* slot bits, above nothing: an extract names
    an element of a register and the register it lands in is the caller's to
    choose, so the cheapest reading is that the packed elements become
    consecutive registers.
    """
    freed = 0
    axes = []
    for bits in layout.axes:
        out, slot = [], 0
        for bit in bits:
            if bit.place is Place.VECTOR:
                out.append(Bit(Place.SLOT, 1 << slot))
                slot += 1
                freed += 1
            else:
                out.append(bit)
        axes.append(tuple(out))
    return BitLayout(tuple(axes)), freed
