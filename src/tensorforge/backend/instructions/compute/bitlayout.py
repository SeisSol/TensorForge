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


def from_lane_axis(block: int, stride: int, extent: int
                   ) -> Optional[Tuple[Bit, ...]]:
    """`LaneAxis(block, stride)` as bits, or `None` where it is not expressible.

    The axis says element `s` lives in slot ``s // block``, held by the threads
    with ``(t // stride) % block == s % block``.  In bits: the low
    ``log2(block)`` bits of `s` are lane bits starting at ``log2(stride)``, and
    everything above them is a slot bit -- one cut point, which is the whole
    of what this language says and a special case of what the fragment tables
    say.

    `None` where `block` or `stride` is not a power of two.  That is not a gap
    to fill in later: an axis that wraps at nine elements does not decompose
    into bits at all, and moving between two such distributions is a shuffle by
    lane index rather than a permutation of bits.  Saying so is the point --
    the fragment side is powers of two throughout because the hardware is, and
    the value side is only sometimes.
    """
    low = _bits_of(block)
    base = _bits_of(stride)
    if low is None or base is None:
        return None
    total = max(1, extent - 1).bit_length()
    out = []
    for position in range(total):
        if position < low:
            out.append(Bit(Place.LANE, 1 << (base + position)))
        else:
            out.append(Bit(Place.SLOT, 1 << (position - low)))
    return tuple(out)


def from_register_layout(layout, extents: Sequence[int]) -> Optional[BitLayout]:
    """A PIR `RegisterLayout` in this vocabulary, or `None`.

    One axis per tensor dimension, in the layout's own order, each through
    :func:`from_lane_axis`.  `None` as soon as one of them does not decompose,
    because a layout is only comparable to a fragment if all of it is.

    The extents come from outside: a `LaneAxis` says how a dimension is spread
    and not how far it reaches, and the number of bits an index needs is the
    second of those.
    """
    if layout is None or len(layout.axes) != len(extents):
        return None
    axes = []
    for axis, extent in zip(layout.axes, extents):
        bits = from_lane_axis(axis.block, axis.stride, extent)
        if bits is None:
            return None
        axes.append(bits)
    return BitLayout(tuple(axes))
