# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which of several instructions serves a shape with the fewest issues.

Every target that has more than one matrix instruction has to pick one, and
picking the widest or the first in a list is a guess dressed as a rule.
Thirteen output columns take four four-wide issues or one sixteen-wide that
wastes three of its sixteen, and which is faster is a property of the two
instructions rather than of the waste.

So the choice is a count, and the count is the same on every target: an
instruction covers so many output columns, so many elements of the leading
dimension and so many contraction steps per issue, and the shape needs so
many of each.  What differs per target is how those three numbers are read
off an entry, which is why :class:`Extent` is what this takes rather than a
catalogue type.

It counts issues, not time.  Two instructions that differ in passes do not
cost the same per issue, which is what :data:`CYCLES` is for and why nothing
here is called a cost.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

from . import packing


@dataclass(frozen=True)
class Extent:
    """What one issue of an instruction covers, in the problem's own units.

    Stated in the generator's dimensions rather than the instruction's, so
    that a target whose mapping puts the contraction in the lanes and a target
    whose mapping puts it in the registers are compared by the same three
    numbers.  Converting is the vendor module's job and it is the only place
    that knows the mapping.
    """

    #: Output columns one issue covers.
    columns: int
    #: Elements of the leading dimension one issue covers.
    lanes: int
    #: Contraction steps one issue covers.
    depth: int
    #: What :data:`CYCLES` is keyed by.
    name: str = ''


def spare_products(extent: Extent, columns: int) -> int:
    """Term products one issue holds on the output axis, the real one included.

    A term product of a split-precision emulation is a whole ``C += A_i x
    B_j`` over the tile, so it occupies every output column the shape has --
    which is why partial spare is no use and this is a division rather than a
    remainder.  An instruction narrower than the shape holds one and no more.

    The output axis is the one worth spending.  The contraction axis has the
    small quantum -- one slot per step -- and no spare, since the contraction
    fills it; the lane axis has spare, but a product needs the whole leading
    dimension there and reclaiming it is a cross-lane reduction rather than an
    add chain inside a lane.
    """
    if columns <= 0:
        return 1
    return max(1, extent.columns // columns)


def issues(extent: Extent, columns: int, lead: int = 0, depth: int = 0,
           products: int = 1) -> int:
    """Issues this instruction takes for the whole contraction.

    The four axes multiplied.  An extent given as 0 is one the caller does not
    know, and counts as a single tile rather than as nothing -- so a caller
    that knows only its output width still ranks by that instead of ranking
    everything equal.
    """
    def tiles(demand, capacity):
        return packing.tiles(demand, capacity) if demand > 0 else 1

    return (tiles(columns, extent.columns)
            * tiles(lead, extent.lanes)
            * tiles(depth, extent.depth)
            * packing.tiles(products, spare_products(extent, columns)))


#: Issue cost per instruction, in cycles.  Empty.
#:
#: The numbers are a hardware fact like every row of a catalogue, and they have
#: to be read off the vendor's own documentation and checked the same way --
#: AMD states them per instruction in the ISA guide; other vendors would need
#: their own source, and one that does not publish them leaves its entries out
#: rather than getting a guess.
#:
#: Guessing is worse than not having them: :func:`rank` would then read as a
#: cost model and is a count.  Until it is filled, ranking by issues alone is
#: exactly wrong where two instructions differ in passes -- fewer issues of a
#: longer instruction can be slower.
CYCLES: Dict[str, int] = {}


def rank(candidates: Iterable, key: Callable[[object], Tuple[Extent, int]],
         columns: int, lead: int = 0, depth: int = 0) -> Tuple:
    """The candidates, cheapest first.

    `key` returns the extent of one candidate and how many term products it
    has to carry, which is the only thing a caller has to say about its own
    types.  Ties break on the extent's name so that a ranking is stable
    between runs, which a snapshot depends on.
    """
    def order(candidate):
        extent, products = key(candidate)
        count = issues(extent, columns, lead, depth, products)
        return (count * CYCLES.get(extent.name, 1), count, extent.name)

    return tuple(sorted(candidates, key=order))
