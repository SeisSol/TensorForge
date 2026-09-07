# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Spending an instruction axis on work the problem does not fill.

A matrix instruction computes ``m * n * k`` products whatever is put in it, so
nothing here makes a full tile cheaper -- the arithmetic is conserved and
rearranging it cannot reduce the count.  What it can do is stop a tile from
computing zeroes.  Every axis has a capacity the entry fixes and a demand the
problem states, and where the demand falls short the difference is issued
either way.

That makes two decisions the same decision.  Padding a partial block and
running its spare lanes against zeroes is one; putting a second term product
of a split-precision emulation into slots the contraction does not reach is
the other.  Both ask what an axis has left over and what is worth putting
there, so both are counted here.

Which axis is available is not.  The mapping from the generator's dimensions
onto an instruction's is `MatrixOp.lane_batched`'s subject and differs per
scheme, and what an axis costs to reclaim differs with it: an instruction sums
over its own **K**, so products laid there need no epilogue at all; **M** is
the accumulator's registers, so summing afterwards is an add chain inside a
lane; ``n * blocks`` is the lanes, so it is a cross-lane reduction and
`reorder.accumulator_cost` is what it costs.  This module takes a capacity and
counts.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


def tiles(demand: int, capacity: int) -> int:
    """Instructions an axis of `capacity` takes to cover `demand`."""
    if capacity <= 0:
        raise ValueError(f'an axis holds at least one position, got {capacity}')
    return -(-max(demand, 0) // capacity)


def waste(demand: int, capacity: int) -> int:
    """Positions the last tile leaves empty.

    Zero where the demand divides, and never `capacity` -- an empty tile is
    one that is not issued rather than one that is wholly wasted.
    """
    return (-max(demand, 0)) % capacity


@dataclass(frozen=True)
class Slot:
    """One term product at one contraction step, in one position of one
    instruction."""

    #: Which instruction, counted from the first of this tile.
    instruction: int
    #: Where in the axis, below its capacity.
    position: int
    #: `(i, j)` term indices, as `split.products` orders them.
    product: Tuple[int, int]
    #: Which step of the contraction this pair belongs to.  Both operands of a
    #: product have to come from the same one; pairing across steps computes a
    #: different sum, not a less accurate one.
    step: int


def _lay(pairs: Sequence[Tuple[Tuple[int, int], int]], capacity: int,
         first: int = 0) -> Tuple[Tuple[Slot, ...], int]:
    """Fill instructions from `first`, `capacity` positions at a time."""
    slots = []
    for index, (product, step) in enumerate(pairs):
        slots.append(Slot(instruction=first + index // capacity,
                          position=index % capacity,
                          product=product, step=step))
    used = tiles(len(pairs), capacity)
    return tuple(slots), first + used


def stages(products: Iterable[Tuple[int, int]], steps: int,
           capacity: int) -> Tuple[Slot, ...]:
    """One term product per instruction, the contraction filling the axis.

    The arrangement `mfma_emu_bf16_f32` sketches: every product issued
    separately into the same accumulator.  It needs no axis of its own and no
    epilogue, which is why it works under any mapping -- and it pays the
    partial last instruction once per product rather than once.
    """
    out, first = [], 0
    for product in products:
        laid, first = _lay([(product, step) for step in range(steps)],
                           capacity, first)
        out += laid
    return tuple(out)


def packed(products: Iterable[Tuple[int, int]], steps: int,
           capacity: int) -> Tuple[Slot, ...]:
    """Products and contraction steps sharing the axis.

    The same pairs, laid end to end instead of restarting per product, so a
    partial instruction is paid once for the whole emulation.  Products come
    in the order `split.products` gives them, which is smallest contribution
    first, and the steps of one product stay together.
    """
    pairs = [(product, step) for product in products for step in range(steps)]
    return _lay(pairs, capacity)[0]


def instructions(slots: Iterable[Slot]) -> int:
    """How many the layout issues."""
    positions = {slot.instruction for slot in slots}
    return len(positions)


def spare(slots: Iterable[Slot], capacity: int) -> int:
    """Positions the layout leaves empty across every instruction it issues.

    What a second term product could have gone into, and what a padded block
    computes against zeroes.
    """
    slots = tuple(slots)
    return instructions(slots) * capacity - len(slots)


def saving(products: Sequence[Tuple[int, int]], steps: int,
           capacity: int) -> int:
    """Instructions `packed` issues fewer than `stages`.

    Never negative and often zero: where the capacity divides the contraction
    there is nothing left over to share, and both lay the same work into the
    same number of full instructions.  It is not simply "zero when it divides"
    either -- three products of a three-deep contraction into four positions
    fill the spare space of one instruction exactly and save nothing, while
    two products of the same contraction save one.  Enumerating is cheaper
    than a closed form that has to be right.
    """
    return (instructions(stages(products, steps, capacity))
            - instructions(packed(products, steps, capacity)))
