# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The relayout that always works, and what it costs to take it.

`bitlayout.moves` serves a gap the swap family reaches and
`amd.relayout.transposes_between` one a transpose reaches.  This is the rung
below both, and the only one `routes.reach` can offer without asking a
target: it needs no instruction that moves bits between a lane index and a
register index, which is what a target that has none hands in nothing about.  Both answer `None`
where their instruction does not close the gap, and both are right to: a
constant XOR per region does not permute bits, and an exchange of lane and
slot bits does not reach an element inside a register.

A round trip through memory closes every gap.  Each lane writes what it holds
to an address that depends only on which element it is, and reads back what it
needs from the address that element has -- so the distribution on the way in
and the one on the way out are unrelated, and no property of either has to
hold for the transfer to work.  That is what makes it the fallback rather than
one more entry in a table.

It is also the expensive one, which is why it is stated as a plan before it is
emitted.  A register path costs the instructions `reorder.fragment_cost`
counts; this costs a store, a barrier and a load per element, plus the buffer.
Having both as numbers is what lets the cheaper one be preferred for a reason.

The address here is the element's own linear index, which is the choice that
needs no argument: dense, one per element, no holes.  It is a choice, though,
and a later pass is free to permute it -- that is what swizzling is for, and
the plan states the address rather than computing it inside an emitter so that
such a pass has something to rewrite.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from .bitlayout import BitLayout, Position


@dataclass(frozen=True)
class Transfer:
    """One element's trip through the buffer."""

    #: Where it rests in between.  The element's linear index in the space the
    #: caller enumerated.
    address: int
    #: Which register of which lane writes it.
    source: Position
    #: Which register of which lane reads it back.
    target: Position


def staged(have: BitLayout, want: BitLayout, indices: Sequence,
           base_have: Position = Position(),
           base_want: Position = Position()) -> Optional[Tuple[Transfer, ...]]:
    """The trip from one distribution to the other, or `None`.

    `None` only where the two do not describe the same index space.  There is
    no shape of gap this declines -- that is the point of it, and a caller
    reaching here has already been told `None` by something cheaper.

    A vector element is carried like any other position: a packed operand
    writes each of its elements to that element's own address, which is
    exactly the case no cross-lane instruction reaches and this one does not
    have to treat specially.
    """
    if len(have.axes) != len(want.axes):
        return None
    out = []
    for address, index in enumerate(indices):
        out.append(Transfer(address,
                            base_have + have.locate(*index),
                            base_want + want.locate(*index)))
    return tuple(out)


def buffer_elements(plan: Iterable[Transfer]) -> int:
    """Elements the buffer has to hold.

    One per address, and the addresses are dense, so this is the element count
    -- the number a reservation has to be made for.  A reservation smaller
    than this is an overrun and one larger is shared memory nobody writes, so
    it is read from the plan rather than computed a second time beside it.
    """
    addresses = {transfer.address for transfer in plan}
    return len(addresses)


def accesses(plan: Iterable[Transfer]) -> Tuple[int, int]:
    """Stores and loads the trip issues, as `(stores, loads)`.

    One of each per element, which is what makes this the expensive answer:
    the register paths move a whole region per instruction and this moves one
    element.  The barrier between them is not counted here -- it is one, and
    what it costs depends on what else the wave is doing.
    """
    plan = tuple(plan)
    return len(plan), len(plan)
