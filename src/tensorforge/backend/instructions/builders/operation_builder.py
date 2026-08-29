# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What building one operation looks like, whatever the operation is.

Four phases, and they are the same four for a contraction, a pointwise
operation and a reduction:

* **resolve** the operands into something the compute instruction can read.
  A contraction stages them -- into registers, into shared memory, or not at
  all -- and a pointwise operation settles them back where their symbols say.
* **allocate** whatever the result is written into: an accumulator, or the
  destination's own symbol when it has one.
* **emit** the compute instruction.
* **record** where the result now is, so the next operation can find it.

Only the second and third differ much between the three, which is why the
first and last are implemented here.  The contraction overrides all four,
because staging an operand and accumulating into a shifted origin are the two
things genuinely specific to it.

The section-level collaborators arrive through the constructor and are shared
by every builder working on one section: the plan, so all three answer
"how wide" the same way; the residency, so a value one of them leaves in
registers is one the next can find; and the temporaries, so `s0` and `r0` are
unique across everything the section produces.
"""

from abc import abstractmethod
from typing import List, Optional

from tensorforge.backend.instructions.builders.abstract_builder import (
    AbstractBuilder)
from tensorforge.backend.residency import Residency
from tensorforge.backend.section_plan import SectionPlan
from tensorforge.backend.symbol import Symbol, SymbolView
from tensorforge.backend.temporaries import Temporaries
from tensorforge.common.context import Context
from tensorforge.generators.descriptions import OperationDescription


class OperationBuilder(AbstractBuilder):
    def __init__(self,
                 context: Context,
                 scopes,
                 shr_mem: Symbol,
                 num_threads: int,
                 plan: SectionPlan,
                 residency: Residency,
                 temporaries: Temporaries,
                 lead_width: int = 1):
        super().__init__(context, scopes)
        self._shr_mem = shr_mem
        self._num_threads = num_threads
        self._lead_width = lead_width
        #: The section's read/write geometry: a statement about the descriptor
        #: list rather than about any one operation.
        self._plan = plan
        #: Which tensors have their newest copy somewhere other than where
        #: their symbol says.  Shared, so a value one operation leaves in
        #: registers is one the next can pick up.
        self._residency = residency
        #: Where the section's own buffers and their names come from.
        self._temporaries = temporaries

    def build(self, descr: OperationDescription) -> None:
        self._reset()
        operands = self.resolve_operands(descr)
        dest = self.alloc_destination(descr, operands)
        self.emit_compute(descr, operands, dest)
        self.record_result(descr, dest)

    # -- the four phases -------------------------------------------------- #

    def resolve_operands(self, descr) -> List:
        """Put every tensor this operation touches back where its symbol says.

        The default, and what an operation that addresses its operands through
        their symbols needs.  A value still sitting in an accumulator is one
        such an operation would read straight past, so it is written out first.

        Destinations settle too, and for the same reason rather than a
        different one.  A pending writeback that is dropped instead loses
        whatever the new operation does not cover; one left in place is emitted
        at the section boundary and overwrites what the new operation just
        produced.  Writing it out first is right either way, and the redundant
        store when the new operation does cover everything is a matter for the
        placement decision, not for correctness.

        No barrier is emitted alongside: `SyncThreadsOpt` discards every sync
        in the section and reinserts them from the shared-memory write/use
        pairs, so one placed here would be removed and one that is needed
        appears without being asked for.
        """
        views = list(descr.reads())
        dest = descr.writes()
        if dest is not None:
            views.append(dest)
        for view in views:
            symbol = self._scopes.get_symbol(view.tensor)
            if symbol is not None:
                self._instructions.extend(self._residency.flush(symbol.name))
        return [self.view_of(view) for view in descr.reads()]

    @abstractmethod
    def alloc_destination(self, descr, operands) -> SymbolView:
        """The view the compute instruction writes its result into."""

    @abstractmethod
    def emit_compute(self, descr, operands, dest) -> None:
        """Append the instruction that does the work."""

    def record_result(self, descr, dest) -> None:
        """Say where the result now is.

        Nothing to do by default: a destination materialised into registers
        got its writeback recorded when the array was allocated, because the
        two are one fact, and a destination that already had a symbol was
        written in place.
        """

    # -- shared machinery ------------------------------------------------- #

    def view_of(self, subtensor) -> SymbolView:
        """A descriptor's view of a tensor, against that tensor's symbol."""
        symbol = self._scopes.get_symbol(subtensor.tensor)
        return SymbolView(symbol, subtensor.bbox, subtensor.offset)

    def materialise_dest(self, descr, lead_pos: int) -> Optional[SymbolView]:
        """A destination this section produces itself, and where it goes.

        A temporary that no operation has written yet has no symbol at all, so
        a view built over it wraps `None`.  Giving it one takes two objects,
        not one: the result is computed into a register array, and a pending
        writeback says that array is the newest copy of the shared buffer it
        belongs in.

        Writing shared memory straight from the compute would be shorter and is
        not available.  `ShrMemOpt` sizes each buffer from its first user and
        requires that user to be a memory instruction able to report a size; a
        compute instruction there fails the check rather than allocating
        nothing.  Going through registers gives the buffer the store it needs
        as its first user, and costs nothing that is not already paid: a
        consumer able to read the image reads it in place, and the flush
        happens only for one that cannot.

        Returns None when the destination already has a symbol and can be
        written where it is.
        """
        dest = descr.writes()
        if dest is None or self._scopes.get_symbol(dest.tensor) is not None:
            return None
        home = self._temporaries.shared_symbol(dest.tensor)
        registers, alloc = self._temporaries.register_array(dest.bbox, lead_pos)
        self._instructions.append(alloc)
        self._residency.record_writeback(home.name, registers, home,
                                         covered=dest.bbox,
                                         shift=[0] * dest.bbox.rank())
        return SymbolView(registers, dest.bbox, [0] * dest.bbox.rank())
