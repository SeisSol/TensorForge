# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What building one operation looks like, whatever the operation is.

Four phases, and they are the same four for a contraction, a pointwise
operation and a reduction:

* **resolve** the operands into something the compute instruction can read.
  A contraction stages them -- into registers, into shared memory, or not at
  all -- and a pointwise operation reads them where their symbols say.
* **allocate** whatever the result is written into: an accumulator, or the
  destination's own symbol when it has one.
* **emit** the compute instruction.
* **record** the result where its symbol says, so the next operation finds
  it there.

Only the second and third differ much between the three, which is why the
first and last are implemented here.  The contraction overrides all four,
because staging an operand and accumulating into a shifted origin are the two
things genuinely specific to it.

The section-level collaborators arrive through the constructor and are shared
by every builder working on one section: the plan, so all three answer
"how wide" the same way, and the temporaries, so `s0` and `r0` are unique
across everything the section produces.  Nothing a build stages or computes
in registers reaches the next: that one reads what this one stored, and what
the round trip costs, the pseudo-IR takes back (`pir.forward`).
"""

from abc import abstractmethod
from typing import List, Optional

from tensorforge.backend.instructions.builders.abstract_builder import (
    AbstractBuilder)
from tensorforge.backend.instructions.memory.store import (StoreRegToGlb,
                                                           StoreRegToShr)
from tensorforge.backend.section_plan import SectionPlan
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.temporaries import Temporaries
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.generators.descriptions import OperationDescription


class OperationBuilder(AbstractBuilder):
    def __init__(self,
                 context: Context,
                 scopes,
                 shr_mem: Symbol,
                 num_threads: int,
                 plan: SectionPlan,
                 temporaries: Temporaries,
                 lead_width: int = 1):
        super().__init__(context, scopes)
        self._shr_mem = shr_mem
        self._num_threads = num_threads
        self._lead_width = lead_width
        #: The section's read/write geometry: a statement about the descriptor
        #: list rather than about any one operation.
        self._plan = plan
        #: Where the section's own buffers and their names come from.
        self._temporaries = temporaries
        #: The store that takes a result from its register image to its
        #: home, once the compute has shaped the image -- see
        #: `pointwise_dest` and `materialize_dest`.
        self._home_store = None

    def build(self, descr: OperationDescription) -> None:
        self._reset()
        self._home_store = None
        operands = self.resolve_operands(descr)
        dest = self.alloc_destination(descr, operands)
        self.emit_compute(descr, operands, dest)
        self.record_result(descr, dest)

    # -- the four phases -------------------------------------------------- #

    def resolve_operands(self, descr) -> List:
        """Every operand where its symbol says: what an operation ahead of
        this one computed, it stored there (`record_result`).

        No barrier is emitted alongside: the barrier placement works on the
        section's body, in the order its accesses end up in (`pir.barriers`),
        so one that is needed appears there without being asked for.
        """
        return [self.view_of(view) for view in descr.reads()]

    @abstractmethod
    def alloc_destination(self, descr, operands) -> SymbolView:
        """The view the compute instruction writes its result into."""

    @abstractmethod
    def emit_compute(self, descr, operands, dest) -> None:
        """Append the instruction that does the work."""

    def record_result(self, descr, dest) -> None:
        """Store the result where its symbol says.

        A destination written in place is where it belongs.  What is left is
        the store `pointwise_dest` or `materialize_dest` decided on: it reads
        the register image's shape, which the compute instruction gives it,
        so it is built only now.
        """
        make, self._home_store = self._home_store, None
        if make is not None:
            self._instructions.append(make())

    # -- shared machinery ------------------------------------------------- #

    def view_of(self, subtensor) -> SymbolView:
        """A descriptor's view of a tensor, against that tensor's symbol."""
        symbol = self._scopes.get_symbol(subtensor.tensor)
        return SymbolView(symbol, subtensor.bbox, subtensor.offset)

    def materialize_dest(self, descr, lead_pos: int) -> Optional[SymbolView]:
        """A destination this section produces itself, and where it goes.

        A temporary that no operation has written yet has no symbol at all, so
        a view built over it wraps `None`.  Giving it one takes two objects,
        not one: the result is computed into a register array, and stored
        behind the compute into the shared buffer it belongs in
        (`record_result`).

        Writing shared memory straight from the compute would be shorter and is
        not available.  A buffer's window is declared by its first user
        (`Generator._declare_buffers`), which has to be a memory instruction
        that knows the buffer's size; a compute instruction there would have no
        window to declare.  Going through registers gives the buffer the store
        it needs as its first user, and costs nothing that is not already paid:
        where the next operation loads the result back, the pseudo-IR reads
        the registers instead (`pir.forward`).

        Returns None when the destination already has a symbol and can be
        written where it is.
        """
        dest = descr.writes()
        if dest is None or self._scopes.get_symbol(dest.tensor) is not None:
            return None
        if self._plan.zero_first(dest.tensor):
            # The store behind the compute writes the register image alone;
            # the zeros a later use needs would never be written.
            raise GenerationError(
                f'{getattr(dest.tensor, "alias", None) or dest.tensor}: a '
                f'later operation uses cells of this temporary that nothing '
                f'defines before it, which are zero, and its first write is '
                f'a {type(descr).__name__} -- only the contraction\'s store '
                f'clears a buffer so far (`SectionPlan.zero_first`).')
        home = self._temporaries.shared_symbol(dest.tensor)
        registers, alloc = self._temporaries.register_array(dest.bbox, lead_pos)
        self._instructions.append(alloc)
        self._home_store = lambda: StoreRegToShr(
            context=self._context, src=registers, dest=home,
            shr_mem=self._shr_mem, num_threads=self._num_threads,
            lead_width=self._lead_width)
        return SymbolView(registers, dest.bbox, [0] * dest.bbox.rank())

    def pointwise_dest(self, descr, lead_pos) -> SymbolView:
        """Where a pointwise operation computes, and how the result gets home.

        An assignment defines its whole promise: the tensor's box, where the
        destination is the tensor with a narrower window, of which the window
        gets the values and the rest zeros (`SubTensor.owed_zeros`); a slice
        only its own box.  The contraction keeps it in its stores
        (`MultilinearBuilder._promised_box`).  A pointwise operation writing
        its window where the destination already is, and nothing else, would
        not: a temporary assigned anew from a narrower operand would keep its
        old values around the new ones, and so would an output.

        So the result is computed into a register image and stored the way
        the contraction stores one:

        * a temporary nothing wrote yet, which one operation writes whole,
          is stored into a buffer of its image's box (`materialize_dest`);
        * one assembled in shared memory from several writes, or one that
          already has a buffer, is stored into the buffer as
          it is laid out (`_buffer_box`), cleared where the plan found a use
          of cells nothing defined (`SectionPlan.zero_first`), and zeroed
          over the rest of its promise.  Through the store and not in place:
          a store tells liveness whether it defines the buffer or a part of
          it (`partial_defs`), and the barrier pass fences shared writes that
          are stores -- a compute instruction writing a buffer in place is
          neither;
        * an output owed zeros goes out through `StoreRegToGlb`, which zero-
          fills the promise; one that is not is written in place.
        """
        dest = descr.writes()
        tensor = dest.tensor
        symbol = self._scopes.get_symbol(tensor)
        owed = None if getattr(descr, 'add', False) else dest.owed_zeros()
        if symbol is None:
            if not self._plan.written_in_slices(tensor):
                return self.materialize_dest(descr, lead_pos)
            symbol = self._temporaries.shared_symbol(tensor)
            first = True
        elif symbol.stype == SymbolType.SharedMem:
            first = False
        elif symbol.stype == SymbolType.Global and owed is not None:
            registers, view, shift = self._register_image(dest, lead_pos)
            self._home_store = lambda: StoreRegToGlb(
                context=self._context, src=registers, dest=symbol,
                num_threads=self._num_threads, lead_width=self._lead_width,
                dest_offset=shift,
                # the promise in the image's coordinates, as `src` is
                dest_bbox=BoundingBox(
                    [l - s for l, s in zip(owed.lower(), shift)],
                    [u - s for u, s in zip(owed.upper(), shift)]),
                zero_fill=True)
            return view
        else:
            return self.view_of(dest)

        registers, view, shift = self._register_image(dest, lead_pos)
        # the first write clears the whole buffer where the plan found cells
        # nothing defines before a use; an assignment at least what it
        # promised and did not compute
        zero_first = first and self._plan.zero_first(tensor)
        buffer = self._buffer_box(symbol, dest, first)
        self._home_store = lambda: StoreRegToShr(
            context=self._context, src=registers, dest=symbol,
            shr_mem=self._shr_mem, num_threads=self._num_threads,
            lead_width=self._lead_width,
            dest_bbox=buffer,
            dest_offset=shift,
            clear=zero_first or owed is not None,
            clear_within=None if zero_first else owed)
        return view

    def _buffer_box(self, symbol, dest, first):
        """The box `symbol`'s shared buffer is laid out over.

        Everything the section writes of it, for the store that introduces it
        (`SectionPlan.dest_union`).  A buffer that already has a layout keeps
        it: its first store sized the window (`Generator._declare_buffers`),
        and one that stored a result whole holds the range its producer
        computed, which can be narrower than the descriptors declare -- a
        contraction whose operands support fewer rows than its destination
        names.  Laid over the declared union instead, the store would
        re-stride the buffer and clear rows past its end.
        """
        union = self._plan.dest_union(dest.tensor)
        view = None if first else symbol.data_view
        if view is None:
            return union
        box = view.get_bbox()
        own = dest.storage_box()
        if box.rank() != own.rank() or any(
                ol < bl or bu < ou for ol, ou, bl, bu in zip(
                    own.lower(), own.upper(), box.lower(), box.upper())):
            return union
        return box

    def _register_image(self, dest, lead_pos):
        """A fresh register array for `dest`'s box, the view to compute into
        it, and where its position 0 sits in the tensor.

        Indexed from 0, and the box's corner and the slicing offset go to the
        store as its `dest_offset` -- the contraction's shifted origin
        (`MultilinearBuilder._store_offset`) with no shift of its own.  A
        register image cannot start its lead axis inside a round of the lanes
        (`Symbol.build_address`), and a piece `_cells` cut from a destination
        -- `exp(trace)` past the three entries `trace` stores -- starts
        wherever an operand's box ended.
        """
        rank = dest.bbox.rank()
        box = BoundingBox([0] * rank, list(dest.bbox.sizes()))
        registers, alloc = self._temporaries.register_array(box, lead_pos)
        self._instructions.append(alloc)
        return registers, SymbolView(registers, box, [0] * rank), \
            list(dest.storage_box().lower())
