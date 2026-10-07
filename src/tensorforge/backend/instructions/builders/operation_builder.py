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

from tensorforge.backend.instructions.abstract_instruction import _explicit_simd
from tensorforge.backend.instructions.builders.abstract_builder import (
    AbstractBuilder)
from tensorforge.backend.instructions.memory.store import (StoreRegToGlb,
                                                           StoreRegToShr)
from tensorforge.backend.residency import Residency
from tensorforge.backend.section_plan import SectionPlan
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.temporaries import Temporaries
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.generators.descriptions import OperationDescription

#: The values of `register_temporaries`, fewest images read in place first.
REGISTER_TEMPORARIES = ('none', 'scalars', 'all')


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
        #: The store that takes a pointwise result from its register image to
        #: its home, once the compute has shaped the image -- see
        #: `pointwise_dest`.
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

        No barrier is emitted alongside: the barrier placement works on the
        section's body, in the order its accesses end up in (`pir.barriers`),
        so one that is needed appears there without being asked for.
        """
        views = list(descr.reads())
        dest = descr.writes()
        if dest is not None:
            views.append(dest)
        for view in views:
            self._settle(view)
        return [self.view_of(view) for view in descr.reads()]

    def resolve_in_place(self, descr, arrays: bool) -> List:
        """`resolve_operands`, for an operation that can read a register image.

        A temporary whose newest copy is still the register image its producer
        computed into is read there; everything else settles.  The image *is*
        the value -- the writeback is pending because the registers are the
        only copy -- so reading it gives what the store and the load back would
        have given, without either of them and without the barrier the store
        needs before another lane may load.

        The entry is left where it is.  A later reader that cannot take the
        image settles it then, one that can reads it too, and at the end of the
        section a temporary's image is dropped (`Residency.flush_all`): its
        shared buffer is then never written, and never sized.  The destination
        settles for the reason `resolve_operands` gives.

        `arrays` says whether the operation can read an image with axes at
        all; `image_in_place` has the rest.
        """
        reads = list(descr.reads())
        images = [self.image_in_place(view, arrays) for view in reads]
        for view, image in zip(reads, images):
            if image is None:
                self._settle(view)
        dest = descr.writes()
        if dest is not None:
            self._settle(dest)
        return [image or self.view_of(view)
                for view, image in zip(reads, images)]

    def image_in_place(self, subtensor, arrays: bool) -> Optional[SymbolView]:
        """The register image `subtensor` can be read from, or None.

        Only a temporary's: a writeback pending to a shared buffer, which is a
        value nothing else holds and nothing after the section reads.  A global
        result has a home other elements and later kernels see, a preload is a
        copy in whatever orientation its contraction staged it, and an atomic
        writeback is this element's share of a sum rather than its value.

        An image without axes is one value, held by every lane, so it serves
        any reader (`register_temporaries=scalars`).  One with axes (`all`)
        spreads a dimension over the lanes, and a pointwise operation lays its
        own loop over them without being asked how.  So the image serves only
        where that loop lands on it as it would on the buffer: the lane axis is
        the buffer's; the window starts on a whole round of the lanes, which is
        all `Symbol.build_address` can apply to a register image; a lane holds
        one element per round; and the lanes are threads -- under explicit SIMD
        the elementwise reads through names (`_body_named`), which carry no
        lane term.
        """
        mode = self._context.get_user_options().register_temporaries
        if mode not in REGISTER_TEMPORARIES:
            raise ValueError(
                f'register_temporaries={mode!r}: expected one of '
                f'{", ".join(REGISTER_TEMPORARIES)}')
        if mode == 'none':
            return None
        symbol = self._scopes.get_symbol(subtensor.tensor)
        entry = self._residency.get(symbol.name) if symbol is not None else None
        if (entry is None or entry.is_preload or entry.atomic
                or entry.home.stype != SymbolType.SharedMem
                or entry.image.stype != SymbolType.Register):
            return None
        # An image is allocated in the kernel's floating-point type
        # (`Temporaries.register_array`) whatever the tensor holds, so a
        # condition is a number there and a boolean again only in its buffer.
        # Read in place, `and(a, b)` over two of them would be `&` on two
        # floats, which CUDA refuses (SeisSol's damage step).
        held = getattr(entry.image, 'datatype', None)
        declared = getattr(symbol.obj, 'datatype', None)
        if held is not None and declared is not None and held != declared:
            return None
        rank = subtensor.bbox.rank()
        if rank == 0:
            return SymbolView(entry.image, subtensor.bbox)
        if mode != 'all' or not arrays or entry.covered is None:
            return None
        offset = list(subtensor.offset or [0] * rank)
        if not entry.holds(subtensor.bbox, offset):
            return None
        image = entry.image
        # The window in the image's own coordinates: position `r` holds
        # element `r + shift`, and the pointwise instructions index by the box.
        move = [o - s for o, s in zip(offset, entry.shift or [0] * rank)]
        box = BoundingBox([l + m for l, m in zip(subtensor.bbox.lower(), move)],
                          [u + m for u, m in zip(subtensor.bbox.upper(), move)])
        lead = list(image.lead_dims)
        threads = self._num_threads
        if (len(lead) != 1 or lead != list(symbol.lead_dims)
                or image.lead_axes is not None
                or getattr(image, 'linear_runs', None)
                or getattr(image, 'lead_width', 1) != 1
                or _explicit_simd(self._context)
                or (threads and box.lower()[lead[0]] % threads)):
            return None
        return SymbolView(image, box, [0] * rank)

    def _settle(self, subtensor) -> None:
        """Write a pending image of `subtensor`'s tensor to where its symbol
        says, so a read through the symbol sees the newest value."""
        symbol = self._scopes.get_symbol(subtensor.tensor)
        if symbol is not None:
            self._instructions.extend(self._residency.flush(symbol.name))

    @abstractmethod
    def alloc_destination(self, descr, operands) -> SymbolView:
        """The view the compute instruction writes its result into."""

    @abstractmethod
    def emit_compute(self, descr, operands, dest) -> None:
        """Append the instruction that does the work."""

    def record_result(self, descr, dest) -> None:
        """Say where the result now is.

        A destination materialized into registers got its writeback recorded
        when the array was allocated, because the two are one fact, and a
        destination written in place is where it belongs.  What is left is the
        store `pointwise_dest` decided on: it reads the register image's shape,
        which the compute instruction gives it, so it is built only now.
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
        not one: the result is computed into a register array, and a pending
        writeback says that array is the newest copy of the shared buffer it
        belongs in.

        Writing shared memory straight from the compute would be shorter and is
        not available.  A buffer's window is declared by its first user
        (`Generator._declare_buffers`), which has to be a memory instruction
        that knows the buffer's size; a compute instruction there would have no
        window to declare.  Going through registers gives the buffer the store
        it needs as its first user, and costs nothing that is not already paid:
        a consumer able to read the image reads it in place, and the flush
        happens only for one that cannot.

        Returns None when the destination already has a symbol and can be
        written where it is.
        """
        dest = descr.writes()
        if dest is None or self._scopes.get_symbol(dest.tensor) is not None:
            return None
        if self._plan.zero_first(dest.tensor):
            # The deferred writeback below stores the register image alone;
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
        self._residency.record_writeback(home.name, registers, home,
                                         covered=dest.bbox,
                                         shift=[0] * dest.bbox.rank())
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
          keeps its image as a pending writeback (`materialize_dest`);
        * one assembled in shared memory from several writes, or one that
          already has a buffer, is stored there at once: into the buffer as
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
                atomic=None, dest_offset=shift,
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
        and one that
        went out as a pending image holds the range its producer computed,
        which can be narrower than the descriptors declare -- a contraction
        whose operands support fewer rows than its destination names.  Laid
        over the declared union instead, the store would re-stride the buffer
        and clear rows past its end.
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
