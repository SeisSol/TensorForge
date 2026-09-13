# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What one section's descriptor list says, before any of it is built.

Three questions are asked of a whole section rather than of one operation, and
all three have to be answered before the first instruction is emitted:

* how wide a staging has to be, since the tensor a first operation stages is
  the one every later operation on that tensor inherits;
* whether a destination is assembled from several partial writes, which decides
  whether its value may stay in registers between operations;
* whether a temporary is used anywhere no earlier write defined it, which is
  zero and has its buffer cleared by the first store (`zero_first`).

None of that is specific to contraction.  It is a function of the descriptor
list and of which tensors already have a symbol, so it lives here and is handed
to whoever builds.  Every descriptor kind states its own geometry through
`OperationDescription.reads`, `writes` and `effective_boxes`, so this module
does not know what kinds there are.

Two coordinate systems appear and they are not interchangeable.  A *declared*
box is what a descriptor states about an operand or a destination; an
*effective* box is what remains after the operation's own range narrowing --
for a contraction, the intersection of everything sharing a target index.
Coverage has to be judged on effective boxes, since a declared read of the
whole tensor from an operand that supports half of it reads half.  Which of the
two a descriptor reports is the descriptor's business; for everything but a
contraction they coincide.

Everything is in tensor storage coordinates: lower bound plus slicing offset.
"""

import itertools
from typing import Optional

from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.generators.descriptions import OperationDescription


def _hull(a: Optional[BoundingBox], b: BoundingBox) -> BoundingBox:
    return b if a is None else a.unite(b)


class SectionPlan:
    """The read/write geometry of one section, keyed by tensor.

    Constructing it runs the initialisation check, so a section that reads a
    temporary somewhere nothing writes fails here rather than several hundred
    lines of generated source later.
    """

    def __init__(self, descr_list, scopes):
        #: symbol name -> union of every operand access.
        #:
        #: Keyed by symbol name and not by tensor, unlike everything else here,
        #: because the consumer -- a builder resolving an operand -- holds a
        #: `SymbolView` and has no way back to the tensor.  The two keys are
        #: not equivalent: a temporary has no symbol until the operation that
        #: first writes it creates one, which is after this runs, so keying by
        #: tensor would give temporaries an entry they do not have today and
        #: change how wide their staging comes out.
        self._operand_union = {}
        #: id(tensor) -> union of what its writes cover.  A temporary written
        #: by one operation covering everything can stay in registers until
        #: someone asks for it; one written in slices has to be assembled in
        #: memory, since each operation only ever holds its own slice.
        self._dest_union = {}
        #: id(tensor) -> union of every read, declared.
        self._read_union = {}
        #: id(tensor) -> every individual declared write box, not their union:
        #: writes to [0,2) and [8,10) union to [0,10), and a union-against-union
        #: test would wave through a read of [2,8) that nothing ever wrote.
        self._dest_boxes = {}
        #: tensor -> effective read box, and id(tensor) -> effective write
        #: boxes.  These are what coverage is judged on.
        self._eff_reads = {}
        self._eff_writes = {}
        #: id(tensor) for anything a guard reads, and for anything written
        #: under one.  Both make a register image unusable, for opposite
        #: reasons -- see `written_in_slices`.
        self._guard_reads = set()
        self._guarded_writes = set()
        #: id(tensor) -> the effective boxes its writes have defined so far,
        #: in section order; and the temporaries an operation uses where no
        #: earlier write defined them -- see `zero_first`.
        self._defined = {}
        self._zero_first = set()

        # Expanded rather than walked: a descriptor that stands for several
        # operations is asked for them, so a section's geometry is the same
        # whether the list states a repetition once or writes it out.  The
        # kinds that contain others stay their own business.
        for outer in descr_list:
            if not isinstance(outer, OperationDescription):
                continue
            for descr in outer.operations():
                self._add_reads(descr, scopes)
                tensor = self._add_writes(descr)
                self._add_effective(descr, tensor)
                self._track_definitions(descr, tensor)

        self._check_initialised()

    # -- construction ---------------------------------------------------- #

    def _add_reads(self, descr, scopes) -> None:
        for op in descr.condition_reads():
            tensor = getattr(op, 'tensor', None)
            if tensor is not None:
                self._guard_reads.add(id(tensor))
        # The guard's operands are read here too. They are not operands of the
        # operation and no builder resolves them as such, but the section has
        # to stage them all the same, and this is what decides that.
        for op in itertools.chain(descr.reads(), descr.condition_reads()):
            tensor = getattr(op, 'tensor', None)
            if tensor is None:
                continue
            box = op.storage_box()

            # Recorded before the symbol lookup: a temporary has no symbol
            # until the operation that first writes it creates one, so
            # guarding this on the lookup left every temporary with an empty
            # read union, and `written_in_slices` then saw nothing to cover.
            self._read_union[id(tensor)] = _hull(
                self._read_union.get(id(tensor)), box)

            symbol = scopes.get_symbol(tensor)
            if symbol is None:
                continue
            self._operand_union[symbol.name] = _hull(
                self._operand_union.get(symbol.name), box)

    def _add_writes(self, descr):
        dest = descr.writes()
        tensor = getattr(dest, 'tensor', None) if dest is not None else None
        if tensor is None:
            return None
        if descr.guarded():
            self._guarded_writes.add(id(tensor))
        box = dest.storage_box()
        self._dest_union[id(tensor)] = _hull(self._dest_union.get(id(tensor)),
                                             box)
        self._dest_boxes.setdefault(id(tensor), []).append(box)
        return tensor

    def _add_effective(self, descr, tensor) -> None:
        eff = descr.effective_boxes()
        if eff is None:
            return
        eff_reads, eff_write = eff
        for t, box in eff_reads.items():
            self._eff_reads[t] = _hull(self._eff_reads.get(t), box)
        if tensor is not None:
            self._eff_writes.setdefault(id(tensor), []).append(eff_write)

    def _track_definitions(self, descr, tensor) -> None:
        """Note every use of a temporary that no earlier write defined.

        In section order, which the union of all writes cannot see: an
        accumulation onto cells an earlier assignment left out adds to
        whatever the buffer held, and it counts as a write all the same.
        yateto means zero there -- an assignment defines its whole
        destination, the window with values and the rest with zeros
        (`initializeWithZero`) -- and so does a read of cells no operation
        writes.  A temporary with such a use has its buffer cleared by the
        store that first writes it (`zero_first`).

        The first write defines what it writes, accumulating or not: there is
        no earlier value for it to add to.
        """
        eff = descr.effective_boxes()
        if eff is None:
            return
        reads, write = eff
        for t, box in reads.items():
            if getattr(t, 'is_tmp', False) and self._undefined(id(t), box):
                self._zero_first.add(id(t))
        if tensor is None or not getattr(tensor, 'is_tmp', False):
            return
        key = id(tensor)
        if (key in self._defined and getattr(descr, 'add', False)
                and self._undefined(key, write)):
            self._zero_first.add(key)
        self._defined.setdefault(key, []).append(write)

    def _undefined(self, key, box) -> bool:
        defined = self._defined.get(key)
        return not defined or self._uncovered_by(defined, box) is not None

    # -- the initialisation check ---------------------------------------- #

    def _uncovered(self, key, read):
        return self._uncovered_by(self._eff_writes.get(key, []), read)

    def _uncovered_by(self, boxes, read):
        """The first sub-box of `read` that none of `boxes` covers, or None.

        Coordinate compression: cut every dimension at all the box boundaries
        that fall inside `read`.  Each resulting cell then lies either wholly
        inside or wholly outside every write box, so "is this cell covered" is
        an exact test and the whole check is exact rather than conservative.
        """
        rank = read.rank()
        if rank == 0 or not boxes or any(b.rank() != rank for b in boxes):
            return None
        cuts = []
        for j in range(rank):
            lo, hi = read.lower()[j], read.upper()[j]
            if lo >= hi:
                return None                    # empty read, nothing to cover
            pts = {lo, hi}
            for b in boxes:
                for v in (b.lower()[j], b.upper()[j]):
                    if lo < v < hi:
                        pts.add(v)
            cuts.append(sorted(pts))
        for corner in itertools.product(*[range(len(c) - 1) for c in cuts]):
            lo = [cuts[j][corner[j]] for j in range(rank)]
            hi = [cuts[j][corner[j] + 1] for j in range(rank)]
            if any(all(b.lower()[j] <= lo[j] and hi[j] <= b.upper()[j]
                       for j in range(rank)) for b in boxes):
                continue
            return BoundingBox(lo, hi)
        return None

    def _check_initialised(self):
        """Refuse to read a temporary where nothing ever wrote.

        A temporary is created by the kernel, so anything read outside what the
        kernel writes is whatever the shared or global allocation happened to
        contain.  Global inputs and outputs are exempt: an input is legitimately
        never written, and an output may hold a value the caller put there.

        Cells some operation reads but none writes are zero, and the store that
        first writes the temporary clears its buffer for them (`zero_first`);
        that store owns the buffer, since nothing touched it before.  A
        temporary no operation writes at all has no such store, and is refused.
        The buffer of a cleared temporary spans what is read as well as what is
        written, so the zeros have somewhere to be.
        """
        for tensor, read in self._eff_reads.items():
            if not getattr(tensor, 'is_tmp', False):
                continue
            key = id(tensor)
            if key not in self._eff_writes:
                raise GenerationError(
                    f'{getattr(tensor, "alias", None) or tensor}: temporary is '
                    f'read over {read} but never written')
            gap = self._uncovered(key, read)
            if gap is not None and key not in self._zero_first:
                raise GenerationError(
                    f'{getattr(tensor, "alias", None) or tensor}: temporary is '
                    f'read over {read} but {gap} is never written by any '
                    f'operation (writes: {self._eff_writes[key]}), and the '
                    f'section order did not see the read')
        for tensor, read in self._eff_reads.items():
            if id(tensor) in self._zero_first:
                self._dest_union[id(tensor)] = _hull(
                    self._dest_union.get(id(tensor)),
                    self._read_union.get(id(tensor), read))

    # -- queries --------------------------------------------------------- #

    def operand_union(self, symbol_name) -> Optional[BoundingBox]:
        """Every access to this symbol as an operand, or None if it has none.

        A staging sized to this serves every consumer in the section, which is
        what lets one be shared rather than refused.
        """
        return self._operand_union.get(symbol_name)

    def dest_union(self, tensor) -> Optional[BoundingBox]:
        """Everything this tensor's writes cover, declared."""
        return self._dest_union.get(id(tensor))

    def zero_first(self, tensor) -> bool:
        """Does the store that first writes this temporary clear its buffer?

        Where some operation reads cells, or accumulates onto cells, that no
        earlier write defined -- see `_track_definitions`.
        """
        return id(tensor) in self._zero_first

    def written_in_slices(self, tensor) -> bool:
        """Does this tensor get assembled from several writes?

        Deferring the store is right while one operation writes the whole
        thing: the value can stay in registers and be handed straight to the
        next consumer.  With several writers each operation holds only its own
        slice, so a deferred entry -- there is one per name -- would keep
        whichever came last and silently lose the rest.  Those have to go into
        the shared buffer as they are produced.

        Several writers are *not* by themselves such a case.  An accumulation
        chain -- `d = a1 b1` followed by `d += a2 b2` and so on, which is what
        a yateto flux or ADER derivative kernel looks like -- has every writer
        covering the same box, each reading what the previous one produced.
        There the last accumulator holds the whole tensor, deferring is exactly
        right, and forcing the store out per term costs a global round trip on
        every term.  So the question is not how many writers there are but
        whether any of them writes less than the union.

        Ask it of what each writer *actually* writes, not of what its
        descriptor declares.  `_analyze` intersects the range down to what the
        operands support, so an accumulation onto the whole box from an operand
        that spans half of it writes half -- the elastic ADER kernels are full
        of `t += Q_face * c`, all declaring the whole tensor and each covering
        the rows its own face touches.  Judged on the declared boxes those look
        like one writer covering everything, and the register image left behind
        holds only the last one's rows; the read that follows then wants the
        union and finds half of it.
        """
        # A guard breaks the deferral in both directions, so neither case gets
        # as far as looking at boxes.
        #
        # Read by a guard: the condition is not an operand, so no builder
        # resolves it through the residency -- the region loads the symbol.
        # A value still sitting in a register has no symbol to load.
        #
        # Written under a guard: whether the register image holds the new
        # value or the old one is decided at run time, and a deferred entry
        # records only that something wrote it.
        key = id(tensor)
        if key in self._guard_reads or key in self._guarded_writes:
            return True
        # A cleared buffer holds zeros no register image does.
        if key in self._zero_first:
            return True
        boxes = (self._eff_writes.get(id(tensor))
                 or self._dest_boxes.get(id(tensor), []))
        union = None
        for b in boxes:
            union = _hull(union, b)
        if union is not None and any(
                b.lower()[j] > union.lower()[j] or b.upper()[j] < union.upper()[j]
                for b in boxes for j in range(union.rank())):
            return True
        # One writer is still not enough if it does not cover everything that
        # gets read back: `_analyze` intersects `_ns` down to what the operands
        # support, so a single store can easily be narrower than the declared
        # destination box.
        written = self._dest_union.get(id(tensor))
        read = self._read_union.get(id(tensor))
        if written is None or read is None:
            return False
        return any(read.lower()[j] < written.lower()[j]
                   or read.upper()[j] > written.upper()[j]
                   for j in range(written.rank()))
