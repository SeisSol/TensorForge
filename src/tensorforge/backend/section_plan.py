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
* whether a temporary is read anywhere the section never writes, which is an
  undefined summand and is refused rather than emitted.

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

        for descr in descr_list:
            if not isinstance(descr, OperationDescription):
                continue
            self._add_reads(descr, scopes)
            tensor = self._add_writes(descr)
            self._add_effective(descr, tensor)

        self._check_initialised()

    # -- construction ---------------------------------------------------- #

    def _add_reads(self, descr, scopes) -> None:
        for op in descr.reads():
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

    # -- the initialisation check ---------------------------------------- #

    def _uncovered(self, key, read):
        """The first sub-box of `read` that no write covers, or None.

        Coordinate compression: cut every dimension at all the box boundaries
        that fall inside `read`.  Each resulting cell then lies either wholly
        inside or wholly outside every write box, so "is this cell covered" is
        an exact test and the whole check is exact rather than conservative.
        """
        boxes = self._eff_writes.get(key, [])
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

        Filling the gap with zeros is the obvious other answer, and the right
        one once a declaration instruction owns the buffer.  Until then this
        refuses, because a silently undefined summand is exactly the failure
        mode that took the longest to find in this area.
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
            if gap is not None:
                raise GenerationError(
                    f'{getattr(tensor, "alias", None) or tensor}: temporary is '
                    f'read over {read} but {gap} is never written by any '
                    f'operation (writes: {self._eff_writes[key]}). '
                    f'Zero-filling the gap needs a declaration instruction '
                    f'that owns the buffer.')

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
