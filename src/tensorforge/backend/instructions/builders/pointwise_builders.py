# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Builders for the two operations that iterate exactly what they declare.

Both are short, and the reason they are short is the point of the base class:
resolving operands and recording the result are the same for them as for a
contraction, so all they have to say is what their destination looks like and
which instruction to emit.

Where they differ from a contraction is a temporary still sitting in the
register image its producer computed into.  A contraction takes the image
where it is, shifted and with the lane on whichever axis it likes.  These lay
their own loop over the lanes without being asked how, so they take an image
only where that loop lands on it the way it would on the buffer -- always, for
a value without axes, which every lane holds -- and settle the rest back to
memory first (`OperationBuilder.image_in_place`, `register_temporaries`).
"""

from typing import List

from tensorforge.backend.instructions.builders.operation_builder import (
    OperationBuilder)
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.symbol import SymbolView
from tensorforge.backend.instructions.compute.elementwise import (
    ElementwiseInstruction, ScalarLike)
from tensorforge.backend.instructions.compute.reduction import (
    ReductionInstruction)
from tensorforge.backend.instructions.compute.scalar import (
    ScalarContractionInstruction)
from tensorforge.generators.descriptions import MultilinearDescr


class ElementwiseBuilder(OperationBuilder):
    def resolve_operands(self, descr) -> List:
        # Scalars are values, not tensors: they settle nothing and are handed
        # to the instruction as they are.
        settled = iter(self.resolve_in_place(descr, arrays=True))
        return [s if isinstance(s, ScalarLike) else next(settled)
                for s in descr.srcs]

    def alloc_destination(self, descr, operands):
        # Iteration axis `i` is axis `i` of every operand here, so the
        # destination spreads whichever axis its sources do.
        lead_pos = ComputeInstruction.shared_lead_dim(
            ElementwiseInstruction,
            [v for v in operands if not isinstance(v, ScalarLike)],
            'elementwise')
        return self.materialize_dest(descr, lead_pos) \
            or self.view_of(descr.dest)

    def emit_compute(self, descr, operands, dest) -> None:
        self._instructions.append(ElementwiseInstruction(
            self._context, descr.op, dest, operands,
            descr.prefer_align, self._num_threads))


class ScalarBuilder(OperationBuilder):
    """A contraction whose destination has no axes.

    A multilinear by its descriptor, and one value by what it computes: see
    `ScalarContractionInstruction`.  Claimed ahead of `MultilinearBuilder`,
    which distributes an axis this destination does not have.
    """

    @staticmethod
    def accepts(descr) -> bool:
        return (isinstance(descr, MultilinearDescr)
                and descr.dest.bbox.rank() == 0)

    def build(self, descr) -> None:
        """A long sum over one axis is a product and a fold, not a loop.

        `alpha = alphaNodal[l] * quadratureWeights[l]` over 125 elements is
        one value, and every lane computed all of it: 125 reads of a shared
        buffer per lane, where the tensor is spread over the lanes to begin
        with.  Written as a pointwise product into a register image and a
        reduction over it, each lane multiplies the four elements it owns and
        the lanes fold their partials (`CrossLaneFold`) -- and with
        `register_temporaries=all` the image never reaches memory at all.

        Both halves are instructions that already exist, which is the point:
        the fold is the reduction's, not a second one written here.  SeisSol's
        damage step has ten such contractions and they are 11 % of its shared
        traffic.
        """
        box = self._foldable(descr)
        if box is None:
            super().build(descr)
            return
        self._reset()
        operands = self.resolve_in_place(descr, arrays=True)
        self._instructions.extend(self._product_and_fold(descr, operands, box))

    def _foldable(self, descr):
        """`(lo, hi)` of the contracted axis where folding is the better shape.

        Narrow on purpose.  One contracted axis, every operand a tensor spread
        over it, nothing accumulated, and an extent past the lane count -- at
        13 elements over 32 lanes the fold would exchange more than the loop
        reads.  `target` says which of an operand's axes the destination has,
        and for these all of them are contracted (`-1`).
        """
        if descr.add or len(descr.ops) != 2:
            return None
        lo, hi = None, None
        for view, axes in zip(descr.ops, descr.target):
            if (list(axes) != [-1] or view.bbox.rank() != 1
                    or any(view.offset or [0])):
                return None
            low, high = view.bbox.lower()[0], view.bbox.upper()[0]
            lo = low if lo is None else max(lo, low)
            hi = high if hi is None else min(hi, high)
        if hi - lo <= self._num_threads:
            return None
        return lo, hi

    def _product_and_fold(self, descr, operands, box) -> List:
        from tensorforge.common.matrix.boundingbox import BoundingBox
        from tensorforge.common.operation import AddOperator, Operation

        lo, hi = box
        extent = BoundingBox([0], [hi - lo])
        registers, alloc = self._temporaries.register_array(extent, 0)
        product = SymbolView(registers, extent, [0])
        # Each operand over the part all of them cover, read at `i + lo`:
        # `ElementwiseInstruction` indexes a source by the loop value plus its
        # box's lower corner, and applies no offset of its own -- which is why
        # `_foldable` refuses a view that carries one.
        srcs = [SymbolView(view.symbol, BoundingBox([lo], [hi]), [0])
                for view in operands]
        dest = self.materialize_dest(descr, ()) or self.view_of(descr.dest)
        return [alloc,
                ElementwiseInstruction(self._context, Operation.MUL, product,
                                       srcs, False, self._num_threads),
                ReductionInstruction(self._context, dest, product, [0],
                                     AddOperator(), False, self._num_threads)]

    def resolve_operands(self, descr) -> List:
        # Values without axes only.  Every lane runs the whole sum, so an
        # image with axes would be read one fixed element at a time -- an
        # exchange with the owning lane per read, where the buffer is a load.
        return self.resolve_in_place(descr, arrays=False)

    def alloc_destination(self, descr, operands):
        return self.materialize_dest(descr, ()) or self.view_of(descr.dest)

    def emit_compute(self, descr, operands, dest) -> None:
        self._instructions.append(ScalarContractionInstruction(
            self._context, dest, operands, descr.target, descr.add,
            self._num_threads))


class ReductionBuilder(OperationBuilder):
    def resolve_operands(self, descr) -> List:
        return self.resolve_in_place(descr, arrays=True)

    def alloc_destination(self, descr, operands):
        """The destination keeps the axes the reduction does not contract.

        Dropping the contracted axes renumbers what is left, so the source's
        lane axis sits at `kept.index(src_lead)` in the destination.  Getting
        it wrong means the lane that computes a value and the lane that stores
        it are different lanes, which is wrong in a way no shape check would
        catch; `ReductionInstruction` checks this one specifically.
        """
        var, = operands
        kept = [d for d in range(var.bbox.rank()) if d not in set(descr.dims)]
        src_lead = ComputeInstruction.lead_dim(var)
        lead_pos = kept.index(src_lead) if src_lead in kept else 0
        return self.materialize_dest(descr, lead_pos) \
            or self.view_of(descr.dest)

    def emit_compute(self, descr, operands, dest) -> None:
        var, = operands
        self._instructions.append(ReductionInstruction(
            self._context, dest, var, descr.dims, descr.op,
            descr.prefer_align, self._num_threads))
