# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Builders for the two operations that iterate exactly what they declare.

Both are short, and the reason they are short is the point of the base class:
resolving operands and recording the result are the same for them as for a
contraction, so all they have to say is what their destination looks like and
which instruction to emit.

What they do not yet do is consult the residency the way a contraction can.
A contraction reading a value another operation left in registers takes the
image where it is; these two settle it back to memory first, because
`ElementwiseInstruction` and `ReductionInstruction` address their operands
through the symbol the descriptor names and have no way to be told "that one,
but shifted, and with the lane on a different axis".  The register image is
already good enough to serve them -- `ew -> ml` proves it, since the
contraction there does exactly that -- so this is a matter of teaching the two
instructions to accept a staged view, not of anything missing underneath.
"""

from typing import List

from tensorforge.backend.instructions.builders.operation_builder import (
    OperationBuilder)
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.compute.elementwise import (
    ElementwiseInstruction, ScalarLike)
from tensorforge.backend.instructions.compute.reduction import (
    ReductionInstruction)


class ElementwiseBuilder(OperationBuilder):
    def resolve_operands(self, descr) -> List:
        # Scalars are values, not tensors: they settle nothing and are handed
        # to the instruction as they are.
        settled = iter(super().resolve_operands(descr))
        return [s if isinstance(s, ScalarLike) else next(settled)
                for s in descr.srcs]

    def alloc_destination(self, descr, operands):
        # Iteration axis `i` is axis `i` of every operand here, so the
        # destination spreads whichever axis its sources do.
        lead_pos = ComputeInstruction.shared_lead_dim(
            ElementwiseInstruction,
            [v for v in operands if not isinstance(v, ScalarLike)],
            'elementwise')
        return self.materialise_dest(descr, lead_pos) \
            or self.view_of(descr.dest)

    def emit_compute(self, descr, operands, dest) -> None:
        self._instructions.append(ElementwiseInstruction(
            self._context, descr.op, dest, operands,
            descr.prefer_align, self._num_threads))


class ReductionBuilder(OperationBuilder):
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
        return self.materialise_dest(descr, lead_pos) \
            or self.view_of(descr.dest)

    def emit_compute(self, descr, operands, dest) -> None:
        var, = operands
        self._instructions.append(ReductionInstruction(
            self._context, dest, var, descr.dims, descr.op,
            descr.prefer_align, self._num_threads))
