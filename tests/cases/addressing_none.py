# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C[b] = A @ B[b]`` — A is batch-constant (:data:`Addressing.NONE`).

SeisSol's static differentiation operators are the textbook example:
the same 56×56 matrix multiplied against every element's solution
vector. With :data:`Addressing.NONE` the kernel skips the
``batchId * volume`` term in ``A``'s pointer (see ``ptr_manip.py:67-71``)
and reads from one shared storage block; the host allocates a single
``volume(shape) * sizeof(T)`` block rather than ``batch *
volume(shape) * sizeof(T)``.

The harness already handles this: ``op_batch = 1 if op.addressing ==
"none"`` (see ``runner.py:136``) — the per-element view has a leading
``1`` axis instead of ``batch``, and the reference broadcasts it
naturally against the per-batch ``B``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_none_operator"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    # A: batch-constant operator matrix.
    a = SubTensor(Tensor([16, 16], Addressing.NONE,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    # B, C: per-batch.
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    # inputs["A"] has shape (1, 16, 16) thanks to op_batch=1 handling;
    # einsum's broadcasting expands it across B's batch axis.
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
