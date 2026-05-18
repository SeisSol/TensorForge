# SPDX-License-Identifier: MIT
"""``C = A_sub @ B`` — only A is sliced, B and C use bbox == shape.

A is stored as 32×16 but the GEMM only reads the 12×16 block from row 4
to row 16 (lead-dim slicing). B is a plain 16×8 (no slicing), C is
plain 12×8 (no slicing).

The case is here to make sure mixed slicing works — the generator must
emit different address arithmetic for the sliced operand than for the
unsliced ones, which is a different code path from "everything sliced
the same way".
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_offset_a"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

# A: 32x16 storage, bbox 12x16 starting at row 4.
A_STORAGE = (32, 16)
A_LO, A_HI = (4, 0), (16, 16)
A_SUB = (slice(4, 16), slice(0, 16))


def descr_list():
    a = SubTensor(Tensor(list(A_STORAGE), Addressing.STRIDED,
                         BoundingBox(list(A_LO), list(A_HI)),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([12, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 8]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    A_sub = inputs["A"][:, *A_SUB]
    return np.einsum("bik,bkj->bij", A_sub, inputs["B"])
