# SPDX-License-Identifier: MIT
"""Chain of two GEMMs over sliced operands.

``D[12×6] = A[12×12] @ (B[12×6] @ C[6×6])`` — each external operand
sits in a 32×32 storage block aligned to ``(0, 0)``. The intermediate
``tmp = B @ C`` is generated via :func:`generate_tmp_matrix` and gets
its bbox/shape derived from B and C (no slicing pressure there).

The case exists to verify that the offset arithmetic survives a chain:
if one GEMM in the middle clamps to wrong strides, the second GEMM's
input is silently corrupted and the comparison fails — much easier to
diagnose than a one-step slicing case where everything else might be
masking the issue.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_chain_three"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)        # two-step chain accumulates more FP error

STORAGE = (32, 32)
A_SUB = (slice(0, 12), slice(0, 12))
B_SUB = (slice(0, 12), slice(0, 6))
C_SUB = (slice(0, 6), slice(0, 6))
D_SUB = (slice(0, 12), slice(0, 6))


def descr_list():
    a = SubTensor(Tensor(list(STORAGE), Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 12]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor(list(STORAGE), Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 6]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor(list(STORAGE), Addressing.STRIDED,
                         BoundingBox([0, 0], [6, 6]),
                         alias="C", datatype=DTYPE))
    d = SubTensor(Tensor(list(STORAGE), Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 6]),
                         alias="D", datatype=DTYPE))
    tmp = SubTensor(generate_tmp_matrix(b, c))
    return [
        GemmDescr(False, False, a=b, b=c, c=tmp),
        GemmDescr(False, False, a=a, b=tmp, c=d, alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    A_sub = inputs["A"][:, *A_SUB]
    B_sub = inputs["B"][:, *B_SUB]
    C_sub = inputs["C"][:, *C_SUB]
    out[:, *D_SUB] = np.einsum("bik,bkj->bij",
                               A_sub,
                               np.einsum("bik,bkj->bij", B_sub, C_sub))
    return out
