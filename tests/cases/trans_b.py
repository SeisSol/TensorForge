# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A @ B^T``. Mirror of ``trans_a.py`` for the second-operand permute.

A is 12×20 (no transpose); B is stored as 16×20 but is transposed for
the GEMM, so the logical contraction is ``A (12×20) @ B^T (20×16)``
into C (12×16).
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_trans_b_12x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([12, 20], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 20]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 20], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 20]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([12, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 16]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(trans_a=False, trans_b=True,
                      a=a, b=b, c=c, alpha=1.0, beta=0.0)]
