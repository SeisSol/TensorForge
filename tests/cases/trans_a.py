# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A^T @ B``. Exercises the permute path in the auto-reference."""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_trans_a_20x12"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    # A is 20x12 in storage; logical contraction is A^T (12x20) @ B (20x16).
    # Result C is 12x16.
    a = SubTensor(Tensor([20, 12], Addressing.STRIDED,
                         BoundingBox([0, 0], [20, 12]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([20, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [20, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([12, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 16]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(trans_a=True, trans_b=False,
                      a=a, b=b, c=c, alpha=1.0, beta=0.0)]
