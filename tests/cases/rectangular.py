# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Rectangular dense GEMM.

A 56x18 = 56x18 * 18x18 GEMM is the canonical SeisSol-shaped operator
(small leading dim, square inner).
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_56x18_x_18x18"
DTYPE = Datatype.F32
BATCH = 8
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([56, 18], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 18]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([18, 18], Addressing.STRIDED,
                         BoundingBox([0, 0], [18, 18]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([56, 18], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 18]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=a, b=b, c=c, alpha=1.0, beta=0.0)]
