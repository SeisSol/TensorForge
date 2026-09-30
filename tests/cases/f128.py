# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Plain dense GEMM ``C = A @ B`` in quadruple precision.

Square 2x2, alpha=1, beta=0, no transposes, STRIDED addressing: the plainest
kernel that carries `__float128` through generation and emission, whose cache
hints `test_nontemporal.py` checks per vendor.
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_square_16_f128"
DTYPE = Datatype.F128
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([2, 2], Addressing.STRIDED,
                         BoundingBox([0, 0], [2, 2]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([2, 2], Addressing.STRIDED,
                         BoundingBox([0, 0], [2, 2]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([2, 2], Addressing.STRIDED,
                         BoundingBox([0, 0], [2, 2]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=a, b=b, c=c, alpha=1.0, beta=0.0)]
