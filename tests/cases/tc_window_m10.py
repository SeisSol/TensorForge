# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A[:10, 1:18] B[1:18, :]`` -- the first product of SeisSol's order-4 time
derivative (`dQ(1) = kDivMT(0) x t0`): ten rows of a sixteen-row tile, a
contraction window from depth 1, through the matrix path."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "tc_window_m10"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


def descr_list():
    a = SubTensor(Tensor([16, 20], Addressing.NONE, BoundingBox([0, 1], [10, 18]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([20, 9], Addressing.STRIDED, BoundingBox([1, 0], [18, 9]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([10, 9], Addressing.STRIDED, BoundingBox([0, 0], [10, 9]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
