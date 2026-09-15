# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A1[:10, 1:18] B1[1:18, :]; C += A2[:10, 1:19] B2[1:19, :]`` -- the
accumulation of SeisSol's order-4 time derivative (`dQ(1) = kDivMT(0) x t0;
dQ(1) += kDivMT(1) x t1`) through the matrix path: ten rows of a sixteen-row
tile, windows from depth 1 of different depths, the second product added."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "tc_window_accum"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


def descr_list():
    a1 = SubTensor(Tensor([16, 20], Addressing.NONE, BoundingBox([0, 1], [10, 18]),
                          alias="A1", datatype=DTYPE))
    b1 = SubTensor(Tensor([20, 9], Addressing.STRIDED, BoundingBox([1, 0], [18, 9]),
                          alias="B1", datatype=DTYPE))
    a2 = SubTensor(Tensor([16, 20], Addressing.NONE, BoundingBox([0, 1], [10, 19]),
                          alias="A2", datatype=DTYPE))
    b2 = SubTensor(Tensor([20, 9], Addressing.STRIDED, BoundingBox([1, 0], [19, 9]),
                          alias="B2", datatype=DTYPE))
    c = Tensor([10, 9], Addressing.STRIDED, BoundingBox([0, 0], [10, 9]),
               alias="C", datatype=DTYPE)
    return [GemmDescr(False, False, a1, b1, SubTensor(c), alpha=1.0, beta=0.0),
            GemmDescr(False, False, a2, b2, SubTensor(c), alpha=1.0, beta=1.0)]


def reference(inputs, dest_in):
    return (np.einsum("Bik,bkj->bij", inputs["A1"], inputs["B1"])
            + np.einsum("Bik,bkj->bij", inputs["A2"], inputs["B2"]))
