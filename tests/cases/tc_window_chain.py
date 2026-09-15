# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Two products in the shape of SeisSol's time derivative, through the matrix path.

``T = A1[:, 1:18] B[1:18, :]`` and then ``C = A2[:, 1:16] T[1:16, :]``: the
second product reads the first one's result -- a temporary, in registers or in
shared memory -- from row 1 on, as `dQ(k+1) = kDivMT(k) x dQ(k)` does.  The
`tc_window_*` cases read their `B` from global memory only.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "tc_window_chain"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


def descr_list():
    a1 = SubTensor(Tensor([16, 20], Addressing.NONE, BoundingBox([0, 1], [16, 18]),
                          alias="A1", datatype=DTYPE))
    b = SubTensor(Tensor([20, 9], Addressing.STRIDED, BoundingBox([1, 0], [18, 9]),
                         alias="B", datatype=DTYPE))
    t = Tensor([16, 9], Addressing.PTR_BASED, BoundingBox([0, 0], [16, 9]),
               alias="T", is_tmp=True, datatype=DTYPE)
    a2 = SubTensor(Tensor([16, 20], Addressing.NONE, BoundingBox([0, 1], [16, 16]),
                          alias="A2", datatype=DTYPE))
    c = SubTensor(Tensor([16, 9], Addressing.STRIDED, BoundingBox([0, 0], [16, 9]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a1, b, SubTensor(t), alpha=1.0, beta=0.0),
            GemmDescr(False, False, a2, SubTensor(t, bbox=BoundingBox([1, 0], [16, 9])),
                      c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    t = np.einsum("Bik,bkj->bij", inputs["A1"], inputs["B"])
    return np.einsum("Bik,bkj->bij", inputs["A2"], t[:, 1:16, :])
