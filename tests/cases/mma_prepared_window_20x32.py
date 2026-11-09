# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C[0:35] = A[0:20, 1:33] B[1:33]`` in double precision on NVIDIA's matrix path,
A batch-constant and prepared: a window of the operator whose contraction starts inside
it, as each order of SeisSol's derivative reads ``kDivMT`` (rows 0 to 20, columns 1 to 33 at order 6, two k tiles of the operand's four; rows 0 to 35, columns 1 to 54
at order 6)."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_prepared_window_20x32"
DTYPE = Datatype.F64
BATCH = 37
TOL = (1e-12, 1e-12)
OPTIONS = {"tensor_cores": True, "mma_chunked_order": True}


def descr_list():
    a = SubTensor(Tensor([56, 56], Addressing.NONE, BoundingBox([0, 0], [56, 56]),
                         alias="A", datatype=DTYPE), BoundingBox([0, 1], [20, 33]))
    b = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0], [56, 9]),
                         alias="B", datatype=DTYPE), BoundingBox([1, 0], [33, 9]))
    c = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0], [56, 9]),
                         alias="C", datatype=DTYPE), BoundingBox([0, 0], [20, 9]))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    out[:, :20, :] = np.einsum("Bik,bkj->bij", inputs["A"][:, :20, 1:33],
                                   inputs["B"][:, 1:33, :])
    return out
