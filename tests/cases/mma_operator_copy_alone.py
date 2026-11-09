# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A batch-constant operator read once, on NVIDIA's matrix path, over a window narrower
than the operator: such a window is offered no order unless the caller asks for prepared
operands by name, so the reader is given a copy in that order instead, a table of the
file (`Options.mma_operator_copies`), and the operator is not passed at all.

``C[0:35] = A[0:35, 1:54] B[1:54]``, as SeisSol's derivative at order 6 reads ``kDivMT``."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_operator_copy_alone"
DTYPE = Datatype.F64
BATCH = 37
TOL = (1e-12, 1e-12)
OPTIONS = {"tensor_cores": True, "mma_operator_copies": True,
           "mma_chunked_order": True}

A_VALUES = np.random.default_rng(62).uniform(0.5, 1.5, size=(56, 56))
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def descr_list():
    a = SubTensor(Tensor([56, 56], Addressing.NONE, BoundingBox([0, 0], [56, 56]),
                         alias="A", datatype=DTYPE, data=A_VALUES.copy()),
                  BoundingBox([0, 1], [35, 54]))
    b = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0], [56, 9]),
                         alias="B", datatype=DTYPE), BoundingBox([1, 0], [54, 9]))
    c = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0], [56, 9]),
                         alias="C", datatype=DTYPE), BoundingBox([0, 0], [35, 9]))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    out[:, :35, :] = np.einsum("ik,bkj->bij", A_VALUES[:35, 1:54], inputs["B"][:, 1:54, :])
    return out
