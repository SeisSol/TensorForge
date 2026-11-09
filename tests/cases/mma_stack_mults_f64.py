# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Two multiplications sharing a warp on NVIDIA's matrix path (16 lanes each), their nine
columns stacked into one wide B (`Options.mma_stack_mults`): a tile of eight takes the ninth
column of the first and seven of the second, and the last tile's padding columns are read
from a real column and never stored.  `A` is batch-constant and read in a prepared order
(an operator copy), as the stacked path asks.

``C[0:35] = A[0:35, 1:54] B[1:54]``, as SeisSol's derivative at order 6 reads ``kDivMT``."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_stack_mults_f64"
DTYPE = Datatype.F64
BATCH = 37
TOL = (1e-12, 1e-12)
OPTIONS = {"tensor_cores": True, "mma_operator_copies": True, "mma_stack_mults": True,
           "lanes_per_mult": 16}

A_VALUES = np.random.default_rng(81).uniform(0.5, 1.5, size=(56, 56))
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
