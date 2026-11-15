# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A product whose contraction starts at step one, on the stacked F64 path.

``C[0:40] = A[0:40, 1:30] (X[1:30] S)``: the operator's window starts a
column into its box, as SeisSol's time derivative reads ``kDivMT(k)[:, 1:]``
against ``dQ(k)[1:, :]``.  The per-element operand is then read from the block
holding step one, a position before it, so the tiles of four steps the
instruction issues straddle the contraction's start: the first tile's first
position is no step of it, and the last tile ends inside.  Both have to read
zeros there (`Options.amd_matrix_k4`), at sixteen lanes a multiplication.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "stacked_from_step_one_f64"
OUTPUT = "C"
DTYPE = Datatype.F64
BATCH = 23
TOL = (1e-11, 1e-11)
OPTIONS = {"lanes_per_mult": 16, "amd_matrix_k4": True}

_M, _K, _N = 40, 30, 9


def _values():
    rng = np.random.default_rng(17)
    a = rng.uniform(-1.0, 1.0, (_M, _K))
    a[:, 0] = 0.0
    a[0:16, 20:30] = 0.0
    return a


A_VALUES = _values()
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def descr_list():
    a = Tensor([_M, _K], Addressing.NONE, BoundingBox([0, 0], [_M, _K]),
               alias="A", datatype=DTYPE, data=A_VALUES.copy())
    x = Tensor([_K, _N], Addressing.STRIDED, BoundingBox([0, 0], [_K, _N]),
               alias="X", datatype=DTYPE)
    s = Tensor([_N, _N], Addressing.STRIDED, BoundingBox([0, 0], [_N, _N]),
               alias="S", datatype=DTYPE)
    t = Tensor([_K, _N], Addressing.STRIDED, BoundingBox([0, 0], [_K, _N]),
               alias="T", datatype=DTYPE, is_tmp=True)
    c = Tensor([_M, _N], Addressing.STRIDED, BoundingBox([0, 0], [_M, _N]),
               alias="C", datatype=DTYPE)
    return [
        MultilinearDescr(dest=SubTensor(t), ops=[SubTensor(x), SubTensor(s)],
                         target=[[0, -1], [-1, 1]],
                         permute=[[0, 1], [0, 1]]),
        MultilinearDescr(dest=SubTensor(c),
                         ops=[SubTensor(a, BoundingBox([0, 1], [_M, _K])),
                              SubTensor(t, BoundingBox([1, 0], [_K, _N]))],
                         target=[[0, -1], [-1, 1]],
                         permute=[[0, 1], [0, 1]]),
    ]


def reference(inputs, dest_in):
    t = np.einsum("bik,bkj->bij", inputs["X"], inputs["S"])
    return np.einsum("ik,bkj->bij", A_VALUES[:, 1:], t[:, 1:, :])
