# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`k_unroll_tail` in double precision: a trip of two doubles is the vector,
so every even count up to the cap of 16 is a candidate, none divides 35, and
16 leaves the last three steps -- written out at a tail of 3."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "k_unroll_tail_f64"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)
OPTIONS = {"k_unroll_max": 16, "k_unroll_max_tail": 3}
M, K, N = 40, 35, 9


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
