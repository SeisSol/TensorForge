# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A B`` on the stacked F64 path, over a batch that ends inside a wave.

Seven elements at sixteen lanes a multiplication: the second wave holds three,
and its fourth multiplication has no element -- it takes the group's first
(`BatchLoop`), whose numbers its lanes then hand the instruction, and stores
nothing.  Eight columns, two whole tiles, so no column is left for the DPP
chain (`Options.amd_matrix_k4`).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "stacked_ragged_f64"
DTYPE = Datatype.F64
BATCH = 7
TOL = (1e-11, 1e-11)
OPTIONS = {"lanes_per_mult": 16, "amd_matrix_k4": True}
M, K, N = 48, 24, 8


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
