# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`C = A B` in double precision on NVIDIA's matrix path with A prepared lane
by lane per sixteen bytes (`Options.mma_chunked_order`): 56 x 56 against 56 x 9,
each lane reading a tile as four 16-byte chunks, the operand staged once per block (`preload_globals`)."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_chunked_f64_preload"
DTYPE = Datatype.F64
BATCH = 37
TOL = (1e-12, 1e-12)
OPTIONS = {"tensor_cores": True, "mma_chunked_order": True, "preload_globals": True}
M, K, N = 56, 56, 9


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
