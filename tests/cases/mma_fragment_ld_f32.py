# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A (X S)`` in single precision (three TF32 products) on NVIDIA's matrix path, the per-element product
``X S`` a temporary stored to shared memory with the leading dimension its
B fragments read without a bank conflict (`Options.mma_fragment_ld`): 56 rows
laid out 60 apart.  A batch-constant, as SeisSol's volume operators are."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_fragment_ld_f32"
DTYPE = Datatype.F32
BATCH = 37
TOL = (1e-4, 5e-3)
OPTIONS = {"tensor_cores": True, "mma_fragment_ld": True}
M, K, N = 56, 56, 9


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE))
    x = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="X", datatype=DTYPE))
    s = SubTensor(Tensor([N, N], Addressing.STRIDED, BoundingBox([0, 0], [N, N]),
                         alias="S", datatype=DTYPE))
    t = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="T", datatype=DTYPE, is_tmp=True))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, x, s, t, alpha=1.0, beta=0.0),
            GemmDescr(False, False, a, t, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    t = np.einsum("bkn,bnj->bkj", inputs["X"], inputs["S"])
    return np.einsum("Bik,bkj->bij", inputs["A"], t)
