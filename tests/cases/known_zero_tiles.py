# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A B`` through the matrix path, with whole tiles of ``A`` known zero.

`skip_known_zeros` on a tensor-core tile: ``A`` is 56x32, batch-constant, and
the description gives its numbers.  Rows 0..15 are zero in columns 8..15 --
one m16 x k8 tile, whose products are left out -- and rows 16..31 are zero in
columns 0..7 but for one entry, so that tile keeps its product.  Rows 48..55
are the ragged last tile.  Without tensor cores the case is the per-lane path
with no whole 32-row slot zero, which it has to leave as it was.

The kernel has to see the numbers the generator was told, so the input for
``A`` is replaced by them (`INPUT_TRANSFORM`); the reference reads the same.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "known_zero_tiles"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
M, K, N = 56, 32, 18


def _values():
    rng = np.random.default_rng(11)
    a = rng.uniform(0.5, 1.5, size=(M, K))
    a[0:16, 8:16] = 0.0
    a[16:32, 0:8] = 0.0
    a[21, 5] = 0.75
    a[48:56, 24:32] = 0.0
    return a


A_VALUES = _values()
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE, data=A_VALUES.copy()))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
