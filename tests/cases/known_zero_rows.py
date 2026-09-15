# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A B`` with a batch-constant ``A`` whose numbers the description gives.

`skip_known_zeros`: the contraction runs 40 rows over the lanes, the first 32
in one slot, and ``A`` is zero in those 32 rows at columns 0, 2 and 5 -- so
the first slot leaves those three steps out and the second, where rows 32..39
hold values everywhere, keeps them.  Column 3 is zero in every row but one of
the first slot, which has to keep the step.

The kernel has to see the numbers the generator was told, so the input for
``A`` is replaced by them (`INPUT_TRANSFORM`); the reference reads the same.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "known_zero_rows"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
M, K, N = 40, 8, 6


def _values():
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 1.5, size=(M, K))
    a[:32, [0, 2, 5]] = 0.0
    a[:32, 3] = 0.0
    a[17, 3] = 1.25
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
