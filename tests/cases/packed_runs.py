# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A B`` with every column of ``A`` stored between its own two ends.

`pack_operands='runs'`.  The eight columns hold runs of different lengths at
different places -- one of them crosses the lane boundary at 32, one lies
wholly in the second slot, one is a single row -- so each column starts at
its own slot offset and carries its own pair of lane bounds.  That is the
difference from `packed_rows`, where one span serves every column: here the
predicate and the offset change from one product of the reduction to the
next, and a read that took the previous column's pair would address inside
the neighbouring run and read a number where the operand is zero.

85 of 320 cells, against 192 for the row span -- and the column whose run is
one row long is why: a form that keeps a rectangle cannot leave it out.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_runs"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "runs"}
M, K, N = 40, 8, 6

#: ``(first row, last row)`` per column.
RUNS = ((0, 7), (3, 20), (28, 35), (33, 39), (17, 17), (9, 31), (31, 32),
        (12, 29))


def _values():
    rng = np.random.default_rng(13)
    a = np.zeros((M, K))
    for c, (lo, hi) in enumerate(RUNS):
        a[lo:hi + 1, c] = rng.uniform(0.5, 1.5, size=hi + 1 - lo)
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
