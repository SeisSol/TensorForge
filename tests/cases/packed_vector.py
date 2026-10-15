# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A packed ``A`` read several rows to a lane at once.

`pack_operands='runs'` with `lead_vectorize`: the lead dimension is wide, so
a lane holds four adjacent rows in one vector and the packing has to give it
four adjacent *slots*.  It does -- consecutive rows are consecutive slots in
any of the forms -- but the lane bounds that hold the access off the rows a
column does not have are per lane, and cannot mask inside one lane's four.
So the runs are rounded out to whole groups of four (`Packing.granule`), and
every column's start and slot offset is a multiple of it.

None of the runs below starts or ends on a multiple of four, which is the
case worth running: rounded, each gains up to six rows, and a lane whose
four rows straddled an end would otherwise read the neighbouring column's
numbers on part of its vector.  The operand is 16-byte aligned because that
is what makes the wide lead available at all.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_vector"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "runs", "lead_vectorize": True}
M, K, N = 64, 8, 6

#: ``(first row, last row)`` per column, none of them on a group of four.
RUNS = ((1, 6), (9, 26), (31, 33), (2, 61), (45, 50), (7, 22), (37, 38),
        (13, 54))


def _values():
    rng = np.random.default_rng(23)
    a = np.zeros((M, K))
    for c, (lo, hi) in enumerate(RUNS):
        a[lo:hi + 1, c] = rng.uniform(0.5, 1.5, size=hi + 1 - lo)
    return a


A_VALUES = _values()
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE, data=A_VALUES.copy(),
                         alignment=16))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE, alignment=16))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE, alignment=16))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
