# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A B`` with ``A`` stored without the rows its numbers never fill.

`pack_operands='rows'`.  ``A`` is 40x8 and zero outside rows 5..28, so the
box stores 320 cells for the 192 that can hold a number.  Packed, the buffer is
those 24 rows -- column ``c`` at slot ``24*c``, row ``r`` at ``r - 5`` inside
it -- and a read of a row outside the span is a lane the access is held off,
which is 8 of 32 lanes in the first slot and all of them in the second.

Both ends of the span are inside a lane slot and neither is a multiple of the
lane count: that is the case worth testing, because rounding the span out to
whole slots is what would make the predicate unnecessary and the saving zero.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_rows"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "rows"}
M, K, N = 40, 8, 6
FIRST, LAST = 5, 28


def _values():
    rng = np.random.default_rng(11)
    a = np.zeros((M, K))
    a[FIRST:LAST + 1, :] = rng.uniform(0.5, 1.5, size=(LAST + 1 - FIRST, K))
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
