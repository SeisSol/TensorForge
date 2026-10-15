# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A packed ``A`` read from a reduction that is a real loop.

`pack_operands='rows'` with `k_roll`: the contraction is a counted loop, so
the column arrives as a loop variable and there is no per-column number to
fold into the address.  A uniform packing needs none.  Every column starts
at the same row and runs the same length, so the slot is ``span * k + r -
start`` -- affine in the column -- and the lane bounds that hold the access
off the dropped rows do not mention the column at all.

That is the whole of what the rolled case needs; `runs` with unequal columns
and `bits` are refused here, because for them the slot is a per-column
number and a loop variable has nowhere to read it from.

96 columns so the reduction is long enough to be worth rolling, and rows
7..45 of 64 so the span is neither the whole box nor aligned to anything.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_rolled"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "rows", "k_roll": 4}
M, K, N = 64, 96, 6
FIRST, LAST = 7, 45


def _values():
    rng = np.random.default_rng(53)
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
