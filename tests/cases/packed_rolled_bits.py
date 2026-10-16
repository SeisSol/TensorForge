# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A bitset-packed ``A`` read from a reduction that rolls.

The other half of `packed_rolled_runs`.  A bitset is never uniform -- its
slot is a population count and not the row plus a constant -- so before
`Options.pack_tables` it was unreachable from a rolled reduction whatever
its columns looked like, and `fit` fell all the way back to `rows`.

Two numbers a column here rather than three: the offset and the word.  The
run bounds say nothing the word does not, because a row inside a column's
span may still be a hole, so the bit itself carries the whole predicate.

32 rows, so that the rows one lane slot reaches fit the 32 bits a word has
-- the condition `_bitset_fits` states and `_tabled_packed_index` refuses
on.  Holes inside every column as well as different ends, so a slot read
with a neighbouring column's word lands on a number.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_rolled_bits"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "bits", "k_roll": 4}
M, K, N = 32, 48, 6


def _values():
    rng = np.random.default_rng(31)
    a = np.zeros((M, K))
    for k in range(K):
        lo = 1 + (5 * k) % 9
        hi = M - (3 * k) % 7
        for r in range(lo, hi):
            # A hole every third row, offset per column: the span is not the
            # set of rows, which is the whole difference from `runs`.
            if (r + k) % 3:
                a[r, k] = rng.uniform(0.5, 1.5)
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
