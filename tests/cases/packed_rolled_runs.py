# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A packed ``A`` whose columns disagree, read from a reduction that rolls.

What `packed_rolled` does not cover.  There the packing is uniform -- every
column holds the same rows -- so the slot is ``span * k + r - start`` and a
loop variable is all the address needs.  Here each column starts and ends
somewhere else, so the slot base and the run bounds are a set of numbers per
column and the one the read wants is picked by the loop.

`Options.pack_tables` is what makes that addressable: the numbers go into a
kernel parameter (`KernelParam.numbers`), which lives where kernel arguments
live -- one copy for the grid, read-only, and indexed by a value every lane
agrees on.  The address is ``offset + r - start`` with two of its terms read
from there, and the lane bounds that held the access off the dropped rows
become a comparison against two more (`Symbol._tabled_packed_index`).

Without it the offer falls back to the loosest form that needs no numbers,
which is `rows`, and the kernel is correct and larger.  Both are built: the
option is off in `OPTIONS_OFF`, so the case pins the fallback as well.

Columns that are each a different run, and none of them aligned to the lanes
or to each other -- a staircase with a varying length, so that a slot read
with another column's numbers lands on a number rather than on padding and
the test cannot pass by accident.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_rolled_runs"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "runs", "k_roll": 4}
M, K, N = 64, 96, 6


def _span(k: int):
    """Column `k`'s run: a staircase whose length varies with it too."""
    first = 3 + (7 * k) % 29
    return first, first + 11 + (5 * k) % 17


def _values():
    rng = np.random.default_rng(97)
    a = np.zeros((M, K))
    for k in range(K):
        lo, hi = _span(k)
        a[lo:hi, k] = rng.uniform(0.5, 1.5, size=hi - lo)
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
