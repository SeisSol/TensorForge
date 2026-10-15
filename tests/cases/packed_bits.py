# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A B`` with only ``A``'s non-zeros stored, addressed by population count.

`pack_operands='bits'`.  Every column keeps its own bitset of the rows it
holds and the running count of the slots before it, so the slot of row ``r``
is ``offset_c + popcount(word_c & ((1 << r) - 1))`` and the lane holds an
element at all only where ``(word_c >> r) & 1``.  That last part is what no
pair of lane bounds can express, and it is the whole difference from
`packed_runs`: here a row *inside* a column's span can be absent, and the
columns below are deliberately built with holes in the middle so that a form
keeping runs would have to store them.

24 rows so the highest row any lane slot reaches fits in the 32-bit word the
shift is counted in; past that the offer declines and the operand stays
whole (`MultilinearInstruction._bitset_fits`).

Staged into shared memory as well (`preload_globals`), which is the other
half of the seam: the image is the storage copied verbatim, so the block
copies the packed slots and the body addresses them there exactly as it
would in global memory.  `packed_rows` and `packed_runs` read from global,
so between the three both sides are covered.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_bits"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "bits", "preload_globals": True}
M, K, N = 24, 7, 6

#: The rows each column holds.  Holes in the middle, a column of one row, a
#: column that is dense and one that is empty but for its two ends.
ROWS = (
    (0, 1, 2, 9, 10, 23),
    (4,),
    tuple(range(24)),
    (0, 23),
    (3, 5, 7, 11, 13, 17, 19),
    (12, 13, 14, 15, 16),
    (1, 22),
)


def _values():
    rng = np.random.default_rng(17)
    a = np.zeros((M, K))
    for c, rows in enumerate(ROWS):
        for r in rows:
            a[r, c] = rng.uniform(0.5, 1.5)
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
