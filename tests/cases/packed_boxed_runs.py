# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A packed operand whose bounding box does not start at row zero.

Two defects met here, and neither needed the other.

**The lead start.**  A window of 32 rows beginning at row 9 covers two
slots of 32 lanes, nine rows of the first and nine of the second.  The
vendor paths take the slot count from the *extent* -- `ceil(32 / 32)` is
one -- and `unwindI` puts the window at slot `start // span`, which rounds
the 9 away a second time.  So the path emitted one slot where the
destination has two: on gfx1150 the first nine rows were read from before
the buffer and the last nine were never computed, and the store wrote them
from registers nothing had assigned (`max_rel_err` 1.0).  The condition
that refuses such a start existed and was gated on `lead_width > 1`, where
it is a statement about the start and not about the width.  Nothing in the
corpus or in the recorded SeisSol kernels reaches it, and it is
independent of packing -- `pack_operands=off` failed the same way.

**The lower corner.**  `Symbol._address` subtracts the view's lower corner
from every axis, so a lead index that hands its address over outright
(`LeadIndex._vaddr`) has to leave room for that.  `_uniform_packed_index`
does, `_bitset_index` did not, and the two cannot both be right.  A table
has to do both at once: the run bounds are compared against the *box* row,
which carries the corner, while the address must not.  All 207940 packed
reads of the recorded SeisSol kernels at orders 6 and 8 start their box at
row zero, where the three forms agree because there is nothing to disagree
about.

So the box is shifted, by 9: neither zero nor a multiple of the lane
count, which is what `_bitset_fits` asks of a bitset and the run forms
deliberately do not.  96 columns with a run apiece, so the reduction rolls
and the columns disagree -- which is what puts the numbers in a table
(`Options.pack_tables`) on the targets that roll.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "packed_boxed_runs"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"pack_operands": "runs", "k_roll": 4}
M, K, N = 56, 96, 5
LO, HI = 9, 41


def _values():
    rng = np.random.default_rng(23)
    a = np.zeros((M, K))
    for k in range(K):
        lo = LO + (3 * k) % 7
        a[lo:lo + 9 + (5 * k) % 13, k] = rng.uniform(
            0.5, 1.5, size=9 + (5 * k) % 13)
    return a


A_VALUES = _values()
#: The host buffer is the stored block, which for a box starting at row
#: `LO` is the box and not the tensor.
A_BOX = A_VALUES[LO:HI, :]
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_BOX, x.shape)}


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE,
                         BoundingBox([LO, 0], [HI, K]),
                         alias="A", datatype=DTYPE, data=A_VALUES.copy()))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED,
                         BoundingBox([LO, 0], [HI, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    # both host buffers are already the stored blocks
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
