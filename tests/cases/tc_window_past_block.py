# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A[:, 9:13] B[9:13, :]`` over eight rows -- eight lanes, and a
contraction window that starts in the second block of them.

The time derivative's one-lane product (`test_regressions`) with a block
wider than one: the matrix paths read `B` from the block holding the
window's first step, slot `9 // 8`, and pair its lane `p` with the lead
operand's step `p - kx`.  With `kx` the step itself, 9, rather than its
place in the block, 1, the step was counted from the next block on and every
product read a row eight further on, past the window.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "tc_window_past_block"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
M, KS, N = 8, 16, 9
LO, HI = 9, 13


def descr_list():
    a = SubTensor(Tensor([M, KS], Addressing.NONE, BoundingBox([0, LO], [M, HI]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([KS, N], Addressing.STRIDED, BoundingBox([LO, 0], [HI, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    # both host buffers are the stored windows
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
