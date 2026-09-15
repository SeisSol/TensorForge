# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A[:, 0:17] B[0:17, :]`` -- the contraction window of SeisSol's time
derivative (`kDivMT(k) x dQ(k)`), batch-constant A, through the matrix path.

The window starts at depth 0 and is 17 deep: a ragged k-tile, and no lane
of the first block before the window.  The control for `tc_window_k1_16` and
`tc_window_k1_17`, which start at depth 1.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "tc_window_k0_17"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)
M, KS, N = 16, 20, 9
LO, HI = 0, 17


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
