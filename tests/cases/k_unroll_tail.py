# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A rolled reduction whose last steps are written out behind the loop.

35 steps, the extent of SeisSol's order-5 reductions, under a cap of 16: no
multiple of the four floats a trip reads at once divides them, so without a
tail the loop runs by 16 and the compiler's remainder loop takes the last
three.  With `k_unroll_max_tail` at 3 the loop runs two whole trips over 32
steps and the last three are written out -- every one of them still a term
of the sum, which is what the comparison holds.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "k_unroll_tail"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"k_unroll_max": 16, "k_unroll_max_tail": 3}
M, K, N = 40, 35, 9


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
