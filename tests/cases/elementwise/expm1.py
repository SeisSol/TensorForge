# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = expm1(A[i, j])`` -- single-unary-op ElementwiseDescr.

The companion of `log1p.py`: the two exist for small arguments, where
``exp(x) - 1`` and ``log(1 + x)`` cancel, and nothing else in the corpus
emits either.

Domain: ``standard_normal`` keeps the argument in ``[-5, 5]``, where
``expm1`` stays below 150 -- well within F32.  Tolerance as for ``exp``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_expm1_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.expm1(b, a)]


def reference(inputs, dest_in):
    return np.expm1(inputs["A"])
