# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = log1p(A[i, j])`` -- single-unary-op ElementwiseDescr.

Nothing else in the corpus emits `log1p`, so this case is what puts each
backend's spelling of it in front of a snapshot and the syntax check.

Domain: ``log1p`` requires ``x > -1``. ``INPUT_TRANSFORM`` maps to ``|x|``,
which keeps the argument where the function is well conditioned.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_log1p_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
INPUT_TRANSFORM = {"A": lambda x: np.abs(x)}


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.log1p(b, a)]


def reference(inputs, dest_in):
    return np.log1p(inputs["A"])
