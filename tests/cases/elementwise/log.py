# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = log(A[i, j])`` — single-unary-op ElementwiseDescr.

Domain: ``log`` requires positive input. ``INPUT_TRANSFORM`` maps to
``|x| + 0.1`` to stay safely off the singularity at zero. Even at
0.1 the log is well-behaved (``log(0.1) ≈ -2.3``).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_log_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
INPUT_TRANSFORM = {"A": lambda x: np.abs(x) + 0.1}


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.log(b, a)]


def reference(inputs, dest_in):
    return np.log(inputs["A"])
