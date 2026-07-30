# SPDX-License-Identifier: MIT
"""``B[i, j] = sin(A[i, j])`` — single-unary-op ElementwiseDescr.

Domain: ``sin`` is bounded and signed; ``standard_normal`` is fine.
Tolerance can stay tight because ``sin`` rounds well in F32.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_sin_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.sin(b, a)]


def reference(inputs, dest_in):
    return np.sin(inputs["A"])
