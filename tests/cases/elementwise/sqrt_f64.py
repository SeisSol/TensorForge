# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""F64 variant of ``elementwise/sqrt.py``.

The square root is one name for both types (``std::sqrt``,
``sycl::sqrt``), and the overload decides.  A regression that narrows the
F64 one to float -- ``sqrtf``, a ``float`` literal, a cast -- ruins F64
accuracy silently; the tolerance below, which F32 cannot meet, catches it.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_sqrt_16x16_f64"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)

INPUT_TRANSFORM = {"A": lambda x: np.abs(x) + 0.1}


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.sqrt(b, a)]


def reference(inputs, dest_in):
    return np.sqrt(inputs["A"])
