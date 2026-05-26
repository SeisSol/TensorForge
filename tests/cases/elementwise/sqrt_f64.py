# SPDX-License-Identifier: MIT
"""F64 variant of ``elementwise/sqrt.py``.

The CUDA lexic dispatches ``sqrt`` → ``sqrtf`` for F32 and ``sqrt``
for F64 (no ``f`` suffix). A regression that emits ``sqrtf`` for both
dtypes silently narrows to F32 and ruins F64 accuracy; the looser-
than-F32-but-stricter-than-F32-allows tolerance below catches it.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import optree
from tensorforge.generators.descriptions import ElementwiseDescr

from harness.optree_helpers import make_tvar

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
    return [ElementwiseDescr(
        [optree.Assignment(make_tvar(b, 2), optree.sqrt(make_tvar(a, 2)))])]


def reference(inputs, dest_in):
    return np.sqrt(inputs["A"])
