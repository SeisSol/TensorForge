# SPDX-License-Identifier: MIT
"""``B[i, j] = exp(A[i, j])`` — single-unary-op ElementwiseDescr.

Domain: ``standard_normal`` puts most mass in ``[-3, 3]``, where
``exp`` ranges over ``[e^-3, e^3] ≈ [0.05, 20]`` — well within F32.
Tail values up to ~5 produce ~150, still fine. No transform needed.

Tolerance: ``exp`` is moderately sensitive to argument noise (1 ulp in
the input ≈ ``exp(x)`` ulps in the output), so we loosen slightly.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import optree
from tensorforge.generators.descriptions import ElementwiseDescr

from harness.optree_helpers import make_tvar

NAME = "elementwise_exp_16x16"
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
    return [ElementwiseDescr(
        [optree.Assignment(make_tvar(b, 2), optree.exp(make_tvar(a, 2)))])]


def reference(inputs, dest_in):
    return np.exp(inputs["A"])
