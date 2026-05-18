# SPDX-License-Identifier: MIT
"""``B[i, j] = abs(A[i, j])`` — single-unary-op ElementwiseDescr.

Domain: signed input desired so the operation is non-trivial; no
transform applied.

This case exists as a counterpart to ``sin``/``tanh`` for a non-
transcendental nonlinear op; the branch-free abs path stresses sign-bit
handling rather than a math-library call.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import optree
from tensorforge.generators.descriptions import ElementwiseDescr

from harness.optree_helpers import make_tvar

NAME = "elementwise_abs_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ElementwiseDescr(
        [optree.Assignment(make_tvar(b, 2), optree.abs(make_tvar(a, 2)))])]


def reference(inputs, dest_in):
    return np.abs(inputs["A"])
