# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = A[i, j] ** 3`` — single-(constant-folded)-op ElementwiseDescr.

This case exercises a binary elementwise op with a scalar operand:
``ew.pow(b, a, 3.0)``.  ``ew.pow`` folds the exponents 2, 1, -1, 0.5,
-0.5, 1/3 and -1/3 into cheaper operations; 3.0 matches none of them,
so it lowers to ``Operation.POW`` and reaches ``powf`` in CUDA.

Two reasons to include it:

* it is the one case whose nonlinear op is the general power, the
  obvious exponents being folded into sqrt/cbrt/rcp and friends;
* it pins the fold table: a fold for ``y == 3`` would change this case's
  generated kernel, and its snapshot would show it.

Domain: signed; ``a**3`` is well-defined everywhere and bounded for
``standard_normal``-magnitude inputs.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_pow3_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)         # powf is the loosest of the unary calls


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    # 3.0 (float) — using int 3 would still take the POW path, but
    # spelling it as float matches what the runtime sees.
    return [ew.pow(b, a, 3.0)]


def reference(inputs, dest_in):
    return np.power(inputs["A"], 3.0)
