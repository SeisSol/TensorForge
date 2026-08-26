# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = A[i, j] ** 3`` — single-(constant-folded)-op ElementwiseDescr.

This case exercises a binary elementwise op with a scalar operand:
``ew.pow(b, a, 3.0)``. The exponent 3.0 matches none of the folding
special cases (2, ±0.5, ±1/3, ±1), so it does lower to a ``POW``
only short-circuits exponents ``-1, 0.5, -0.5, 1, 2, 1/3, -1/3`` —
exponent 3 falls through to ``LexicOpNode([x, 3], Operation.POW)`` and
will hit ``powf`` in CUDA.

Two reasons to include it:

* it is genuinely the only nonlinear op left (after sqrt/cbrt/rcp
  swallowed the obvious short-circuits);
* it provides a stable fixed point to detect regressions in the
  fold-table — if someone adds a fold for ``y == 3``, this case's
  generated kernel will change, which a golden-output check would
  catch (not in scope for the MVP, but the case is the prerequisite).

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
