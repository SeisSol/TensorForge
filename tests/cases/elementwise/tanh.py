# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = tanh(A[i, j])`` — single-unary-op ElementwiseDescr.

Domain: ``tanh`` saturates at ±1 and is signed-safe.

The case also guards the operator values in ``common/operation.py``. An
``Operation.TANH`` sharing its value with ``Operation.TAN`` would be
collapsed into it by Python's :class:`enum.Enum`, so ``ew.tanh`` would
lower to an ``Operation.TAN`` node, which the CUDA lexic table emits as
``tanf``: the kernel would compute ``tan`` while the reference computes
``tanh``, and the numerical comparison would fail. ``sinh``/``sin``,
``cosh``/``cos``, ``asinh``/``asin``, ``acosh``/``acos`` and
``atanh``/``atan`` pair up the same way, and the tanh cases are the only
ones that run a hyperbolic function.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_tanh_16x16"
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
    return [ew.tanh(b, a)]


def reference(inputs, dest_in):
    return np.tanh(inputs["A"])
