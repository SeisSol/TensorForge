# SPDX-License-Identifier: MIT
"""``B[i, j] = tanh(A[i, j])`` — single-unary-op ElementwiseDescr.

Domain: ``tanh`` saturates at ±1 and is signed-safe.

Known issue exposed by this case (separate from the LeadLoop bug):
``Operation.TANH`` aliases ``Operation.TAN`` in ``common/operation.py``
because both have value ``102``. Python's :class:`enum.Enum` collapses
duplicate-valued members, so ``optree.tanh(x)`` lowers to a
``Operation.TAN`` node, which the CUDA lexic table emits as ``tanf``.
The kernel therefore computes ``tan`` while the reference computes
``tanh``, and the numerical comparison fails by design. The same
collision exists for ``sinh``/``sin``, ``cosh``/``cos``,
``asinh``/``asin``, ``acosh``/``acos``, and ``atanh``/``atan`` — only
this case tests it; once the collision is fixed by giving each operator
a unique value, the case starts passing.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import optree
from tensorforge.generators.descriptions import ElementwiseDescr

from harness.optree_helpers import make_tvar

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
    return [ElementwiseDescr(
        [optree.Assignment(make_tvar(b, 2), optree.tanh(make_tvar(a, 2)))])]


def reference(inputs, dest_in):
    return np.tanh(inputs["A"])
