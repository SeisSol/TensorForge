# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``b[i] = prod_j a[i, j]`` — product reduction via :class:`MulOperator`.

Numerically the weakest reduction in the family: a product of 16
standard-normal values can underflow or overflow F32 quickly, so we
constrain the input domain to ``(0.95, 1.05)`` via ``INPUT_TRANSFORM``.
Even there the product accumulates non-trivial FP rounding error,
hence the looser tolerance.

The case exercises a different ``neutral()`` value (``1`` rather than
``0`` or ``±inf``); a reduction implementation that hard-codes one
neutral element will fail this case while the others pass. It is also
the one operator in the family that ``Op.ACCUM`` cannot express, since
that lowers to ``+=``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import MulOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_prod_axis1"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-3, 1e-3)         # 16-fold product of values near 1 amplifies rounding
INPUT_TRANSFORM = {"A": lambda x: 1.0 + 0.05 * np.tanh(x)}



def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
               BoundingBox([0, 0], [16, 16]),
               alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([16], Addressing.STRIDED,
                 BoundingBox([0], [16]),
                 alias="OUT", datatype=DTYPE))
    return [ReductionDescr(out, a, [1], MulOperator())]


def reference(inputs, dest_in):
    return np.prod(inputs["A"], axis=2)
