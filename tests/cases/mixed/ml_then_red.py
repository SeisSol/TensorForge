# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``tmp = A @ B`` then ``out[i] = sum_j tmp[i, j]``.

The same residency gap as `ml_then_ew`, reached through the other consumer.
Worth its own case because a reduction resolves its operand through
`ReductionInstruction`, not `ElementwiseInstruction`, so a fix that only
routes one of the two would still pass the other.

The contracted axis is not the lead axis, so this lowers to the register-local
fold and needs no cross-lane traffic of its own.

Shapes are 8x8 throughout: the elementwise descriptor pins the lane count to the
vector unit length regardless of the tensors, so nothing is gained by going
smaller, and 8x8 keeps a snapshot diff readable.

`abs` is the pointwise operation because it is total and exact -- no input
domain to shape, no tolerance spent on a transcendental, and every backend has
it.  ESIMD, for one, has no `tanh` intrinsic at all.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import GemmDescr, ReductionDescr

DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
N = 8


def _t(alias, shape=(N, N), tmp=False):
    return Tensor(list(shape),
                  Addressing.PTR_BASED if tmp else Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=tmp, datatype=DTYPE)


def _s(alias, shape=(N, N), tmp=False):
    return SubTensor(_t(alias, shape, tmp))


NAME = "mixed_ml_then_red"


def descr_list():
    a, b = _s("A"), _s("B")
    out = _s("OUT", shape=(N,))
    tmp = _s("TMP", tmp=True)
    return [GemmDescr(False, False, a=a, b=b, c=tmp),
            ReductionDescr(out, tmp, [1], AddOperator())]


def reference(inputs, dest_in):
    return np.sum(np.einsum("bik,bkj->bij", inputs["A"], inputs["B"]), axis=2)
