# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``tmp = A @ B`` then ``out[i] = sum_j tmp[i, j]``.

The same residency question as `ml_then_ew`, asked by the other consumer:
the contraction may leave its result in registers, and the reduction has to
read the newest copy.  Worth its own case because a reduction resolves its
operand through `ReductionInstruction`, not `ElementwiseInstruction`, so
asking the residency in only one of the two would leave the other reading a
buffer the writeback has not reached.

The contracted axis is not the lead axis, so this lowers to the register-local
fold and needs no cross-lane traffic of its own.

Shapes are 8x8 throughout, which keeps a snapshot diff readable.
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
