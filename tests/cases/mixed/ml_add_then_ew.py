# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An accumulation chain into a temporary, then read pointwise.

    tmp  = A1 @ B1
    tmp += A2 @ B2
    C    = abs(tmp)

The accumulation is the case the register residency exists for: both writers
cover the whole box, so the second one accumulates into the array the first one
left behind and nothing goes to memory in between.  That makes it the strongest
form of the gap in `ml_then_ew` -- there are two operations' worth of value in
registers when the elementwise asks for the tensor -- and the case that says
whether a fix flushes the newest copy or merely the last one.

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
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

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


NAME = "mixed_ml_add_then_ew"


def descr_list():
    a1, b1, a2, b2, c = _s("A1"), _s("B1"), _s("A2"), _s("B2"), _s("C")
    tmp = _s("TMP", tmp=True)
    return [GemmDescr(False, False, a=a1, b=b1, c=tmp),
            GemmDescr(False, False, a=a2, b=b2, c=tmp, alpha=1.0, beta=1.0),
            ew.abs(c, tmp)]


def reference(inputs, dest_in):
    return np.abs(np.einsum("bik,bkj->bij", inputs["A1"], inputs["B1"])
                   + np.einsum("bik,bkj->bij", inputs["A2"], inputs["B2"]))
