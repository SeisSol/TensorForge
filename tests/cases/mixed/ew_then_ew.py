# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``tmp = abs(A)`` then ``C = neg(tmp)`` — no multilinear anywhere.

The minimal statement of the gap: a temporary only ever gets a symbol inside
`MultilinearBuilder._make_store`, so with no multilinear in the section nothing
ever creates one.  `Generator._emit_ir` then builds a `SymbolView` over `None`.

This is also the case that shows the fix does not belong in the multilinear
builder: neither descriptor here has anything to do with contraction.

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


NAME = "mixed_ew_then_ew"


def descr_list():
    a, c = _s("A"), _s("C")
    tmp = _s("TMP", tmp=True)
    return [ew.abs(tmp, a),
            ew.neg(c, tmp)]


def reference(inputs, dest_in):
    return -np.abs(inputs["A"])
