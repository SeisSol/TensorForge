# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``tmp = abs(A)`` then ``C = tmp @ B`` — an elementwise feeding a multilinear.

The other direction, and it fails earlier and for a different reason.
`MultilinearBuilder.plan` walks the descriptor list to decide which temporaries
are written where, and skips every descriptor that is not a `MultilinearDescr`.
An elementwise write is therefore invisible to it, so `_check_initialised` sees
a temporary that is read and never written and refuses to generate.

Nothing about the refusal is wrong except its premise: the write exists, the
analysis just cannot see it.

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


NAME = "mixed_ew_then_ml"


def descr_list():
    a, b, c = _s("A"), _s("B"), _s("C")
    tmp = _s("TMP", tmp=True)
    return [ew.abs(tmp, a),
            GemmDescr(False, False, a=tmp, b=b, c=c)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", np.abs(inputs["A"]), inputs["B"])
