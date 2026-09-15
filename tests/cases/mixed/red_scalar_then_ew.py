# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``m = max_i |A[i]|`` into a temporary without axes, then ``C[i] = A[i] * m``.

The reduction contracts the lane axis, so its fold crosses lanes, and its
destination is a temporary: a register image without axes, one value that
every lane of the multiplication reads afterwards.  So every lane has to hold
it.  The fold used to store its answer from lane 0 alone -- enough for an
address, which wants one writer, and wrong for a register array, of which each
lane has its own: the other lanes kept their zeros, and the image went to its
shared buffer from every lane at once, the answer and thirty-one zeros to one
address.  SeisSol's damage step takes ten `max`es this way.

40 rows over 32 lanes: two slots, the second ragged, so lanes past the end fold
the neutral element.  `abs` keeps the maximum away from zero, which is what a
losing lane would have stored.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import MaxOperator
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import ReductionDescr

DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)
N = 40

NAME = "mixed_red_scalar_then_ew"


def _s(alias, shape, tmp=False):
    return SubTensor(Tensor(list(shape),
                            Addressing.PTR_BASED if tmp else Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, is_tmp=tmp, datatype=DTYPE))


def descr_list():
    a, c = _s("A", (N,)), _s("C", (N,))
    pos = _s("P", (N,), tmp=True)
    m = _s("M", (), tmp=True)
    return [ew.abs(pos, a),
            ReductionDescr(m, pos, [0], MaxOperator()),
            ew.mul(c, a, m)]


def reference(inputs, dest_in):
    A = inputs["A"]
    return A * np.max(np.abs(A), axis=1, keepdims=True)
