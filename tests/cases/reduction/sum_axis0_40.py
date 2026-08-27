# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``b[j] = sum_i a[i, j]`` with 40 rows — more lead elements than lanes.

``sum_axis0`` contracts the lead axis too, but its extent is 16 against a
thread count of 32: one element per lane, and every lane that holds one
holds exactly one. That is the shape every other reduction case has, and
it hides the question this one asks.

40 over 32 threads is two slots. Lane ``t`` owns element ``t`` and element
``t + 32``, and the second of those exists only for ``t < 8``. A lowering
that addresses the lead axis with a single ``LeadIndex(0, ...)`` folds the
first slot and drops the rest, which here means summing 32 of 40 rows —
a number in the right range, with the right sign, off by the tail.

The ragged slot is also the one place where a lane must be kept away from
the load rather than merely have its contribution discarded: element
``t + 32`` for ``t >= 8`` is past the end of the tensor, so the guard has
to contain the read.

40 is deliberately not a multiple of 32 and not a power of two. The
exchange width is rounded up from the number of lanes that hold data, and
with two slots every lane does, so this case pins the full width where
``sum_axis0`` pins the narrowed one.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_sum_axis0_40"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)         # 40-term sums in F32


def descr_list():
    a = SubTensor(Tensor([40, 3], Addressing.STRIDED,
                         BoundingBox([0, 0], [40, 3]),
                         alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([3], Addressing.STRIDED,
                           BoundingBox([0], [3]),
                           alias="OUT", datatype=DTYPE))
    return [ReductionDescr(out, a, [0], AddOperator())]


def reference(inputs, dest_in):
    # axis 0 of the tensor is axis 1 of the batched array
    return np.sum(inputs["A"], axis=1)
