# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``b[j] = sum_i a[i, j]`` — contract the *lead* axis, keep the other.

The three other axis cases contract axis 1 and keep axis 0, which is the
thread-distributed one, so each lane owns a row and folds it alone. This
one is the mirror image: axis 0 is contracted, so the fold crosses lanes,
and axis 1 survives.

That combination is what separates a full reduction from a partial one.
``max_all`` also crosses lanes, but it contracts everything, so there is
exactly one cross-lane exchange and one store. Here the exchange and the
store happen once per kept element, inside a sequential loop, and the
destination is written entirely by lane 0. Any lowering that assumes the
kept axes are still distributed over the lanes — which they cannot be,
the lanes being spoken for by the axis under contraction — produces a
result that is right in lane 0's column and wrong everywhere else.

Sum rather than max, so that the case also exercises the pairing of a
cross-lane exchange with an operator whose neutral element is 0: a lane
that owns no element must contribute the identity, and for ``max`` a
mistake there is visible only with negative data, while for ``sum`` any
non-zero seed shows up immediately.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_sum_axis0"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)         # 16-term sums in F32


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([16], Addressing.STRIDED,
                           BoundingBox([0], [16]),
                           alias="OUT", datatype=DTYPE))
    return [ReductionDescr(out, a, [0], AddOperator())]


def reference(inputs, dest_in):
    # axis 0 of the tensor is axis 1 of the batched array
    return np.sum(inputs["A"], axis=1)
