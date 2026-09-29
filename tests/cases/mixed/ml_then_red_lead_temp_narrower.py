# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary assigned anew by a reduction over the lead axis, narrower.

    TMP = X                 <- defines all 16 entries
    TMP = sum_i A[i, :]     <- A has columns 0..6 only, and the view is the
                               tensor itself: entries 0..6 get sums, the rest
                               zeros
    OUT = TMP

Both halves of `OperationBuilder.pointwise_dest` on a buffer that already has
a value: the reduction goes through a register image into a store that
zeroes the rest of the promise, and that image is one whose elements a fold
across the lanes writes one by one (`ReductionInstruction._image_axis`).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import MultilinearDescr, ReductionDescr

NAME = "mixed_ml_then_red_lead_temp_narrower"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

N, M, K = 16, 24, 6


def _t(shape, bbox, alias, tmp=False):
    return Tensor(list(shape), Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(bbox)),
                  alias=alias, datatype=DTYPE, is_tmp=tmp)


def descr_list():
    tmp = _t([N], [N], "TMP", tmp=True)
    return [
        MultilinearDescr(SubTensor(tmp), [SubTensor(_t([N], [N], "X"))],
                         [[0]], [[0]]),
        ReductionDescr(SubTensor(tmp, BoundingBox([0], [K])),
                       SubTensor(_t([M, N], [M, K], "A")), [0], AddOperator()),
        MultilinearDescr(SubTensor(_t([N], [N], "OUT")), [SubTensor(tmp)],
                         [[0]], [[0]]),
    ]


def reference(inputs, dest_in):
    out = np.zeros_like(inputs["X"])
    out[:, :K] = np.sum(inputs["A"], axis=1)
    return out
