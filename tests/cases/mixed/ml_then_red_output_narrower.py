# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An output assigned anew by a reduction over an operand narrower than it.

    V = X               <- defines all 16 entries
    V = sum_j N[:, j]   <- N has rows 4..8 only, and the view is the tensor
                           itself: entries 4..8 get sums, the rest zeros

The reduction's side of `mixed/ml_then_ew_output_narrower`: writing only its
four entries in place, it would leave the other twelve holding `X`.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import MultilinearDescr, ReductionDescr

NAME = "mixed_ml_then_red_output_narrower"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, K, LO, HI = 16, 16, 4, 8


def _t(shape, bbox, alias, lower=None):
    return Tensor(list(shape), Addressing.STRIDED,
                  BoundingBox(list(lower or [0] * len(shape)), list(bbox)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    v = _t([M], [M], "V")
    x = _t([M], [M], "X")
    n = _t([M, K], [HI, K], "N", lower=[LO, 0])
    return [
        MultilinearDescr(SubTensor(v), [SubTensor(x)], [[0]], [[0]]),
        ReductionDescr(SubTensor(v, BoundingBox([LO], [HI])), SubTensor(n),
                       [1], AddOperator()),
    ]


def reference(inputs, dest_in):
    out = np.zeros_like(inputs["X"])
    out[:, LO:HI] = np.sum(inputs["N"], axis=2)
    return out
