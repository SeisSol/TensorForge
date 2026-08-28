# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A lead dimension of 35 over aligned operands: one element is left over.

35 is an order-4 basis-function count and the shape that makes the peel run:
at width 2 the whole vectors cover 34 elements and the 35th has no partner.
Every other aligned case in this directory divides, so without this one the
scalar tail is unreachable and only the unit tests speak to it.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "aligned_odd_lead"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, K = 35, 4, 8
ALIGNMENT = 16


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE,
                            alignment=ALIGNMENT))


def descr_list():
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=_t([M, K], "A"), b=_t([K, N], "B"),
                      c=_t([M, N], "D"),
                      alpha=1.0, beta=0.0)]
