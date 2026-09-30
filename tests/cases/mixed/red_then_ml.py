# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``t[i] = sum_j A[i, j]`` then ``C[i, j] = t[i] * B[i, j]``.

A reduction producing a temporary that a multilinear then consumes, with the
temporary carried as a rank-1 broadcast operand.  Two things are being asked at
once: whether the write is visible to `SectionPlan` at all, and whether a
rank-1 temporary is given a symbol of the right shape by whoever ends up
creating it.

Shapes are 8x8 throughout, which keeps a snapshot diff readable.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import MultilinearDescr, ReductionDescr

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


NAME = "mixed_red_then_ml"


def descr_list():
    a, b, c = _s("A"), _s("B"), _s("C")
    tmp = _s("TMP", shape=(N,), tmp=True)
    return [ReductionDescr(tmp, a, [1], AddOperator()),
            MultilinearDescr(c, [tmp, b], [[0], [0, 1]], [[0], [0, 1]])]


def reference(inputs, dest_in):
    return np.sum(inputs["A"], axis=2)[:, :, None] * inputs["B"]
