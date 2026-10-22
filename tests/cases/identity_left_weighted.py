# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = S X`` with a batch-constant ``S`` that is the identity pattern, with weights.

The lanes run along the rows of ``S``, so a step of the reduction is issued
by every lane and has a nonzero in only one of them: computed the usual way
it is the dense product.  The reduction collapses to the lane's own row
(`MultilinearInstruction._identity_reduction`), each output one multiply by the lane's weight.  Forty rows, so a
wave of 32 lanes takes two slots.

The kernel has to see the numbers the generator was told, so the input for
``S`` is replaced by them (`INPUT_TRANSFORM`); the reference reads the same.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "identity_left_weighted"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
K, N = 40, 9


def _values():
    rng = np.random.default_rng(21)
    return np.diag(rng.uniform(0.5, 1.5, K) * np.where(np.arange(K) % 2, 1.0, -1.0))


S_VALUES = _values()
INPUT_TRANSFORM = {"S": lambda x: np.broadcast_to(S_VALUES, x.shape)}


def descr_list():
    s = SubTensor(Tensor([K, K], Addressing.NONE, BoundingBox([0, 0], [K, K]),
                         alias="S", datatype=DTYPE, data=S_VALUES.copy()))
    x = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="X", datatype=DTYPE))
    c = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, s, x, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["S"], inputs["X"])
