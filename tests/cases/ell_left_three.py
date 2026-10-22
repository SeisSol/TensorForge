# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = S X`` with a batch-constant ``S`` of at most three nonzeros per row.

Rows with one, two and three nonzeros side by side in a slot.

The lanes run along the rows of ``S``, so a step of the reduction is issued
by every lane and has a nonzero in only some of them: computed the usual way
it is the dense product.  Gathered (`MultilinearInstruction._ell_plan`), a
lane takes its own row's nonzeros and reads ``X`` at their columns -- at a
per-lane address where ``X`` is in memory, across the lanes where it is a
register image.  Forty rows, so a wave of 32 lanes takes two slots, and an
image of ``X`` over 32 lanes is gathered one slot at a time.

The kernel has to see the numbers the generator was told, so the input for
``S`` is replaced by them (`INPUT_TRANSFORM`); the reference reads the same.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "ell_left_three"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
K, N = 40, 9


def _values():
    rng = np.random.default_rng(34)
    perm = rng.permutation(K)
    s = np.zeros((K, K))
    for row in range(K):
        s[row, perm[row]] = rng.uniform(0.5, 1.5)
        if row % 2 == 0:
            s[row, (perm[row] + 7) % K] = rng.uniform(-1.0, -0.25)
        if row % 3 == 0:
            s[row, (perm[row] + 13) % K] = rng.uniform(0.25, 1.0)
    return s


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
