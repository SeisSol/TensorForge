# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D[:, 6:13] = Q[:, 10:13] @ S[10:13, 6:13]`` --- one sliced write, beta=0.

The shape the poroelastic space-time predictor produces: a descriptor whose
destination is a sub-box of the tensor, with the transposition carried in
``target`` and an identity ``permute`` (the yateto convention).

Two things went wrong here and neither leaves a trace in the source beyond
wrong numbers.  The accumulator adopted its data view from ``next``, which for
a plain global destination is the *whole tensor* rather than an image with a
layout worth adopting; the store then believed the seven-column accumulator
spanned thirteen columns, read past the end of the register array, and wrote
the slice at column 0 instead of column 6.  And the store treated the tensor's
box as what it had to cover, so the columns outside the slice were zero-filled
--- overwriting whatever else had been written there.

``D`` is a pure output here, so the reference defines the untouched columns as
zero; what must not happen is the *slice* landing anywhere else.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "sliced_write"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, LO = 32, 13, 6
KLO = 10


def _t(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    q = _t([M, N], "Q")
    s = _t([N, N], "S")
    d = _t([M, N], "D")
    return [MultilinearDescr(
        dest=SubTensor(d, bbox=BoundingBox([0, LO], [M, N])),
        ops=[SubTensor(q, bbox=BoundingBox([0, KLO], [M, N])),
             SubTensor(s, bbox=BoundingBox([KLO, LO], [N, N]))],
        target=[[0, -1], [-1, 1]],
        permute=[[0, 1], [0, 1]])]


def reference(inputs, dest_in):
    out = np.zeros_like(dest_in)
    out[:, :, LO:N] = np.einsum("bik,bkj->bij",
                                inputs["Q"][:, :, KLO:N],
                                inputs["S"][:, KLO:N, LO:N])
    return out
