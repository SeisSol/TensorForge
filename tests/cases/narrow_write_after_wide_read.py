# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A narrow write to a tensor that was read wide earlier in the same kernel.

    t          = D[0:16, :]        # reads D over a wide box
    t         += A @ B
    D[:, 4:5]  = t[:, 4:5]         # writes one column back
    O          = D @ C

``_deferred_stores`` is keyed by symbol name and lives for the whole kernel,
so the register image staged for the *read* of ``D`` is what the later write
finds when it asks where its destination lives.  The accumulator then adopted
that image's data view --- the whole tensor --- although it only ever computes
one column.  The store believed it held 13 columns, wrote all of them, and
read past the end of the register array to do it.

The space-time predictor does this five times, once per quantity: a
one-row-one-column write picked up the image of the whole 32x13x4 tensor and
clobbered it.

``D`` is read back afterwards, so the damage shows up in the result and not
only in the generated source.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, MultilinearDescr

NAME = "narrow_write_after_wide_read"
OUTPUT = "O"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

M, N, K, COL = 32, 13, 12, 4


def _t(shape, alias, is_tmp=False):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=is_tmp, datatype=DTYPE)


def descr_list():
    d = _t([M, N], "D")
    t = _t([M, N], "t", is_tmp=True)
    return [
        # read D over the whole box: this is what stages the register image
        MultilinearDescr(dest=SubTensor(t), ops=[SubTensor(d)],
                         target=[[0, 1]], permute=[[0, 1]]),
        GemmDescr(False, False, a=SubTensor(_t([M, K], "A")),
                  b=SubTensor(_t([K, N], "B")), c=SubTensor(t),
                  alpha=1.0, beta=1.0),
        # ...and write one column of D back
        MultilinearDescr(
            dest=SubTensor(d, bbox=BoundingBox([0, 0], [M, 1]),
                           offset=[0, COL]),
            ops=[SubTensor(t, bbox=BoundingBox([0, 0], [M, 1]),
                           offset=[0, COL])],
            target=[[0, 1]], permute=[[0, 1]]),
        GemmDescr(False, False, a=SubTensor(d), b=SubTensor(_t([N, N], "C")),
                  c=SubTensor(_t([M, N], "O")), alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    d = np.array(inputs["D"], copy=True)
    t = d + np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    d[:, :, COL:COL + 1] = t[:, :, COL:COL + 1]
    return np.einsum("bik,bkj->bij", d, inputs["C"])
