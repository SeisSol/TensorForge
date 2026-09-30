# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary re-assigned from an operand narrower than the temporary.

    tmp  = B @ C        <- defines all 12 rows
    u    = tmp @ E      <- last read of that value
    tmp  = N @ C        <- N has rows 0..4 only: the assignment defines rows
                           0..4 with values and rows 4..12 with zeros
    tmp += u @ F        <- reads all 12 rows back
    D    = A @ tmp

The shape of SeisSol's free-surface-gravity step `MPrev = U - invImp*(rhoG*MPrev
+ P)`.  The third store is narrowed by `_analyze` to rows 0..4; written to the
shared buffer without the zeros, rows 4..12 would keep whatever the buffer
holds -- the first `tmp`, or `u` where the allocator overlays it -- and the
`+=` would add to that.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "temp_reassign_narrower"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)
ROWS = 4


def _t(bbox, alias):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox([0, 0], list(bbox)), alias=alias, datatype=DTYPE)


def descr_list():
    a = SubTensor(_t((12, 12), "A"))
    b = SubTensor(_t((12, 12), "B"))
    c = SubTensor(_t((12, 12), "C"))
    e = SubTensor(_t((12, 12), "E"))
    f = SubTensor(_t((12, 12), "F"))
    n = SubTensor(_t((ROWS, 12), "N"))
    d = SubTensor(_t((12, 12), "D"))
    tmp = generate_tmp_matrix(b, c)
    u = generate_tmp_matrix(SubTensor(tmp), e)
    return [
        GemmDescr(False, False, a=b, b=c, c=SubTensor(tmp)),
        GemmDescr(False, False, a=SubTensor(tmp), b=e, c=SubTensor(u)),
        GemmDescr(False, False, a=n, b=c, c=SubTensor(tmp)),
        GemmDescr(False, False, a=SubTensor(u), b=f, c=SubTensor(tmp),
                  alpha=1.0, beta=1.0),
        GemmDescr(False, False, a=a, b=SubTensor(tmp), c=d,
                  alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    ein = lambda x, y: np.einsum("bik,bkj->bij", x, y)
    tmp = ein(inputs["B"], inputs["C"])
    u = ein(tmp, inputs["E"])
    tmp = np.zeros_like(tmp)
    tmp[:, :ROWS, :] = ein(inputs["N"], inputs["C"])
    tmp = tmp + ein(u, inputs["F"])
    return ein(inputs["A"], tmp)
