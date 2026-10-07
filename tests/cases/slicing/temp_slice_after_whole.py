# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A slice written into a temporary a whole write defined before it.

    tmp        = B @ C              <- whole
    x[0:6,:]   = F1 @ G             <- another temporary, assembled in between
    x[6:12,:]  = F2 @ G
    D          = x @ H              <- last read of x
    tmp[0:4,:] = N @ C              <- a slice: rows 4..12 are still the first write's
    D         += A @ tmp            <- reads all 12 rows

Taken as a fresh start, the slice would kill `tmp`, so between the two writes
it would be dead and the allocator would lay `x`'s operands over it; rows 4..12
would come back as whatever those held.  The slice defines nothing whole and
does not say it does (`Generator._declare_buffers`), which keeps it from
killing `tmp`.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "temp_slice_after_whole"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)


def _t(bbox, alias):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox([0, 0], list(bbox)), alias=alias, datatype=DTYPE)


def descr_list():
    a, b, c, g, h = (SubTensor(_t((12, 12), n)) for n in "ABCGH")
    f1, f2 = SubTensor(_t((6, 12), "F1")), SubTensor(_t((6, 12), "F2"))
    n = SubTensor(_t((4, 12), "N"))
    d = SubTensor(_t((12, 12), "D"))
    tmp = generate_tmp_matrix(b, c)
    x = generate_tmp_matrix(b, c)
    rows = lambda t, lo, hi: SubTensor(t, BoundingBox([0, 0], [hi - lo, 12]),
                                       [lo, 0], sliced=True)
    return [
        GemmDescr(False, False, a=b, b=c, c=SubTensor(tmp)),
        GemmDescr(False, False, a=f1, b=g, c=rows(x, 0, 6)),
        GemmDescr(False, False, a=f2, b=g, c=rows(x, 6, 12)),
        GemmDescr(False, False, a=SubTensor(x), b=h, c=d),
        GemmDescr(False, False, a=n, b=c, c=rows(tmp, 0, 4)),
        GemmDescr(False, False, a=a, b=SubTensor(tmp), c=d, alpha=1.0, beta=1.0),
    ]


def reference(inputs, dest_in):
    ein = lambda x, y: np.einsum("bik,bkj->bij", x, y)
    tmp = ein(inputs["B"], inputs["C"])
    tmp[:, :4, :] = ein(inputs["N"], inputs["C"])
    x = np.concatenate([ein(inputs["F1"], inputs["G"]),
                        ein(inputs["F2"], inputs["G"])], axis=1)
    return ein(x, inputs["H"]) + ein(inputs["A"], tmp)
