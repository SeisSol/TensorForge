# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A pointwise write into a slice of a shared temporary a whole write defined.

    tmp        = B @ C              <- whole, into shared memory
    x[0:6,:]   = F1 @ G ; x[6:12,:] = F2 @ G
    D          = x @ H              <- last read of x
    tmp[0:4,:] = abs(K)             <- written in place, rows 0..4 only
    D         += A @ tmp            <- reads rows 4..12 of the first write

The pointwise instruction writes the buffer where it is, so it reports the
slice in `partial_defs`.  Reporting none, it would kill `tmp` like a whole
write, and the allocator would lay `x`'s operands over rows 4..12.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

NAME = "elementwise_slice_after_whole"
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
    k = SubTensor(_t((4, 12), "K"))
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
        ew.abs(rows(tmp, 0, 4), k),
        GemmDescr(False, False, a=a, b=SubTensor(tmp), c=d, alpha=1.0, beta=1.0),
    ]


def reference(inputs, dest_in):
    ein = lambda x, y: np.einsum("bik,bkj->bij", x, y)
    tmp = ein(inputs["B"], inputs["C"])
    tmp[:, :4, :] = np.abs(inputs["K"])
    x = np.concatenate([ein(inputs["F1"], inputs["G"]),
                        ein(inputs["F2"], inputs["G"])], axis=1)
    return ein(x, inputs["H"]) + ein(inputs["A"], tmp)
