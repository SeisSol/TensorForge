# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Two temporaries assembled from slices, the second where the first was.

    x[0:6,:] = F1 @ G ; x[6:12,:] = F2 @ G
    D        = x @ H                <- x dead from here
    y[0:6,:] = B1 @ C ; y[6:12,:] = B2 @ C
    D       += y @ E

Legitimate reuse, and the allocation makes it.  An emit-time check that
flattened the nest would refuse it: the batch loop -- whose `defs` are its
body's -- would stand ahead of the body as a write of both buffers, each first
slice would count as a second one and not kill, and `x` would look live to the
end.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "two_assembled_reuse"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)


def _t(bbox, alias):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox([0, 0], list(bbox)), alias=alias, datatype=DTYPE)


def descr_list():
    c, e, g, h = (SubTensor(_t((12, 12), n)) for n in "CEGH")
    f1, f2, b1, b2 = (SubTensor(_t((6, 12), n)) for n in ("F1", "F2", "B1", "B2"))
    d = SubTensor(_t((12, 12), "D"))
    x = generate_tmp_matrix(c, g)
    y = generate_tmp_matrix(c, g)
    half = lambda t, lo: SubTensor(t, BoundingBox([0, 0], [6, 12]), [lo, 0],
                                   sliced=True)
    return [
        GemmDescr(False, False, a=f1, b=g, c=half(x, 0)),
        GemmDescr(False, False, a=f2, b=g, c=half(x, 6)),
        GemmDescr(False, False, a=SubTensor(x), b=h, c=d),
        GemmDescr(False, False, a=b1, b=c, c=half(y, 0)),
        GemmDescr(False, False, a=b2, b=c, c=half(y, 6)),
        GemmDescr(False, False, a=SubTensor(y), b=e, c=d, alpha=1.0, beta=1.0),
    ]


def reference(inputs, dest_in):
    ein = lambda x, y: np.einsum("bik,bkj->bij", x, y)
    x = np.concatenate([ein(inputs["F1"], inputs["G"]),
                        ein(inputs["F2"], inputs["G"])], axis=1)
    y = np.concatenate([ein(inputs["B1"], inputs["C"]),
                        ein(inputs["B2"], inputs["C"])], axis=1)
    return ein(x, inputs["H"]) + ein(y, inputs["E"])
