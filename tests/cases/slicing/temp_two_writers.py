# SPDX-License-Identifier: MIT
"""A temporary assembled from two writes, then read whole.

    tmp[0:6,  :] = B1 @ C
    tmp[6:12, :] = B2 @ C
    D            = A  @ tmp      <- reads all 12 rows

With one deferred entry per tensor, keeping this in registers can only retain
whichever write came last; the other half would be lost. `_written_in_slices`
has to notice the second writer and force both halves into the shared buffer as
they are produced.

The corpus otherwise contains no tensor with more than one writer, so the
writer-count branch of that predicate is dead in tests.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "temp_two_writers"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)
SPLIT = 6


def _t(bbox, alias):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox([0, 0], list(bbox)), alias=alias, datatype=DTYPE)


def descr_list():
    a = SubTensor(_t((12, 12), "A"))
    b1 = SubTensor(_t((SPLIT, 12), "B1"))
    b2 = SubTensor(_t((SPLIT, 12), "B2"))
    c = SubTensor(_t((12, 12), "C"))
    d = SubTensor(_t((12, 12), "D"))
    tmp = generate_tmp_matrix(SubTensor(_t((12, 12), "Bfull")), c)
    top = SubTensor(tmp, bbox=BoundingBox([0, 0], [SPLIT, 12]), offset=[0, 0])
    bot = SubTensor(tmp, bbox=BoundingBox([0, 0], [SPLIT, 12]), offset=[SPLIT, 0])
    return [
        GemmDescr(False, False, a=b1, b=c, c=top),
        GemmDescr(False, False, a=b2, b=c, c=bot),
        GemmDescr(False, False, a=a, b=SubTensor(tmp), c=d,
                  alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    tmp = np.concatenate(
        [np.einsum("bik,bkj->bij", inputs["B1"], inputs["C"]),
         np.einsum("bik,bkj->bij", inputs["B2"], inputs["C"])], axis=1)
    return np.einsum("bik,bkj->bij", inputs["A"], tmp)
