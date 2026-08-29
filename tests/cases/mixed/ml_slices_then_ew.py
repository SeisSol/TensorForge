# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary assembled from two writes, then read pointwise.

    tmp[:, 0:4] = A @ B1
    tmp[:, 4:8] = A @ B2
    C           = abs(tmp)

`_written_in_slices` is true here, so the multilinear stores each half into
shared memory as it is produced rather than deferring it.  The residency is
therefore empty by the time the elementwise runs, and this case separates
"the consumer cannot see the residency" from "the consumer cannot address a
shared-memory temporary at all".  Whichever of the two this case lands on says
which half of the gap is which.

Shapes are 8x8 throughout: the elementwise descriptor pins the lane count to the
vector unit length regardless of the tensors, so nothing is gained by going
smaller, and 8x8 keeps a snapshot diff readable.

`abs` is the pointwise operation because it is total and exact -- no input
domain to shape, no tolerance spent on a transcendental, and every backend has
it.  ESIMD, for one, has no `tanh` intrinsic at all.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

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


NAME = "mixed_ml_slices_then_ew"
HALF = N // 2


def descr_list():
    a, c = _s("A"), _s("C")
    b1 = _s("B1", shape=(N, HALF))
    b2 = _s("B2", shape=(N, HALF))
    tmp = _t("TMP", tmp=True)
    left = SubTensor(tmp, BoundingBox([0, 0], [N, HALF]), [0, 0], sliced=True)
    right = SubTensor(tmp, BoundingBox([0, 0], [N, HALF]), [0, HALF])
    return [GemmDescr(False, False, a=a, b=b1, c=left),
            GemmDescr(False, False, a=a, b=b2, c=right),
            ew.abs(c, SubTensor(tmp))]


def reference(inputs, dest_in):
    whole = np.concatenate(
        [np.einsum("bik,bkj->bij", inputs["A"], inputs["B1"]),
         np.einsum("bik,bkj->bij", inputs["A"], inputs["B2"])], axis=2)
    return np.abs(whole)
