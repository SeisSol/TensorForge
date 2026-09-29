# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary assigned anew, pointwise, from an operand narrower than it.

    tmp = B @ C         <- defines all 12 rows
    u   = tmp @ E       <- last read of that value
    tmp = abs(N)        <- N has rows 4..8 only, and the view is the tensor
                           itself: rows 4..8 get values, the rest zeros
    D   = u @ tmp       <- reads all 12 rows back

The pointwise side of SeisSol's free-surface-gravity step, which re-assigns a
temporary from a product one row of its operand supports.  The pointwise
operation wrote its four rows into the shared buffer in place and nothing
else, so the other eight kept the old `tmp` -- or `u`, where the allocator
overlaid it.  Now the result goes through a store that zeroes the rest of the
promise (`OperationBuilder.pointwise_dest`).  The window starts past row 0,
so the zeros go on both sides of it.

The step accumulates onto the temporary next; that is left out here, since on
the AMD targets an accumulation onto a shared temporary loses what the buffer
held whether or not it was re-assigned (`gemm_add_accumulate_f64` fails on
gfx1150 as well).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

NAME = "mixed_ml_then_ew_temp_narrower"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)
LO, HI = 4, 8


def _t(bbox, alias, lower=(0, 0)):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox(list(lower), list(bbox)), alias=alias,
                  datatype=DTYPE)


def descr_list():
    b = SubTensor(_t((12, 12), "B"))
    c = SubTensor(_t((12, 12), "C"))
    e = SubTensor(_t((12, 12), "E"))
    n = SubTensor(_t((HI, 12), "N", lower=(LO, 0)))
    d = SubTensor(_t((12, 12), "D"))
    tmp = generate_tmp_matrix(b, c)
    u = generate_tmp_matrix(SubTensor(tmp), e)
    return [
        GemmDescr(False, False, a=b, b=c, c=SubTensor(tmp)),
        GemmDescr(False, False, a=SubTensor(tmp), b=e, c=SubTensor(u)),
        ew.abs(SubTensor(tmp, BoundingBox([LO, 0], [HI, 12])), n),
        GemmDescr(False, False, a=SubTensor(u), b=SubTensor(tmp), c=d),
    ]


def reference(inputs, dest_in):
    ein = lambda x, y: np.einsum("bik,bkj->bij", x, y)
    tmp = ein(inputs["B"], inputs["C"])
    u = ein(tmp, inputs["E"])
    tmp = np.zeros_like(tmp)
    tmp[:, LO:HI, :] = np.abs(inputs["N"])
    return ein(u, tmp)
