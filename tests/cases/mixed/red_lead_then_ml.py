# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``TMP[j] = sum_i A[i, j]`` into a temporary, then ``OUT = TMP``.

The reduction contracts the lead axis, so the fold crosses the lanes and the
kept axis `j` is walked sequentially -- and the temporary is computed into a
register image that spreads `j` over the lanes (`materialize_dest`).  A loop
variable on the image's lead axis addresses a slot, not an element: lane 0
wrote every `j` into its own slot `j`, past the end of a one-slot image, and
the image went to the buffer as one wrong value.  Each element is now written
by number, by the lane that owns it (`ReductionInstruction._image_axis`).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import MultilinearDescr, ReductionDescr

NAME = "mixed_red_lead_then_ml"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

M, K = 24, 6


def _t(shape, alias, tmp=False):
    return Tensor(list(shape), Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE, is_tmp=tmp)


def descr_list():
    tmp = _t([K], "TMP", tmp=True)
    return [
        ReductionDescr(SubTensor(tmp), SubTensor(_t([M, K], "A")), [0],
                       AddOperator()),
        MultilinearDescr(SubTensor(_t([K], "OUT")), [SubTensor(tmp)],
                         [[0]], [[0]]),
    ]


def reference(inputs, dest_in):
    return np.sum(inputs["A"], axis=1)          # batch axis is 0
