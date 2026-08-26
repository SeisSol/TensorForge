# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary written in successive partial pieces, then read whole.

    t          = Q                 (whole box)
    t         += F0                (declared whole, F0 spans half of it)
    t         += F1
    O          = t x M             (reads the whole box)

Every descriptor declares the whole of ``t`` --- that is what yateto emits ---
but ``_analyze`` intersects the range down to what the operand supports, so the
second and third write half.  This is the shape of the elastic ADER kernels,
where each face contributes to the rows it touches.

Judged on the declared boxes those look like one writer covering everything,
and deferring the value in registers looks safe.  It is not: the image left
behind holds only the last writer's rows, and the read that follows wants the
union.  It used to be refused outright --- the elastic build stopped there --
which is why the boxes that matter are the ones actually written.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "partial_writes_read_whole"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, HALF, TERMS = 32, 9, 16, 2


def _t(shape, alias, is_tmp=False, addressing=Addressing.PTR_BASED):
    return Tensor(shape, addressing,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=is_tmp, datatype=DTYPE)


def descr_list():
    t = _t([M, N], "t", is_tmp=True, addressing=Addressing.STRIDED)
    out = [MultilinearDescr(dest=SubTensor(t), ops=[SubTensor(_t([M, N], "Q"))],
                            target=[[0, 1]], permute=[[0, 1]])]
    for i in range(TERMS):
        out.append(MultilinearDescr(
            dest=SubTensor(t), ops=[SubTensor(_t([HALF, N], f"F{i}"))],
            target=[[0, 1]], permute=[[0, 1]], add=True))
    out.append(MultilinearDescr(
        dest=SubTensor(_t([M, N], "O")),
        ops=[SubTensor(t), SubTensor(_t([N, N], "M"))],
        target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]))
    return out


def reference(inputs, dest_in):
    t = np.array(inputs["Q"], copy=True)
    for i in range(TERMS):
        t[:, :HALF, :] = t[:, :HALF, :] + inputs[f"F{i}"]
    return np.einsum("bik,bkj->bij", t, inputs["M"])
