# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``t4[32x3] = t2[32]`` --- a broadcast --- then ``t4 += t3[32x3]``.

The shape the boundary-condition kernels produce: a rank-1 temporary spread
across a rank-2 destination, accumulated onto, and read back.

The operand carries only index 0, so nothing in the operation mentions index
1.  Deriving the operation's rank from the operands alone dropped it: the loop
nest ran over ``n0``, one slot per lead block was written where three were
needed, and the broadcast never happened.  The accumulation that follows then
took its array size from the rank-1 image left behind and came out with one
slot where the store walks three --- one array too short, one too long, which
is how this shows up in a dump.

Neither is visible in the numbers: the host interpreter keeps registers in an
unbounded dict, so a short array reads and writes past its end and still
produces the right answer, while on a GPU ``float r5[1]`` is one register and
index 2 is whatever follows it.  ``tools/host/check_structure.py`` compares a
store's indices against the declared length, which is what catches it.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "broadcast_then_accumulate"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N = 32, 3


def _t(shape, alias, is_tmp=False, addressing=Addressing.PTR_BASED):
    return Tensor(shape, addressing,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=is_tmp, datatype=DTYPE)


def descr_list():
    t2 = _t([M], "t2", is_tmp=True, addressing=Addressing.STRIDED)
    t3 = _t([M, N], "t3", is_tmp=True, addressing=Addressing.STRIDED)
    t4 = _t([M, N], "t4", is_tmp=True, addressing=Addressing.STRIDED)
    return [
        MultilinearDescr(dest=SubTensor(t2), ops=[SubTensor(_t([M], "A"))],
                         target=[[0]], permute=[[0]]),
        MultilinearDescr(dest=SubTensor(t3), ops=[SubTensor(_t([M, N], "B"))],
                         target=[[0, 1]], permute=[[0, 1]]),
        # the broadcast: one operand, carrying index 0 only
        MultilinearDescr(dest=SubTensor(t4), ops=[SubTensor(t2)],
                         target=[[0]], permute=[[0]]),
        MultilinearDescr(dest=SubTensor(t4), ops=[SubTensor(t3)],
                         target=[[0, 1]], permute=[[0, 1]], add=True),
        MultilinearDescr(dest=SubTensor(_t([M, N], "O")), ops=[SubTensor(t4)],
                         target=[[0, 1]], permute=[[0, 1]]),
    ]


def reference(inputs, dest_in):
    return inputs["A"][:, :, None] + inputs["B"]
