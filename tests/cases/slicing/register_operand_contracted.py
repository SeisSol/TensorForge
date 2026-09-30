# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A register-resident intermediate whose lead dimension is *contracted*.

The shape of SeisSol's space-time predictor at order 6: rows 35..54 of a
64-row intermediate ``x`` are multiplied into ``t``, and ``t`` is then
contracted over those rows against columns 35..54 of ``K``.

Writing ``t`` from the slice of ``x`` pins the lead loop's origin to the
slice's lane residue (`_lead_origin_shift`, 35 % 32 = 3), so ``t`` holds its
row ``l`` in lane ``(l + 3) % 32``.  The last GEMM reads ``t`` as its second
operand, where the lead dimension is the reduction ``k`` rather than the
output's lead ``n0`` -- and a vendor path that reads ``B`` a register slot at
a time would find the lanes three places off, which it cannot address. The
reduction has a free origin just like ``n0`` has; shifting it by the same
residue makes ``t``'s effective offset vanish and moves the compensating
shift onto ``K``, which lives in memory and takes any offset.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "register_operand_contracted_lead"
OUTPUT = "D"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

M, N, OFF, W = 64, 13, 35, 19
GEMM = dict(target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]])


def _t(shape, alias, is_tmp=False, addressing=Addressing.STRIDED):
    return Tensor(shape, addressing,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=is_tmp, datatype=DTYPE)


def descr_list():
    x = _t([M, N], "x", is_tmp=True)
    t = _t([W, N], "t", is_tmp=True)
    k = _t([M, 56], "K")
    rows = SubTensor(x, BoundingBox([0, 0], [W, N]), [OFF, 0], sliced=True)
    cols = SubTensor(k, BoundingBox([0, 0], [M, W]), [0, OFF], sliced=True)
    return [
        MultilinearDescr(dest=SubTensor(x),
                         ops=[SubTensor(_t([M, N], "A")), SubTensor(_t([N, N], "B0"))],
                         **GEMM),
        MultilinearDescr(dest=SubTensor(t),
                         ops=[rows, SubTensor(_t([N, N], "B"))], **GEMM),
        MultilinearDescr(dest=SubTensor(_t([M, N], "D")),
                         ops=[cols, SubTensor(t)], **GEMM),
    ]


def reference(inputs, dest_in):
    x = np.einsum("bik,bkj->bij", inputs["A"], inputs["B0"])
    t = np.einsum("bik,bkj->bij", x[:, OFF:OFF + W, :], inputs["B"])
    return np.einsum("bik,bkj->bij", inputs["K"][:, :, OFF:OFF + W], t)
