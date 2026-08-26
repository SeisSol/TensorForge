# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A register-resident intermediate sliced along the *lead* dimension.

Same chain as ``register_operand``, but the last GEMM takes rows 8..16 of
``tmp2`` instead of a column range.  ``tmp2`` enters as the first operand, so
dim 0 carries the lead index --- and the lead index is distributed across
lanes, element ``s`` living in lane ``s % T``.  With ``T = 32`` an offset of 8
is not a whole thread-block, so it cannot be applied to the register address:
the data thread 0 would want sits in lane 8.

What makes it work anyway is that ``n0`` is an internal loop variable with a
free origin.  The builder shifts it by 8 (`_lead_origin_shift`), which makes
the operand's effective offset zero --- the loop simply runs where the data
already is --- and pushes the compensating shift onto the destination, which
is global and takes any offset.  No shuffle, and here not even an extra
register slot, since [8,16) still lies inside one block.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "register_operand_lead_slice"
DTYPE = Datatype.F32
BATCH = 2
TOL = (1e-3, 1e-3)

N_SLICE = slice(8, 16)


def descr_list():
    a = SubTensor(Tensor([32, 32], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 32]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([32, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([32, 32], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 32]),
                         alias="C", datatype=DTYPE))
    d_ = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                          BoundingBox([0, 0], [16, 8]),
                          alias="D", datatype=DTYPE))
    q = SubTensor(Tensor([8, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [8, 8]),
                         alias="Q", datatype=DTYPE))

    tmp0 = SubTensor(generate_tmp_matrix(a, b))              # 32x16
    tmp1 = SubTensor(generate_tmp_matrix(c, b))              # 32x16
    tmp2_t = generate_tmp_matrix(tmp0, tmp1, trans_a=True)   # 16x16
    tmp2 = SubTensor(tmp2_t)
    # rows 8..16 only, stated as a logical [0,8) plus an offset of 8
    tmp2_slice = SubTensor(tmp2_t,
                           bbox=BoundingBox([0, 0], [8, 16]),
                           offset=[8, 0])

    return [
        GemmDescr(False, False, a=a, b=b, c=tmp0),
        GemmDescr(False, False, a=c, b=b, c=tmp1),
        GemmDescr(True,  False, a=tmp0, b=tmp1, c=tmp2),
        GemmDescr(False, False, a=tmp2_slice, b=d_, c=q, alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    t0 = np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    t1 = np.einsum("bik,bkj->bij", inputs["C"], inputs["B"])
    t2 = np.einsum("bki,bkj->bij", t0, t1)
    return np.einsum("bik,bkj->bij", t2[:, N_SLICE, :], inputs["D"])
