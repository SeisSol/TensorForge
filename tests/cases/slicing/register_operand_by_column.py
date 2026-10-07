# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A second operand held a row per lane and read by column, from column 6 on.

``Q = M0 @ T[6:9, 6:9]^T``: the product of SeisSol's free-surface kernel at
order 6 that takes the velocity block of ``Tinv``.  The generator holds the
small block of ``T`` in registers, a row per lane.  The product reads it
transposed: its reduction ``k`` runs along a row of ``T``, across the columns,
from column 6 on.

Every vendor path reads the second operand across the lanes by the
reduction: lane ``s`` reads the element at ``k = s``.  In a register image
spread by rows, that element is in the registers of another lane, at an index
that differs from lane to lane, which no shuffle reaches; a shift by 6 moves
the data between the lanes besides.  The product goes to the nest, which reads
``T`` an element at a time.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "register_operand_by_column"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)

ROWS, OFF, W = 21, 6, 3
# Q[i, j] = sum_k M0[i, k] * T[j, k]: the second operand enters transposed
PRODUCT = dict(target=[[0, -1], [1, -1]], permute=[[0, 1], [0, 1]])


def descr_list():
    m0 = Tensor([32, W], Addressing.PTR_BASED, BoundingBox([0, 0], [32, W]),
                alias="M0", datatype=DTYPE)
    t = Tensor([9, 9], Addressing.PTR_BASED, BoundingBox([0, 0], [9, 9]),
               alias="T", datatype=DTYPE)
    q = Tensor([32, W], Addressing.PTR_BASED, BoundingBox([0, 0], [32, W]),
               alias="Q", datatype=DTYPE)
    block = SubTensor(t, BoundingBox([0, 0], [W, W]), [OFF, OFF], sliced=True)
    return [
        MultilinearDescr(dest=SubTensor(q, BoundingBox([0, 0], [ROWS, W])),
                         ops=[SubTensor(m0, BoundingBox([0, 0], [ROWS, W])), block],
                         **PRODUCT),
    ]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    out[:, :ROWS, :] = np.einsum("bik,bjk->bij", inputs["M0"][:, :ROWS, :],
                                 inputs["T"][:, OFF:OFF + W, OFF:OFF + W])
    return out
