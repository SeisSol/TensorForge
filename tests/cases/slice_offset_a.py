# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A[4:16, :] @ B`` — expressed as a *slicing offset*, not a bbox.

``offset_a`` states the same computation the other way round: there, A's
storage bounding box starts at row 4, so the index space is global and the
address arithmetic subtracts 4.  Here A's stored region is the full 32×16 and
the operand is a ``SubTensor`` whose logical bbox is ``[0,0]..[12,16]`` with
offset ``[4, 0]`` — the loop index runs 0..12 and 4 is added on the way to the
address.

The two must agree numerically, which makes this the direct test that the
offset never leaks into a loop range and never goes missing from an address.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_offset_a_via_offset"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

A_STORAGE = (32, 16)
A_SUB = (slice(4, 16), slice(0, 16))


def descr_list():
    a = SubTensor(Tensor(list(A_STORAGE), Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 16]),
                         alias="A", datatype=DTYPE),
                  bbox=BoundingBox([0, 0], [12, 16]),
                  offset=[4, 0])
    b = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([12, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 8]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    A_sub = inputs["A"][:, *A_SUB]
    return np.einsum("bik,bkj->bij", A_sub, inputs["B"])
