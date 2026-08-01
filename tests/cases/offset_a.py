# SPDX-License-Identifier: MIT
"""``C = A_sub @ B`` — both shifts active on A at once, and cancelling.

A's *memory* bounding box is ``[4,16) x [0,16)``: the buffer holds 12x16
reals and address 0 is row 4.  That box lives in A's own coordinates, so
A's index space is 4..16 while C's is 0..12 — and a bounding box does not
remap an index space, it only says what is stored.  Contracting the two
against a shared ``n0`` therefore needs the slicing offset to reconcile
the origins: A gets a logical bbox of ``[0,12)`` and offset 4.

The two shifts then run in opposite directions and cancel exactly:
``address = index - lower + offset = n0 - 4 + 4 = n0``.  That makes this
the case where conflating them is invisible in the address but fatal in
the loop range, which is why it is worth having.

``slice_offset_a`` is the same computation with the other knob only (full
storage box, pure offset); ``bbox_shared_lower`` is the third quadrant
(memory box, no offset, consistent across operands).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_offset_a"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

# A: declared 32x16, memory box 12x16 starting at row 4.  The host buffer
# is that 12x16 block, so there is nothing to slice host-side.
A_STORAGE = (32, 16)
A_LO, A_HI = (4, 0), (16, 16)


def descr_list():
    a = SubTensor(Tensor(list(A_STORAGE), Addressing.STRIDED,
                         BoundingBox(list(A_LO), list(A_HI)),
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
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
