# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A @ B`` where A and C share a memory box that starts above zero.

A bounding box lives in its tensor's own coordinates and says only what is
stored — it does not remap an index space.  So operands that share a target
index have to agree on the origin: here both A's and C's dim-0 box is
``[4,16)``, the contraction index runs 4..16, and no slicing offset is
involved anywhere.  ``address = index - lower`` puts row 4 at address 0 in
both buffers.

This is the counterpart to ``offset_a``, which reaches the same numbers by
disagreeing on the origin and reconciling it with an offset.  If the two
concepts ever get conflated again, exactly one of the two cases breaks —
which one tells you in which direction.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "bbox_shared_lower"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([4, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([4, 0], [16, 8]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    # both host buffers are already the stored 12-row blocks
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
