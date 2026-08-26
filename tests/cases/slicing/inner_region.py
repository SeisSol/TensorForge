# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A[8:24, 8:24] @ B`` — an inner 16x16 window of a 32x32 operand.

Unlike ``offset_a``/``chain``, the window is not flush with either edge: the
offset is non-zero *and* the slice ends before the storage does.  A's bounding
box stays the full 32x32 (memory spans upper - lower, so the host buffer is
32x32 and nothing is compacted away); the window is expressed purely as a
slicing offset of ``[8, 8]`` on a logical bbox of ``[0,0]..[16,16]``.

Transposing A pushes it through the shared-memory staging path, where the
offset is *not* absorbed by the loader but has to survive onto the staged
symbol --- and where the staging buffer is padded against bank conflicts, so
its stride basis is the padded extent while the offset is still expressed in
the tensor's own coordinates.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_inner_16in32"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

A_SUB = (slice(8, 24), slice(8, 24))


def descr_list():
    a = SubTensor(Tensor([32, 32], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 32]),
                         alias="A", datatype=DTYPE),
                  bbox=BoundingBox([0, 0], [16, 16]),
                  offset=[8, 8])
    b = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["A"][:, *A_SUB], inputs["B"])
