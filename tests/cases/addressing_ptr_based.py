# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C[b] = A[b] @ B[b]`` with :data:`Addressing.PTR_BASED` operands.

Pointer-based addressing means the per-batch operands aren't laid out
contiguously: each batch element has its own buffer, and the kernel
receives a ``T**`` (array of base pointers, one per element). The
address becomes ``&m1[batchId][sub_offset]`` rather than the STRIDED
form ``&m1[batchId * volume + sub_offset]`` (``ptr_manip.py``).

The test driver allocates the operand as one block, builds a device-side
``T**`` of per-element base pointers into it, and passes that in place of
a ``T*`` (``driver_emit.py``).  Reads and writes both go through it: the
case has a PTR_BASED source and a PTR_BASED sink.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_ptr_based"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.PTR_BASED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.PTR_BASED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 16], Addressing.PTR_BASED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
