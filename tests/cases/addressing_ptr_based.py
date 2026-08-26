# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C[b] = A[b] @ B[b]`` with :data:`Addressing.PTR_BASED` operands.

Pointer-based addressing means the per-batch operands aren't laid out
contiguously: each batch element has its own buffer, and the kernel
receives a ``T**`` (array of base pointers, one per element). See
``ptr_manip.py:54-66`` — the address becomes
``&m1[batchId][sub_offset]`` rather than the STRIDED form
``&m1[batchId * volume + sub_offset]``.

Generation itself works on dev2 (the case constructs and the
:class:`Generator` emits code). The test driver, however, only knows
how to populate STRIDED and NONE buffers — see ``driver_emit.py:257``
where PTR_BASED hits a deliberate :class:`NotImplementedError`. To
make this case green, the harness needs:

1. Per-batch host allocations (``batch`` separate ``malloc`` calls
   rather than one large contiguous one).
2. A device-side ``T**`` array populated with the individual base
   pointers and passed in place of the current ``T*``.
3. Symmetric handling for reads and writes (the case below tests both
   SOURCE and SINK in PTR_BASED).
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
