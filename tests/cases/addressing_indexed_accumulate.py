# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An accumulation into a destination reached through the index list.

``C[b] += A[b] @ B[b]``, ``C`` :data:`Addressing.INDEXED_POINTER` -- read
and written through a cell's pointer -- over the harness's permutation of
the batch, so every cell is written once.  ``A`` is batch-constant, ``B``
indexed with an offset into its cell.  Two elements of one cell would add
up neither sum completely, which is why the list a destination is written
through has to name every cell once.
"""

import numpy as np

from tensorforge.common.basic_types import (Addressing, Datatype, indexed,
                                            indexed_pointer)
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_indexed_accumulate"
DTYPE = Datatype.F32
BATCH = 50
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([20, 9], Addressing.NONE,
                         BoundingBox([0, 0], [20, 9]), alias="A",
                         datatype=DTYPE))
    b = SubTensor(Tensor([9, 9], indexed(offset=12),
                         BoundingBox([0, 0], [9, 9]), alias="B",
                         datatype=DTYPE))
    c = SubTensor(Tensor([20, 9], indexed_pointer(slots=2, slot=1),
                         BoundingBox([0, 0], [20, 9]), alias="C",
                         datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=1.0)]


def reference(inputs, dest_in):
    return dest_in + np.einsum("ik,bkj->bij", inputs["A"][0], inputs["B"])
