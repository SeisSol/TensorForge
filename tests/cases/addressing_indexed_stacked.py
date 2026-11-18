# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Indexed operands in a stack of two elements (`Options.stack_elements`).

``C[b] = A[b] @ B[b]`` with ``A`` indexed, ``B`` reached through an indexed
pointer, ``C`` indexed with a stride of its own: each element of a stack
reads its own entry of the index list, and a stack the batch ends inside
reads its first element's (`GetElementPtr._gen_stacked`).  An odd batch
ends inside one.
"""

import numpy as np

from tensorforge.common.basic_types import (Addressing, Datatype, indexed,
                                            indexed_pointer)
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_indexed_stacked"
DTYPE = Datatype.F32
BATCH = 33
TOL = (1e-5, 1e-5)
OPTIONS = {"stack_elements": 2}


def descr_list():
    box = BoundingBox([0, 0], [16, 8])
    a = SubTensor(Tensor([16, 16], Addressing.NONE,
                         BoundingBox([0, 0], [16, 16]), alias="A",
                         datatype=DTYPE))
    b = SubTensor(Tensor([16, 8], indexed_pointer(slots=3, slot=2), box,
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 8], indexed(stride=136, offset=4), box,
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("ik,bkj->bij", inputs["A"][0], inputs["B"])
