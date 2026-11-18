# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Inputs read through an index list that repeats cells, out of order.

``C[b] = A[b] @ B[b]``: ``A`` and ``B`` indexed through a list that names
some cells two and three times and skips others, ``C`` strided.  An input
may repeat (`Addressing.INDEXED`); every element reads its cell, and the
elements naming one cell compute from the same numbers -- which the harness
gives them (`runner.run_case`), so the reference is the plain product.
"""

import numpy as np

from tensorforge.common.basic_types import RUNTIME, Addressing, Datatype, indexed
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_indexed_inputs_repeat"
DTYPE = Datatype.F32
BATCH = 40
TOL = (1e-5, 1e-5)


def INDICES(batch):
    """Cells 2, 5, 5, 0, 9, 2, 2, ...: repeated, unsorted, with holes."""
    return (np.arange(batch) * 7 // 3) % 11


def descr_list():
    a = SubTensor(Tensor([12, 20], indexed(stride=RUNTIME),
                         BoundingBox([0, 0], [12, 20]), alias="A",
                         datatype=DTYPE))
    b = SubTensor(Tensor([20, 7], Addressing.INDEXED,
                         BoundingBox([0, 0], [20, 7]), alias="B",
                         datatype=DTYPE))
    c = SubTensor(Tensor([12, 7], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 7]), alias="C",
                         datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
