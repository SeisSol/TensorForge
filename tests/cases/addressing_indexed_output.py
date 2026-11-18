# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A destination written through the index list, its inputs read through
the same one.

``C[b] = A[b] @ B[b]`` with all three indexed.  Writing through the list is
only defined where no two entries name one cell (`Addressing.INDEXED`), so
the list is a permutation of the cells -- reversed, so that element ``n``
lands in another cell than a strided write would put it in.  ``A`` reads
64 bytes into its cell and ``B`` steps by a stride of its own.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype, indexed
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_indexed_output"
DTYPE = Datatype.F64
BATCH = 24
TOL = (1e-10, 1e-10)


def INDICES(batch):
    """A permutation of the cells, reversed: no cell twice."""
    return np.arange(batch)[::-1].copy()


def descr_list():
    box = BoundingBox([0, 0], [9, 9])
    a = SubTensor(Tensor([9, 9], indexed(offset=8), box, alias="A",
                         datatype=DTYPE))
    b = SubTensor(Tensor([9, 9], indexed(stride=96), box, alias="B",
                         datatype=DTYPE))
    c = SubTensor(Tensor([9, 9], Addressing.INDEXED, box, alias="C",
                         datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
