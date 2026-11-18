# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C[b] = A[b] @ B[b]`` with every kind of address an element can take.

* ``A`` is :data:`Addressing.INDEXED` with a stride bound at run time: the
  launch's index list names the cell each element reads, and the cells are
  the driver's stride apart (`driver_emit.RUNTIME_STRIDE_PAD`).
* ``B`` is :data:`Addressing.INDEXED_POINTER` over four pointers a cell, the
  slot bound at run time, sixteen bytes into what the pointer points at: a
  face's neighbor reached through a per-cell pointer variable.
* ``C`` is strided with a stride and an offset of its own -- a member of a
  struct of 300 floats, eight into it.

The index list is the harness's permutation of the batch
(`runner.run_case`), so a kernel that read element ``n`` where the list
names another cell fails.
"""

import numpy as np

from tensorforge.common.basic_types import (RUNTIME, Datatype, indexed,
                                            indexed_pointer, strided)
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_indexed"
DTYPE = Datatype.F32
BATCH = 37
TOL = (1e-5, 1e-5)


def descr_list():
    box = BoundingBox([0, 0], [16, 16])
    a = SubTensor(Tensor([16, 16], indexed(stride=RUNTIME), box, alias="A",
                         datatype=DTYPE))
    b = SubTensor(Tensor([16, 16],
                         indexed_pointer(slots=4, slot=RUNTIME, offset=4),
                         box, alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 16], strided(stride=300, offset=8), box,
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
