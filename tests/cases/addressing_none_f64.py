# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""F64 variant of ``addressing_none.py``.

The :data:`Addressing.NONE` device path emits a different pointer
offset (no ``batchId * volume``); the F64 variant catches a regression
where the dtype mismatches the elem-size literal in the offset
expression (``ptr_manip.py`` doesn't use ``sizeof(T)``, so the
generator relies on the C++ pointer arithmetic doing the right thing
— a bug there would be silent for F32 if the literal happened to
match).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_addressing_none_operator_f64"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.NONE,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
