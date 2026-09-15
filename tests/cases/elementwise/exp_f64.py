# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""F64 variant of ``elementwise/exp.py``.

ESIMD's ``exp`` takes float and half only; for double the SYCL lexic spells
``tensorforge::expF64``, a range reduction and a thirteenth-order polynomial
(``isycl.h``).  A regression that routed double through the float intrinsic
would be a compile error, and one that narrowed through float would lose
about nine digits -- the F64 tolerance below catches the second.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_exp_f64_16x16"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-13, 1e-13)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.exp(b, a)]


def reference(inputs, dest_in):
    return np.exp(inputs["A"])
