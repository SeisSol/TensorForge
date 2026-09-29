# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An output assigned anew, pointwise, from an operand narrower than it.

    D = A @ B           <- defines all 12 rows
    D = abs(N)          <- N has rows 4..8 only, and the view is the tensor
                           itself: rows 4..8 get values, the rest zeros

The pointwise operation wrote its four rows of `D` in place and nothing else,
so the other eight kept the product.  The window starts past row 0, so the
zeros go on both sides of it.  A contraction assigning `D` from `N` zero-
fills them (`StoreRegToGlb`, `zero_fill`); the pointwise result now goes out
through the same store (`OperationBuilder.pointwise_dest`).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

NAME = "mixed_ml_then_ew_output_narrower"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)
LO, HI = 4, 8


def _t(bbox, alias, lower=(0, 0)):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox(list(lower), list(bbox)), alias=alias,
                  datatype=DTYPE)


def descr_list():
    a = SubTensor(_t((12, 12), "A"))
    b = SubTensor(_t((12, 12), "B"))
    n = SubTensor(_t((HI, 12), "N", lower=(LO, 0)))
    d = _t((12, 12), "D")
    return [
        GemmDescr(False, False, a=a, b=b, c=SubTensor(d)),
        ew.abs(SubTensor(d, BoundingBox([LO, 0], [HI, 12])), n),
    ]


def reference(inputs, dest_in):
    out = np.zeros((inputs["A"].shape[0], 12, 12), dtype=inputs["A"].dtype)
    out[:, LO:HI, :] = np.abs(inputs["N"])
    return out
