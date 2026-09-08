# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = A B`` whose staged operand needs every width to cover it.

`aligned_operands` is the only other case that promises an aligned stride,
and its extent divides evenly: 128 elements over 16 lanes is two hops of four
and nothing else. So the cascade in `GlbToShrLoader._write_datatransfer` --
whole hops at four, then at two, then at one -- was reachable in principle
and never generated. This case makes it happen: 176 elements over 16 lanes is
two hops of four, one of two, one of one, and no remainder.

No predicated tail here, deliberately. The extent divides by the lane count,
so every lane copies in every hop, and what is under test is the width
arithmetic alone. `wide_cascade_tail` adds the lanes dropping out.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "wide_cascade"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, K = 16, 11, 16

#: What `yateto.py` sets when the layout reports an aligned stride.
ALIGNMENT = 16


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE,
                            alignment=ALIGNMENT))


def descr_list():
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=_t([M, K], "A"), b=_t([K, N], "B"),
                      c=_t([M, N], "D"),
                      alpha=1.0, beta=0.0)]


def reference(a, b):
    return a @ b
