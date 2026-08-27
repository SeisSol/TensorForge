# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``b[i] = max_j a[i, j]`` — max-reduction along axis 1.

The case that is *not* expressible as a :class:`MultilinearDescr`:
``max`` is not a multilinear operator. This is the genuine reason for
``ReductionDescr`` to exist alongside the multilinear path.

It is also the case that pins the neutral element down to a spelling:
``MaxOperator.neutral()`` is ``-math.inf``, and an input that happens
to be entirely negative is the only thing that separates a correct
``-INFINITY`` from a wrong ``numeric_limits<float>::min()``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import MaxOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_max_axis1"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)         # max is exact when implemented correctly



def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
               BoundingBox([0, 0], [16, 16]),
               alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([16], Addressing.STRIDED,
                 BoundingBox([0], [16]),
                 alias="OUT", datatype=DTYPE))
    return [ReductionDescr(out, a, [1], MaxOperator())]


def reference(inputs, dest_in):
    return np.max(inputs["A"], axis=2)
