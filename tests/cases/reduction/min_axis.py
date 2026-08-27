# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``b[i] = min_j a[i, j]`` — min-reduction along axis 1.

Dual to ``max_axis``; tests that ``MinOperator`` survives the same path
``MaxOperator`` does. The neutral element differs (``+inf`` vs.\\
``-inf``), so a reduction that threads ``neutral()`` through correctly
for one sign and not the other fails exactly one of this pair.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import MinOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_min_axis1"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)



def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
               BoundingBox([0, 0], [16, 16]),
               alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([16], Addressing.STRIDED,
                 BoundingBox([0], [16]),
                 alias="OUT", datatype=DTYPE))
    return [ReductionDescr(out, a, [1], MinOperator())]


def reference(inputs, dest_in):
    return np.min(inputs["A"], axis=2)
