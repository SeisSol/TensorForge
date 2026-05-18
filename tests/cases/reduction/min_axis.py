# SPDX-License-Identifier: MIT
"""``b[i] = min_j a[i, j]`` — min-reduction along axis 1.

Dual to ``max_axis``; tests that ``MinOperator`` survives the same path
``MaxOperator`` does. The neutral element differs (``+inf`` vs.\\
``-inf``) so the implementation has to thread the operator's
``neutral()`` value through correctly. ``MaxOperator.neutral()`` returns
``-math.inf``, ``MinOperator.neutral()`` returns ``math.inf`` — both
will need handling in F32 (``-inf``/``+inf`` representable).

XFAIL: same reason as the other reduction cases.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.operation import MinOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_min_axis1"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)

XFAIL = True
XFAIL_REASON = (
    "ReductionDescr / ReductionInstruction are scaffold-only."
)


def descr_list():
    a = Tensor([16, 16], Addressing.STRIDED,
               BoundingBox([0, 0], [16, 16]),
               alias="A", datatype=DTYPE)
    out = Tensor([16], Addressing.STRIDED,
                 BoundingBox([0], [16]),
                 alias="OUT", datatype=DTYPE)
    return [ReductionDescr(out, a, [1], MinOperator())]


def reference(inputs, dest_in):
    return np.min(inputs["A"], axis=2)
