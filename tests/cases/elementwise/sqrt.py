# SPDX-License-Identifier: MIT
"""``B[i, j] = sqrt(A[i, j])`` — single-unary-op ElementwiseDescr.

This is the canonical smoke-test for the elementwise pipeline: one
:class:`Assignment`, one source, one sink, one unary nonlinear op.

Domain: ``sqrt`` requires non-negative input; ``INPUT_TRANSFORM`` maps
``standard_normal`` samples to ``|x| + 0.1`` before they hit either the
kernel or the reference.

Known issue exposed by this case:
``ElementwiseInstruction._assignment_loop`` constructs ``LeadLoop``
without the required ``stride`` argument (see
``backend/instructions/compute/elementwise.py:57`` vs.\\
``backend/symbol.py:LeadLoop.__init__``). Until that is fixed, every
elementwise case raises ``TypeError`` at generation time.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


NAME = "elementwise_sqrt_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

# Force non-negative input. The kernel receives the same bytes the
# reference sees, so this only changes the input distribution.
INPUT_TRANSFORM = {"A": lambda x: np.abs(x) + 0.1}


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.sqrt(b, a)]


def reference(inputs, dest_in):
    return np.sqrt(inputs["A"])
