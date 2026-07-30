# SPDX-License-Identifier: MIT
"""``s = max_{i, j} a[i, j]`` — full (all-axis) reduction to a scalar.

This case stresses the boundary in two ways:

* the sink is rank zero (or, equivalently, a single-element tensor) —
  the generator's "what does the launcher emit for a 0-rank output?"
  question;
* both data axes are contracted, so the lead-dim heuristics in
  :class:`ReductionInstruction` cannot fall back to "iterate one axis
  while threading parallelize the other".

We pass a single-element output tensor (``shape=[1]``) rather than a
true 0-rank tensor — :class:`Tensor` allows any shape but the rest of
the pipeline assumes ``rank() >= 1`` in several places (see
``boundingbox.py`` whose ``rank()`` returns ``len(self._lower)``).
``BoundingBox([], [])`` works but is untested elsewhere; rank-1 with
a single element is the conservative choice.

XFAIL: same as the other reduction cases.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import MaxOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_max_all"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)

XFAIL = True
XFAIL_REASON = (
    "ReductionDescr / ReductionInstruction are scaffold-only; "
    "full-reduction shape (rank-0 sink) is an additional edge case."
)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
               BoundingBox([0, 0], [16, 16]),
               alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([1], Addressing.STRIDED,
                 BoundingBox([0], [1]),
                 alias="OUT", datatype=DTYPE))
    return [ReductionDescr(out, a, [0, 1], MaxOperator())]


def reference(inputs, dest_in):
    A = inputs["A"]
    return np.max(A.reshape(A.shape[0], -1), axis=1, keepdims=True)
