# SPDX-License-Identifier: MIT
"""``b[i] = sum_j a[i, j]`` — sum-reduction along axis 1, via ReductionDescr.

This is the same arithmetic as ``cases/trace.py``, which uses a bare
:class:`MultilinearDescr`. Here we use the dedicated
:class:`ReductionDescr` API so that the reduction path itself is
exercised — both descriptors should produce identical numerical
output once the reduction pipeline lights up.

XFAIL because:

* ``ReductionDescr.__init__`` (see ``descriptions.py:102``) accepts
  ``dims`` and ``op`` but stores neither — the reduction structure is
  lost on construction.
* ``ReductionInstruction.__init__`` (``reduction.py:9``) is
  literally ``pass``.
* The :class:`Generator` does not dispatch on ``ReductionDescr`` —
  ``isinstance`` checks only cover ``Multilinear`` and ``Elementwise``.

Construction signature: ``ReductionDescr(dest, var, dims, op)`` takes
bare :class:`Tensor` objects (the constructor calls
``set_data_flow_direction`` directly on them — see
``descriptions.py:108-109``). This is inconsistent with the rest of
the API, which uses :class:`SubTensor`; once the reduction pipeline
is implemented this signature is likely to shift. When that happens,
the test will need updating along with the harness's handling for
non-SubTensor operands.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_sum_axis1"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

XFAIL = True
XFAIL_REASON = (
    "ReductionDescr / ReductionInstruction are scaffold-only; the "
    "Generator does not dispatch on them (no isinstance branch in "
    "generator.py)."
)


def descr_list():
    a = Tensor([16, 16], Addressing.STRIDED,
               BoundingBox([0, 0], [16, 16]),
               alias="A", datatype=DTYPE)
    out = Tensor([16], Addressing.STRIDED,
                 BoundingBox([0], [16]),
                 alias="OUT", datatype=DTYPE)
    return [ReductionDescr(out, a, [1], AddOperator())]


def reference(inputs, dest_in):
    return np.sum(inputs["A"], axis=2)         # batch axis is 0; data axes 1, 2
