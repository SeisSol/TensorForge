# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = alpha * A @ B + beta * C_in`` with ``alpha=1`` and ``beta=0.5``.

This is a deterministic reproducer for the silently-dropped ``beta``
argument: :class:`GemmDescr` accepts ``beta`` as a keyword argument
(``descriptions.py:135``) but never forwards it to the parent
:class:`MultilinearDescr`. The original assert (``# assert beta ==
0.0`` at line 144) was commented out rather than replaced with handling,
so callers can pass any value without warning and silently get
``alpha * A @ B`` back.

Promoting C to ``SOURCESINK`` is necessary for the same reason as in
``cases/add_true.py``: without a non-zero initial value the
``beta * C_in`` term contributes nothing and the bug becomes
invisible.

Marked XFAIL with strict=True. Once :class:`GemmDescr` either rejects
non-zero ``beta`` or forwards it correctly, the case turns into a hard
failure on xpass and prompts updating the marker.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, DataFlowDirection, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_beta_half"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
BETA = 0.5

XFAIL = True
XFAIL_REASON = (
    "GemmDescr accepts beta but does not forward it; descriptions.py:144 "
    "has a commented-out 'assert beta == 0'. Kernel computes alpha*A*B, "
    "reference computes alpha*A*B + beta*C_in — comparison fails."
)


def descr_list():
    a = SubTensor(Tensor([12, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([12, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 8]),
                         alias="C", datatype=DTYPE))
    descr = GemmDescr(trans_a=False, trans_b=False,
                      a=a, b=b, c=c, alpha=1.0, beta=BETA)
    # Same SOURCESINK promotion as cases/add_true.py — the test
    # has to give C a non-zero initial value for the beta term to be
    # observable at all.
    c.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)
    return [descr]


def reference(inputs, dest_in):
    return (np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
            + BETA * dest_in)
