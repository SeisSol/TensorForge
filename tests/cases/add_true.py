# SPDX-License-Identifier: MIT
"""``C += sum_k A[i, k] * B[k, j]`` — bare :class:`MultilinearDescr`
with ``add=True``.

This is the only case in the suite that exercises the accumulator
branch in :mod:`tensorforge.backend.instructions.builders.multilinear_builder`
(``self._add`` gates two distinct code paths at lines 216 and 269):
register-array sizing reads from the *destination's existing*
``data_view._bbox`` rather than ``dest_obj.bbox``, and the store-back
loads the previous value of C via ``_get_target_symbol(True)``.

Why not :class:`GemmDescr`? Because the ``GemmDescr.__init__`` call to
``super().__init__`` (descriptions.py:147 / :158) is positional, and the
parent's signature is ``(dest, ops, target, permute, add=False,
strict_match=False, ...)`` — so :class:`GemmDescr` accidentally passes
``strict_match`` into the ``add`` slot. Every ``GemmDescr`` therefore
has ``add == strict_match`` (in practice ``False``). To actually test
``add=True`` we have to drop down to a bare :class:`MultilinearDescr`.

The destination direction is promoted to ``SOURCESINK`` after
construction. Without this, ``MultilinearDescr.__init__`` would force
``DataFlowDirection.SINK`` (descriptions.py:26) and the harness would
zero-initialize C — at which point ``C += A*B`` and ``C = A*B``
become numerically indistinguishable and the case wouldn't actually
exercise the accumulator. ``set_data_flow_direction(SOURCE)`` after
the descr sets it to ``SINK`` triggers the auto-promotion path in
``tensor.py:51-55``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, DataFlowDirection, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "gemm_add_accumulate"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


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
    descr = MultilinearDescr(
        dest=c, ops=[a, b],
        target=[[0, -1], [-1, 1]],
        permute=[[0, 1], [0, 1]],
        add=True,
    )
    # Promote C from SINK to SOURCESINK so the harness gives the kernel
    # a non-zero initial value to accumulate into. Without this, the
    # accumulator path can't actually be observed.
    c.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)
    return [descr]


def reference(inputs, dest_in):
    return dest_in + np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
