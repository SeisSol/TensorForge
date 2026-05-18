# SPDX-License-Identifier: MIT
"""Square 16×16 GEMM contained inside 32×32 storage.

This is the cleanest slicing case: storage = 32×32 for every operand,
bbox = 16×16 inside, starting at lower index ``(8, 8)``. ``alpha=1``,
``beta=0`` so the result equals ``A_sub @ B_sub`` inside the bbox and
stays zero everywhere else.

The harness already ships full-storage buffers (driver_emit derives
``shape`` and ``volume`` from ``Tensor.shape``, not from the bbox), so
the host-side setup needs no changes. ``reference()`` is just expected
to put the result in the right window and leave the rest as the
sink's initial value (zero, for a SINK-only operand).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_inner_16in32"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

# Storage is 32×32; bbox is 16×16 starting at (8, 8). One window, same
# for all operands, so the GEMM contracts a 16-element inner dim.
STORAGE = (32, 32)
LO, HI = (8, 8), (24, 24)
SUB = (slice(LO[0], HI[0]), slice(LO[1], HI[1]))


def descr_list():
    def t(alias):
        return SubTensor(Tensor(list(STORAGE), Addressing.STRIDED,
                                BoundingBox(list(LO), list(HI)),
                                alias=alias, datatype=DTYPE))
    return [GemmDescr(False, False, t("A"), t("B"), t("C"),
                      alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)        # preserves non-bbox cells
    A_sub = inputs["A"][:, *SUB]
    B_sub = inputs["B"][:, *SUB]
    out[:, *SUB] = np.einsum("bik,bkj->bij", A_sub, B_sub)
    return out
