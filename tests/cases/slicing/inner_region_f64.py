# SPDX-License-Identifier: MIT
"""F64 variant of ``slicing/inner_region.py``.

Verifies bbox arithmetic at F64. The slicing pipeline uses the same
``bbox.lower()``-based offsets regardless of dtype; this case catches
elem-size-dependent regressions in the kernel emitter (e.g.\\
``glb_m0[batchId * 1024 + 0 + ...]`` literal that hard-codes
``volume * 4`` somewhere downstream).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "slice_inner_16in32_f64"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)

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
    out = np.array(dest_in, copy=True)
    A_sub = inputs["A"][:, *SUB]
    B_sub = inputs["B"][:, *SUB]
    out[:, *SUB] = np.einsum("bik,bkj->bij", A_sub, B_sub)
    return out
