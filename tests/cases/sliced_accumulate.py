# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = A B``, then ``D[:, 0:8] += A0 B0`` and ``D[:, 8:16] += A1 B1``.

Slicing and accumulation at the same time.  Each is handled on its own:
several writers covering different boxes must go out as they are produced,
because ``_deferred_stores`` holds one entry per symbol name; several writers
covering the same box are an accumulation chain and belong in registers.
Together they used to fall between the two.

On a vendor with atomic updates the accumulating writes took the
``can_use_atomic`` path, which *deferred* them --- so the second slice's
pending update displaced the first, and one term's contribution was computed
into a register array that nothing ever read.  Two writes reached memory where
three were produced.

The slices are on the second dimension deliberately.  On the lead dimension
they would pin the accumulator's origin (see
``MultilinearBuilder._lead_origin_shift``) against the operands' and be
refused, which is a separate and honest limitation.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "sliced_accumulate"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

M, N, K, HALF = 32, 16, 12, 8


def _t(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    d = _t([M, N], "D")
    out = [GemmDescr(False, False, a=SubTensor(_t([M, K], "A")),
                     b=SubTensor(_t([K, N], "B")), c=SubTensor(d),
                     alpha=1.0, beta=0.0)]
    for half in range(2):
        out.append(GemmDescr(
            False, False,
            a=SubTensor(_t([M, K], f"A{half}")),
            b=SubTensor(_t([K, HALF], f"B{half}")),
            c=SubTensor(d, bbox=BoundingBox([0, 0], [M, HALF]),
                        offset=[0, half * HALF]),
            alpha=1.0, beta=1.0))
    return out


def reference(inputs, dest_in):
    out = np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    for half in range(2):
        cols = slice(half * HALF, (half + 1) * HALF)
        out[:, :, cols] = out[:, :, cols] + np.einsum(
            "bik,bkj->bij", inputs[f"A{half}"], inputs[f"B{half}"])
    return out
