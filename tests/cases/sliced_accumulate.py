# SPDX-License-Identifier: MIT
"""``D[0:8] = A0 B0; D[0:8] += A1 B1; D[8:16] = A2 B2; D[8:16] += A3 B3``

Slicing and accumulation at the same time.  Each is handled on its own:
several writers covering different boxes must go out as they are produced,
because ``_deferred_stores`` holds one entry per symbol name; several writers
covering the same box are an accumulation chain and belong in registers.
Together they used to fall between the two.

On a vendor with atomic updates the accumulating writes took the
``can_use_atomic`` path, which *deferred* them --- so the second slice's
pending update displaced the first, and one term's contribution was computed
into a register array that nothing ever read.  Three writes reached memory
where four were produced.
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

M, N, K, HALF = 16, 8, 12, 8


def _t(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    d = _t([M, N], "D")
    out = []
    for half in range(2):
        dest = SubTensor(d, bbox=BoundingBox([0, 0], [HALF, N]),
                         offset=[half * HALF, 0])
        for term in range(2):
            out.append(GemmDescr(
                False, False,
                a=SubTensor(_t([HALF, K], f"A{half}{term}")),
                b=SubTensor(_t([K, N], f"B{half}{term}")),
                c=dest, alpha=1.0, beta=0.0 if term == 0 else 1.0))
    return out


def reference(inputs, dest_in):
    out = np.zeros_like(dest_in)
    for half in range(2):
        rows = slice(half * HALF, (half + 1) * HALF)
        acc = np.zeros_like(out[:, rows, :])
        for term in range(2):
            acc = acc + np.einsum("bik,bkj->bij", inputs[f"A{half}{term}"],
                                  inputs[f"B{half}{term}"])
        out[:, rows, :] = acc
    return out
