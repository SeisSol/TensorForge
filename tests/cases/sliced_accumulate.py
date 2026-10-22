# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = A B``, then ``D[:, 0:8] += A0 B0`` and ``D[:, 8:16] += A1 B1``.

Slicing and accumulation at the same time.  Each is handled on its own:
several writers covering different boxes each store their slice in its place;
several writers covering the same box are an accumulation chain, each adding
onto what the one before it stored.  Together they fall between the two.

On a vendor with atomic updates, an atomic path that held the accumulating
writes back, one per tensor, would let the second slice's update displace the
first, and one term's contribution would be computed into a register array
that nothing ever reads.  Two writes would reach memory where three are
produced.

The slices are on the second dimension deliberately.  On the lead dimension
an operand whose rows the accumulator computes in other lanes goes through
memory (``MultilinearBuilder._image_serves``), which is a path of its own,
and ``tests/test_canonical_lanes.py`` has it.
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
