# SPDX-License-Identifier: MIT
"""``D = A0 B0 + A1 B1 + A2 B2 + A3 B3`` --- one write, three accumulations.

The shape of an ADER derivative or flux kernel: several descriptors whose
destination is the same tensor, the first with ``beta=0`` and the rest with
``beta=1``.

Two things used to go wrong here, and both are invisible in a single-writer
case.  ``_written_in_slices`` counted writers rather than boxes, so a chain
that writes the *same* box every time was classified as "assembled from
pieces" and stored out eagerly per term.  Each eager store then left the
register image ``_get_target_symbol`` had preloaded in place, so every later
``+=`` read the same stale bias --- the destination ended up holding the first
write plus the *last* term, with everything in between overwritten.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "accumulate_chain"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)         # four accumulated products

TERMS = 4
M, N, K = 12, 8, 12


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE))


def descr_list():
    d = _t([M, N], "D")
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=_t([M, K], f"A{k}"), b=_t([K, N], f"B{k}"), c=d,
                      alpha=1.0, beta=0.0 if k == 0 else 1.0)
            for k in range(TERMS)]


def reference(inputs, dest_in):
    total = np.zeros_like(dest_in)
    for k in range(TERMS):
        total = total + np.einsum("bik,bkj->bij", inputs[f"A{k}"],
                                  inputs[f"B{k}"])
    return total
