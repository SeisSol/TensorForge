# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``E = P @ (Q @ (P @ B))`` -- two batch-constant operators, read in turn.

SeisSol's derivative reads its three stiffness operators so: one product
each, then the three again on the derivative just computed, down to the last
order.  At order 8 in single precision the three are 140 kB, past a block's
shared memory, and what does not fit for the whole batch loop can be copied
in a product at a time instead (`Options.stream_shards`,
`pir.shards._rest`).  ``P`` here is read twice with ``Q`` between, so it is
two uses, and the room the uses share is overwritten between them.

``C`` and ``D`` are written too, and only ``E`` compared.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "operators_in_turn"
OUTPUT = "E"
DTYPE = Datatype.F32
BATCH = 24
TOL = (1e-4, 1e-4)


def _matrix(alias, addressing):
    return SubTensor(Tensor([16, 16], addressing,
                            BoundingBox([0, 0], [16, 16]),
                            alias=alias, datatype=DTYPE))


def descr_list():
    p = _matrix("P", Addressing.NONE)
    q = _matrix("Q", Addressing.NONE)
    b, c, d, e = (_matrix(x, Addressing.STRIDED) for x in "BCDE")
    return [GemmDescr(False, False, p, b, c, alpha=1.0, beta=0.0),
            GemmDescr(False, False, q, c, d, alpha=1.0, beta=0.0),
            GemmDescr(False, False, p, d, e, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    c = np.einsum("Bik,bkj->bij", inputs["P"], inputs["B"])
    d = np.einsum("Bik,bkj->bij", inputs["Q"], c)
    return np.einsum("Bik,bkj->bij", inputs["P"], d)
