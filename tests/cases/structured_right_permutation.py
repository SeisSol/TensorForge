# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = X S`` with a batch-constant ``S`` that permutes the columns of ``X``.

`skip_known_zeros` on an operand whose numbers are in the code: ``S`` is
small enough to be taken as literals (`inline_constants`), and every zero
of it used to be multiplied in as ``x * 0.0f``, which a compiler may not
fold.  Unweighted, so a term is a multiply by one, which is exact and
folds -- the product is a relabelling of registers and no arithmetic at all.

The kernel has to see the numbers the generator was told, so the input for
``S`` is replaced by them (`INPUT_TRANSFORM`); the reference reads the same.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "structured_right_permutation"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
M, K = 24, 16


def _values():
    rng = np.random.default_rng(11)
    perm = rng.permutation(K)
    s = np.zeros((K, K))
    for col in range(K):
        s[perm[col], col] = 1.0
    return s


S_VALUES = _values()
INPUT_TRANSFORM = {"S": lambda x: np.broadcast_to(S_VALUES, x.shape)}


def descr_list():
    x = SubTensor(Tensor([M, K], Addressing.STRIDED, BoundingBox([0, 0], [M, K]),
                         alias="X", datatype=DTYPE))
    s = SubTensor(Tensor([K, K], Addressing.NONE, BoundingBox([0, 0], [K, K]),
                         alias="S", datatype=DTYPE, data=S_VALUES.copy()))
    c = SubTensor(Tensor([M, K], Addressing.STRIDED, BoundingBox([0, 0], [M, K]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, x, s, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,Bkj->bij", inputs["X"], inputs["S"])
