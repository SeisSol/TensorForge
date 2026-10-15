# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An interleaved operand whose last slot is ragged.

`prepare_operands` stores a batch-constant ``A`` in the SIMT interleave: each
lane's rows side by side, so it reads them in one vector instead of one load
each.  40 rows over 32 lanes is two slots, the second holding rows 32..39 --
so its read is held to eight lanes, and the other twenty-four reach the
order's padding, which is stored and is zero.

That tail is the case worth pinning.  A read carrying a lane bound used to
decline the interleaved address and fall back to the logical one, which for
a buffer the host wrote interleaved is not slower but wrong: `known_zero_rows`
read row ``lane + 32`` of a 40-row column out of a buffer that is 64 wide.
Three corpus kernels were off by a factor of eighty, invisibly, because the
option was off by default.

Stated as its own case, with the option named, so the coverage does not
depend on what that default happens to be.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "interleave_ragged_tail"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"prepare_operands": True}
#: 40 rows over 32 lanes: two slots, the second eight rows deep.
M, K, N = 40, 8, 6


def _values():
    rng = np.random.default_rng(41)
    # Every row carries a number, so a wrong address cannot be hidden by a
    # zero standing where the right one would have been.
    return rng.uniform(0.5, 1.5, size=(M, K))


A_VALUES = _values()
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE, data=A_VALUES.copy()))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
