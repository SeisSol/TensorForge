# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = A @ (B @ C)`` — a two-step GEMM chain.

The chain forces the generator to allocate a temporary inside the
kernel; nothing about the harness side changes since both temporaries
and intermediate results stay on-device.

Note: the original ``example/three_matrices.py`` uses ``beta=1.0``,
which currently has no effect in the generator (beta is silently
dropped from the kernel — separate bug). We keep ``beta=0.0`` here
so the case validates an actual chain rather than masking a known
arithmetic error.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "chain_three_matrices"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)         # chains accumulate FP rounding more than a single GEMM


def descr_list():
    d = SubTensor(Tensor([56, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 9]),
                         alias="D", datatype=DTYPE))
    a = SubTensor(Tensor([56, 56], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 56]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([56, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 9]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([9, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [9, 9]),
                         alias="C", datatype=DTYPE))
    tmp_t = generate_tmp_matrix(b, c)
    tmp = SubTensor(tmp_t)
    return [
        GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=tmp),
        GemmDescr(trans_a=False, trans_b=False,
                  a=a, b=tmp, c=d, alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    A, B, C = inputs["A"], inputs["B"], inputs["C"]
    return np.einsum("bik,bkj->bij", A, np.einsum("bik,bkj->bij", B, C))
