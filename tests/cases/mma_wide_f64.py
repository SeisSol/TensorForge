# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One batch-constant operator read by three products on NVIDIA's matrix path at 64 lanes a
multiplication, two warps each (`nvidia.WIDE_WARPS`): rows 0 to 31 are the first warp's, 32 to
63 the second's, each warp running the instructions for its own block of rows, `B` staged once
for both behind a barrier of the multiplication and `D` back through a tile of each warp.  The
operator comes as copies in the order over both warps' lanes (`_wide_fragment_order`); the
short product stays off the path.

``D[0:35] = A[0:35, 1:54] B[1:54]``, ``D[0:10] += A[0:10, 1:9] E[1:9]``,
``D[0:20] += A[0:20, 1:33] F[1:33]`` -- the windows of orders 1, 4 and 2 at order 6."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_wide_f64"
DTYPE = Datatype.F64
BATCH = 37
TOL = (1e-12, 1e-12)
OPTIONS = {"tensor_cores": True, "mma_operator_copies": True,
           "mma_chunked_order": True, "lanes_per_mult": 64}
OUTPUT = "D"
#: Several waves a multiplication: NVIDIA's alone (`test_kernel`).
BACKENDS = ("cuda",)

#: Every cell a number, so a copy read out of order cannot hide behind a zero.
A_VALUES = np.random.default_rng(61).uniform(0.5, 1.5, size=(56, 56))
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def _strided(alias, rows):
    return Tensor([rows, 9], Addressing.STRIDED, BoundingBox([0, 0], [rows, 9]),
                  alias=alias, datatype=DTYPE)


def descr_list():
    a = Tensor([56, 56], Addressing.NONE, BoundingBox([0, 0], [56, 56]),
               alias="A", datatype=DTYPE, data=A_VALUES.copy())
    b, e, f, d = (_strided("B", 56), _strided("E", 56), _strided("F", 56),
                  _strided("D", 56))
    return [
        GemmDescr(False, False, SubTensor(a, BoundingBox([0, 1], [35, 54])),
                  SubTensor(b, BoundingBox([1, 0], [54, 9])),
                  SubTensor(d, BoundingBox([0, 0], [35, 9])), alpha=1.0, beta=0.0),
        GemmDescr(False, False, SubTensor(a, BoundingBox([0, 1], [10, 9])),
                  SubTensor(e, BoundingBox([1, 0], [9, 9])),
                  SubTensor(d, BoundingBox([0, 0], [10, 9])), alpha=1.0, beta=1.0),
        GemmDescr(False, False, SubTensor(a, BoundingBox([0, 1], [20, 33])),
                  SubTensor(f, BoundingBox([1, 0], [33, 9])),
                  SubTensor(d, BoundingBox([0, 0], [20, 9])), alpha=1.0, beta=1.0),
    ]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    out[:, :35, :] = np.einsum("ik,bkj->bij", A_VALUES[:35, 1:54], inputs["B"][:, 1:54, :])
    out[:, :10, :] += np.einsum("ik,bkj->bij", A_VALUES[:10, 1:9], inputs["E"][:, 1:9, :])
    out[:, :20, :] += np.einsum("ik,bkj->bij", A_VALUES[:20, 1:33], inputs["F"][:, 1:33, :])
    return out
