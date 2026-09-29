# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A slice nothing reads, then the whole temporary assigned from a narrower operand.

    tmp[6:12, :] = B2 @ C      <- dead: the next line defines every cell
    tmp          = N  @ C      <- N has rows 0..6: rows 6..12 are zeros
    tmp[6:12, :] = Y
    D            = tmp         == [N @ C ; Y]

The assignment clears what it promised and did not compute (`StoreRegToShr`,
`clear_within`), which makes it a whole write of the buffer, so the first slice
is a store whose value nothing reads.  It still writes its bytes: recorded only
from the live-out, its buffer was live nowhere at that store, the allocator laid
`tmp` over the staged `C` -- read by the very next multiplication -- and the
dead slice overwrote it (`LivenessAnalysis._forward`).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "temp_dead_slice_reassign"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


def _g(alias, shape):
    return SubTensor(Tensor(list(shape), Addressing.STRIDED,
                            BoundingBox([0, 0], list(shape)),
                            alias=alias, datatype=DTYPE))


def _gemm(dest, a, b):
    return MultilinearDescr(dest=dest, ops=[a, b], target=[[0, -1], [-1, 1]],
                            permute=[[0, 1], [0, 1]])


def _copy(dest, a):
    return MultilinearDescr(dest=dest, ops=[a], target=[[0, 1]],
                            permute=[[0, 1]])


def descr_list():
    tmp = Tensor([12, 12], Addressing.PTR_BASED, BoundingBox([0, 0], [12, 12]),
                 alias="T", is_tmp=True, datatype=DTYPE)

    def bottom():
        return SubTensor(tmp, BoundingBox([0, 0], [6, 12]), [6, 0])

    c, d = _g("C", (12, 12)), _g("D", (12, 12))
    return [_gemm(bottom(), _g("B2", (6, 12)), c),
            _gemm(SubTensor(tmp), _g("N", (6, 12)), c),
            _copy(bottom(), _g("Y", (6, 12))),
            _copy(d, SubTensor(tmp))]


def reference(inputs, dest_in):
    top = np.einsum("bik,bkj->bij", inputs["N"], inputs["C"])
    return np.concatenate([top, inputs["Y"]], axis=1)
