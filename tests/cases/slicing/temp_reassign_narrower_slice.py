# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary assembled from two slices, one of them re-assigned from fewer rows.

    tmp[0:6,  :] = B1 @ C      (sliced, offset 0)
    tmp[6:12, :] = B2 @ C
    X            = tmp
    tmp[6:12, :] = N2 @ C      <- N2 has 2 rows: rows 8..12 of the slice are zero
    D            = tmp         == [B1 @ C ; N2 @ C ; 0]

The re-assignment owes zeros inside its own slice and nowhere else: the top
half has to survive it.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "temp_reassign_narrower_slice"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)
OUTPUT = "D"


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

    def top():
        return SubTensor(tmp, BoundingBox([0, 0], [6, 12]), [0, 0], sliced=True)

    def bottom():
        return SubTensor(tmp, BoundingBox([0, 0], [6, 12]), [6, 0])

    c = _g("C", (12, 12))
    return [_gemm(top(), _g("B1", (6, 12)), c),
            _gemm(bottom(), _g("B2", (6, 12)), c),
            _copy(_g("X", (12, 12)), SubTensor(tmp)),
            _gemm(bottom(), _g("N2", (2, 12)), c),
            _copy(_g("D", (12, 12)), SubTensor(tmp))]


def reference(inputs, dest_in):
    ein = lambda x, y: np.einsum("bik,bkj->bij", x, y)
    out = np.zeros((inputs["C"].shape[0], 12, 12), dtype=inputs["C"].dtype)
    out[:, 0:6] = ein(inputs["B1"], inputs["C"])
    out[:, 6:8] = ein(inputs["N2"], inputs["C"])
    return out
