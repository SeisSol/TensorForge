# SPDX-License-Identifier: MIT
"""``b[i] = sum_j a[i, j]`` — partial trace via raw MultilinearDescr.

Not a GEMM: the descriptor is a bare ``MultilinearDescr`` where the
destination has rank 1 and we contract one of the two axes of A.
Adapted from ``example/trace.py`` with shape simplified to a clean
power of two.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "trace_partial"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    out = SubTensor(Tensor([16], Addressing.STRIDED,
                           BoundingBox([0], [16]),
                           alias="OUT", datatype=DTYPE))
    # target=[[0, -1]] keeps axis 0, sums over axis 1.
    return [MultilinearDescr(out, [a], [[0, -1]], [[0, 1]])]


def reference(inputs, dest_in):
    A = inputs["A"]
    return np.einsum("bij->bi", A)
