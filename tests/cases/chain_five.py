# SPDX-License-Identifier: MIT
"""``Q = (((A @ B) ^ T @ (C @ B)) @ D)`` — four chained GEMMs.

Adapted from ``example/five_multiplies.py``. Demonstrates that the
generator can fuse a non-trivial chain (one of the GEMMs is
transposed, others are plain) into a single kernel.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "chain_five_multiplies"
DTYPE = Datatype.F32
BATCH = 2
TOL = (1e-3, 1e-3)         # four chained GEMMs amplify FP error noticeably


def descr_list():
    q = SubTensor(Tensor([9, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [9, 9]),
                         alias="Q", datatype=DTYPE))
    a = SubTensor(Tensor([56, 56], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 56]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([56, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 9]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([56, 56], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 56]),
                         alias="C", datatype=DTYPE))
    d_ = SubTensor(Tensor([9, 9], Addressing.STRIDED,
                          BoundingBox([0, 0], [9, 9]),
                          alias="D", datatype=DTYPE))

    tmp0 = SubTensor(generate_tmp_matrix(a, b))
    tmp1 = SubTensor(generate_tmp_matrix(c, b))
    tmp2 = SubTensor(generate_tmp_matrix(tmp0, tmp1, trans_a=True))

    return [
        GemmDescr(False, False, a=a, b=b, c=tmp0),
        GemmDescr(False, False, a=c, b=b, c=tmp1),
        GemmDescr(True,  False, a=tmp0, b=tmp1, c=tmp2),
        GemmDescr(False, False, a=tmp2, b=d_, c=q, alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    A, B, C, D = inputs["A"], inputs["B"], inputs["C"], inputs["D"]
    AB = np.einsum("bik,bkj->bij", A, B)
    CB = np.einsum("bik,bkj->bij", C, B)
    ABT_CB = np.einsum("bki,bkj->bij", AB, CB)         # tmp0^T @ tmp1
    return np.einsum("bik,bkj->bij", ABT_CB, D)
