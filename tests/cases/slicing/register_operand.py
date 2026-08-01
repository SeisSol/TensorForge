# SPDX-License-Identifier: MIT
"""A register-resident intermediate consumed through a slice.

Four chained GEMMs, the same shape as ``chain_five``: the last operand
``tmp2`` is produced into registers and read back from there, which is the
one configuration where an operand reaches the compute site as
``SymbolType.Register``.  Here it is read through a slicing offset.

``tmp2`` enters the last GEMM as its first operand, so dim 0 carries the lead
index and dim 1 is the contraction index.  This case slices dim 1 --- a
non-lead dimension, where the offset is a plain constant in the address and
any value is representable.  ``register_operand_lead`` covers the other
dimension, where only whole thread-blocks work.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "register_operand_nonlead_slice"
DTYPE = Datatype.F32
BATCH = 2
TOL = (1e-3, 1e-3)         # four chained GEMMs amplify FP error

K_SLICE = slice(8, 16)


def descr_list():
    a = SubTensor(Tensor([32, 32], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 32]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([32, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([32, 32], Addressing.STRIDED,
                         BoundingBox([0, 0], [32, 32]),
                         alias="C", datatype=DTYPE))
    d_ = SubTensor(Tensor([8, 8], Addressing.STRIDED,
                          BoundingBox([0, 0], [8, 8]),
                          alias="D", datatype=DTYPE))
    q = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="Q", datatype=DTYPE))

    tmp0 = SubTensor(generate_tmp_matrix(a, b))          # 32x16
    tmp1 = SubTensor(generate_tmp_matrix(c, b))          # 32x16
    tmp2_t = generate_tmp_matrix(tmp0, tmp1, trans_a=True)   # 16x16
    tmp2 = SubTensor(tmp2_t)
    # same tensor, read through columns 8..16 only
    tmp2_slice = SubTensor(tmp2_t,
                           bbox=BoundingBox([0, 0], [16, 8]),
                           offset=[0, 8])

    return [
        GemmDescr(False, False, a=a, b=b, c=tmp0),
        GemmDescr(False, False, a=c, b=b, c=tmp1),
        GemmDescr(True,  False, a=tmp0, b=tmp1, c=tmp2),
        GemmDescr(False, False, a=tmp2_slice, b=d_, c=q, alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    t0 = np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    t1 = np.einsum("bik,bkj->bij", inputs["C"], inputs["B"])
    t2 = np.einsum("bki,bkj->bij", t0, t1)
    return np.einsum("bik,bkj->bij", t2[:, :, K_SLICE], inputs["D"])
