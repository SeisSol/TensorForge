# SPDX-License-Identifier: MIT
"""Control for ``register_operand``: the same chain, nothing sliced.

``register_operand`` and ``register_operand_lead`` differ from this case in
exactly one place --- the last GEMM reads a sub-range of ``tmp2`` instead of
all of it.  Diffing the generated code of the two confirms that: the only
difference is the constant added to the register index.

So if this case fails too, the fault is in the chain at these shapes and not
in the slicing.  ``chain_five`` runs the identical four-GEMM pattern but with
56/9 dimensions, where the lead range fills the whole thread block; here the
lead range is 16 with T = 32, so half the lanes idle throughout.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "register_operand_control"
DTYPE = Datatype.F32
BATCH = 2
TOL = (1e-3, 1e-3)


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
    d_ = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                          BoundingBox([0, 0], [16, 8]),
                          alias="D", datatype=DTYPE))
    q = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="Q", datatype=DTYPE))

    tmp0 = SubTensor(generate_tmp_matrix(a, b))              # 32x16
    tmp1 = SubTensor(generate_tmp_matrix(c, b))              # 32x16
    tmp2 = SubTensor(generate_tmp_matrix(tmp0, tmp1, trans_a=True))   # 16x16

    return [
        GemmDescr(False, False, a=a, b=b, c=tmp0),
        GemmDescr(False, False, a=c, b=b, c=tmp1),
        GemmDescr(True,  False, a=tmp0, b=tmp1, c=tmp2),
        GemmDescr(False, False, a=tmp2, b=d_, c=q, alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    t0 = np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    t1 = np.einsum("bik,bkj->bij", inputs["C"], inputs["B"])
    t2 = np.einsum("bki,bkj->bij", t0, t1)
    return np.einsum("bik,bkj->bij", t2, inputs["D"])
