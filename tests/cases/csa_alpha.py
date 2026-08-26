# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = alpha * A @ B`` — exercises the synthetic-scalar operand path.

This is the regression test for the symbol/datatype fix:

* Before fix: ``Symbol.get_fptype()`` raised ``assert False`` because
  ``GemmDescr`` constructed the synthetic ``alpha`` tensor without a
  datatype.
* After fix: the launcher gains a ``float`` parameter for alpha, the
  driver bakes the constant in, and the kernel multiplies by it.
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_alpha_9x9"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def descr_list():
    a = SubTensor(Tensor([9, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [9, 9]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([9, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [9, 9]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([9, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [9, 9]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=a, b=b, c=c, alpha=13.0, beta=0.0)]
