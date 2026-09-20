# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A1 B1; C += A2 B2`` with fewer destination rows than lanes.

Ten rows over sixteen or thirty-two lanes, two products accumulated, both
operators batch-constant and both right-hand operands staged into registers
and read across the lanes.  Nothing about the shapes is unusual --- it is the
order-4 time derivative's first step with the windows taken out --- and that
is the point: the combination is what the Intel device compiler miscompiles.

The operator is read under a lane guard, and predicating that *load* is what
the Intel device compiler carries backwards: it concludes the lanes above the
tenth produce nothing, drops their share of the register fill, and the
broadcasts read exactly those lanes.  The contraction then loses the steps
whose source lane lies between the active count and the vector width ---
silently, and only with the optimizer on.  `split_predicated_load` is the
remedy, and this is what says whether it still works.

One product, or as many rows as lanes, computes correctly either way; both
variants were measured before this case was written down as the narrowest one
that fails.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "tw_split_load"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


def descr_list():
    a1 = SubTensor(Tensor([10, 17], Addressing.NONE, BoundingBox([0, 0], [10, 17]),
                          alias="A1", datatype=DTYPE))
    b1 = SubTensor(Tensor([17, 9], Addressing.STRIDED, BoundingBox([0, 0], [17, 9]),
                          alias="B1", datatype=DTYPE))
    a2 = SubTensor(Tensor([10, 17], Addressing.NONE, BoundingBox([0, 0], [10, 17]),
                          alias="A2", datatype=DTYPE))
    b2 = SubTensor(Tensor([17, 9], Addressing.STRIDED, BoundingBox([0, 0], [17, 9]),
                          alias="B2", datatype=DTYPE))
    c = Tensor([10, 9], Addressing.STRIDED, BoundingBox([0, 0], [10, 9]),
               alias="C", datatype=DTYPE)
    return [GemmDescr(False, False, a1, b1, SubTensor(c), alpha=1.0, beta=0.0),
            GemmDescr(False, False, a2, b2, SubTensor(c), alpha=1.0, beta=1.0)]


def reference(inputs, dest_in):
    return (np.einsum("Bik,bkj->bij", inputs["A1"], inputs["B1"])
            + np.einsum("Bik,bkj->bij", inputs["A2"], inputs["B2"]))
