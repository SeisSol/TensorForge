# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``b[i] = sum_j a[i, j]`` — sum-reduction along axis 1, via ReductionDescr.

This is the same arithmetic as ``cases/trace.py``, which uses a bare
:class:`MultilinearDescr`. Here we use the dedicated
:class:`ReductionDescr` API so that the reduction path itself is
exercised — both descriptors should produce identical numerical
output, which is what makes the pair worth keeping.

The contracted axis is not the lead axis, so this lowers to the
register-local fold: each lane owns one ``i`` and folds ``j`` into an
accumulator, with no cross-lane traffic.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators.descriptions import ReductionDescr

NAME = "reduction_sum_axis1"
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
    return [ReductionDescr(out, a, [1], AddOperator())]


def reference(inputs, dest_in):
    return np.sum(inputs["A"], axis=2)         # batch axis is 0; data axes 1, 2
