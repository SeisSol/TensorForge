# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = A B`` with a hop cascade and a tail that not every lane joins.

The other half of `wide_cascade`: 216 elements over 24 lanes leaves a
remainder, so after the whole hops at four, two and one there is a tail that
only some lanes copy.

That combination is what makes it worth a case of its own. A width decision
and a lane predicate are separately simple; together they are where an offset
computed in one unit meets a bound computed in the other, and the corpus had
no kernel where both happened to the same transfer.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "wide_cascade_tail"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, K = 24, 9, 24

#: What `yateto.py` sets when the layout reports an aligned stride.
ALIGNMENT = 16


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE,
                            alignment=ALIGNMENT))


def descr_list():
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=_t([M, K], "A"), b=_t([K, N], "B"),
                      c=_t([M, N], "D"),
                      alpha=1.0, beta=0.0)]


def reference(a, b):
    return a @ b
