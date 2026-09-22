# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``IA[:, :17] += I[:, :17]; IA[:, 17:] = max(IA[:, 17:], I[:, 17:])``.

SeisSol's damage ``accumulateIntegrals``: most columns are sums, and the last
two are bounds that fold by a maximum.  The maximum reads two *slices*, views
with a slicing offset of 17 columns, into a temporary that a multilinear then
copies back through a view with the same offset.

The elementwise operation used to index its operands by their box alone and
drop the offset, so it read columns 0 and 1 -- which the sum had just
overwritten -- while the store back applied the 17.  Every other entry came
out right, so only a check on the two bound columns saw it.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "elementwise_sliced_max"
OUTPUT = "IA"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)

M, N, CUT = 64, 19, 17


def _t(shape, alias, **kw):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE, **kw)


def descr_list():
    ia = _t([M, N], "IA")
    i = _t([M, N], "I")
    t = _t([M, N - CUT], "t", is_tmp=True)
    head = BoundingBox([0, 0], [M, CUT])
    tail = BoundingBox([0, 0], [M, N - CUT])
    return [
        MultilinearDescr(SubTensor(ia, head), [SubTensor(i, head)],
                         [[0, 1]], [[0, 1]], add=True),
        ew.max(SubTensor(t, tail),
               SubTensor(ia, tail, [0, CUT]),
               SubTensor(i, tail, [0, CUT])),
        MultilinearDescr(SubTensor(ia, tail, [0, CUT]), [SubTensor(t, tail)],
                         [[0, 1]], [[0, 1]]),
    ]


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    out[:, :, :CUT] += inputs["I"][:, :, :CUT]
    out[:, :, CUT:] = np.maximum(dest_in[:, :, CUT:], inputs["I"][:, :, CUT:])
    return out
