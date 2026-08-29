# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``tmp = A @ B`` then ``C = abs(tmp)`` — a multilinear feeding an elementwise.

The shape a plasticity kernel has: a contraction produces stresses, a pointwise
operation evaluates a yield criterion on them.

`MultilinearBuilder` leaves the result of the contraction in a register array
and records a pending writeback, and `Generator._emit_ir` hands the elementwise
a `SymbolView` on the shared-memory symbol that writeback has not reached yet.
The residency is private to the builder, so the elementwise cannot ask where
the newest copy is.

Shapes are 8x8 throughout: the elementwise descriptor pins the lane count to the
vector unit length regardless of the tensors, so nothing is gained by going
smaller, and 8x8 keeps a snapshot diff readable.

`abs` is the pointwise operation because it is total and exact -- no input
domain to shape, no tolerance spent on a transcendental, and every backend has
it.  ESIMD, for one, has no `tanh` intrinsic at all.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
N = 8


def _t(alias, shape=(N, N), tmp=False):
    return Tensor(list(shape),
                  Addressing.PTR_BASED if tmp else Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=tmp, datatype=DTYPE)


def _s(alias, shape=(N, N), tmp=False):
    return SubTensor(_t(alias, shape, tmp))


NAME = "mixed_ml_then_ew"


def descr_list():
    a, b, c = _s("A"), _s("B"), _s("C")
    tmp = _s("TMP", tmp=True)
    return [GemmDescr(False, False, a=a, b=b, c=tmp),
            ew.abs(c, tmp)]


def reference(inputs, dest_in):
    return np.abs(np.einsum("bik,bkj->bij", inputs["A"], inputs["B"]))
