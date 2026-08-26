# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D[:, 8:9] = Q[:, 10:13] @ S[10:13, 8:9]`` --- written through a *view*.

The counterpart to :mod:`sliced_write`.  Both descriptors write a narrow box
of their destination, and they want opposite treatment:

* :mod:`sliced_write` has no slicing offset.  It addresses the tensor itself
  and its box is the eqspp window --- the range yateto knows the result can be
  nonzero in --- so everything outside is zero and the store has to say so.
* here the destination carries an offset, which makes it a slice of ``D`` with
  its own index space.  The rest of ``D`` belongs to whoever else writes it,
  and zero-filling it would destroy their work.  The poroelastic
  ``kernel_0bf208a83b`` writes ``m2`` this way thirteen times, once per column.

``D`` is read back afterwards, so anything the store touches outside its slice
shows up in the result rather than only in the generated source.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, MultilinearDescr

NAME = "sliced_write_view"
OUTPUT = "O"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, COL, KLO = 32, 13, 8, 10


def _t(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    q = _t([M, N], "Q")
    s = _t([N, N], "S")
    d = _t([M, N], "D")
    return [
        MultilinearDescr(
            dest=SubTensor(d, bbox=BoundingBox([0, 0], [M, 1]),
                           offset=[0, COL]),
            ops=[SubTensor(q, bbox=BoundingBox([0, KLO], [M, N])),
                 SubTensor(s, bbox=BoundingBox([KLO, 0], [N, 1]),
                           offset=[0, COL])],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]),
        GemmDescr(False, False, a=SubTensor(d), b=SubTensor(_t([N, N], "C")),
                  c=SubTensor(_t([M, N], "O")), alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    d = np.array(inputs["D"], copy=True)
    d[:, :, COL:COL + 1] = np.einsum(
        "bik,bkj->bij", inputs["Q"][:, :, KLO:N], inputs["S"][:, KLO:N, COL:COL + 1])
    return np.einsum("bik,bkj->bij", d, inputs["C"])
