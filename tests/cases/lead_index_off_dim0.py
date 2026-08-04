# SPDX-License-Identifier: MIT
"""``C[i,j] = sum_k A[k,i] * B[k,j]`` with the contraction on the lead axis.

The transposition lives in ``target`` and ``permute`` is the identity ---
the convention yateto emits, and the one :class:`GemmDescr` was corrected to
in ``fix: fix target/permute mismatch``.  Operand ``A`` therefore carries the
destination's lead index on its *second* dimension while its first, of extent
one, carries the contraction index.

That is the shape that used to be miscompiled: the register staging spread
dimension 0 across lanes regardless, so the lane-distributed index ended up in
the register axis, ``Symbol.load`` saw a loop constant on what it believed was
the lane axis, and every lane got element ``[0,0]`` broadcast from lane 0 ---
nineteen of the twenty input values were discarded.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "lead_index_off_dim0"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE))


def descr_list():
    a = _t([1, 20], "A")
    b = _t([1, 9], "B")
    c = _t([20, 9], "C")
    return [MultilinearDescr(dest=c, ops=[a, b],
                             target=[[-1, 0], [-1, 1]],
                             permute=[[0, 1], [0, 1]])]


def reference(inputs, dest_in):
    return np.einsum("bki,bkj->bij", inputs["A"], inputs["B"])
