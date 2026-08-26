# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary written narrower than it is read back.

    tmp[:, 0:6] = B @ C          <- one writer, half the columns
    D           = A @ tmp        <- reads all 12 columns

The point is `_written_in_slices`. Deferring a store is right while one
operation writes the whole tensor: the value stays in registers and goes
straight to the next consumer. Here the single writer covers only half, so the
deferred registers hold half of what the consumer reads, and the rest is
whatever the buffer happened to contain.

No case in the corpus reaches either of that predicate's True branches --- it
returns False for every case, either because a single writer covers everything
or because no read union was recorded at all. So the branch that decides
"materialise this, do not keep it in registers" is exercised by nothing, while
it is on the hot path for any chain of yateto-style contractions.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "temp_partial_write"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

STORAGE = (32, 32)
WRITTEN_COLS = 6


def _t(bbox, alias):
    return Tensor(list(STORAGE), Addressing.STRIDED,
                  BoundingBox([0, 0], list(bbox)), alias=alias, datatype=DTYPE)


def descr_list():
    a = SubTensor(_t((12, 12), "A"))
    b = SubTensor(_t((12, 12), "B"))
    c = SubTensor(_t((12, 12), "C"))
    d = SubTensor(_t((12, 12), "D"))
    tmp = generate_tmp_matrix(b, c)
    half = SubTensor(tmp, bbox=BoundingBox([0, 0], [12, WRITTEN_COLS]),
                     offset=[0, 0])
    return [
        GemmDescr(False, False, a=b, b=c, c=half),
        GemmDescr(False, False, a=a, b=SubTensor(tmp), c=d,
                  alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    # The temporary is zero-initialised and only its first WRITTEN_COLS columns
    # are produced; the rest stay zero and must contribute nothing.
    tmp = np.zeros((inputs["B"].shape[0], 12, 12), dtype=inputs["B"].dtype)
    tmp[:, :, :WRITTEN_COLS] = np.einsum(
        "bik,bkj->bij", inputs["B"], inputs["C"])[:, :, :WRITTEN_COLS]
    return np.einsum("bik,bkj->bij", inputs["A"], tmp)
