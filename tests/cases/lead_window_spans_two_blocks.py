# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D[20:35, 12] += s * t[20:35, 12]`` --- a window across two register blocks.

With 32 lanes a lead window of 15 rows starting at row 20 covers lanes 20..31
of one register block and lanes 0..2 of the next, so the accumulator needs two
slots per remaining index.  ``_analyze`` works that out and the store walks
both blocks; ``_alloc_register_array`` sized the array for one, because it
added theta to a box that already carried it --- the bias image is staged in
the tensor's own lead coordinates.  The store then read past the end of the
register array, which on a GPU is whatever register happens to follow.

Order 4 never showed this: with 32 or fewer rows every window fell inside a
single block, and the double-counted theta cancelled.  Order 6 has 64 rows.

Nothing in the emitted numbers gives it away either --- the host interpreter
does not enforce array bounds, so this needs the structural check that
compares a store's indices against the declared length.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "lead_window_spans_two_blocks"
OUTPUT = "D"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, T = 64, 13, 6
THETA, ROWS, COL = 20, 15, 12


def _t(shape, alias, is_tmp=False, addressing=Addressing.PTR_BASED):
    return Tensor(shape, addressing,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=is_tmp, datatype=DTYPE)


def descr_list():
    t = _t([M, N, T], "t", is_tmp=True, addressing=Addressing.STRIDED)
    d = _t([M, N, T], "D")

    def window(tensor):
        return SubTensor(tensor, BoundingBox([0, 0, 0], [ROWS, 1, T]),
                         [THETA, COL, 0], sliced=True)

    return [
        MultilinearDescr(dest=SubTensor(t),
                         ops=[SubTensor(_t([M, N], "A")),
                              SubTensor(_t([T], "v", addressing=Addressing.NONE))],
                         target=[[0, 1], [2]], permute=[[0, 1], [0]]),
        MultilinearDescr(dest=window(d), ops=[window(t)],
                         target=[[0, 1, 2]], permute=[[0, 1, 2]], add=True),
    ]


def reference(inputs, dest_in):
    t = np.einsum("bij,bk->bijk", inputs["A"],
                  np.asarray(inputs["v"]).reshape(inputs["A"].shape[0], -1))
    d = np.array(inputs["D"], copy=True)
    rows = slice(THETA, THETA + ROWS)
    d[:, rows, COL:COL + 1] += t[:, rows, COL:COL + 1]
    return d
