# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = Q; D += F0; D += F1; D += F2`` --- each term over fewer rows.

The shape of SeisSol's ADER Taylor expansion, ``I = dQ(0) c_0`` followed by
``I += dQ(k) c_k``: every descriptor declares the whole destination, and
``_analyze`` narrows each accumulation to the rows its operand has.  The
assignment covers the whole box, so its register image holds everything and
each later term adds into it.  Held to its own rows instead, a term's image
would lose the others; the destination went through global memory on every
term to avoid that.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "accumulate_narrowing_chain"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N = 20, 9
#: The rows each accumulation reaches, fewer every time.
ROWS = (10, 4, 1)


def _tensor(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    d = _tensor([M, N], "D")
    copy = dict(target=[[0, 1]], permute=[[0, 1]])
    out = [MultilinearDescr(dest=SubTensor(d),
                            ops=[SubTensor(_tensor([M, N], "Q"))], **copy)]
    for i, rows in enumerate(ROWS):
        out.append(MultilinearDescr(
            dest=SubTensor(d), ops=[SubTensor(_tensor([rows, N], f"F{i}"))],
            add=True, **copy))
    return out


def reference(inputs, dest_in):
    d = np.array(inputs["Q"], copy=True)
    for i, rows in enumerate(ROWS):
        d[:, :rows, :] = d[:, :rows, :] + inputs[f"F{i}"]
    return d
