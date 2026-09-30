# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``O[i] = B[i]``, then ``O[i] += 3 * 2`` --- a term whose operands are all scalars.

A multilinear moves every rank-0 operand into its scalar factor, so a term made
of scalars alone has no tensor operand left.  What such a term accumulates is
the empty product, the product's neutral element.  An accumulator left at the
sum's neutral element would never be written, and the epilogue would multiply
that zero by the scalars: every broadcast of a scalar onto an index would come
out zero, and SeisSol's damage step would lose ``1 - B`` (to ``-B``) and
``2 * mu0``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "scalar_broadcast"
OUTPUT = "O"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-6, 1e-6)

M = 32
A, B = 3.0, 2.0


def _t(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def _literal(value):
    return SubTensor(tensor=Tensor([], Addressing.SCALAR, data=np.array(value)))


def descr_list():
    b = _t([M], "B")
    out = _t([M], "O")
    return [
        MultilinearDescr(SubTensor(out), [SubTensor(b)], [[0]], [[0]]),
        MultilinearDescr(SubTensor(out), [_literal(A), _literal(B)], [[], []], [[], []],
                         add=True),
    ]


def reference(inputs, dest_in):
    return inputs["B"] + A * B
