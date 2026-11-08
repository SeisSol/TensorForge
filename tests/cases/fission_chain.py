# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A chain cut in two runs of the batch (`Options.fission`).

``X = A (B C)`` is an operand of the kernel, written; ``D = A (X C)`` reads it
back.  The one point where no temporary is live is between the two halves,
so `fission = 2` cuts there: the second run reads ``X`` from memory where the
first stored it, every block over the same elements, ordered by the block
barrier at the head of the second run.  ``D`` holds both halves.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "fission_chain"
DTYPE = Datatype.F32
BATCH = 37
#: Four products of 56-step sums: values near 3e3, rounding near 1e-3 absolute.
TOL = (1e-4, 5e-3)
OPTIONS = {"fission": 2}
OUTPUT = "D"


def _t(rows, cols, alias):
    return SubTensor(Tensor([rows, cols], Addressing.STRIDED,
                            BoundingBox([0, 0], [rows, cols]),
                            alias=alias, datatype=DTYPE))


def descr_list():
    a, b, c = _t(56, 56, "A"), _t(56, 9, "B"), _t(9, 9, "C")
    x, d = _t(56, 9, "X"), _t(56, 9, "D")
    t1 = SubTensor(generate_tmp_matrix(b, c))
    t2 = SubTensor(generate_tmp_matrix(x, c))
    return [
        GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=t1),
        GemmDescr(trans_a=False, trans_b=False, a=a, b=t1, c=x,
                  alpha=1.0, beta=0.0),
        GemmDescr(trans_a=False, trans_b=False, a=x, b=c, c=t2),
        GemmDescr(trans_a=False, trans_b=False, a=a, b=t2, c=d,
                  alpha=1.0, beta=0.0),
    ]


def reference(inputs, dest_in):
    A, B, C = inputs["A"], inputs["B"], inputs["C"]
    x = np.einsum("bik,bkj->bij", A, np.einsum("bik,bkj->bij", B, C))
    return np.einsum("bik,bkj->bij", A, np.einsum("bik,bkj->bij", x, C))
