# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Three operators read over one window each, by three products merged into one run, as the
viscoelastic derivative's kDivMT products are: the run forms over the operators' copies
(`Options.mma_operator_copies`), one per member, each in the order the run's one body is
offered, and the run selects them through a table the kernel builds -- while a short product
still reads the first operator as passed.

``D[0:35] (+)= A_d[0:35, 1:54] B_d[1:54]`` for d = 0, 1, 2, and ``D[0:10] += A_0[0:10, 1:9] E[1:9]``."""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "mma_operator_copies_run"
DTYPE = Datatype.F64
BATCH = 37
TOL = (1e-12, 1e-12)
OPTIONS = {"tensor_cores": True, "mma_operator_copies": True,
           "mma_chunked_order": True, "merge_variants": True}
OUTPUT = "D"

A_VALUES = [np.random.default_rng(70 + d).uniform(0.5, 1.5, size=(56, 56)) for d in range(3)]
INPUT_TRANSFORM = {f"A{d}": (lambda x, d=d: np.broadcast_to(A_VALUES[d], x.shape))
                   for d in range(3)}


def _strided(alias):
    return Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0], [56, 9]),
                  alias=alias, datatype=DTYPE)


def descr_list():
    a = [Tensor([56, 56], Addressing.NONE, BoundingBox([0, 0], [56, 56]),
                alias=f"A{d}", datatype=DTYPE, data=A_VALUES[d].copy()) for d in range(3)]
    b = [_strided(f"B{d}") for d in range(3)]
    e, d_ = _strided("E"), _strided("D")
    # The first overwrites: a destination only ever added onto is no source
    # to the harness, which then hands the kernel memory it never filled.
    out = [GemmDescr(False, False, SubTensor(a[k], BoundingBox([0, 1], [35, 54])),
                     SubTensor(b[k], BoundingBox([1, 0], [54, 9])),
                     SubTensor(d_, BoundingBox([0, 0], [35, 9])), alpha=1.0,
                     beta=0.0 if k == 0 else 1.0)
           for k in range(3)]
    out.append(GemmDescr(False, False, SubTensor(a[0], BoundingBox([0, 1], [10, 9])),
                         SubTensor(e, BoundingBox([1, 0], [9, 9])),
                         SubTensor(d_, BoundingBox([0, 0], [10, 9])), alpha=1.0, beta=1.0))
    return out


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    out[:, :35, :] = 0.0
    for k in range(3):
        out[:, :35, :] += np.einsum("ik,bkj->bij", A_VALUES[k][:35, 1:54],
                                    inputs[f"B{k}"][:, 1:54, :])
    out[:, :10, :] += np.einsum("ik,bkj->bij", A_VALUES[0][:10, 1:9], inputs["E"][:, 1:9, :])
    return out
