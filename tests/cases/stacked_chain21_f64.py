# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's local flux chain at order 6, on the stacked F64 path.

``R += rDivM (fMrT (I F))``: the face operators reduce 56 basis functions to
21 and back, so the first operator product has 21 rows -- two slots at
sixteen lanes, the second mostly past the window -- and the second contracts
over those 21 steps, ending inside a tile of four.  Both are batch-constant
products the stacked path takes (`Options.amd_matrix_k4`); the first reads
the element's own data out of shared memory or gathers it, the second the
first's result.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "stacked_chain21_f64"
OUTPUT = "R"
DTYPE = Datatype.F64
BATCH = 19
TOL = (1e-11, 1e-11)
OPTIONS = {"lanes_per_mult": 16, "amd_matrix_k4": True}

_M, _F, _N = 56, 21, 9


def _operators():
    rng = np.random.default_rng(23)
    fmrt = rng.uniform(-1.0, 1.0, (_F, _M))
    rdivm = rng.uniform(-1.0, 1.0, (_M, _F))
    rdivm[0:16, 12:21] = 0.0
    return fmrt, rdivm


FMRT, RDIVM = _operators()
INPUT_TRANSFORM = {"fMrT": lambda x: np.broadcast_to(FMRT, x.shape),
                   "rDivM": lambda x: np.broadcast_to(RDIVM, x.shape)}


def descr_list():
    i = Tensor([_M, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_M, _N]),
               alias="I", datatype=DTYPE)
    f = Tensor([_N, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_N, _N]),
               alias="F", datatype=DTYPE)
    r = Tensor([_M, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_M, _N]),
               alias="R", datatype=DTYPE)
    fmrt = Tensor([_F, _M], Addressing.NONE, BoundingBox([0, 0], [_F, _M]),
                  alias="fMrT", datatype=DTYPE, data=FMRT.copy())
    rdivm = Tensor([_M, _F], Addressing.NONE, BoundingBox([0, 0], [_M, _F]),
                   alias="rDivM", datatype=DTYPE, data=RDIVM.copy())
    t = Tensor([_M, _N], Addressing.STRIDED, BoundingBox([0, 0], [_M, _N]),
               alias="T", datatype=DTYPE, is_tmp=True)
    u = Tensor([_F, _N], Addressing.STRIDED, BoundingBox([0, 0], [_F, _N]),
               alias="U", datatype=DTYPE, is_tmp=True)
    target = [[0, -1], [-1, 1]]
    permute = [[0, 1], [0, 1]]
    return [
        MultilinearDescr(dest=SubTensor(t), ops=[SubTensor(i), SubTensor(f)],
                         target=target, permute=permute),
        MultilinearDescr(dest=SubTensor(u), ops=[SubTensor(fmrt), SubTensor(t)],
                         target=target, permute=permute),
        MultilinearDescr(dest=SubTensor(r), ops=[SubTensor(rdivm), SubTensor(u)],
                         target=target, permute=permute, add=True),
    ]


def reference(inputs, dest_in):
    t = np.einsum("bik,bkj->bij", inputs["I"], inputs["F"])
    u = np.einsum("ik,bkj->bij", FMRT, t)
    return dest_in + np.einsum("ik,bkj->bij", RDIVM, u)
