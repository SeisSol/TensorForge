# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Four products into one output, merged into one loop, on the stacked F64
matrix path with the products chained.

``Q += K_f (I S_f)`` for four faces `f`, the operators `K_f` batch-constant
and `S_f` per element -- SeisSol's `localFluxAll` in its order-4 shape.  The
operators come without values: dense ones are read from their buffers, which
the harness fills, so the reference takes them from there.  The
runs are merged (`Options.merge_variants`), so the four products into `Q`
are one product in a loop's body, and the chain (`Options.mma_d_layout`,
`mma_d_layout_loops`) spans the loop's trips: `Q` read into the accumulators
ahead of the loop, every trip issuing into them, `Q` written behind it
(`dchain.DChain.loop`).

Thirty-nine elements: the last wave holds three.  Without the matrix core
(gfx1150, sm_120) the case is the DPP chain and the nest, merged, and has to
be right there too.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "stacked_merged_chain_f64"
OUTPUT = "Q"
DTYPE = Datatype.F64
BATCH = 39
TOL = (1e-11, 1e-11)
OPTIONS = {"lanes_per_mult": 16, "amd_matrix_k4": True, "mma_d_layout": True,
           "mma_d_layout_loops": True, "merge_variants": True}

_M, _K, _N = 20, 10, 9
FACES = 4


def descr_list():
    i = Tensor([_M, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_M, _N]),
               alias="I", datatype=DTYPE)
    q = Tensor([_M, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_M, _N]),
               alias="Q", datatype=DTYPE)
    descrs = []
    for f in range(FACES):
        s = Tensor([_N, _N], Addressing.PTR_BASED,
                   BoundingBox([0, 0], [_N, _N]), alias=f"S{f}",
                   datatype=DTYPE)
        t = Tensor([_K, _N], Addressing.STRIDED, BoundingBox([0, 0], [_K, _N]),
                   alias=f"T{f}", datatype=DTYPE, is_tmp=True)
        k = Tensor([_M, _K], Addressing.NONE, BoundingBox([0, 0], [_M, _K]),
                   alias=f"K{f}", datatype=DTYPE)
        descrs.append(MultilinearDescr(
            dest=SubTensor(t), ops=[SubTensor(i, BoundingBox([0, 0], [_K, _N])),
                                    SubTensor(s)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]))
        descrs.append(MultilinearDescr(
            dest=SubTensor(q), ops=[SubTensor(k), SubTensor(t)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]], add=True))
    return descrs


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    for f in range(FACES):
        t = np.einsum("bik,bkj->bij", inputs["I"][:, :_K, :],
                      inputs[f"S{f}"])
        out += np.einsum("ik,bkj->bij", inputs[f"K{f}"][0], t)
    return out
