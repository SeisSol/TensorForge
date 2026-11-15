# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's order-6 volume in single precision on the stacked F32 matrix path.

``Q[1:r1] += kDivM_d[1:r1, :35] (I[:35] star_d)`` for three operators, at 16
lanes a multiplication with `Options.amd_matrix_k4`: every operator product
issues `mfma_f32_16x16x4f32` over two tiles of four columns, the ninth column
on the DPP chain, the four multiplications of a wave stacked along the
instruction's M -- the element major there, where the F64 instruction
(`stacked_volume_f64`) stacks it minor, so the lanes a fragment reads its
elements and columns from are the other way round.  Without the option the
products are the lane-batched `mfma_f32_4x4x1f32` tile's.

Thirty-nine elements: the last wave holds three, so a multiplication past the
batch shares the wave whose elements it reads.  Without the matrix core
(gfx1150, sm_120) the case is the DPP chain and the nest, and has to be right
there too.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "stacked_volume_f32"
OUTPUT = "Q"
DTYPE = Datatype.F32
BATCH = 39
TOL = (1e-4, 1e-4)
OPTIONS = {"lanes_per_mult": 16, "amd_matrix_k4": True}

_M, _K, _N = 56, 35, 9
#: The rows each operator writes, `[r0, r1)`, as SeisSol's `kDivM(d)` do.
WINDOWS = ((1, 54), (1, 55), (1, 56))


def _operators():
    rng = np.random.default_rng(31)
    out = []
    for d, (r0, r1) in enumerate(WINDOWS):
        a = np.zeros((_M, _M))
        a[r0:r1, :_K] = rng.uniform(-1.0, 1.0, (r1 - r0, _K))
        # The derivative's staircase: a row of a higher degree reaches the
        # basis functions of a lower one, so the blocks above it are zero.
        for r in range(_M):
            a[r, min(_K, 8 + 4 * (r // 8) + 4 * d):] = 0.0
        out.append(a)
    return out


OPERATORS = _operators()
#: The kernel reads the operators' box, which the harness hands over.
INPUT_TRANSFORM = {f"kDivM{d}": (lambda x, v=v: np.broadcast_to(
                       v[:x.shape[-2], :x.shape[-1]], x.shape))
                   for d, v in enumerate(OPERATORS)}


def descr_list():
    i = Tensor([_M, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_M, _N]),
               alias="I", datatype=DTYPE)
    q = Tensor([_M, _N], Addressing.PTR_BASED, BoundingBox([0, 0], [_M, _N]),
               alias="Q", datatype=DTYPE)
    descrs = []
    for d, (r0, r1) in enumerate(WINDOWS):
        star = Tensor([_N, _N], Addressing.PTR_BASED,
                      BoundingBox([0, 0], [_N, _N]), alias=f"star{d}",
                      datatype=DTYPE)
        t = Tensor([_M, _N], Addressing.STRIDED, BoundingBox([0, 0], [40, _N]),
                   alias=f"T{d}", datatype=DTYPE, is_tmp=True)
        k = Tensor([_M, _M], Addressing.NONE, BoundingBox([0, 0], [_M, _K]),
                   alias=f"kDivM{d}", datatype=DTYPE,
                   data=OPERATORS[d].copy())
        descrs.append(MultilinearDescr(
            dest=SubTensor(t, BoundingBox([0, 0], [_K, _N])),
            ops=[SubTensor(i, BoundingBox([0, 0], [_K, _N])), SubTensor(star)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]))
        descrs.append(MultilinearDescr(
            dest=SubTensor(q, BoundingBox([r0, 0], [r1, _N])),
            ops=[SubTensor(k, BoundingBox([r0, 0], [r1, _K])),
                 SubTensor(t, BoundingBox([0, 0], [_K, _N]))],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]], add=True))
    return descrs


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    for d, (r0, r1) in enumerate(WINDOWS):
        t = np.einsum("bik,bkj->bij", inputs["I"][:, :_K, :],
                      inputs[f"star{d}"])
        k = OPERATORS[d][r0:r1, :_K]
        out[:, r0:r1, :] += np.einsum("ik,bkj->bij", k, t)
    return out
