# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""F64 variant of ``sparsity_band.py``.

The sparsity-aware compute path emits unrolled multiply-adds whose
literal-zero handling is dtype-dependent (``0.0f`` vs ``0.0``); this
case catches a regression that would only show up in F64 sparsity.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import MaskSPP
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_sparse_band_B_f64"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)

_SIZE = 16
_MASK = np.zeros((_SIZE, _SIZE), dtype=bool, order="F")
for _i in range(_SIZE):
    for _j in range(_SIZE):
        if abs(_i - _j) <= 1:
            _MASK[_i, _j] = True


def _apply_band_mask(x):
    return x * _MASK


INPUT_TRANSFORM = {"B": _apply_band_mask}


def descr_list():
    a = SubTensor(Tensor([_SIZE, _SIZE], Addressing.STRIDED,
                         BoundingBox([0, 0], [_SIZE, _SIZE]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([_SIZE, _SIZE], Addressing.STRIDED,
                         BoundingBox([0, 0], [_SIZE, _SIZE]),
                         alias="B", datatype=DTYPE,
                         spp=MaskSPP(_MASK)))
    c = SubTensor(Tensor([_SIZE, _SIZE], Addressing.STRIDED,
                         BoundingBox([0, 0], [_SIZE, _SIZE]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
