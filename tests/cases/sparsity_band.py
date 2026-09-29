# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``C = A @ B`` with a banded sparsity pattern on B.

B carries a :class:`MaskSPP` mask: the cells with ``|i - j| <= 1``
are non-zero, everything else is structurally zero. This triggers the
sparsity-aware code path in :mod:`multilinear` (cf.\\
``multilinear.py:111`` where ``_sparseN`` is set, and ``:432`` where
the k-loop is unrolled for sparse operands).

Host-side treatment: B is stored compressed, 46 cells per batch
element rather than 256, in the order ``Tensor.linear_index``
assigns. The harness packs the dense view into that buffer through
the operand's ``pack_index``, so the structural zeros are never
written and never read. ``INPUT_TRANSFORM`` still zeroes them in the
dense view, which is what makes the plain dense reference below the
right answer to compare against.

The reference is a plain dense GEMM: since the masked-out cells of B
are zero, ``A @ B_masked`` equals what the sparse kernel computes.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import MaskSPP
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "gemm_sparse_band_B"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

# Banded mask: keep cells where |i - j| <= 1 (tridiagonal block).
_SIZE = 16
_MASK = np.zeros((_SIZE, _SIZE), dtype=bool, order="F")
for _i in range(_SIZE):
    for _j in range(_SIZE):
        if abs(_i - _j) <= 1:
            _MASK[_i, _j] = True


def _apply_band_mask(x):
    """Zero out cells where the mask is false. ``x`` is ``(batch, 16, 16)``."""
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
    # B has been zeroed outside the band by INPUT_TRANSFORM, so the
    # dense matmul gives the same result as the sparsity-aware kernel.
    return np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
