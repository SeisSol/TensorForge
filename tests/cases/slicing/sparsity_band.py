# SPDX-License-Identifier: MIT
"""``C = A @ B`` with a banded sparsity pattern on B.

B carries a :class:`MaskSPP` mask: the cells with ``|i - j| <= 1``
are non-zero, everything else is structurally zero. This triggers the
sparsity-aware code path in :mod:`multilinear` (cf.\\
``multilinear.py:111`` where ``_sparseN`` is set, and ``:432`` where
the k-loop is unrolled for sparse operands).

Host-side treatment: the buffer is still 256 cells (full storage —
``Tensor.get_real_volume`` returns 256 regardless of the mask), but
the values outside the mask are forced to zero via
``INPUT_TRANSFORM``. This matters because the generator's load path
copies *all* 256 cells into shared memory; only the compute loop is
sparsity-aware. If the masked-out cells contained noise, the
generated kernel would still produce the right answer (those values
are never multiplied), but a future-fold optimization that bypasses
masked loads entirely would silently produce a different result —
zeroing them here mirrors the SeisSol convention and avoids that
ambiguity.

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
