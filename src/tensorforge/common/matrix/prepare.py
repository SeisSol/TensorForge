# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Turning a tensor's values into the scalars a prepared operand holds.

An operand the kernel reads prepared holds something other than one value
per element: the halves a matrix instruction multiplies, for one.  Deciding
that is the instruction's business; producing the numbers is this module's,
so that whoever fills the buffer and whoever reads it work from one
statement of what is in it.
"""

from __future__ import annotations

import numpy as np

#: Scalars one value occupies when it is stored as the two halves 3xTF32
#: multiplies.
TF32_PARTS = 2


def split_tf32(flat, planar: int = 0) -> np.ndarray:
    """Store each scalar as the two TF32 halves a matrix instruction multiplies.

    What ``splitFloatTF32`` in ``tensorforge_device/cuda.h`` computes for an
    operand the kernel splits itself, done here for one it reads prepared --
    though not bit for bit: the kernel rounds `upper` to nearest-even and
    leaves `lower` unrounded, which is as accurate and cheaper there, and
    either is a valid pair of halves.  Done here, once, for an operand that is
    constant across the batch, the kernel reads the pair instead of computing
    it, which is the whole point.

    Interleaved, ``[hi0, lo0, hi1, lo1, ...]``, because that is what
    ``DataView.get_dim_strides`` produces for ``storage_parts == 2``: the part
    index is the innermost stride, so the halves of one element are adjacent
    and a single wide access fetches both.  Planar, ``[hi0, hi1, ..., lo0,
    lo1, ...]`` per group of ``planar`` elements, where the operand is stored
    in fragment order (``Tensor.storage_planar``): a lane reads several
    neighbouring slots there, and each part's have to be one run.

    Both halves are stored as ``float`` and not as ``uint32``.  A TF32 value
    *is* a float with its low thirteen mantissa bits zero, so the kernel loads
    them through the accessor it already has and reinterprets -- no
    conversion, which is the arithmetic this exists to remove.

    The rounding is ``cvt.rna``: nearest, ties **away from zero**.  Not
    ties-to-even, however much the mnemonic looks like ``rne``.
    """

    def _rna(x: np.ndarray) -> np.ndarray:
        # The bits below the sign are a magnitude, so adding half an ulp of
        # the kept width and truncating rounds away from zero for both signs.
        u = x.view(np.uint32).astype(np.uint64)
        return ((u + 0x1000) & 0xFFFFE000).astype(np.uint32)

    x = np.ascontiguousarray(flat, dtype=np.float32)
    hi = _rna(x)
    lo = _rna((x - hi.view(np.float32)).astype(np.float32))
    if planar:
        out = np.stack([hi.view(np.float32).reshape(-1, planar),
                        lo.view(np.float32).reshape(-1, planar)], axis=1)
        return np.ascontiguousarray(out.ravel())
    out = np.empty(x.size * 2, dtype=np.float32)
    out[0::2] = hi.view(np.float32)
    out[1::2] = lo.view(np.float32)
    return out
