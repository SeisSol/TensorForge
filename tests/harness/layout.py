# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Mapping between TensorForge's on-device layout and NumPy arrays.

What the generator actually emits (see ``glb_m0 = &m0[batchId0 * volume + …]``
and the inner indexing ``i0 * 1 + i1 * shape[0]`` in a 2D tensor):

* **batch axis outermost**, contiguous (stride = product of shape).
* inside each element, **column-major** (Fortran order) over ``shape``.
* bounding-box and offsets are ignored here — the MVP only exercises
  cases where the full shape is used (dense bbox, zero offset).

Wrapping NumPy arrays with ``order='F'`` per element and a leading batch
axis makes ``tobytes()`` produce exactly the layout the kernel expects,
so host<->device transfers are a plain memcpy.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Tuple

import numpy as np

from tensorforge.common.matrix import prepare

from tensorforge.common.basic_types import Datatype

#from numpy_quaddtype import QuadPrecDType

_DTYPE_MAP = {
    Datatype.F16: np.float16,
    Datatype.F32: np.float32,
    Datatype.F64: np.float64,
    Datatype.F128: np.float128,
    # conditions and comparisons (yateto's `where`, guards) and integer data
    Datatype.BOOL: np.bool_,
    Datatype.I32: np.int32,
    Datatype.I64: np.int64,
}

# maintain a different set of datatypes for export/import
# (to support more variants)
_DTYPE_EXPORT_MAP = {
    Datatype.F16: np.float16,
    Datatype.F32: np.float32,
    Datatype.F64: np.float64,
    Datatype.F128: np.float128,#QuadPrecDType(),
    Datatype.BOOL: np.bool_,
    Datatype.I32: np.int32,
    Datatype.I64: np.int64,
}

_CTYPE_MAP = {
    Datatype.F16: "__half",
    Datatype.F32: "float",
    Datatype.F64: "double",
    Datatype.F128: "__float128",
    Datatype.BOOL: "bool",
    Datatype.I32: "int32_t",
    Datatype.I64: "int64_t",
}

def np_dtype(dt: Datatype) -> np.dtype:
    if dt not in _DTYPE_MAP:
        raise NotImplementedError(f"dtype {dt!r} not wired in the MVP harness")
    return np.dtype(_DTYPE_MAP[dt])

def np_export_dtype(dt: Datatype) -> np.dtype:
    if dt not in _DTYPE_EXPORT_MAP:
        raise NotImplementedError(f"dtype {dt!r} not wired in the MVP harness")
    return np.dtype(_DTYPE_EXPORT_MAP[dt])

def ctype(dt: Datatype) -> str:
    if dt not in _CTYPE_MAP:
        raise NotImplementedError(f"dtype {dt!r} not wired in the MVP harness")
    return _CTYPE_MAP[dt]


def volume(shape: Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, shape, 1)


def _strided_view(flat: np.ndarray, shape: Tuple[int, ...], batch: int) -> np.ndarray:
    """Per-element F-order view over a flat C-contiguous batch buffer."""
    itemsize = flat.dtype.itemsize
    per_elem = [itemsize]
    for d in shape[:-1]:
        per_elem.append(per_elem[-1] * d)
    batch_stride = itemsize * volume(shape)
    return np.lib.stride_tricks.as_strided(
        flat, shape=(batch, *shape), strides=(batch_stride, *per_elem),
        writeable=flat.flags.writeable,
    )


def make_batch(rng: np.random.Generator,
               shape: Tuple[int, ...],
               batch: int,
               dt: Datatype) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(view, flat)``.

    * ``view`` is a ``(batch, *shape)`` array, F-contiguous per element,
      suitable for direct use with :func:`numpy.einsum`.
    * ``flat`` is the underlying 1-D ``batch * prod(shape)`` array whose
      ``tobytes()`` is the exact byte stream the kernel indexes into.

    Keeping both around explicitly avoids the ``as_strided`` ``.base``
    aliasing pitfalls and makes host<->device transfers obviously correct.
    """

    # need to cast explicitly, since standard_normal only supports F32 and F64
    flat = rng.standard_normal(batch * volume(shape)).astype(np_dtype(dt), copy=False)
    return _strided_view(flat, shape, batch), flat


def zeros_batch(shape: Tuple[int, ...], batch: int,
                dt: Datatype) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.zeros(batch * volume(shape), dtype=np_dtype(dt))
    return _strided_view(flat, shape, batch), flat


def view_of(flat: np.ndarray, shape: Tuple[int, ...], batch: int) -> np.ndarray:
    """Attach a ``(batch, *shape)`` per-element F-order view to ``flat``."""
    return _strided_view(flat, shape, batch)


def pack(view: np.ndarray, pack_index: np.ndarray,
         dt: Datatype) -> np.ndarray:
    """Compress a ``(batch, *shape)`` dense view into the kernel's buffer.

    ``pack_index`` says, for each storage slot, which F-order cell of one
    batch element belongs there.  Cells no slot points at are the structural
    zeros: they are not stored, and whatever the dense view holds for them is
    dropped here rather than silently reaching the kernel.

    A slot naming ``-1`` is the other direction, and it is the one thing a
    gather cannot express on its own: a slot with no cell.  A tiled storage
    order has them wherever the tiling runs past the end of the matrix, and
    they read zero -- which is what the kernel's own padding registers held
    before the order moved into memory, so the product is unchanged and not
    merely harmless.  Written as a gather from cell 0 followed by a mask,
    because ``dense[:, -1]`` is a legal read of the last cell and would put
    the wrong value there rather than fail.
    """
    batch = view.shape[0]
    # Each element is F-contiguous, so the flat cell order within one element
    # is F-order over ``shape`` -- which is the order ``pack_index`` speaks.
    dense = np.stack([np.asarray(view[b]).ravel(order='F')
                      for b in range(batch)])
    pack_index = np.asarray(pack_index)
    empty = pack_index < 0
    out = dense[:, np.where(empty, 0, pack_index)]
    if empty.any():
        out = np.where(empty[None, :], np.zeros((), dtype=out.dtype), out)
    return np.ascontiguousarray(out.ravel(), dtype=np_dtype(dt))


def split_tf32(flat: np.ndarray, dt: Datatype, planar: int = 0) -> np.ndarray:
    """Store each scalar as the two TF32 halves a matrix instruction multiplies.

    The library's own, so that what the harness writes into a buffer and what
    a prepared operand is filled with are one routine.
    """
    if np_dtype(dt) != np.float32:
        raise ValueError(
            f"the TF32 split is defined for F32 operands; got {dt}")
    return prepare.split_tf32(flat, planar=planar)


def unpack(flat: np.ndarray, pack_index: np.ndarray,
           shape: Tuple[int, ...], batch: int) -> np.ndarray:
    """Expand the kernel's buffer back into a dense ``(batch, *shape)`` array.

    Structural zeros come back as zeros, which is what they are.
    """
    out = np.zeros((batch, volume(shape)), dtype=flat.dtype)
    out[:, pack_index] = flat.reshape(batch, -1)
    return np.stack([out[b].reshape(shape, order='F') for b in range(batch)])
