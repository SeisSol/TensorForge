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

from tensorforge.common.basic_types import Datatype

#from numpy_quaddtype import QuadPrecDType

_DTYPE_MAP = {
    Datatype.F16: np.float16,
    Datatype.F32: np.float32,
    Datatype.F64: np.float64,
    Datatype.F128: np.float128,
}

# maintain a different set of datatypes for export/import
# (to support more variants)
_DTYPE_EXPORT_MAP = {
    Datatype.F16: np.float16,
    Datatype.F32: np.float32,
    Datatype.F64: np.float64,
    Datatype.F128: np.float128,#QuadPrecDType(),
}

_CTYPE_MAP = {
    Datatype.F16: "__half",
    Datatype.F32: "float",
    Datatype.F64: "double",
    Datatype.F128: "__float128",
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
    """
    batch = view.shape[0]
    # Each element is F-contiguous, so the flat cell order within one element
    # is F-order over ``shape`` -- which is the order ``pack_index`` speaks.
    dense = np.stack([np.asarray(view[b]).ravel(order='F')
                      for b in range(batch)])
    return np.ascontiguousarray(dense[:, pack_index].ravel(),
                                dtype=np_dtype(dt))


def unpack(flat: np.ndarray, pack_index: np.ndarray,
           shape: Tuple[int, ...], batch: int) -> np.ndarray:
    """Expand the kernel's buffer back into a dense ``(batch, *shape)`` array.

    Structural zeros come back as zeros, which is what they are.
    """
    out = np.zeros((batch, volume(shape)), dtype=flat.dtype)
    out[:, pack_index] = flat.reshape(batch, -1)
    return np.stack([out[b].reshape(shape, order='F') for b in range(batch)])
