# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Metrics over the sparsity pattern of a globally constant matrix.

Phase 1 of the sparse-constant plan: measure the corpus before modelling it.
Nothing here touches the code generator, so a pattern whose kernel cannot yet
be generated -- orders 7 and 8, where the lane ladder caps at 32 and the
unrolled body runs to five figures -- is measurable all the same.

What each number is for:

``runs``
    Maximal stretches along one axis that are contiguous in *both* the dense
    and the compressed index space.  A run needs no predicate and can carry a
    wide access, so the run-length distribution decides between encoding the
    pattern as immediates and encoding it as a bitmap.

``blocks``
    Occupied tiles for one candidate tile shape.  The shape comes from the
    matrix-instruction catalogue: the contraction extent is the instruction's
    ``K``, the free extent is what the lane block gives.  ``padding`` is the
    structural zeros a tile-dense layout has to store, and it is a price worth
    paying when it buys a wider access.

``metadata``
    Bytes the *index* side costs under either encoding.  Both are lane-uniform,
    so neither belongs in shared memory; the number decides between immediates
    in the instruction stream and a scalar load from constant space.

``values``
    How many distinct values there are, and how many of them the hardware
    encodes for free.  AMD's inline constants cover a set that the ADER-DG
    operators hit often, and a value that is inline is a value that never
    needs a load at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Values GCN/CDNA/RDNA encode directly in the operand field, at no cost in
#: instruction bytes and with no literal slot consumed.  The integers are
#: omitted: an operator matrix that holds one holds it as a float.
AMD_INLINE_FP = (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0)

#: Reciprocal of 2*pi, the one irrational AMD encodes inline.  Kept apart
#: because matching it needs a tolerance and the others do not.
AMD_INLINE_RCP_2PI = 0.15915494309189535


@dataclass
class RunStats:
    """Runs along one axis, aggregated over all lines parallel to it."""

    axis: int
    count: int
    lengths: List[int] = field(repr=False, default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.lengths)) if self.lengths else 0.0

    def fraction_at_least(self, w: int) -> float:
        """Share of non-zeros sitting in a run of at least ``w`` elements.

        The quantity the vector width cares about: a run shorter than ``w``
        cannot fill a ``w``-wide access however the base is aligned.
        """
        total = sum(self.lengths)
        if total == 0:
            return 0.0
        return sum(l for l in self.lengths if l >= w) / total


@dataclass
class BlockStats:
    """Occupancy for one tile shape."""

    shape: Tuple[int, ...]
    occupied: int
    total: int
    stored: int
    nnz: int

    @property
    def padding(self) -> int:
        """Structural zeros a tile-dense layout has to store."""
        return self.stored - self.nnz

    @property
    def fill(self) -> float:
        return self.nnz / self.stored if self.stored else 0.0


@dataclass
class PatternMetrics:
    name: str
    shape: Tuple[int, ...]
    nnz: int
    volume: int
    runs: Dict[int, RunStats]
    blocks: Dict[Tuple[int, ...], BlockStats]
    distinct_values: Optional[int]
    inline_fraction: Optional[float]

    @property
    def density(self) -> float:
        return self.nnz / self.volume if self.volume else 0.0

    def metadata_bytes(self, axis: int, group: int = 1) -> Dict[str, int]:
        """Index-side cost under either encoding, in bytes.

        ``group`` raises the granularity: one bit per group of ``group``
        adjacent elements rather than per element, which is what makes a
        bitmap compatible with a wide access.  The run encoding is unaffected
        by it, so the comparison stays honest.
        """
        runs = self.runs[axis]
        # Two 32-bit immediates per run: where it starts, and the constant
        # that turns a dense index into a compressed one.
        run_bytes = 8 * runs.count
        lines = self.volume // self.shape[axis]
        words = math.ceil(self.shape[axis] / group / 32)
        # One mask word per 32 groups, plus a 32-bit base per line.
        mask_bytes = 4 * words * lines + 4 * lines
        return {'runs': run_bytes, 'bitmap': mask_bytes}


def _mask_of(pattern) -> np.ndarray:
    """A boolean occupancy mask, from whatever the caller has."""
    if hasattr(pattern, 'is_nz'):
        shape = getattr(pattern, 'shape', None)
        if shape is None:
            indexmask = getattr(pattern, 'indexmask', None)
            if indexmask is not None:
                shape = indexmask.shape
            elif hasattr(pattern, 'bbox'):
                # A box says where the non-zeros are, not how large the tensor
                # is; its upper corner is the smallest shape that holds them.
                shape = tuple(pattern.bbox.upper())
            else:
                raise ValueError(f'cannot infer a shape from {pattern!r}')
        mask = np.zeros(shape, dtype=bool, order='F')
        for idx in np.ndindex(*shape):
            mask[idx] = bool(pattern.is_nz(idx))
        return mask
    arr = np.asarray(pattern)
    return arr != 0 if arr.dtype != bool else arr


def _runs_along(mask: np.ndarray, axis: int) -> RunStats:
    moved = np.moveaxis(mask, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    lengths: List[int] = []
    for line in flat:
        run = 0
        for cell in line:
            if cell:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return RunStats(axis=axis, count=len(lengths), lengths=lengths)


def _blocks_of(mask: np.ndarray, shape: Sequence[int]) -> BlockStats:
    if len(shape) != mask.ndim:
        raise ValueError(f'tile shape {tuple(shape)} does not match a '
                         f'rank-{mask.ndim} pattern')
    tiles = [math.ceil(s / b) for s, b in zip(mask.shape, shape)]
    occupied = 0
    for tile in np.ndindex(*tiles):
        sl = tuple(slice(t * b, min((t + 1) * b, s))
                   for t, b, s in zip(tile, shape, mask.shape))
        if mask[sl].any():
            occupied += 1
    per_tile = math.prod(shape)
    return BlockStats(shape=tuple(shape),
                      occupied=occupied,
                      total=math.prod(tiles),
                      stored=occupied * per_tile,
                      nnz=int(mask.sum()))


def _value_stats(values, mask: np.ndarray,
                 tol: float) -> Tuple[Optional[int], Optional[float]]:
    if values is None:
        return None, None
    vals = np.asarray(values)[mask]
    if vals.size == 0:
        return 0, 0.0
    distinct = int(np.unique(vals).size)
    inline = np.zeros(vals.shape, dtype=bool)
    for c in AMD_INLINE_FP:
        inline |= np.isclose(vals, c, rtol=0.0, atol=tol)
    inline |= np.isclose(vals, AMD_INLINE_RCP_2PI, rtol=0.0, atol=tol)
    return distinct, float(inline.mean())


def measure(pattern, name: str = '', values=None,
            tile_shapes: Sequence[Sequence[int]] = (),
            tol: float = 1e-12) -> PatternMetrics:
    """Every Phase-1 number for one pattern.

    ``pattern`` is a ``SparsityPattern``, a boolean array, or a value array
    whose zeros mark the structural zeros.  ``values`` is the value array when
    the pattern was given separately; without it the value columns stay empty
    rather than guessing.
    """
    mask = _mask_of(pattern)
    if values is None and not hasattr(pattern, 'is_nz'):
        # A float array carries both halves: the zeros mark the pattern and
        # the rest are the values.  Asking a `SparsityPattern` for values
        # would only produce a zero-dimensional object array.
        arr = np.asarray(pattern)
        if arr.dtype.kind == 'f':
            values = arr
    distinct, inline = _value_stats(values, mask, tol)
    return PatternMetrics(
        name=name,
        shape=tuple(mask.shape),
        nnz=int(mask.sum()),
        volume=int(mask.size),
        runs={a: _runs_along(mask, a) for a in range(mask.ndim)},
        blocks={tuple(t): _blocks_of(mask, t) for t in tile_shapes},
        distinct_values=distinct,
        inline_fraction=inline,
    )


def table(metrics: Sequence[PatternMetrics], widths=(2, 4)) -> str:
    """One line per pattern, for a terminal and for the paper's first figure."""
    head = (f'{"name":<24} {"shape":>12} {"nnz":>7} {"dens":>6} '
            f'{"runs0":>6} {"len0":>6} ' +
            ' '.join(f'{"≥" + str(w) + "/0":>7}' for w in widths) +
            f' {"vals":>6} {"inline":>7}')
    lines = [head, '-' * len(head)]
    for m in metrics:
        r0 = m.runs[0]
        cols = ' '.join(f'{r0.fraction_at_least(w):>7.2f}' for w in widths)
        vals = '-' if m.distinct_values is None else str(m.distinct_values)
        inl = '-' if m.inline_fraction is None else f'{m.inline_fraction:.2f}'
        lines.append(
            f'{m.name:<24} {"×".join(str(s) for s in m.shape):>12} '
            f'{m.nnz:>7} {m.density:>6.2f} {r0.count:>6} {r0.mean:>6.1f} '
            f'{cols} {vals:>6} {inl:>7}')
    return '\n'.join(lines)
