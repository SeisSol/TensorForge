# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Derived forms of an operator: the parts an emulated product multiplies,
made where the numbers are known.

An operand that is the same for every element of the batch -- SeisSol's
operators -- need not be taken apart in the kernel: its parts can be made
once, at generation from the description's numbers or on the host by
whoever stores it, and read as they are.  What the parts are is the form's:
how many, the narrow format each is exact in, and how each is rounded --
which decides whether the prepared parts are the ones the kernel would have
made itself, bit for bit, or merely a valid decomposition as accurate.  The
per-element side is always split at run time, by each target's routine
(`device` names it); the form is the statement both sides are held to.

Three forms exist, each the arithmetic one path already had:

* `TF32_STORED` -- the pair an operand read prepared is stored as on the
  multilinear path (`Tensor.storage_values`, the offering and
  `Options.mma_operator_copies`): both halves rounded to TF32, ties away
  (`prepare.split_tf32`).
* `tf32_device(ties)` -- the pair `splitFloatTF32` makes on NVIDIA: the
  upper half rounded to nearest (ties to even on sm_90 on, away before),
  the lower the exact remainder, which the instruction reads the upper bits
  of (`prepare.tf32_halves`) -- what the asynchronous engines' prepared
  ring holds, so that a product is the converting ring's bit for bit.
* `BF16_TRIPLE` -- three BF16 terms, each rounded to nearest even and the
  remainder carried, whose sum is the value exactly (`prepare.split_bf16`,
  `splitFloatBF16` on HIP).

A table of a form (`lay`) holds each part of a piece in a layout of its
own, the parts one after the other -- the canonical K-major layout of the
asynchronous engines (`canonical_k_major`), or planar or adjacent by a
storage order (`DerivedForm.store`).  Ozaki's slices with their scales and
its residues are forms of the same kind, with integer parts and side
tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from tensorforge.common.basic_types import Datatype

from .prepare import split_bf16, split_tf32, tf32_halves


class Rounding(Enum):
    """How a part is made of what is left of the value.

    `EVEN` and `AWAY`: rounded to the narrow format's nearest, ties to even
    (`cvt.rn.tf32.f32`, `static_cast<__bf16>`) or away from zero
    (`cvt.rna.tf32.f32`).  `EXACT`: what is left, unrounded -- the
    instruction reads its upper bits (`splitFloatTF32`'s lower half)."""

    EVEN = 'even'
    AWAY = 'away'
    EXACT = 'exact'


@dataclass(frozen=True)
class DerivedForm:
    """A value as `parts` numbers whose sum it is (or, under `EXACT`, whose
    sum the instruction reads it as): each exact in `narrow`, the first
    rounded by `leading`, each later one by `rest`, held as `stored` in a
    table.  `device` is the routine that makes the same parts at run time,
    by the family that has it -- the statement a host model of the kernel
    reproduces (`tests/test_derived_forms.py`)."""

    name: str
    parts: int
    narrow: Datatype
    leading: Rounding
    rest: Rounding
    stored: Datatype = Datatype.F32
    device: Tuple[Tuple[str, str], ...] = ()
    #: `values -> (parts, n)` float32: the arithmetic, as `prepare` has it.
    make: Callable = field(default=None, compare=False, repr=False)

    def derive(self, values) -> np.ndarray:
        """The parts of each of `values` (taken as float32): an array of
        `parts` rows, a value a column."""
        x = np.ascontiguousarray(values, dtype=np.float32).ravel()
        out = np.asarray(self.make(x), dtype=np.float32)
        if out.shape != (self.parts, x.size):
            raise ValueError(f'{self.name}: {out.shape} parts of {x.size} '
                             f'values')
        return out

    def combine(self, parts) -> np.ndarray:
        """The parts summed, the smallest first, in double precision: the
        value back, where the form is exact (`BF16_TRIPLE`, the TF32 pairs'
        sums are the float32 value)."""
        parts = np.asarray(parts, dtype=np.float64)
        total = np.zeros(parts.shape[1:])
        for row in parts[::-1]:
            total = total + row
        return total

    def store(self, values, planar: int = 0) -> np.ndarray:
        """`values` stored with their parts: adjacent -- a value's parts
        side by side, `[p0 v0, p1 v0, ..., p0 v1, ...]` -- or, `planar`, per
        group of `planar` values every part's run in turn: what
        `Tensor.storage_values` holds where an element is spread over
        several slots (`Tensor.storage_planar`)."""
        parts = self.derive(values)
        if planar:
            out = np.stack([p.reshape(-1, planar) for p in parts], axis=1)
            return np.ascontiguousarray(out.ravel())
        return np.ascontiguousarray(parts.T.ravel())

    def __repr__(self):
        return f'DerivedForm({self.name})'


def _tf32_stored(x):
    return split_tf32(x).reshape(-1, 2).T


def _bf16_triple(x):
    return split_bf16(x).reshape(-1, 3).T


#: The TF32 pair an operand read prepared is stored as (`split_tf32`).
TF32_STORED = DerivedForm(
    'tf32 pair, rounded', 2, Datatype.TF32, Rounding.AWAY, Rounding.AWAY,
    device=(('cuda', 'tensorforge::splitFloatTF32 (cvt.rna, both halves '
                     'rounded: as accurate, not the same numbers)'),),
    make=_tf32_stored)

#: Three BF16 terms (`split_bf16`; `tensorforge::splitFloatBF16` on HIP).
BF16_TRIPLE = DerivedForm(
    'bf16 triple', 3, Datatype.BF16, Rounding.EVEN, Rounding.EVEN,
    device=(('hip', 'tensorforge::splitFloatx4BF16'),),
    make=_bf16_triple)


def tf32_device(ties: str) -> DerivedForm:
    """The TF32 pair `splitFloatTF32` makes on NVIDIA (`tf32_halves`): the
    upper half to nearest, ties to even (`'even'`, `cvt.rn`, sm_90 on) or
    away (`'away'`, `cvt.rna`), the lower the exact remainder."""
    leading = {'even': Rounding.EVEN, 'away': Rounding.AWAY}[ties]
    routine = 'cvt.rn' if ties == 'even' else 'cvt.rna'
    return DerivedForm(
        f'tf32 pair, upper {ties}, lower exact', 2, Datatype.TF32, leading,
        Rounding.EXACT,
        device=(('cuda', f'tensorforge::splitFloatTF32 ({routine})'),),
        make=lambda x: np.stack(tf32_halves(x, ties)))


def tf32_device_for(sm: int) -> DerivedForm:
    """`tf32_device` of the target: ties to even from sm_90 on."""
    return tf32_device('even' if sm >= 90 else 'away')


def stored(parts: int) -> Optional[DerivedForm]:
    """The form an operand stored in `parts` slots an element holds
    (`Tensor.storage_parts`): the count says which, since no target reads
    three TF32 halves or two BF16 terms; None for one this cannot make."""
    return {TF32_STORED.parts: TF32_STORED,
            BF16_TRIPLE.parts: BF16_TRIPLE}.get(int(parts))


def canonical_k_major(rows: int, width: int) -> np.ndarray:
    """Where cell `(k, n)` of a piece of `rows` rows of K and `width` of N
    lies in the canonical K-major layout the asynchronous engines read
    through a descriptor: core matrices of eight rows of N by four of K,
    128 bytes apart along K and `rows` x 32 elements along N -- `(n / 8)
    rows 8 + (k / 4) 32 + (n % 8) 4 + k % 4` -- as an array over `k * width
    + n`."""
    k, n = np.divmod(np.arange(rows * width), width)
    return (n // 8) * rows * 8 + (k // 4) * 32 + (n % 8) * 4 + k % 4


def lay(form: DerivedForm, block, slots) -> np.ndarray:
    """A piece of a table of `form`: `block`'s cells (any shape, read in
    C order) taken apart, each part laid out by `slots` -- the place of
    each cell in a part's run -- and the parts' runs one after the
    other."""
    values = np.asarray(block, dtype=np.float64).ravel()
    slots = np.asarray(slots).ravel()
    if slots.size != values.size:
        raise ValueError(f'{slots.size} slots for {values.size} cells')
    parts = form.derive(values)
    size = values.size
    laid = np.zeros(form.parts * size, dtype=np.float32)
    for p in range(form.parts):
        laid[p * size + slots] = parts[p]
    return laid


__all__ = ['BF16_TRIPLE', 'DerivedForm', 'Rounding', 'TF32_STORED',
           'canonical_k_major', 'lay', 'stored', 'tf32_device',
           'tf32_device_for']
