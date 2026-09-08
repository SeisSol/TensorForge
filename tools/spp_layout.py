# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The storage layout, in the form the frontend already takes.

Two problems from the staging work turn out to be one, and it has a socket
on the yateto side already.

A run's alignment has to be worked out per run, because each one starts
wherever its non-zeros happen to start -- unless the layout guarantees
otherwise.  `yateto.memory.PatternMemoryLayout` with `alignStride` does
guarantee it: every stored cell is padded out to its aligned block of
`alignedReals`, so every run begins on an alignment boundary and one width
answers for all of them.  That is what `sparse="aligned"` selects in a
`memLayout` file, and it is what `config/gpu/tensorforge.xml` asks for on
every operator whose line is currently commented out.

Swapping a tensor for a dense twin at staging time does not work from the
frontend's side -- what arrives is what yateto decided.  But
`PatternMemoryLayout(spp, alignStride, pattern=...)` takes an explicit slot
per non-zero rather than numbering them itself, so the order is the caller's
to choose.  Between the two, the frontend can be told the layout instead of
being second-guessed after the fact.

What this module produces is exactly those two arguments: the padded pattern
and the slot assignment.  It does not wire them up -- something still has to
carry them from here to the layout decision -- but it settles what would be
carried.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np

#: The widest single load either ISA issues, in bytes.  What a padded block
#: has to reach for the padding to buy anything, and what it should not
#: exceed, since nothing addresses more than this in one go.
LOAD_BYTES = 16


def natural_align(fp_bytes: int) -> int:
    """Elements one load covers: the width worth padding to.

    Not the width the layout uses today.  `deriveArchitecture` gives a device
    build 64 bytes on NVIDIA and 128 on AMD, and `PatternMemoryLayout` pads
    to that -- eight FP64 elements, or sixteen.  That figure is a cacheline,
    which is the right question for a dense stride and the wrong one for a
    sparse pattern: a block wider than one load cannot be read in one go, so
    everything past this is stored and never addressed as a unit.
    """
    return max(1, LOAD_BYTES // fp_bytes)


@dataclass
class Layout:
    """A storage layout for one tensor."""

    #: Which cells are stored.  A superset of the non-zeros when padded.
    stored: np.ndarray
    #: Slot per stored cell, 1-based and 0 elsewhere -- the `pattern`
    #: argument of `PatternMemoryLayout`, which reads it exactly this way.
    slots: np.ndarray
    #: Alignment the padding guarantees, in elements.  1 means none.
    align: int

    @property
    def volume(self) -> int:
        return int(self.stored.sum())

    @property
    def padding(self) -> int:
        return self.volume - int(np.count_nonzero(self.values_mask))

    @property
    def grew_by(self) -> int:
        """Rows the aligned box added past the tensor's own."""
        if self.values_mask is None:
            return 0
        return self.stored.shape[0] - self.values_mask.shape[0]

    values_mask: Optional[np.ndarray] = None

    def runs(self) -> Tuple[Tuple[int, int, int], ...]:
        """`(slot, cell, length)` over the flattened bounding box."""
        flat = self.stored.ravel(order='F')
        out, start, slot = [], None, 0
        for cell, occupied in enumerate(flat):
            if occupied and start is None:
                start = cell
            elif not occupied and start is not None:
                out.append((slot, start, cell - start))
                slot += cell - start
                start = None
        if start is not None:
            out.append((slot, start, len(flat) - start))
        return tuple(out)


def align_pattern(mask: np.ndarray, width: int) -> np.ndarray:
    """Pad every non-zero out to its aligned block along axis 0.

    The same rule `PatternMemoryLayout` applies under `alignStride`: round the
    row index down to a multiple of `width` and store the whole block, up to
    the bounding box's aligned upper end.

    That end can lie past the tensor's own rows, and the layout lets it: it
    sizes the pattern to `max(shape[0], bbox.stop)` rather than clipping.  So
    the result here may be taller than what came in, which is the difference
    between agreeing with the frontend and being 3% under it on every
    operator whose row count is not a multiple of the width.
    """
    if width <= 1:
        return mask.copy()
    occupied = np.flatnonzero(mask.any(axis=tuple(range(1, mask.ndim))))
    if occupied.size == 0:
        return np.zeros_like(mask)
    stop = int(occupied[-1]) + 1
    stop += (-stop) % width
    shape = (max(mask.shape[0], stop),) + mask.shape[1:]
    out = np.zeros(shape, dtype=bool, order='F')
    for column in np.ndindex(*mask.shape[1:]):
        col = mask[(slice(None),) + column]
        for row in np.flatnonzero(col):
            lo = int(row) - int(row) % width
            out[(slice(lo, min(lo + width, stop)),) + column] = True
    return out


def merge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Close every gap shorter than `max_gap`, along the flattened box.

    Optimal for a cost of the form `alpha * stored + beta * runs`, and the
    argument is short: closing a gap of `g` stores `g` more cells and removes
    one run, so it pays exactly when `alpha * g < beta`.  The gaps do not
    interact -- closing one changes nothing about what any other is worth --
    so the whole decision is that comparison applied to each, and `max_gap`
    is `beta / alpha` spelled as a length.

    A fill ratio would not do: it closes larger gaps in a dense stretch than
    in a sparse one, though a gap costs the same either way.
    """
    if max_gap <= 0:
        return mask.copy()
    flat = mask.ravel(order='F').copy()
    run_start = None
    last_end = None
    for cell, occupied in enumerate(flat):
        if not occupied:
            continue
        if last_end is not None and cell - last_end < max_gap:
            flat[last_end:cell] = True
        last_end = cell + 1
        run_start = run_start if run_start is not None else cell
    return flat.reshape(mask.shape, order='F')


def layout_for(mask: np.ndarray, align: int = 1,
               max_gap: int = 0) -> Layout:
    """The padded pattern and its slot assignment, F-order.

    Slots run in the order the flattened bounding box does, which is the
    order `PatternMemoryLayout` would have chosen for itself.  Handing it
    back explicitly is what makes a different order possible later without
    changing anything else.
    """
    # Gaps first, boundaries second: closing a gap is about how many runs
    # there are, rounding one is about where a run may start, and doing it
    # the other way round would round boundaries that then disappear.
    stored = align_pattern(merge_gaps(mask, max_gap), align)
    slots = np.zeros(stored.shape, dtype=np.int64, order='F')
    flat_stored = stored.ravel(order='F')
    flat_slots = slots.ravel(order='F')
    flat_slots[np.flatnonzero(flat_stored)] = np.arange(
        1, int(flat_stored.sum()) + 1)
    slots = flat_slots.reshape(stored.shape, order='F')
    return Layout(stored=stored, slots=slots, align=align, values_mask=mask)


def compare(mask: np.ndarray, widths: Sequence[int] = (1, 2, 4, 8, 16),
            fp_bytes: int = 8) -> str:
    """What each alignment costs in storage and saves in runs.

    The width one load covers is marked; wider ones are padding that nothing
    reads as a unit.
    """
    natural = natural_align(fp_bytes)
    head = (f'{"":1} {"align":>6} {"stored":>8} {"×nnz":>6} {"runs":>6} '
            f'{"mean len":>9}')
    lines = [head, '-' * len(head)]
    nnz = int(mask.sum())
    for width in widths:
        lay = layout_for(mask, width)
        runs = lay.runs()
        mean = (sum(r[2] for r in runs) / len(runs)) if runs else 0.0
        mark = '<' if width == natural else ' '
        lines.append(f'{mark:1} {width:>6} {lay.volume:>8} '
                     f'{lay.volume / nnz:>6.2f} {len(runs):>6} {mean:>9.1f}')
    return '\n'.join(lines)


def gap_sweep(masks, gaps: Sequence[int] = (0, 2, 3, 4, 6, 8, 12, 16),
              align: int = 1, group: int = 32,
              w_run: float = 3.0, w_bit: float = 6.0,
              fp_bytes: int = 8, bytes_per_cycle: float = 64.0) -> str:
    """Cost against `max_gap`, with the bitset as the alternative per column.

    A column is costed either as its runs -- one iteration each, most lanes
    idle on a short one -- or as a bitset over `group`-cell words, where every
    lane works but the index has to be computed.  `w_run` and `w_bit` are
    instructions per iteration and are estimates until the loader exists to
    count them from.

    Issue and traffic are put in one currency the way `spp_plan` does it, so
    the total has a minimum rather than falling forever: closing gaps buys
    iterations and costs bytes, and the gap that minimises the sum is
    `beta / alpha` for the corpus rather than for one matrix, which is the
    number the generator would carry.
    """
    head = (f'{"gap":>5} {"stored":>9} {"runs":>8} {"issue":>10} '
            f'{"traffic":>9} {"total":>10} {"bitset cols":>12}')
    lines = [head, '-' * len(head)]
    for gap in gaps:
        stored = runs = cost = bitcols = 0
        for mask in masks.values():
            merged = align_pattern(merge_gaps(mask, gap), align)
            stored += int(merged.sum())
            rows = merged.shape[0]
            words = math.ceil(rows / group)
            for column in np.ndindex(*merged.shape[1:]):
                col = merged[(slice(None),) + column]
                nz = np.flatnonzero(col)
                if nz.size == 0:
                    continue
                breaks = int(np.count_nonzero(np.diff(nz) > 1)) + 1
                runs += breaks
                run_cost = breaks * w_run
                bit_cost = words * w_bit
                cost += min(run_cost, bit_cost)
                bitcols += bit_cost < run_cost
        traffic = stored * fp_bytes / bytes_per_cycle
        lines.append(f'{gap:>5} {stored:>9} {runs:>8} {cost:>10.0f} '
                     f'{traffic:>9.0f} {cost + traffic:>10.0f} '
                     f'{bitcols:>12}')
    return '\n'.join(lines)


def sweep(masks, widths: Sequence[int] = (1, 2, 4, 8, 16),
          fp_bytes: int = 8) -> str:
    """The same over a set of tensors, with the totals that decide.

    `masks` is a mapping of name to occupancy mask.
    """
    natural = natural_align(fp_bytes)
    head = (f'{"name":<12} {"nnz":>7} ' +
            ' '.join(f'{"w=" + str(w):>9}' for w in widths) + f' {"dense":>9}')
    lines = [head, '-' * len(head)]
    totals = {w: 0 for w in widths}
    dense = 0
    for name, mask in masks.items():
        row = []
        for width in widths:
            volume = layout_for(mask, width).volume
            totals[width] += volume
            row.append(f'{volume:>9}')
        dense += mask.size
        lines.append(f'{name:<12} {int(mask.sum()):>7} ' + ' '.join(row) +
                     f' {mask.size:>9}')
    lines.append('-' * len(head))
    lines.append(f'{"total":<12} {"":>7} ' +
                 ' '.join(f'{totals[w]:>9}' for w in widths) +
                 f' {dense:>9}')
    lines.append('of dense:    ' +
                 '  '.join(f'w={w}{"*" if w == natural else ""}: '
                           f'{totals[w] / dense:.2f}' for w in widths))
    return '\n'.join(lines)
