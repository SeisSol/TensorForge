# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Storing a batch-constant operand without the cells its numbers never fill.

Not `backend.instructions.compute.packing`, which answers a different
question with the same word: how much of an instruction's axis a tile
leaves over.  This one is about a buffer in memory.

A dense operand is stored over its bounding box, and the box is the smallest
*rectangle* around the non-zeros -- which for SeisSol's operators is a good
deal larger than the non-zeros themselves: 20627 cells hold 11545 numbers.
The difference is padding the kernel reads, multiplies by zero and adds.

Three ways to leave it out, in increasing order of what they cost to address:

``rows``
    The rows no column fills at all are dropped.  One pair of numbers for the
    whole operand, the column stride shrinks from the box's rows to the span,
    and a read of a dropped row is a lane the access is held off.  13963
    cells, and the only one of the three that carries no table.
``runs``
    Per column, the span between its own first and last non-zero.  Two
    numbers per column; the slot is still the row plus a constant, so a read
    is as cheap as ``rows``.  12341 cells.
``bits``
    Only the non-zeros.  A bitset per column and the running count before it;
    the slot is a masked population count of the row, which is two
    instructions on every target that has the opcode.  11545 cells, the
    floor.

The three are the same statement at three resolutions -- ``rows`` is ``runs``
with every column sharing one run, and ``runs`` is ``bits`` with the holes
inside a column filled in -- so one class describes all three and the kind
only says which construction produced it.

What a `Packing` does *not* say is where the tables live; that is the
reader's business (`Symbol._packed_index`), and for a column known at build
time there is no table at all, only a literal.
"""

from typing import Sequence, Tuple, Union

import numpy as np

#: The three kinds, in the order they compress.
KINDS = ('rows', 'runs', 'bits')


class Packing:
    """Which slot of the stored buffer holds box cell ``(row, column)``.

    ``column`` is the box's non-lead axes flattened in F-order, so a rank-3
    operand is a matrix here and the lead axis is the one the lanes split.

    Every column holds a contiguous *run* of rows ``[start, start + length)``,
    beginning at slot ``offset``; for ``bits`` the run is where the column's
    non-zeros lie and ``word`` says which of them are actually stored.  A row
    outside its column's run is in no slot, and reading it gives zero -- that
    is what makes the form smaller than the box, and the reason a read of a
    packed operand carries a predicate.
    """

    __slots__ = ('kind', 'rows', 'columns', 'starts', 'lengths', 'offsets',
                 'words', 'elements')

    def __init__(self, kind: str, rows: int, columns: int,
                 starts: Sequence[int], lengths: Sequence[int],
                 offsets: Sequence[int],
                 words: Union[Sequence[int], None] = None):
        if kind not in KINDS:
            raise ValueError(f'unknown packing {kind!r}; one of {KINDS}')
        self.kind = kind
        self.rows = int(rows)
        self.columns = int(columns)
        self.starts = tuple(int(x) for x in starts)
        self.lengths = tuple(int(x) for x in lengths)
        self.offsets = tuple(int(x) for x in offsets)
        self.words = None if words is None else tuple(int(x) for x in words)
        if not (len(self.starts) == len(self.lengths) == len(self.offsets)
                == self.columns):
            raise ValueError('a packing says start, length and offset for '
                             'every column')
        if (self.words is None) != (kind != 'bits'):
            raise ValueError('a bitset packing carries one word per column, '
                             'and the other two carry none')
        self.elements = (int(self.offsets[-1] + self.lengths[-1])
                         if self.columns else 0)
        if kind == 'bits':
            self.elements = (int(self.offsets[-1]
                                 + int(self.words[-1]).bit_count())
                             if self.columns else 0)

    # -- what the kernel asks ------------------------------------------------

    @property
    def uniform(self) -> bool:
        """Whether every column holds the same rows at the same spacing.

        Then the address is affine in the column -- ``offset = stride * c`` --
        and a read needs no per-column number at all, which is the one case
        where a column the kernel computes rather than knows is addressable.
        """
        if self.kind == 'bits' or not self.columns:
            return False
        if len(set(self.starts)) != 1 or len(set(self.lengths)) != 1:
            return False
        stride = self.lengths[0]
        return all(off == stride * c for c, off in enumerate(self.offsets))

    def slot(self, row: int, column: int) -> int:
        """The slot holding ``(row, column)``, or ``-1`` where it is in none.

        The definition the kernel's addressing and the host's packing both
        follow, so the two agree by construction rather than by two
        derivations of the same arithmetic.
        """
        row, column = int(row), int(column)
        if not 0 <= column < self.columns:
            return -1
        start, length = self.starts[column], self.lengths[column]
        if not start <= row < start + length:
            return -1
        if self.kind != 'bits':
            return self.offsets[column] + row - start
        word = self.words[column]
        if not (word >> row) & 1:
            return -1
        return self.offsets[column] + (word & ((1 << row) - 1)).bit_count()

    def order(self) -> Tuple[int, ...]:
        """`Tensor.storage_order` for this packing: the box cell each slot
        holds, in F-order over ``(rows, columns)``."""
        out = [-1] * self.elements
        for c in range(self.columns):
            for r in range(self.starts[c], self.starts[c] + self.lengths[c]):
                s = self.slot(r, c)
                if s >= 0:
                    out[s] = r + self.rows * c
        if any(s < 0 for s in out):
            raise ValueError(f'{self.kind} packing left a slot unfilled')
        return tuple(out)

    def __repr__(self):
        return (f'Packing({self.kind!r}, {self.rows}x{self.columns}, '
                f'{self.elements} slots)')

    def __eq__(self, other):
        return (isinstance(other, Packing) and self.kind == other.kind
                and self.rows == other.rows and self.columns == other.columns
                and self.starts == other.starts
                and self.lengths == other.lengths
                and self.offsets == other.offsets
                and self.words == other.words)

    def __hash__(self):
        return hash((self.kind, self.rows, self.columns, self.starts,
                     self.lengths, self.offsets, self.words))


def _mask(tensor) -> Union[np.ndarray, None]:
    """The bounding box's own non-zeros, as ``rows x columns`` in F-order.

    From `Tensor.nonzero_pattern`, which is the numbers where the operand has
    them and the storage pattern where it does not -- so an operand nothing
    knows the values of packs to its box and this returns no saving rather
    than a wrong one.
    """
    box = [int(x) for x in tensor.get_actual_shape()]
    if len(box) < 1 or box[0] < 1:
        return None
    lower = [int(x) for x in tensor.bbox.lower()]
    pattern = tensor.nonzero_pattern
    rows = box[0]
    columns = int(np.prod(box[1:])) if len(box) > 1 else 1
    whole = getattr(pattern, 'mask', None)
    if whole is not None and tuple(whole.shape) == tuple(tensor.get_real_shape()):
        # The derivation already walked the index space once; slicing the box
        # out of its answer beats walking it a second time cell by cell.
        cut = whole[tuple(slice(lo, lo + n) for lo, n in zip(lower, box))]
        return np.asarray(cut, dtype=bool).reshape(rows, columns, order='F')
    mask = np.zeros((rows, columns), dtype=bool)
    rest = box[1:]
    for c in range(columns):
        tail = np.unravel_index(c, rest, order='F') if rest else ()
        for r in range(rows):
            cell = (r,) + tuple(int(t) for t in tail)
            mask[r, c] = pattern.is_nz(tuple(i + lo for i, lo
                                             in zip(cell, lower)))
    return mask


def pack(tensor, kind: str) -> Union[Packing, None]:
    """How ``tensor`` would be stored under ``kind``, or `None`.

    `None` where the form saves nothing -- a packing that stores the whole box
    is the box, and offering it would only put an order and a predicate on
    reads that do not need them.
    """
    return pack_union((tensor,), kind)


def pack_union(tensors: Sequence, kind: str) -> Union[Packing, None]:
    """One packing for several operands at once, holding every cell any of
    them fills.

    What a merged run needs.  Its members are stored one per iteration and
    the one body reads whichever the counter names, so they are stored alike
    or not at all -- and alike means *one* packing, not one per member that
    happens to have the same shape.  Packed each to its own non-zeros, the
    four faces of `fMrT` give four different run tables and the body would
    read the first member's slots out of the fourth member's buffer.

    The union is sound for the same reason a single packing is: a cell no
    member fills is zero in all of them, so leaving it out leaves out a zero.
    It costs whatever the members disagree about, which for `localFluxAll`
    is the difference between 1600 slots and 951.
    """
    if kind not in KINDS:
        raise ValueError(f'unknown packing {kind!r}; one of {KINDS}')
    tensors = tuple(tensors)
    # Asked repeatedly and answered from the operand alone: the generator
    # weighs all three forms against the block's shared memory before it
    # offers one, and does it again on every rebuild of the section.
    cache = None
    if len(tensors) == 1:
        cache = getattr(tensors[0], '_packings', None)
        if cache is None:
            cache = {}
            try:
                tensors[0]._packings = cache
            except AttributeError:
                cache = None
        elif kind in cache:
            return cache[kind]
    masks = [_mask(t) for t in tensors]
    def nothing():
        if cache is not None:
            cache[kind] = None
        return None
    if not masks or any(m is None for m in masks):
        return nothing()
    if len({m.shape for m in masks}) != 1:
        return nothing()
    mask = masks[0]
    for other in masks[1:]:
        mask = mask | other
    if not mask.any():
        return nothing()
    rows, columns = mask.shape
    if kind == 'rows':
        used = np.flatnonzero(mask.any(axis=1))
        first, span = int(used[0]), int(used[-1] - used[0] + 1)
        packing = Packing('rows', rows, columns, [first] * columns,
                          [span] * columns,
                          [span * c for c in range(columns)])
    elif kind == 'runs':
        starts, lengths, offsets, at = [], [], [], 0
        for c in range(columns):
            nz = np.flatnonzero(mask[:, c])
            start = int(nz[0]) if len(nz) else 0
            length = int(nz[-1] - nz[0] + 1) if len(nz) else 0
            starts.append(start)
            lengths.append(length)
            offsets.append(at)
            at += length
        packing = Packing('runs', rows, columns, starts, lengths, offsets)
    else:
        starts, lengths, offsets, words, at = [], [], [], [], 0
        for c in range(columns):
            nz = np.flatnonzero(mask[:, c])
            start = int(nz[0]) if len(nz) else 0
            length = int(nz[-1] - nz[0] + 1) if len(nz) else 0
            word = 0
            for r in nz:
                word |= 1 << int(r)
            starts.append(start)
            lengths.append(length)
            offsets.append(at)
            words.append(word)
            at += len(nz)
        packing = Packing('bits', rows, columns, starts, lengths, offsets,
                          words)
    found = packing if packing.elements < rows * columns else None
    if cache is not None:
        cache[kind] = found
    return found
