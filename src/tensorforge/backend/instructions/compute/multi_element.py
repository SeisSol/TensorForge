# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A section in the multi-element layout (`generators.element_group`),
lowered: a block is a group of E warps, one a multiplication and one an
element, and the batch loop hands it E consecutive elements a trip
(`Trip.ELEMENT_GROUP`).  Two instructions write the section:

* `MultiElementOperators`, ahead of the loop: the section's batch-constant
  operators, copied once into the block's shared memory by every thread of
  the block, as they are stored.
* `MultiElementSection`, the loop's body.  A trip:

  1. each warp copies its element's operands -- the group operands and the
     small ones the section reads -- into the group's buffers, element-outer
     (`element_group.Buffer`), through its own pointers;
  2. the section's steps run in order, each by its lowering (`LOWERINGS`),
     reading the staged operators and the group's buffers, writing an
     intermediate's buffer or the section's output;
  3. an output is written back per element, under the element's mask, as
     read-modify-write where the step accumulates.

The lowerings so far are the FMA forms, `fma` for an operator product and
`rows` for a mixing one: the warp of an element computes that element's
outputs of the step, a lane an output, each product of the step adding only
into its own window of the destination.  The matrix engines' tiling of an
operator product is the next lowering; it reads the same buffers, and its D
reaches a per-row consumer or an output through shared memory.

The instructions place their barriers themselves, as every builder does:
the block's barrier after the operators' copy, one at the head of each trip
(the last trip's reads before this trip's writes -- the barrier the loop
ends a trip with is the multiplication's, a warp's), and one ahead of a step
that reads a buffer written since the last.  They are what a build without
the barrier optimization (`Options.enable_sync_block_opt`) keeps; with it,
the barrier placement (`pir.barriers`) derives them from the buffers'
accesses.  They are block barriers.  With the FMA forms a warp reads only
its own element's rows, but a buffer is the block's, and the matrix
engines' products read across warps.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from tensorforge.backend import elementmask
from tensorforge.backend.instructions.abstract_instruction import \
    AbstractInstruction
from tensorforge.backend.pir.core import (BOOL, INDEX, MemSpace,
                                          Participants, ScalarType,
                                          Uniformity)
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.common.context import Context

#: The longest contraction a lowering writes out rather than loops over.
UNROLL = 16


def _hint(prefix: str, tensor) -> str:
    """A name for a buffer of `tensor` in the generated code."""
    return prefix + re.sub(r'\W', '', tensor.alias)


_FOLD = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y,
         'mul': lambda x, y: x * y}


def _index(builder, op: str, x, y, hint: str):
    """`x op y` over indices: a number where both are, else a value --
    and nothing where the operation is the identity."""
    if isinstance(x, int) and isinstance(y, int):
        return _FOLD[op](x, y)
    if isinstance(y, int) and (y == 0 and op in ('add', 'sub')
                               or y == 1 and op == 'mul'):
        return x
    return builder.op(op, INDEX, x, y, hint=hint)


def _box_index(builder, row, col, lower, height):
    """The index of `(row, col)` in a box stored column by column from
    `lower`, `height` rows to a column."""
    r = _index(builder, 'sub', row, lower[0], 'r')
    c = _index(builder, 'sub', col, lower[1], 'c')
    return _index(builder, 'add', r, _index(builder, 'mul', c, height, 'c'),
                  'a')


def _address(builder, tensor, row, col):
    """The storage index of `(row, col)` of a dense tensor stored column by
    column over its bounding box -- what `Tensor.linear_index` reads --
    `row` and `col` values or numbers."""
    lower = tuple(int(x) for x in tensor.bbox.lower())
    return _box_index(builder, row, col, lower, int(tensor.bbox.sizes()[0]))


def _bound(builder, symbol):
    """The global binding of a symbol in this body, or the symbol, which the
    builder resolves to it."""
    bound = symbol.pir_buffer(builder)
    return bound if bound is not None else symbol


class MultiElementOperators(AbstractInstruction):
    """The section's operators, copied into the block's shared memory once,
    ahead of the batch loop, as they are stored: every thread of the block
    a share of the entries, then the block's barrier.

    The copies are handed to the section (`buffers`, by tensor), which reads
    them in every trip."""

    def __init__(self, context: Context, group, symbols: Sequence):
        super().__init__(context)
        self._group = group
        #: The operators' symbols, in the section's order of its operators.
        self.symbols = tuple(symbols)
        self.buffers: Optional[Dict[int, object]] = None
        self._is_ready = True
        for symbol in self.symbols:
            symbol.add_user(self)

    def uses(self):
        return self.symbols

    def barrier_scope(self) -> Uniformity:
        return Uniformity.BLOCK

    def gen_ir(self, builder) -> None:
        section = self._group.section
        fp = ScalarType(section.dtype)
        threads = self._group.threads * self._group.elements
        tid = builder.op('add', INDEX, builder.thread_id('x'),
                         builder.op('mul', INDEX, builder.thread_id('y'),
                                    self._group.threads, hint='w'),
                         hint='tid')
        buffers = {}
        for tensor, symbol in zip(section.operators, self.symbols):
            volume = tensor.storage_volume()
            shared = builder.alloc(section.dtype, (volume,), MemSpace.SHARED,
                                   arena=GeneralLexicon.TOTAL_SHR_MEM,
                                   hint=_hint('k', tensor))
            source = _bound(builder, symbol)
            with builder.for_(tid, volume, threads, hint='i',
                              uniform=Uniformity.LANE) as loop:
                value = builder.load(source, loop.induction, type_=fp,
                                     hint='k')
                builder.store(shared, value, loop.induction)
            buffers[id(tensor)] = shared
        builder.barrier(Participants.BLOCK)
        self.buffers = buffers

    def __str__(self) -> str:
        return (f'multi-element operators '
                f'{", ".join(s.name for s in self.symbols)}')


class _Trip:
    """What the lowerings of one trip share: the warp's element and its
    mask, the group's buffers, and how to read and write an operand."""

    def __init__(self, builder, group, symbols, operators):
        self.builder = builder
        self.group = group
        self.section = group.section
        self.fp = ScalarType(self.section.dtype)
        self.lane = builder.thread_id('x')
        self.slot = builder.op('rem', INDEX, builder.thread_id('y'),
                               group.elements, hint='slot')
        self.symbols = symbols
        self.operators = operators
        self.buffers: Dict[int, object] = {}
        # The warp's element's mask, which every write of an output runs
        # under.  The element group's traversal is text, and its mask a
        # name: the warp's own, since a warp is its element.
        mask, name = elementmask.active()
        if mask is None and name is not None and self.section.outputs:
            mask = builder.extern_value(name, BOOL, uniform=Uniformity.MULT,
                                        hint='active')
        self.mask = mask

    def at(self, buffer, row, col):
        """The index of `(row, col)` of the warp's element in `buffer`."""
        b = self.builder
        base = b.op('mul', INDEX, self.slot, buffer.per_element, hint='b')
        return _index(b, 'add', base, _box_index(
            b, row, col, (buffer.rows[0], buffer.cols[0]),
            buffer.rows[1] - buffer.rows[0]), 'a')

    def read(self, sub, row, col):
        """`(row, col)` of an operand of a product: the staged operator, or
        the warp's element's rows of a buffer."""
        tensor = sub.tensor
        staged = self.operators.get(id(tensor))
        if staged is not None:
            return self.builder.load(staged,
                                     _address(self.builder, tensor, row,
                                              col),
                                     type_=self.fp, hint='k')
        buffer = self.section.buffer(tensor)
        return self.builder.load(self.buffers[id(tensor)],
                                 self.at(buffer, row, col), type_=self.fp,
                                 hint='g')

    def write(self, step, row, col, total) -> None:
        """The warp's element's `(row, col)` of the step's destination: its
        intermediate's buffer, or its output in memory under its mask --
        added to what is there where the step accumulates."""
        b = self.builder
        tensor = step.dest
        if tensor.is_tmp:
            target = self.buffers[id(tensor)]
            at = self.at(self.section.buffer(tensor), row, col)
            predicate = None
        else:
            target = _bound(b, self.symbols[id(tensor)])
            at = _address(b, tensor, row, col)
            predicate = self.mask
        if not step.assigns:
            old = b.load(target, at, type_=self.fp, hint='q')
            total = b.op('add', self.fp, old, total, hint='q')
        b.store(target, total, at, predicate=predicate)


def _product(trip, product, row, col):
    """`sum_k first(row, k) second(k, col)` over the product's contraction:
    written out where it is short, a loop where it is not."""
    b = trip.builder
    lo, hi = product.depth
    if hi - lo <= UNROLL:
        total = None
        for k in range(lo, hi):
            a = trip.read(product.first, row, k)
            x = trip.read(product.second, k, col)
            total = (b.op('mul', trip.fp, a, x, hint='x') if total is None
                     else b.op('fma', trip.fp, a, x, total, hint='x'))
        return total
    with b.for_(lo, hi, 1, inits=(0.0,), types=(trip.fp,),
                hint='l') as inner:
        k = inner.induction
        a = trip.read(product.first, row, k)
        x = trip.read(product.second, k, col)
        inner.yield_(b.op('fma', trip.fp, a, x, inner.iter_args[0],
                          hint='x'))
    return inner.result


def _per_output(trip, step) -> None:
    """The FMA forms: the warp of an element computes the element's outputs
    of the step, a lane an output, summing the step's products -- each only
    in its own window of the destination, and reading nothing outside it."""
    b = trip.builder
    rows, cols = step.rows, step.cols
    height = rows[1] - rows[0]
    with b.for_(trip.lane, height * (cols[1] - cols[0]),
                trip.group.threads, hint='o',
                uniform=Uniformity.LANE) as loop:
        row = _index(b, 'add', b.op('rem', INDEX, loop.induction, height,
                                    hint='i'), rows[0], 'i')
        col = _index(b, 'add', b.op('div', INDEX, loop.induction, height,
                                    hint='j'), cols[0], 'j')
        total = None
        for product in step.products:
            if (product.rows, product.cols) == (rows, cols):
                contrib = _product(trip, product, row, col)
            else:
                inside = None
                for value, (lo, hi), whole in (
                        (row, product.rows, rows), (col, product.cols, cols)):
                    if (lo, hi) == whole:
                        continue
                    edge = b.op('and', BOOL,
                                b.op('ge', BOOL, value, lo, hint='in'),
                                b.op('lt', BOOL, value, hi, hint='in'),
                                hint='in')
                    inside = edge if inside is None else b.op(
                        'and', BOOL, inside, edge, hint='in')
                guard = b.if_else(inside, types=(trip.fp,))
                with guard.then():
                    guard.yield_(_product(trip, product, row, col))
                with guard.otherwise():
                    guard.yield_(b.const(0.0, trip.fp))
                contrib = guard.result
            total = contrib if total is None else b.op(
                'add', trip.fp, total, contrib, hint='q')
        trip.write(step, row, col, total)


#: Each step's lowering (`element_group.Step.lowering`).
LOWERINGS = {'fma': _per_output, 'rows': _per_output}


class MultiElementSection(AbstractInstruction):
    """One trip of a section over a group of elements (see the module).

    `symbols` are the section's symbols of its inputs and outputs, by
    tensor (`id`); the operators come staged
    (`MultiElementOperators.buffers`)."""

    def __init__(self, context: Context, group, symbols: Dict[int, object],
                 operators: MultiElementOperators):
        super().__init__(context)
        self._group = group
        self._symbols = dict(symbols)
        self._operators = operators
        self._is_ready = True
        section = group.section
        self._inputs = tuple(self._symbols[id(b.tensor)]
                             for b in section.inputs)
        self._outputs = tuple(self._symbols[id(t)] for t in section.outputs)
        for symbol in self._inputs + self._outputs:
            symbol.add_user(self)

    def defs(self):
        return self._outputs

    def uses(self):
        return self._inputs + self._outputs

    def barrier_scope(self) -> Uniformity:
        return Uniformity.BLOCK

    def gen_ir(self, builder) -> None:
        section = self._group.section
        trip = _Trip(builder, self._group, self._symbols,
                     self._operators.buffers)
        elements = self._group.elements
        for buffer in section.inputs + section.intermediates:
            trip.buffers[id(buffer.tensor)] = builder.alloc(
                section.dtype, (elements * buffer.per_element,),
                MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                hint=_hint('g', buffer.tensor))
        # The block meets before any warp overwrites the group's buffers, so
        # that the last trip's reads of them are done.
        builder.barrier(Participants.BLOCK)
        for buffer, symbol in zip(section.inputs, self._inputs):
            self._stage(trip, buffer, symbol)
        # What was written and read since the block last met: a step that
        # reads what is written, or writes what is written or read, waits.
        written = {id(b.tensor) for b in section.inputs}
        read = set()
        for step in section.steps:
            reads = {id(t) for t in step.reads()}
            if written & reads or id(step.dest) in written | read:
                builder.barrier(Participants.BLOCK)
                written, read = set(), set()
            LOWERINGS[step.lowering](trip, step)
            written.add(id(step.dest))
            read |= reads

    def _stage(self, trip, buffer, symbol) -> None:
        """The warp's element's box of an operand, from its own memory into
        the group's buffer."""
        b = trip.builder
        source = _bound(b, symbol)
        height = buffer.rows[1] - buffer.rows[0]
        base = b.op('mul', INDEX, trip.slot, buffer.per_element, hint='b')
        with b.for_(trip.lane, buffer.per_element, trip.group.threads,
                    hint='j', uniform=Uniformity.LANE) as loop:
            row = _index(b, 'add', b.op('rem', INDEX, loop.induction,
                                        height, hint='r'),
                         buffer.rows[0], 'r')
            col = _index(b, 'add', b.op('div', INDEX, loop.induction,
                                        height, hint='c'),
                         buffer.cols[0], 'c')
            value = b.load(source, _address(b, buffer.tensor, row, col),
                           type_=trip.fp, hint='s')
            b.store(trip.buffers[id(buffer.tensor)], value,
                    b.op('add', INDEX, base, loop.induction, hint='j'))

    def __str__(self) -> str:
        section = self._group.section
        return (f'multi-element section, {len(section.steps)} steps into '
                f'{", ".join(t.alias for t in section.outputs)}, '
                f'{self._group.elements} elements a block')


__all__ = ['LOWERINGS', 'MultiElementOperators', 'MultiElementSection',
           'UNROLL']
