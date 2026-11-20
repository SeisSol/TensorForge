# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A section in the multi-element layout (`generators.element_group`),
lowered: a block is a group of E warps, one a multiplication and one an
element, and the batch loop hands it E consecutive elements a trip
(`Trip.ELEMENT_GROUP`).  Two instructions write the section:

* `MultiElementOperators`, ahead of the loop: the section's batch-constant
  operators, copied once into the block's shared memory by every thread of
  the block, as they are stored -- or only their nonzeros, packed, where a
  step walks those (`element_group.Terms`).
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
into its own window of the destination.  Where the operators' nonzeros are
known, `fma` walks only those (`_sparse`): a lane executes the nonzeros of
its output's row, the rows sorted so that a warp's lanes walk about as many
each.  The matrix engines' tiling of an
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
from tensorforge.common.basic_types import Datatype, GeneralLexicon
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError

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
    ahead of the batch loop, as the section stages them
    (`element_group.Stage`): as stored, or as a matrix product's B -- its
    window, zero elsewhere -- and, for a step that walks its operators'
    nonzeros (`element_group.Terms`), those nonzeros packed.  Every thread of
    the block a share of the entries, then the block's barrier.

    The copies are handed to the section (`buffers`, by tensor and window;
    `tables`, by a step's terms), which reads them in every trip."""

    def __init__(self, context: Context, group, symbols: Sequence):
        super().__init__(context)
        self._group = group
        #: The operators' symbols, in the section's order of its operators.
        self.symbols = tuple(symbols)
        self.buffers: Optional[Dict[int, object]] = None
        #: A sparse step's tables, by `id` of its terms: the values, the
        #: contraction indices and the row data (`_pack`).
        self.tables: Optional[Dict[int, tuple]] = None
        #: The generated file's tables the packing copies from, by name
        #: (`module_tables`).
        self._module_tables: Dict[str, tuple] = {}
        self._is_ready = True
        for symbol in self.symbols:
            symbol.add_user(self)

    def module_tables(self):
        """The numbers this instruction reads by name at namespace scope:
        the nonzeros' cells in the operators, their contraction indices, and
        the row data (`_pack`), as `MultilinearInstruction.module_tables`
        hands its own out."""
        return dict(self._module_tables)

    def _table(self, prefix: str, numbers) -> str:
        """The name of a table of integers in the generated file, which says
        which numbers it holds (`multilinear._module_table_name`)."""
        from tensorforge.backend.instructions.compute.multilinear import \
            _module_table_name
        numbers = tuple(int(n) for n in numbers)
        name = _module_table_name(prefix, Datatype.I32, numbers)
        self._module_tables[name] = (Datatype.I32, numbers)
        self._context.module_tables[name] = numbers
        return name

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
        symbols = {id(t): s for t, s in zip(section.operators, self.symbols)}
        buffers = {}
        for stage in section.stages:
            tensor = stage.tensor
            shared = builder.alloc(section.dtype, (stage.volume,),
                                   MemSpace.SHARED,
                                   arena=GeneralLexicon.TOTAL_SHR_MEM,
                                   hint=_hint('k' if stage.window is None
                                              else 'b', tensor))
            source = _bound(builder, symbols[id(tensor)])
            with builder.for_(tid, stage.volume, threads, hint='i',
                              uniform=Uniformity.LANE) as loop:
                if stage.window is None:
                    value = builder.load(source, loop.induction, type_=fp,
                                         hint='k')
                else:
                    value = self._entry(builder, stage, source, fp,
                                        loop.induction)
                builder.store(shared, value, loop.induction)
            buffers[(id(tensor), stage.window)] = shared
        tables = {id(step.terms): self._pack(builder, step, symbols, tid,
                                             threads)
                  for step in section.steps if step.terms is not None}
        builder.barrier(Participants.BLOCK)
        self.buffers = buffers
        self.tables = tables

    def _pack(self, builder, step, symbols, tid, threads) -> tuple:
        """A sparse step's tables in the block's shared memory
        (`element_group.Terms`): the values, gathered from each operator's
        memory at its nonzeros' cells -- the numbers the kernel is handed,
        the pattern the description's -- and the contraction indices and the
        row data, copied from the generated file's tables."""
        b = builder
        section = self._group.section
        terms = step.terms
        fp = ScalarType(section.dtype)

        def shared(elem, length, hint):
            return b.alloc(elem, (max(1, length),), MemSpace.SHARED,
                           arena=GeneralLexicon.TOTAL_SHR_MEM,
                           hint=_hint(hint, step.dest))

        def copy(length, store):
            with b.for_(tid, length, threads, hint='e',
                        uniform=Uniformity.LANE) as loop:
                store(loop.induction)

        values = shared(section.dtype, terms.entries, 'v')
        cols = shared(Datatype.I32, terms.entries, 'l')
        rows = shared(Datatype.I32, len(terms.order) * terms.stride, 'r')
        for p, product in enumerate(step.products):
            cells = terms.cells[p]
            if not cells:
                continue
            name = self._table('mec', cells)
            source = _bound(b, symbols[id(product.first.tensor)])
            base = terms.base(p)

            def gather(e, name=name, source=source, base=base):
                at = b.rawexpr(f'{name}[{{0}}]', e, type_=INDEX, pure=True,
                               movable=True, hint='a')
                b.store(values, b.load(source, at, type_=fp, hint='k'),
                        _index(b, 'add', e, base, 'e'))
            copy(len(cells), gather)
        for buffer, numbers, prefix in (
                (cols, [k for ks in terms.cols for k in ks], 'mek'),
                (rows, terms.rows_table(), 'mer')):
            if not numbers:
                continue
            name = self._table(prefix, numbers)

            def fill(e, name=name, buffer=buffer):
                b.store(buffer, b.rawexpr(f'{name}[{{0}}]', e, type_=INDEX,
                                          pure=True, movable=True, hint='n'),
                        e)
            copy(len(numbers), fill)
        return values, cols, rows

    @staticmethod
    def _entry(builder, stage, source, fp, index):
        """Entry `index` of a matrix product's B, `B(k, n)` at `k * ld + n`:
        the operator's `(n0 + n, k0 + k)` inside the product's window, zero
        outside it -- read from a clamped address, so that no lane reads past
        the operator."""
        b = builder
        (r0, r1), (d0, d1) = stage.window
        k = b.op('div', INDEX, index, stage.ld, hint='k')
        n = b.op('rem', INDEX, index, stage.ld, hint='n')
        row = _index(b, 'add', n, stage.n0, 'r')
        col = _index(b, 'add', k, d0, 'c')
        inside = b.op('and', BOOL, b.op('and', BOOL,
                                        b.op('ge', BOOL, row, r0, hint='in'),
                                        b.op('lt', BOOL, row, r1, hint='in'),
                                        hint='in'),
                      b.op('lt', BOOL, col, d1, hint='in'), hint='in')
        at = b.op('select', INDEX, inside,
                  _address(b, stage.tensor, row, col), 0, hint='a')
        value = b.load(source, at, type_=fp, hint='k')
        return b.op('select', fp, inside, value, b.const(0.0, fp), hint='k')

    def __str__(self) -> str:
        return (f'multi-element operators '
                f'{", ".join(s.name for s in self.symbols)}')


class _Trip:
    """What the lowerings of one trip share: the warp's element and its
    mask, the group's buffers, and how to read and write an operand."""

    def __init__(self, builder, context, group, symbols, operators,
                 tables=None):
        self.builder = builder
        self.context = context
        self.group = group
        self.section = group.section
        self.fp = ScalarType(self.section.dtype)
        self.lane = builder.thread_id('x')
        self.slot = builder.op('rem', INDEX, builder.thread_id('y'),
                               group.elements, hint='slot')
        self.symbols = symbols
        self.operators = operators
        #: A sparse step's tables, by its terms (`MultiElementOperators`).
        self.tables = tables or {}
        self.buffers: Dict[int, object] = {}
        self.results: Dict[int, object] = {}
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
        staged = self.operators.get((id(tensor), None))
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


def _lanes_active(active: int, of: int) -> tuple:
    """A guard's statement of how many of `of` lane slots take it, which the
    emitter's flop count reads (`BuildMetrics.emitted_flops`)."""
    return (('lanes_active', (active, of)),)


def _chunks(trip, count: int, body) -> None:
    """`body(o)` for the outputs `o < count`, a lane an output: the warp's
    lanes in chunks, a loop of constant bounds over them -- which the
    emitter counts the trips of, as it cannot a loop from the lane -- and
    in the last chunk the lanes past the end held off."""
    b = trip.builder
    lanes = trip.group.threads
    chunks = -(-count // lanes)
    with b.for_(0, chunks, 1, hint='c') as loop:
        o = b.op('add', INDEX, trip.lane,
                 b.op('mul', INDEX, loop.induction, lanes, hint='o'),
                 hint='o')
        if count % lanes == 0:
            body(o)
            return
        with b.if_(b.op('lt', BOOL, o, count, hint='in'),
                   attrs=_lanes_active(count, chunks * lanes)):
            body(o)


def _per_output(trip, step) -> None:
    """The FMA forms: the warp of an element computes the element's outputs
    of the step, a lane an output, summing the step's products -- each only
    in its own window of the destination, and reading nothing outside it.
    An operator product reads its operator as stored, every entry of its
    window; a step whose operators' nonzeros are known walks those instead
    (`_sparse`)."""
    b = trip.builder
    rows, cols = step.rows, step.cols
    height = rows[1] - rows[0]
    count = height * (cols[1] - cols[0])

    def outputs(index):
        row = _index(b, 'add', b.op('rem', INDEX, index, height,
                                    hint='i'), rows[0], 'i')
        col = _index(b, 'add', b.op('div', INDEX, index, height,
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
                window = ((product.rows[1] - product.rows[0])
                          * (product.cols[1] - product.cols[0]))
                guard = b.if_else(inside, types=(trip.fp,),
                                  attrs=_lanes_active(window, count))
                with guard.then():
                    guard.yield_(_product(trip, product, row, col))
                with guard.otherwise():
                    guard.yield_(b.const(0.0, trip.fp))
                contrib = guard.result
            total = contrib if total is None else b.op(
                'add', trip.fp, total, contrib, hint='q')
        trip.write(step, row, col, total)

    _chunks(trip, count, outputs)


def _sparse(trip, step) -> None:
    """The `fma` form over the operators' nonzeros (`element_group.Terms`):
    a lane an output, as `_per_output`, each product's operator read only
    where it is not zero.

    The lanes take the outputs in the terms' order -- output `o` is column
    `o % width` of row `order[o / width]`, the rows sorted by how many
    nonzeros they have -- a warp's chunk at a time, written out, since what
    each chunk costs is its own.  For a product, a chunk walks as many steps
    as the longest of its rows has nonzeros there, a loop of constant
    bounds; step `t` adds the `t`-th of the lane's row, `value * G(k, col)`
    with the value and the contraction index `k` from the step's tables,
    and holds off the lanes whose row has fewer.  So a lane executes its
    row's nonzeros and no more, and the guard says how many lanes take it
    (`lanes_active`), which is what the emitter's count reads."""
    b = trip.builder
    section = trip.section
    terms = step.terms
    values, cols, info = trip.tables[id(terms)]
    lanes = trip.group.threads
    c0, c1 = step.cols
    width = c1 - c0
    count = len(terms.order) * width
    fp = trip.fp
    for first in range(0, count, lanes):
        live = min(lanes, count - first)
        o = _index(b, 'add', trip.lane, first, 'o')
        # The chunk's lanes, as the terms' row and the column each takes.
        mine = [divmod(first + lane, width) for lane in range(live)]

        def chunk():
            s = b.op('div', INDEX, o, width, hint='s')
            col = _index(b, 'add', b.op('rem', INDEX, o, width, hint='j'),
                         c0, 'j')
            at = b.op('mul', INDEX, s, terms.stride, hint='r')
            row = b.load(info, at, type_=INDEX, hint='i')
            total = b.const(0.0, fp)
            for p, product in enumerate(step.products):
                p0, p1 = product.cols
                held = [terms.counts[p][r] for r, j in mine
                        if p0 <= c0 + j < p1]
                steps = max(held, default=0)
                if not steps:
                    continue
                first_term = b.load(info, _index(b, 'add', at, 1 + 2 * p,
                                                 'r'), type_=INDEX, hint='f')
                terms_here = b.load(info, _index(b, 'add', at, 2 + 2 * p,
                                                 'r'), type_=INDEX, hint='n')
                group = section.buffer(product.second.tensor)
                source = trip.buffers[id(product.second.tensor)]
                # The lane's column of the group operand, from its row 0:
                # a term reads it at its contraction index.
                base = _index(b, 'sub', trip.at(group, group.rows[0], col),
                              group.rows[0], 'b')
                inside = None
                if (p0, p1) != (c0, c1):
                    inside = b.op('and', BOOL,
                                  b.op('ge', BOOL, col, p0, hint='in'),
                                  b.op('lt', BOOL, col, p1, hint='in'),
                                  hint='in')
                with b.for_(0, steps, 1, inits=(total,), types=(fp,),
                            unroll=True, hint='t') as walk:
                    t = walk.induction
                    held_here = b.op('lt', BOOL, t, terms_here, hint='in')
                    if inside is not None:
                        held_here = b.op('and', BOOL, held_here, inside,
                                         hint='in')
                    guard = b.if_else(held_here, types=(fp,),
                                      attrs=_lanes_active(sum(held),
                                                          steps * live))
                    with guard.then():
                        e = b.op('add', INDEX, first_term, t, hint='e')
                        k = b.load(cols, e, type_=INDEX, hint='l')
                        g = b.load(source, b.op('add', INDEX, base, k,
                                                hint='a'),
                                   type_=fp, hint='g')
                        guard.yield_(b.op('fma', fp,
                                          b.load(values, e, type_=fp,
                                                 hint='k'),
                                          g, walk.iter_args[0], hint='x'))
                    with guard.otherwise():
                        guard.yield_(walk.iter_args[0])
                    walk.yield_(guard.result)
                total = walk.result
            trip.write(step, row, col, total)

        if live < lanes:
            with b.if_(b.op('lt', BOOL, o, count, hint='in'),
                       attrs=_lanes_active(live, lanes)):
                chunk()
        else:
            chunk()


def _fma(trip, step) -> None:
    """An operator step on the FMA units: over its operators' nonzeros where
    they are known (`_sparse`), over their windows where not."""
    (_sparse if step.terms is not None else _per_output)(trip, step)


def _atom(lowering: str, shape, dtype):
    """The `nvidia.MMAInstr` a matrix step issues: of `shape`, over TF32
    halves for `mma_sync`, in full for `dmma`."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    mode = (nvidia.MMAMode.TF32 if lowering == 'mma_sync'
            else nvidia.MMAMode.DIRECT)
    for atom in nvidia.INSTRS:
        if ((atom.m, atom.n, atom.k) == tuple(shape) and atom.d is dtype
                and atom.mode == mode):
            return atom
    raise InternalError(f'no {lowering} instruction of shape {shape} for '
                        f'{dtype}')


def _a_lane_cells(atom):
    """`slot -> (row, column)` of lane 0's A registers, the rest of A's
    layout a shift: lane `l` holds the cell `l / 4` rows and `l % 4`
    columns past lane 0's (`nvidia.a_fragment_bits`), checked for every
    lane and register rather than assumed."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    cells = nvidia._fragment_cells(atom)
    first = {slot: cell for (slot, lane), cell in cells.items() if lane == 0}
    for (slot, lane), (row, col) in cells.items():
        row0, col0 = first[slot]
        if (row, col) != (row0 + lane // 4, col0 + lane % 4):
            raise InternalError(
                f'the A layout of {atom.name} puts register {slot} of lane '
                f'{lane} at {(row, col)}, which is not a shift of lane 0\'s')
    return dict(sorted(first.items()))


def _matrix(trip, step) -> None:
    """The matrix tiling of an operator step on `MATRIX_MAC`.

    The step's products are one product over the group: M the group's rows
    -- `m = e * width + j`, element `e`'s column `j` of the destination,
    element after element -- N the destination's rows, K each product's
    contraction, the products one after another (`element_group.Step`).  A
    is the group operand's buffer as it stands, row `m` of it an element's
    column, the contraction contiguous; B the product's stage, zero outside
    its window and in the padding.  A warp takes the tiles of D in turn, D
    in registers across the products, and hands each to the destination's
    buffer -- the intermediate's, or the output's result buffer, which each
    warp then adds into its own element's output under its mask."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    from tensorforge.generators.element_group import matrix_shape
    b = trip.builder
    section = trip.section
    fp = trip.fp
    shape = matrix_shape(step.lowering, nvidia.sm_of(trip.context))
    atom = _atom(step.lowering, shape, section.dtype)
    m_, n_, k_ = shape
    rows, cols = step.rows, step.cols
    width = cols[1] - cols[0]
    height = rows[1] - rows[0]
    M = trip.group.elements * width
    mtiles, ntiles = -(-M // m_), -(-height // n_)
    a_cells = _a_lane_cells(atom)
    d_cells = nvidia.d_lane_cells(atom)
    kregs = k_ // 4
    if step.dest.is_tmp:
        dest = section.buffer(step.dest)
        target = trip.buffers[id(step.dest)]
    else:
        dest = section.result(step.dest)
        target = trip.results[id(step.dest)]
    quad = b.op('div', INDEX, trip.lane, 4, hint='q')
    inquad = b.op('rem', INDEX, trip.lane, 4, hint='q')

    def rows_of(m, buffer, col_lo, row):
        """The index in `buffer` of row `row` of the group's row `m`: element
        `m / width`, the element's column `col_lo + m % width`."""
        e = b.op('div', INDEX, m, width, hint='e')
        j = _index(b, 'add', b.op('rem', INDEX, m, width, hint='j'),
                   col_lo - buffer.cols[0], 'j')
        base = b.op('mul', INDEX, e, buffer.per_element, hint='b')
        return _index(b, 'add', base, _box_index(
            b, row, j, (buffer.rows[0], 0),
            buffer.rows[1] - buffer.rows[0]), 'a')

    with b.for_(trip.slot, mtiles * ntiles, trip.group.elements, hint='t',
                uniform=Uniformity.MULT) as tiles:
        mt = b.op('div', INDEX, tiles.induction, ntiles, hint='mt')
        nt = b.op('rem', INDEX, tiles.induction, ntiles, hint='nt')
        m0 = b.op('mul', INDEX, mt, m_, hint='m')
        n0 = b.op('mul', INDEX, nt, n_, hint='n')
        accs = [b.declare(fp, hint='c', uniform=Uniformity.LANE)
                for _ in d_cells]
        # The lane's rows of A, the same for every product and step of K.
        a_rows = {}
        for slot, (r0, _) in a_cells.items():
            if r0 not in a_rows:
                m = b.op('add', INDEX, m0, _index(b, 'add', quad, r0, 'm'),
                         hint='m')
                live = (b.op('lt', BOOL, m, M, hint='in') if M % m_
                        else None)
                a_rows[r0] = (m, live)
        for product in step.products:
            group = section.buffer(product.second.tensor)
            source = trip.buffers[id(product.second.tensor)]
            stage = section.stage(product.first.tensor,
                                  (product.rows, product.depth))
            staged = trip.operators[(id(stage.tensor), stage.window)]
            k_lo, k_hi = product.depth
            depth = k_hi - k_lo
            for kt in range(0, depth, k_):
                a = []
                for slot in range(len(a_cells)):
                    r0, c0 = a_cells[slot]
                    m, live = a_rows[r0]
                    kk = _index(b, 'add', inquad, kt + c0, 'k')
                    if kt + c0 + 4 > depth:
                        deep = b.op('lt', BOOL, kk, depth, hint='in')
                        live = deep if live is None else b.op(
                            'and', BOOL, live, deep, hint='in')
                    at = rows_of(m, group, cols[0],
                                 _index(b, 'add', kk, k_lo, 'k'))
                    if live is None:
                        a.append(b.load(source, at, type_=fp, hint='a'))
                        continue
                    at = b.op('select', INDEX, live, at, 0, hint='a')
                    value = b.load(source, at, type_=fp, hint='a')
                    a.append(b.op('select', fp, live, value,
                                  b.const(0.0, fp), hint='a'))
                bb = []
                for j in range(kregs):
                    k = _index(b, 'add', inquad, kt + 4 * j, 'k')
                    bb.append(b.load(staged, b.op(
                        'add', INDEX, b.op('mul', INDEX, k, stage.ld,
                                           hint='k'),
                        b.op('add', INDEX, n0, quad, hint='n'), hint='a'),
                        type_=fp, hint='b'))
                for term, (fa, fb) in enumerate(atom.products(b, a, bb)):
                    atom.asmcall(b, accs, fa, fb, accs, term=atom.term(term))
        # D, to the destination's buffer: the cells of the group's rows and
        # the step's rows, the padding's left out.
        for slot, (r0, c0) in d_cells.items():
            m = b.op('add', INDEX, m0, _index(b, 'add', quad, r0, 'm'),
                     hint='m')
            n = b.op('add', INDEX, n0, _index(b, 'add', b.op(
                'mul', INDEX, inquad, 2, hint='n'), c0, 'n'), hint='n')
            live = b.op('lt', BOOL, n, height, hint='in')
            if M % m_:
                live = b.op('and', BOOL, live,
                            b.op('lt', BOOL, m, M, hint='in'), hint='in')
            at = b.op('select', INDEX, live, rows_of(
                m, dest, cols[0], _index(b, 'add', n, rows[0], 'i')), 0,
                hint='a')
            value = accs[slot]
            if step.dest.is_tmp and not step.assigns:
                value = b.op('add', fp, b.load(target, at, type_=fp,
                                               hint='q'), value, hint='q')
            b.store(target, value, at, predicate=live)
    if step.dest.is_tmp:
        return
    # The output's D is in the result buffer, every warp's tiles of it: the
    # block meets, and each warp adds its own element's into its output.
    b.barrier(Participants.BLOCK)
    with b.for_(trip.lane, height * width, trip.group.threads, hint='o',
                uniform=Uniformity.LANE) as loop:
        row = _index(b, 'add', b.op('rem', INDEX, loop.induction, height,
                                    hint='i'), rows[0], 'i')
        col = _index(b, 'add', b.op('div', INDEX, loop.induction, height,
                                    hint='j'), cols[0], 'j')
        total = b.load(target, trip.at(dest, row, col), type_=fp, hint='d')
        trip.write(step, row, col, total)


#: Each step's lowering (`element_group.Step.lowering`).
LOWERINGS = {'fma': _fma, 'rows': _per_output,
             'mma_sync': _matrix, 'dmma': _matrix}


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
        trip = _Trip(builder, self._context, self._group, self._symbols,
                     self._operators.buffers, self._operators.tables)
        elements = self._group.elements
        for buffer in section.inputs + section.intermediates:
            trip.buffers[id(buffer.tensor)] = builder.alloc(
                section.dtype, (elements * buffer.per_element,),
                MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                hint=_hint('g', buffer.tensor))
        for buffer in section.results:
            trip.results[id(buffer.tensor)] = builder.alloc(
                section.dtype, (elements * buffer.per_element,),
                MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                hint=_hint('d', buffer.tensor))
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
