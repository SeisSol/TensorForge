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
     (`element_group.Buffer`), through its own pointers, by asynchronous
     copies (`cp.async`) that `pir.wrap` takes a trip ahead where it can:
     the first `Options.move_distance` of them, into a second stage where
     `Options.enable_multibuffer` asks and the plan has the room
     (`element_group.second_stages`);
  2. the section's steps run in order, each by its lowering (`LOWERINGS`),
     reading the staged operators and the group's buffers, writing an
     intermediate's buffer or the section's output;
  3. an FMA form writes an output per element, under the element's mask,
     as read-modify-write where the step accumulates; the matrix tiling
     writes it from the accumulators, a cell through its row's element's
     pointer, which each warp shares with the block in the trip
     (`_Trip.publish`), null where its element is masked.

The lowerings are the FMA forms, `fma` for an operator product and `rows`
for a mixing one -- the warp of an element computes that element's outputs
of the step, a lane an output, each product of the step adding only into
its own window of the destination -- and the matrix tiling of an operator
step (`mma_sync`, `dmma`; `_matrix`), which reads the same buffers: A from
the group operand's (by ldmatrix in F32, where the buffer is pitched for
it), B from the operator's stage, and D reaching its consumer's buffer, or
the output from the accumulators: one reduction a cell (`red.global.add`),
a store where the step assigns.  Where the operators' nonzeros are known,
`fma` walks only those (`_sparse`): a lane executes the nonzeros of its
output's row, the rows sorted so that a warp's lanes walk about as many
each.

The lanes form (`element_group.Lanes`) is lowered by `_lanes` instead: a
lane its element, read and written through its own pointers -- 16 bytes of
a column at a time where that is one aligned access -- the warps of the
block each a share of the output, nothing staged and no barrier.  Every
step's lowering there is `lane`, and a column is evaluated backwards from
the output (`_LaneTrip`).

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

import contextlib
import re
from typing import Dict, List, Optional, Sequence

from tensorforge.backend import elementmask
from tensorforge.backend.instructions.abstract_instruction import \
    AbstractInstruction
from tensorforge.backend.pir.core import (BOOL, INDEX, BufferType, Effect,
                                          MemSpace, Participants, ScalarType,
                                          Uniformity)
from tensorforge.common.basic_types import Datatype, GeneralLexicon
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError

#: The longest contraction a lowering writes out rather than loops over.
UNROLL = 16

#: A global address as a pointer table holds it (`_Trip.publish`).
WORD = ScalarType(Datatype.I64)


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


def _as_float(builder, half):
    """A TF32 half's bits as the float shared memory stores them, without
    arithmetic."""
    from tensorforge.common.basic_types import Datatype
    return builder.rawexpr('__uint_as_float({0})', half,
                           type_=ScalarType(Datatype.F32), hint='h',
                           pure=True, movable=True)


def _bound(builder, symbol):
    """The global binding of a symbol in this body, or the symbol, which the
    builder resolves to it."""
    bound = symbol.pir_buffer(builder)
    return bound if bound is not None else symbol


def _positions(tensor) -> tuple:
    """Where each cell of a tensor stored sparse sits among its stored
    entries (`Tensor.linear_index`), column by column over its box -- the
    order `_address` counts in -- and -1 where the cell is not stored."""
    lower = [int(x) for x in tensor.bbox.lower()]
    sizes = [int(x) for x in tensor.bbox.sizes()]
    out = []
    for c in range(sizes[1]):
        for r in range(sizes[0]):
            cell = (lower[0] + r, lower[1] + c)
            out.append(int(tensor.linear_index(cell))
                       if tensor.spp.is_nz(cell) else -1)
    return tuple(out)


def _sparse_read(builder, context, tables, source, tensor, index, fp,
                 uniform=None, hint='s'):
    """`tensor`'s cell at `index` -- its place in the box, column by column
    (`_address`) -- read from its compressed memory: where the table of the
    cells' places (`_positions`, a table of the generated file, which
    `tables` takes by name) puts it, and zero where the cell is not stored.
    An element's small operand stored sparse: SeisSol's `star` under its GPU
    memory layout keeps 24 of its 81 cells."""
    from tensorforge.backend.instructions.compute.multilinear import \
        _module_table_name
    b = builder
    numbers = _positions(tensor)
    kwargs = {} if uniform is None else {'uniform': uniform}
    if isinstance(index, int):
        at = numbers[index]
        if at < 0:
            return b.const(0.0, fp)
        return b.load(source, at, type_=fp, hint=hint, **kwargs)
    name = _module_table_name('mep', Datatype.I32, numbers)
    tables[name] = (Datatype.I32, numbers)
    context.module_tables[name] = numbers
    at = b.rawexpr(f'{name}[{{0}}]', index, type_=INDEX, pure=True,
                   movable=True, hint='p')
    stored = b.op('ge', BOOL, at, 0, hint='in')
    value = b.load(source, b.op('select', INDEX, stored, at, 0, hint='p'),
                   type_=fp, hint=hint, **kwargs)
    return b.op('select', fp, stored, value, b.const(0.0, fp), hint=hint)


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
                                              else 'b', tensor),
                                   place_align=(None if stage.window is None
                                                else 4))
            source = _bound(builder, symbols[id(tensor)])
            if stage.window is None:
                with builder.for_(tid, stage.volume, threads, hint='i',
                                  uniform=Uniformity.LANE) as loop:
                    value = builder.load(source, loop.induction, type_=fp,
                                         hint='k')
                    builder.store(shared, value, loop.induction)
            else:
                self._fragments(builder, stage, source, shared, fp, tid,
                                threads)
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
    def _fragments(builder, stage, source, shared, fp, tid, threads):
        """A matrix product's B in the order the instruction's fragments are
        read (`element_group.Stage`): cell after cell of B -- lane `l`'s
        register `j` of tile `(kt, nt)` is `B(kt k + l % 4 + 4 j, 8 nt +
        l / 4)`, the operator's `(n0 + n, k0 + k)` inside the product's
        window and zero outside it -- written where the lane reads it, its
        two TF32 halves apart where the stage is split.  The operator is read
        at a clamped address, so that no lane reads past it."""
        from tensorforge.backend.instructions.compute.primitives import \
            nvidia
        b = builder
        (r0, r1), (d0, d1) = stage.window
        kregs = stage.k // 4
        ntiles = stage.width // 8
        cells = stage.depth * stage.width
        with b.for_(tid, cells, threads, hint='i',
                    uniform=Uniformity.LANE) as loop:
            fragment = b.op('div', INDEX, loop.induction, kregs, hint='f')
            j = b.op('rem', INDEX, loop.induction, kregs, hint='j')
            lane = b.op('rem', INDEX, fragment, 32, hint='l')
            tile = b.op('div', INDEX, fragment, 32, hint='t')
            k = b.op('add', INDEX, b.op('add', INDEX, b.op(
                'mul', INDEX, b.op('div', INDEX, tile, ntiles, hint='t'),
                stage.k, hint='k'), b.op('rem', INDEX, lane, 4, hint='k'),
                hint='k'), b.op('mul', INDEX, j, 4, hint='k'), hint='k')
            n = b.op('add', INDEX, b.op('mul', INDEX, b.op(
                'rem', INDEX, tile, ntiles, hint='t'), 8, hint='n'),
                b.op('div', INDEX, lane, 4, hint='n'), hint='n')
            row = _index(b, 'add', n, stage.n0, 'r')
            col = _index(b, 'add', k, d0, 'c')
            inside = b.op('and', BOOL, b.op(
                'and', BOOL, b.op('ge', BOOL, row, r0, hint='in'),
                b.op('lt', BOOL, row, r1, hint='in'), hint='in'),
                b.op('lt', BOOL, col, d1, hint='in'), hint='in')
            at = b.op('select', INDEX, inside,
                      _address(b, stage.tensor, row, col), 0, hint='a')
            value = b.op('select', fp, inside,
                         b.load(source, at, type_=fp, hint='k'),
                         b.const(0.0, fp), hint='k')
            slot = b.op('add', INDEX, b.op('mul', INDEX, fragment,
                                           stage.regs, hint='s'), j,
                        hint='s')
            if stage.split:
                (upper, lower), = nvidia.tfconvert(b, [value])
                b.store(shared, _as_float(b, upper), slot)
                b.store(shared, _as_float(b, lower),
                        b.op('add', INDEX, slot, kregs, hint='s'))
            else:
                b.store(shared, value, slot)

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
        #: The block's pointer tables, by output (`publish`).
        self.pointers: Dict[int, object] = {}
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
            b, row, col, (buffer.rows[0], buffer.cols[0]), buffer.pitch),
            'a')

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

    def publish(self, tensor) -> None:
        """The warp's element's pointer to the output `tensor`, into the
        block's table at the warp's slot, by one lane: null where the
        element is masked, so that nothing is written through it."""
        b = self.builder
        word = b.rawexpr('reinterpret_cast<int64_t>({0})',
                         _bound(b, self.symbols[id(tensor)]), type_=WORD,
                         hint='p', pure=True, movable=True)
        if self.mask is not None:
            word = b.op('select', WORD, self.mask, word, 0, hint='p')
        b.store(self.pointers[id(tensor)], word, self.slot,
                predicate=b.op('eq', BOOL, self.lane, 0, hint='in'))

    def pointer(self, tensor, element):
        """Element `element`'s pointer to the output `tensor`, from the
        block's table (`publish`), and whether it has one -- null where the
        element is masked.  Read as the binding it is, a pointer to the
        output (`IRBuilder.decl_expr`)."""
        b = self.builder
        word = b.load(self.pointers[id(tensor)], element, type_=WORD,
                      hint='p')
        symbol = self.symbols[id(tensor)]
        bound = _bound(b, symbol)
        if not isinstance(getattr(bound, 'type', None), BufferType):
            raise InternalError(f'{tensor.alias}: no pointer bound in the '
                                f'body to read another element\'s like')
        ctype = self.section.dtype.ctype()
        name = f'{_hint("e", tensor)}_{b.next_index()}'
        pointer = b.decl_expr(f'{ctype} *const {name}',
                              f'reinterpret_cast<{ctype}*>({{0}})',
                              bound.type, symbol, args=(word,),
                              kind=Effect.READ, hint=name, extern=name,
                              alias_root=b.alias_root(symbol))
        return pointer, b.op('ne', BOOL, word, 0, hint='in')


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
    in registers across the products, and hands each to the destination:
    the intermediate's buffer, or the output, straight from the
    accumulators -- a lane's cells through the pointers of their rows'
    elements (`_Trip.pointer`), one reduction a cell where the step
    accumulates, so that each output gets one add of the step whichever
    warp holds it, a store where it assigns, and nothing where the element
    is masked."""
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
    dest = target = None
    if step.dest.is_tmp:
        dest = section.buffer(step.dest)
        target = trip.buffers[id(step.dest)]
    quad = b.op('div', INDEX, trip.lane, 4, hint='q')
    inquad = b.op('rem', INDEX, trip.lane, 4, hint='q')
    # A by ldmatrix where the step is F32 and its group operands' buffers
    # are pitched for it (`element_group._pitched`): lane `l` names row
    # `l % 8 + 8 ((l / 8) % 2)` of the tile at column `4 (l / 16)`, and the
    # four registers are A's in the instruction's order (`ldmatrixX4`).
    ldmatrix = step.lowering == 'mma_sync' and all(
        section.buffer(p.second.tensor).stride for p in step.products)
    if ldmatrix:
        if [a_cells[slot] for slot in sorted(a_cells)] != [
                (0, 0), (8, 0), (0, 4), (8, 4)]:
            raise InternalError(f'the A layout of {atom.name} is not the '
                                f'one ldmatrix.x4 loads')
        lrow = b.op('add', INDEX, b.op('rem', INDEX, trip.lane, 8, hint='l'),
                    b.op('mul', INDEX, b.op('rem', INDEX, b.op(
                        'div', INDEX, trip.lane, 8, hint='l'), 2, hint='l'),
                        8, hint='l'), hint='l')
        lcol = b.op('mul', INDEX, b.op('div', INDEX, trip.lane, 16,
                                       hint='l'), 4, hint='l')

    def rows_of(m, buffer, col_lo, row):
        """The index in `buffer` of row `row` of the group's row `m`: element
        `m / width`, the element's column `col_lo + m % width`."""
        e = b.op('div', INDEX, m, width, hint='e')
        j = _index(b, 'add', b.op('rem', INDEX, m, width, hint='j'),
                   col_lo - buffer.cols[0], 'j')
        base = b.op('mul', INDEX, e, buffer.per_element, hint='b')
        return _index(b, 'add', base, _box_index(
            b, row, j, (buffer.rows[0], 0), buffer.pitch), 'a')

    with b.for_(trip.slot, mtiles * ntiles, trip.group.elements, hint='t',
                uniform=Uniformity.MULT) as tiles:
        mt = b.op('div', INDEX, tiles.induction, ntiles, hint='mt')
        nt = b.op('rem', INDEX, tiles.induction, ntiles, hint='nt')
        m0 = b.op('mul', INDEX, mt, m_, hint='m')
        n0 = b.op('mul', INDEX, nt, n_, hint='n')
        accs = [b.declare(fp, hint='c', uniform=Uniformity.LANE)
                for _ in d_cells]
        if ldmatrix:
            lm = b.op('add', INDEX, m0, lrow, hint='m')
            if M % m_:
                lm = b.op('select', INDEX, b.op('lt', BOOL, lm, M, hint='in'),
                          lm, 0, hint='m')
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
                if ldmatrix:
                    # The lane's row of the tile, 16 bytes of it: the
                    # padding and the clamped rows past the group read
                    # numbers, zeros or rows whose D is not written.
                    at = rows_of(lm, group, cols[0],
                                 _index(b, 'add', lcol, k_lo + kt, 'k'))
                    a = list(b.split_op('tensorforge::ldmatrixX4',
                                        (fp,) * 4, source, at,
                                        hints=('a',) * 4, reads=(source,)))
                else:
                    a = []
                for slot in (() if ldmatrix else range(len(a_cells))):
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
                # The lane's registers of B for this tile and step of K,
                # side by side: one load (`element_group.Stage`).
                at = b.op('mul', INDEX, b.op('add', INDEX, b.op(
                    'mul', INDEX, b.op('add', INDEX, nt,
                                       (kt // k_) * ntiles, hint='f'),
                    32, hint='f'), trip.lane, hint='f'), stage.regs,
                    hint='f')
                if stage.regs == 1:
                    regs = [b.load(staged, at, type_=fp, hint='b')]
                else:
                    vec = b.load(staged, at, type_=ScalarType(
                        section.dtype, stage.regs), hint='b',
                        align=stage.regs * section.dtype.size())
                    regs = [b.extract(vec, r, hint='b')
                            for r in range(stage.regs)]
                if stage.split:
                    halves = [(nvidia._as_tf32(b, regs[j]),
                               nvidia._as_tf32(b, regs[kregs + j]))
                              for j in range(kregs)]
                    products = atom.products(b, a, None, b_split=halves)
                else:
                    products = atom.products(b, a, regs)
                for term, (fa, fb) in enumerate(products):
                    atom.asmcall(b, accs, fa, fb, accs, term=atom.term(term))
        # D, to the intermediate's buffer, or to the output from the
        # accumulators: the cells of the group's rows and the step's rows,
        # the padding's left out.
        pointers = {}
        for slot, (r0, c0) in d_cells.items():
            m = b.op('add', INDEX, m0, _index(b, 'add', quad, r0, 'm'),
                     hint='m')
            n = b.op('add', INDEX, n0, _index(b, 'add', b.op(
                'mul', INDEX, inquad, 2, hint='n'), c0, 'n'), hint='n')
            live = b.op('lt', BOOL, n, height, hint='in')
            if M % m_:
                live = b.op('and', BOOL, live,
                            b.op('lt', BOOL, m, M, hint='in'), hint='in')
            if target is None:
                # The row's element's pointer and its column; a row past
                # the group reads the first entry and writes nothing.
                if r0 not in pointers:
                    e = b.op('div', INDEX, m, width, hint='e')
                    if M % m_:
                        e = b.op('select', INDEX, b.op(
                            'lt', BOOL, m, M, hint='in'), e, 0, hint='e')
                    col = _index(b, 'add', b.op('rem', INDEX, m, width,
                                                hint='j'), cols[0], 'j')
                    pointers[r0] = trip.pointer(step.dest, e) + (col,)
                pointer, held, col = pointers[r0]
                b.store(pointer, accs[slot], _address(
                    b, step.dest, _index(b, 'add', n, rows[0], 'i'), col),
                    predicate=b.op('and', BOOL, live, held, hint='in'),
                    atomic=not step.assigns)
                continue
            at = b.op('select', INDEX, live, rows_of(
                m, dest, cols[0], _index(b, 'add', n, rows[0], 'i')), 0,
                hint='a')
            value = accs[slot]
            if step.dest.is_tmp and not step.assigns:
                value = b.op('add', fp, b.load(target, at, type_=fp,
                                               hint='q'), value, hint='q')
            b.store(target, value, at, predicate=live)


class _LaneTrip:
    """What the lanes form's lowering of a trip shares (`element_group.
    Lanes`): the lane's element -- its own pointers and its mask, as the
    traversal binds them, a lane each -- the column the warp computes and
    the rows of its range, and what of the column it holds.

    The section is evaluated backwards from the output, a column at a
    time: an operator step's intermediate is computed into registers where
    a product is about to read it (`walk`), a mixing step's entry where it
    is read (`mixed`), and an element's own operand is read from its memory
    (`entry`).  An operator's nonzero is a literal or read from the
    operator's memory (`number`)."""

    def __init__(self, builder, context, group, symbols):
        from tensorforge.generators import element_group
        self.kinds = element_group.Kind
        self.builder = builder
        self.context = context
        self.group = group
        self.section = group.section
        self.fp = ScalarType(self.section.dtype)
        self.symbols = symbols
        self.cones = element_group.Cones(self.section)
        #: Where the operators' nonzeros come from: `literal`, `memory` or
        #: `table` (`element_group.lane_values`).
        self.values = element_group.lane_values(context)
        #: The column's anchor (`element_group.lane_anchors`): a zero the
        #: compiler cannot see through, added into the addresses the column
        #: reads, or None.
        self.anchor = None
        #: The `table` form's numbers, in the order the column multiplies by
        #: them, by table: name -> numbers (`module_tables`); the table of
        #: the range being written and its block of a warp's width read last.
        self.tables: Dict[str, tuple] = {}
        #: The tables of a sparse operand's cells' places, by name:
        #: (Datatype.I32, numbers) (`_sparse_read`).
        self.int_tables: Dict[str, tuple] = {}
        self.table = None
        self.sequence: List[float] = []
        self.block = None
        #: What the FMAs carry where the target spells an FMA as the one
        #: operation it is (`Lexic.fused_fma`): that they are (`fused`), and
        #: that each is a statement of its own where it is computed
        #: (`no_inline`).  A compiler that may reassociate would otherwise
        #: move the accumulator chains away from the reads they consume,
        #: and a chain written as one expression at its last use puts them
        #: there before it does.
        self.fma_attrs = ((('fused', True), ('no_inline', True))
                          if context.target.lexic.fused_fma('a', 'b', 'c')
                          is not None else ())
        # The lane's element's mask, a lane's own here.
        mask, name = elementmask.active()
        if mask is None and name is not None:
            mask = builder.extern_value(name, BOOL, uniform=Uniformity.LANE,
                                        hint='active')
        self.mask = mask
        #: The columns the warp computes in this pass (values,
        #: `element_group.Lanes.columns` of them) and the one being written
        #: (`slot`); its rows, and the rows of every tensor upstream they
        #: need (`Cones.need`).
        self.cols: List[object] = []
        self.slot = 0
        self.rows = (0, 0)
        self.need: Dict[int, set] = {}
        #: Operator columns held in registers, by the step that writes
        #: them (`Cones.source`): (row, slot) -> value.
        self.columns: Dict[int, Dict[tuple, object]] = {}
        #: The small operands' columns, by `(tensor, row, slot)`, for this
        #: pass.
        self.small: Dict[tuple, object] = {}
        #: Entries read or computed at one contraction index.
        self.here: Dict[tuple, object] = {}
        #: The output's columns at the warp's rows, as its steps add into
        #: them (`_column`): (row, slot) -> value, None for zero.
        self.acc: Dict[tuple, object] = {}
        #: Rows of a column a lane reads in one access (`own`): 16 bytes.
        self.width = max(1, 16 // self.section.dtype.size())
        #: Those reads, by `(tensor, first row, column)`, while their rows
        #: are being used.
        self.tiles: Dict[tuple, object] = {}

    @property
    def col(self):
        """The column of the slot being written (`slot`)."""
        return self.cols[self.slot]

    def claim(self, tensor, row) -> Optional[int]:
        """The alignment, in bytes, of `width` rows of a column of
        `tensor` from `row` -- or None where they are not one access: past
        the column's box, or proving less than the access needs.  The base's
        promise (`Symbol.linear_align_bytes`), the row's offset into the
        column and the column's stride bound it."""
        width = self.width
        lower = int(tensor.bbox.lower()[0])
        height = int(tensor.bbox.sizes()[0])
        if width < 2 or (row - lower) % width or row + width > lower + height:
            return None
        if not tensor.is_dense():
            # Its rows are not where a box would put them.
            return None
        elem = self.section.dtype.size()

        def bound(nbytes):
            return nbytes & -nbytes if nbytes else 1 << 20

        claim = min(self.symbols[id(tensor)].linear_align_bytes(),
                    bound((row - lower) * elem), bound(height * elem))
        return claim if claim >= width * elem else None

    def own(self, tensor, row, col):
        """`(row, col)` of the lane's element's own `tensor`, from its
        memory -- a lane's own value: `width` rows of the column at a time
        where they are one aligned access (`claim`), an element where not."""
        b = self.builder
        source = _bound(b, self.symbols[id(tensor)])
        if not tensor.is_dense():
            return _sparse_read(b, self.context, self.int_tables, source,
                                tensor, self.anchored(
                                    _address(b, tensor, row, col)),
                                self.fp, uniform=Uniformity.LANE, hint='e')
        lower = int(tensor.bbox.lower()[0])
        first = row - (row - lower) % self.width
        claim = self.claim(tensor, first)
        if claim is None:
            return b.load(source, self.anchored(_address(b, tensor, row, col)),
                          type_=self.fp, uniform=Uniformity.LANE, hint='e')
        key = (id(tensor), first,
               col if isinstance(col, int) else ('col', self.slot))
        if key not in self.tiles:
            # The anchor is zero: the claim is the address's without it.
            self.tiles[key] = b.load(
                source, self.anchored(_address(b, tensor, first, col)),
                type_=ScalarType(self.section.dtype, self.width),
                align=claim, uniform=Uniformity.LANE, hint='e')
        return b.extract(self.tiles[key], row - first, hint='e')

    def anchored(self, index):
        """`index` plus the column's anchor, where there is one."""
        if self.anchor is None:
            return index
        return _index(self.builder, 'add', index, self.anchor, 'a')

    def number(self, product, row, k, value):
        """The operator's nonzero `(row, k)` of `product` (`values`):
        `value`, the description's number, as a literal; the number the
        kernel is handed, read from the operator's memory at an address
        every lane shares, which the compiler reads once for the wave (a
        scalar load on AMD); or `value` from the range's table, read a
        warp's width at a time, an element a lane, and handed to every lane
        from the lane that holds it."""
        b = self.builder
        if self.values == 'literal':
            return b.const(value, self.fp)
        if self.values == 'memory':
            tensor = product.first.tensor
            return b.load(_bound(b, self.symbols[id(tensor)]),
                          self.anchored(_address(b, tensor, row, k)),
                          type_=self.fp, uniform=Uniformity.GRID, hint='k')
        lanes = self.group.elements
        at = len(self.sequence)
        self.sequence.append(value)
        first = at - at % lanes
        if self.block is None or self.block[0] != first:
            # A block read where the target has one -- one message for the
            # sub-group, at an address all of it shares -- and otherwise
            # each lane its element of the block.
            run = self.context.target.lexic.sub_group_run(
                self.fp.base.ctype(), 1, f'{self.table} + {{0}}')
            if run is not None:
                read = b.rawexpr(f'{run[1]}[0]', self.anchored(first),
                                 type_=self.fp, hint='kt', pure=True,
                                 movable=False, crosslane=True,
                                 uniform=Uniformity.LANE)
            else:
                index = _index(b, 'add', b.thread_id('x'), first, 't')
                read = b.rawexpr(f'{self.table}[{{0}}]', self.anchored(index),
                                 type_=self.fp, hint='kt', pure=True,
                                 movable=False, uniform=Uniformity.LANE)
            self.block = (first, read)
        return b.lane_broadcast(self.block[1], at - first, lanes, hint='k')

    def begin_table(self, rows) -> None:
        """The `table` form's table for the range `rows`: named by what
        decides its numbers -- the section, the range, the warp's width and
        the operators' nonzeros -- before they are known, since every read
        names it (`end_table`)."""
        if self.values != 'table':
            return
        from tensorforge.backend.instructions.compute.multilinear import \
            _module_table_name
        # In the section's order of its products: an `id` says nothing
        # another process would repeat.
        nonzeros = tuple(
            tuple((i, tuple(ks)) for i, ks in
                  sorted(self.cones.nonzeros[id(p)].items()))
            for step in self.section.steps for p in step.products
            if p.kind is not self.kinds.MIXING)
        key = (self.section.shape(), tuple(rows), self.group.elements,
               self.cones.cols, nonzeros)
        self.table = _module_table_name('mel', self.section.dtype, key)
        self.sequence = []
        self.block = None

    def end_table(self) -> None:
        """The range's table, padded to the warp's width: what the last
        block's lanes past its numbers read."""
        if self.values != 'table' or self.table is None:
            return
        lanes = self.group.elements
        numbers = tuple(self.sequence) + (0.0,) * (-len(self.sequence)
                                                   % lanes)
        known = self.tables.get(self.table)
        if known is not None and known != numbers:
            raise InternalError(f'{self.table}: two sequences of the lanes '
                                f'form\'s numbers under one name')
        self.tables[self.table] = numbers
        self.context.module_tables[self.table] = numbers
        self.table, self.block = None, None

    def entry(self, product, row):
        """An operator product's operand at `row` of the column, or None
        where it is zero: held, computed by the mixing step that writes it,
        or read from the element's memory (`Cones.source`)."""
        source = self.cones.source(product)
        if source is None:
            key = (id(product.second.tensor), row, self.slot)
            if key not in self.here:
                self.here[key] = self.own(product.second.tensor, row,
                                          self.col)
            return self.here[key]
        held = self.columns.get(id(source))
        if held is not None:
            return held.get((row, self.slot))
        if source.kind is not self.kinds.MIXING:
            raise InternalError(f'{source.dest.alias} at row {row} read by '
                                f'the lanes form before it holds it')
        key = (id(source), row, self.slot)
        if key not in self.here:
            self.here[key] = self.mixed(source, row)
        return self.here[key]

    def mixed(self, step, row):
        """A mixing step's destination at `row` of the column, or None where
        none of its products adds into the row: each product's row of the
        element's operand times the column of its small one."""
        b, fp = self.builder, self.fp
        total = None
        for p in step.products:
            if not p.rows[0] <= row < p.rows[1]:
                continue
            part = None
            for q in range(*p.depth):
                key = (id(p.first.tensor), row, q)
                if key not in self.here:
                    self.here[key] = self.own(p.first.tensor, row, q)
                g = self.here[key]
                key = (id(p.second.tensor), q, self.slot)
                if key not in self.small:
                    self.small[key] = self.own(p.second.tensor, q, self.col)
                s = self.small[key]
                part = (b.op('mul', fp, g, s, hint='t') if part is None
                        else b.op('fma', fp, g, s, part, hint='t',
                                  attrs=self.fma_attrs))
            part = self.windowed(part, p, step)
            total = part if total is None else b.op('add', fp, total, part,
                                                    hint='t')
        return total

    def windowed(self, value, product, step):
        """`value`, or zero where the column is outside the product's
        columns -- a product adding into a window of the step's."""
        if product.cols == step.cols:
            return value
        b = self.builder
        c0, c1 = product.cols
        inside = b.op('and', BOOL, b.op('ge', BOOL, self.col, c0, hint='in'),
                      b.op('lt', BOOL, self.col, c1, hint='in'), hint='in')
        return b.op('select', self.fp, inside, value,
                    b.const(0.0, self.fp), hint='w')

    def materialized(self, product) -> bool:
        """Whether an operator step writes the product's operand."""
        source = self.cones.source(product)
        return source is not None and source.kind is self.kinds.OPERATOR

    def walk(self, step, rows, acc):
        """`acc` -- (row, slot) -> value, None for zero -- plus an operator
        step's products at `rows` of the pass's columns, by the operators'
        nonzeros (`number`): each number taken once for all the columns.

        A product whose operand an operator step computes runs alone: the
        operand's needed rows into registers, the product, the registers
        dropped.  The others run together by contraction index, so that an
        index's entries -- a row of the element's operand, the entry a
        mixing step computes from it -- are read once for all of them."""
        b, fp = self.builder, self.fp
        nonzeros = self.cones.nonzeros
        slots = range(len(self.cols))
        # A walk reads the element's operands for itself: one walk's reads
        # held through the next would be every row of them at once, which the
        # IR would share otherwise (`load_cse`).  Not a fence of the schedule
        # as well: a `__syncwarp()` under the warp's guard of its range has
        # ptxas write the code out twice.
        b.begin_operation(force=True)

        def add(row, v, x, product, j):
            if x is None:
                return
            x = self.windowed(x, product, step)
            old = acc.get((row, j))
            acc[(row, j)] = (b.op('mul', fp, v, x, hint='x') if old is None
                             else b.op('fma', fp, v, x, old, hint='x',
                                       attrs=self.fma_attrs))

        def each(product, i, k, value, operand):
            """The number `(i, k)` of `product`, once, into every column's
            row `i`, times the column's operand at `k` (`operand`)."""
            v = None
            for j in slots:
                self.slot = j
                x = operand(j)
                if x is None:
                    continue
                if v is None:
                    v = self.number(product, i, k, value)
                add(i, v, x, product, j)

        held = [p for p in step.products if self.materialized(p)]
        for p in held:
            source = self.cones.source(p)
            need = sorted(self.need.get(id(source), ()))
            self.columns[id(source)] = self.walk(
                source, need, {(i, j): None for i in need for j in slots})
            column = self.columns[id(source)]
            for i in rows:
                for k, value in nonzeros[id(p)].get(i, ()):
                    each(p, i, k, value,
                         lambda j, k=k, column=column: column.get((k, j)))
            del self.columns[id(source)]
            b.begin_operation(force=True)
        together: Dict[int, list] = {}
        for p in step.products:
            if any(p is h for h in held):
                continue
            for i in rows:
                for k, value in nonzeros[id(p)].get(i, ()):
                    together.setdefault(k, []).append((p, i, value))
        tile = None
        for k in sorted(together):
            self.here = {}
            if k // self.width != tile:
                # A tile of rows read is held while its rows are used.
                tile, self.tiles = k // self.width, {}
            for p, i, value in together[k]:
                each(p, i, k, value, lambda j, p=p, k=k: self.entry(p, k))
        self.here, self.tiles = {}, {}
        self.slot = 0
        return acc


def _lane(trip, step) -> None:
    """The lanes form's step into the output: its products added into the
    column's registers (`_LaneTrip.acc`) at the warp's rows, a lane for its
    own element (`_lanes`)."""
    b, fp = trip.builder, trip.fp
    rows = [i for i in range(*trip.rows) if step.rows[0] <= i < step.rows[1]]
    if step.kind is trip.kinds.OPERATOR:
        trip.acc = trip.walk(step, rows, trip.acc)
        return
    for i in rows:
        trip.here = {}
        for j in range(len(trip.cols)):
            trip.slot = j
            part = trip.mixed(step, i)
            if part is not None:
                old = trip.acc.get((i, j))
                trip.acc[(i, j)] = (part if old is None
                                    else b.op('add', fp, old, part,
                                              hint='q'))
    trip.here, trip.slot = {}, 0


def _column(trip) -> None:
    """The output's columns of the pass at the warp's rows: the rows its
    steps write read once where they accumulate -- every row where the
    first assigns -- each step added in turn (`LOWERINGS`), and written
    back through the lane's pointers under its mask, column by column."""
    b, fp = trip.builder, trip.fp
    cones = trip.cones
    steps = cones.outputs
    tensor = steps[0].dest
    target = _bound(b, trip.symbols[id(tensor)])
    lo, hi = cones.rows
    rows = [i for i in range(*trip.rows) if lo <= i < hi]
    if steps[0].assigns:
        write = rows
    else:
        touched = set()
        for step in steps:
            for p in step.products:
                if p.kind is trip.kinds.OPERATOR:
                    touched |= {i for i in rows
                                if cones.nonzeros[id(p)].get(i)}
                else:
                    touched |= {i for i in rows if p.rows[0] <= i < p.rows[1]}
        write = sorted(touched)
    slots = range(len(trip.cols))
    trip.tiles, trip.acc = {}, {}
    for j in slots:
        trip.slot = j
        for i in write:
            trip.acc[(i, j)] = (None if steps[0].assigns
                                else trip.own(tensor, i, trip.col))
    trip.tiles, trip.slot = {}, 0
    for step in steps:
        LOWERINGS[step.lowering](trip, step)
    width = trip.width
    lower = int(tensor.bbox.lower()[0])
    for j in slots:
        col = trip.cols[j]
        done = set()
        for i in write:
            if i in done:
                continue
            first = i - (i - lower) % width
            tile = list(range(first, first + width))
            claim = trip.claim(tensor, first)
            if claim is not None and all((r, j) in trip.acc for r in tile):
                # A whole tile written: one access.
                values = [trip.acc[(r, j)] if trip.acc[(r, j)] is not None
                          else b.const(0.0, fp) for r in tile]
                b.store(target, b.pack(ScalarType(trip.section.dtype, width),
                                       *values, hint='q'),
                        _address(b, tensor, first, col),
                        predicate=trip.mask, align=claim)
                done.update(tile)
                continue
            value = (trip.acc[(i, j)] if trip.acc[(i, j)] is not None
                     else b.const(0.0, fp))
            b.store(target, value, _address(b, tensor, i, col),
                    predicate=trip.mask)
            done.add(i)


def _lanes(trip) -> None:
    """A trip of the lanes form: warp `w` takes the output's columns of
    class `w % classes`, one after another, at the rows of range
    `w / classes`, each lane for its own element (`element_group.Lanes`).

    The loop over a class's columns has constant bounds and the ranges
    are guards that say what share of the threads takes them
    (`lanes_active`), so that the emitter counts the average warp's
    work (`analysis.pipeline.executed_flops`)."""
    b = trip.builder
    plan = trip.group.lanes
    c0, c1 = trip.cones.cols
    width = c1 - c0
    classes = plan.classes
    warp = b.thread_id('y')
    kind = (b.op('rem', INDEX, warp, classes, hint='cls') if classes > 1
            else 0)
    part = (b.op('div', INDEX, warp, classes, hint='part')
            if len(plan.ranges) > 1 else 0)
    trips = -(-width // classes)
    # A pass takes `columns` of a class's columns at once; the plan has them
    # divide its trips, and every class's columns exist where they are more
    # than one (`element_group.lanes_plan`).
    columns = plan.columns
    passes = trips // columns

    from tensorforge.generators import element_group
    anchors = element_group.lane_anchors(trip.context)
    lexic = trip.context.target.lexic

    def range_body(rows):
        trip.rows = rows
        trip.need = trip.cones.need(range(*rows))
        trip.begin_table(rows)
        # Rolled: the columns could be written out, and what one reads of the
        # element is read again by the next rather than held across all of
        # them (`passes.licm`) -- by the compiler too, which would otherwise
        # write the few trips out and overlap the columns (`unroll=1`).
        with b.for_(0, passes, 1, hint='c', rolled=True, unroll=1) as loop:
            first = b.op('mul', INDEX, loop.induction, columns * classes,
                         hint='p')
            cols = [_index(b, 'add', _index(b, 'add', first, kind, 'p'),
                           c0 + j * classes, 'p') for j in range(columns)]
            spelled = lexic.opaque_zero('{0}') if anchors else None
            trip.anchor = (None if spelled is None else b.rawexpr(
                spelled, *(() if '{0}' not in spelled else (loop.induction,)),
                type_=INDEX, hint='anc', pure=False, movable=False,
                crosslane='{0}' in spelled, uniform=Uniformity.MULT))

            def column():
                trip.cols, trip.slot = cols, 0
                trip.small = {}
                _column(trip)

            if width % classes == 0:
                column()
            else:
                with b.if_(b.op('lt', BOOL, cols[0], c1, hint='in'),
                           attrs=_lanes_active(width, classes * trips)):
                    column()
        trip.anchor = None
        trip.end_table()

    if len(plan.ranges) == 1:
        range_body(plan.ranges[0])
        return
    for n, rows in enumerate(plan.ranges):
        with b.if_(b.op('eq', BOOL, part, n, hint='part'),
                   attrs=_lanes_active(1, len(plan.ranges))):
            range_body(rows)


#: Each step's lowering (`element_group.Step.lowering`).
LOWERINGS = {'fma': _fma, 'rows': _per_output,
             'mma_sync': _matrix, 'dmma': _matrix, 'lane': _lane}


class MultiElementSection(AbstractInstruction):
    """One trip of a section over a group of elements (see the module).

    `symbols` are the section's symbols of its inputs and outputs, by
    tensor (`id`); the operators come staged
    (`MultiElementOperators.buffers`) -- but in the lanes form, which has
    none: its operators' nonzeros are literals, or read where the operators
    are stored (`element_group.lane_literals`), through their symbols."""

    def __init__(self, context: Context, group, symbols: Dict[int, object],
                 operators: Optional[MultiElementOperators]):
        super().__init__(context)
        self._group = group
        self._symbols = dict(symbols)
        self._operators = operators
        #: The tables of numbers the section reads, by name: (dtype,
        #: numbers) (`module_tables`).
        self._tables: Dict[str, tuple] = {}
        self._is_ready = True
        section = group.section
        self._inputs = tuple(self._symbols[id(b.tensor)]
                             for b in section.inputs)
        from tensorforge.generators import element_group
        if (group.lanes is not None
                and element_group.lane_values(context) == 'memory'):
            self._inputs += tuple(self._symbols[id(t)]
                                  for t in section.operators)
        self._outputs = tuple(self._symbols[id(t)] for t in section.outputs)
        for symbol in self._inputs + self._outputs:
            symbol.add_user(self)

    def defs(self):
        return self._outputs

    def uses(self):
        return self._inputs + self._outputs

    def module_tables(self):
        """The numbers the section reads by name at namespace scope -- the
        lanes form's `table` (`_LaneTrip.number`), the places of a sparse
        operand's cells (`_sparse_read`) -- as
        `MultiElementOperators.module_tables` hands its own out."""
        return dict(self._tables)

    def barrier_scope(self) -> Uniformity:
        return Uniformity.BLOCK

    def gen_ir(self, builder) -> None:
        section = self._group.section
        if self._group.lanes is not None:
            # The lanes form: nothing staged, a lane its element's
            # (`_lanes`).
            trip = _LaneTrip(builder, self._context, self._group,
                             self._symbols)
            _lanes(trip)
            dtype = section.dtype
            self._tables = {name: (dtype, numbers)
                            for name, numbers in trip.tables.items()}
            self._tables.update(trip.int_tables)
            return
        self._tables = {}
        trip = _Trip(builder, self._context, self._group, self._symbols,
                     self._operators.buffers, self._operators.tables)
        elements = self._group.elements
        for buffer in section.inputs + section.intermediates:
            trip.buffers[id(buffer.tensor)] = builder.alloc(
                section.dtype, (elements * buffer.per_element,),
                MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                hint=_hint('g', buffer.tensor))
        # A pointer table's entries on 8 bytes, in the arena's elements.
        unit = max(1, WORD.base.size() // self._context.fp_type.size())
        for tensor in section.pointer_tables:
            trip.pointers[id(tensor)] = builder.alloc(
                Datatype.I64, (elements,), MemSpace.SHARED,
                arena=GeneralLexicon.TOTAL_SHR_MEM, hint=_hint('p', tensor),
                place_align=unit if unit > 1 else None)
        # The block meets before any warp overwrites the group's buffers, so
        # that the last trip's reads of them are done -- a barrier of the
        # instruction's own where nothing places them (`Options.
        # enable_sync_block_opt`), where the trip stages by loads and stores
        # (`element_group.async_staging`: the batch loop is text then, whose
        # back edge the placement does not see), or where the plan declined
        # the inputs' second stage (`element_group.second_stages`): ahead of
        # the staging it keeps `pir.wrap` from taking the copies a trip
        # ahead.  Otherwise the placement's, which meets them where the pass
        # put them.
        from tensorforge.generators.element_group import (async_staging,
                                                          second_stages)
        if (not self._context.get_user_options().enable_sync_block_opt
                or not async_staging(self._context)
                or second_stages(section, self._context) > section.doubled):
            builder.barrier(Participants.BLOCK)
        tokens = []
        for buffer, symbol in zip(section.inputs, self._inputs):
            tokens += self._stage(trip, buffer, symbol)
        for buffer in section.inputs + section.intermediates:
            if buffer.pitch > buffer.height:
                self._pad(trip, buffer)
        if tokens:
            builder.wait(*tokens)
        for tensor in section.pointer_tables:
            trip.publish(tensor)
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

    def _stage(self, trip, buffer, symbol) -> list:
        """The warp's element's box of an operand, from its own memory into
        the group's buffer; their tokens, where the copies are asynchronous.

        A dense operand where the target's copies are
        (`element_group.async_staging`) goes by asynchronous copies
        (`IRBuilder.copy_async`), a cell a lane a hop, every hop issued
        before the trip waits for any, so that `pir.wrap` can take them a
        trip ahead.  The last hop's lanes past the box stand outside a guard
        block, as a loader's ragged tail does: a copy predicated itself is
        one the pass does not take either.  Otherwise by loads and stores,
        and a sparse operand's cells read through its table
        (`_sparse_read`)."""
        from tensorforge.generators.element_group import async_staging
        b = trip.builder
        source = _bound(b, symbol)
        height = buffer.height
        width = buffer.cols[1] - buffer.cols[0]
        if buffer.tensor.is_dense() and async_staging(self._context):
            total = height * width
            lanes = trip.group.threads
            tokens = []
            for hop in range(0, total, lanes):
                index = _index(b, 'add', trip.lane, hop, 'j')
                guard = (contextlib.nullcontext() if hop + lanes <= total
                         else b.if_(b.op('lt', BOOL, index, total,
                                         hint='in')))
                with guard:
                    row = _index(b, 'add', b.op('rem', INDEX, index, height,
                                                hint='r'), buffer.rows[0], 'r')
                    col = _index(b, 'add', b.op('div', INDEX, index, height,
                                                hint='c'), buffer.cols[0], 'c')
                    tokens.append(b.copy_async(
                        trip.buffers[id(buffer.tensor)], source,
                        dst_index=(trip.at(buffer, row, col),),
                        src_index=(_address(b, buffer.tensor, row, col),),
                        elems=1, threads=lanes))
            return tokens
        with b.for_(trip.lane, height * width, trip.group.threads,
                    hint='j', uniform=Uniformity.LANE) as loop:
            row = _index(b, 'add', b.op('rem', INDEX, loop.induction,
                                        height, hint='r'),
                         buffer.rows[0], 'r')
            col = _index(b, 'add', b.op('div', INDEX, loop.induction,
                                        height, hint='c'),
                         buffer.cols[0], 'c')
            if buffer.tensor.is_dense():
                value = b.load(source, _address(b, buffer.tensor, row, col),
                               type_=trip.fp, hint='s')
            else:
                value = _sparse_read(b, self._context, self._tables, source,
                                     buffer.tensor,
                                     _address(b, buffer.tensor, row, col),
                                     trip.fp)
            b.store(trip.buffers[id(buffer.tensor)], value,
                    trip.at(buffer, row, col))
        return []

    @staticmethod
    def _pad(trip, buffer) -> None:
        """Zeros past the rows of every column of the warp's element, where
        the buffer has a pitch (`element_group.Buffer`): what the matrix
        tiling's padded steps of K read, to multiply zeros."""
        b = trip.builder
        pad = buffer.pitch - buffer.height
        width = buffer.cols[1] - buffer.cols[0]
        with b.for_(trip.lane, pad * width, trip.group.threads, hint='z',
                    uniform=Uniformity.LANE) as loop:
            row = _index(b, 'add', b.op('rem', INDEX, loop.induction, pad,
                                        hint='r'), buffer.rows[1], 'r')
            col = _index(b, 'add', b.op('div', INDEX, loop.induction, pad,
                                        hint='c'), buffer.cols[0], 'c')
            b.store(trip.buffers[id(buffer.tensor)], b.const(0.0, trip.fp),
                    trip.at(buffer, row, col))

    def __str__(self) -> str:
        section = self._group.section
        return (f'multi-element section, {len(section.steps)} steps into '
                f'{", ".join(t.alias for t in section.outputs)}, '
                f'{self._group.elements} elements a block')


__all__ = ['LOWERINGS', 'MultiElementOperators', 'MultiElementSection',
           'UNROLL']
