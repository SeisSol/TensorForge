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
     (`element_group.second_stages`).  Or, where the plan has a ring
     (`GroupSection.ring`, `Options.element_producer`), a producer warp
     behind the element warps fills its slots trips ahead -- a bulk copy a
     column where the copy takes the box, a cell a copy otherwise --
     and the element warps, a role of their own, take each slot as its
     bytes land and give it back once done (`MultiElementSection._ring`);
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
it -- both TF32 halves where the star step wrote them,
`element_group.Buffer.halves`), B from the operator's stage, and D
reaching its consumer's buffer, or the output from the accumulators: one
reduction a cell (`red.global.add`), a store where the step assigns.
Where the operators' nonzeros are known, `fma` walks only those
(`_sparse`): a lane executes the nonzeros of its output's row, the rows
sorted so that a warp's lanes walk about as many each.

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
import dataclasses
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


#: The first of the two named barriers of a block's roles (`IRBuilder.roles`):
#: the last two of the sixteen, apart from the ids the multiplications take
#: from 1 on.
ROLE_BARRIER = 14


def _role_split(builder, context, group):
    """The block in its two roles (`IRBuilder.roles`): the producer's
    warps behind the element warps, and theirs, each meeting on a named
    barrier of its own -- one id a role, which every thread computes as its
    role's."""
    from tensorforge.backend.pir.build import Role
    lexic = context.target.lexic
    elements = group.elements
    ident = f'({ROLE_BARRIER} + ({lexic.thread_idx_y} >= {elements}))'
    return builder.roles(
        builder.op('ge', BOOL, builder.thread_id('y'), elements,
                   hint='producer'),
        Role('producer', group.threads * group.producers, ident),
        Role('elements', group.threads * elements, ident))


def _element_pointer(builder, context, symbol, element, dtype):
    """Element `element`'s pointer to the operand of `symbol`, as the batch
    traversal binds a warp's own (`GetElementPtr._element_address`): what
    a producer's lane copies its element from."""
    import types
    from tensorforge.backend.instructions.ptr_manip import GetElementPtr
    from tensorforge.common.helper import get_extra_offset_name
    bound = _bound(builder, symbol)
    if not isinstance(getattr(bound, 'type', None), BufferType):
        raise InternalError(f'{symbol.name}: no pointer bound in the body to '
                            f'read another element\'s like')
    # The kernel's parameter, whose name is the operand's: the symbol in
    # scope here is the warp's binding of it.
    param = types.SimpleNamespace(name=symbol.obj.name, obj=symbol.obj)
    address = GetElementPtr(context, param, None)._element_address(
        '{0}', f' + {get_extra_offset_name(param)}',
        symbol.obj.storage_volume())
    ctype = dtype.ctype()
    name = f'{_hint("e", symbol.obj)}_{builder.next_index()}'
    return builder.decl_expr(f'const {ctype} *const {name}', address,
                             bound.type, symbol, args=(element,),
                             kind=Effect.READ, space=MemSpace.GLOBAL,
                             hint=name, extern=name)


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


def _stored_cells(buffer) -> list:
    """The cells of an input stored sparse that its storage keeps, over the
    group's buffer's box: `(at, place)`, where the box holds the cell in an
    element's share of the buffer -- column by column over its pitch -- and
    its place among the stored entries (`_positions`).  What the producer
    copies of the input (`MultiElementSection._produce_copies`), and what
    `_sparse_read` reads of it without the ring."""
    tensor = buffer.tensor
    lower = [int(x) for x in tensor.bbox.lower()]
    height = int(tensor.bbox.sizes()[0])
    places = _positions(tensor)
    out = []
    for col in range(*buffer.cols):
        for row in range(*buffer.rows):
            place = places[(row - lower[0]) + (col - lower[1]) * height]
            if place >= 0:
                out.append(((row - buffer.rows[0])
                            + (col - buffer.cols[0]) * buffer.pitch, place))
    return out


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
def _tcgen05_columns(section, step):
    """`(columns, at, apart)`: the columns of tensor memory a tcgen05 step
    takes, where each product's A starts, by product -- its upper TF32
    halves there and its lower ones the depth behind -- and whether the
    products are apart.  D's columns first, N rounded to 16.  Every
    product's A apart, where B is resident and the 512 columns hold them,
    so that the step stores them all ahead of one round of its products
    (`_tcgen05`); else every product's at D's end, one product a round.

    Apart is this decision, not columns that differ: a step of one product
    has its own columns either way, and with B streamed its pieces are the
    ring's chunks, each its own round."""
    from tensorforge.generators.element_group import matrix_shape
    _, n_, _ = matrix_shape('tcgen05', 100)
    n = -(-(step.rows[1] - step.rows[0]) // n_) * n_
    depths = [section.stage(p.first.tensor, (p.rows, p.depth)).depth
              for p in step.products]
    streamed = any(section.stage(p.first.tensor, (p.rows, p.depth)).streamed
                   for p in step.products)
    columns = n + 2 * sum(depths)
    if not streamed and columns <= 512:
        at, column = [], n
        for depth in depths:
            at.append(column)
            column += 2 * depth
        return columns, tuple(at), True
    return n + 2 * max(depths), tuple(n for _ in depths), False


def tmem_columns(section) -> int:
    """The columns of tensor memory the section's tcgen05 steps take, the
    widest of them -- a power of two of at least 32 -- or 0 where none is
    tcgen05's: D's columns, N rounded to 16, then both TF32 halves of A's
    deepest product (`_tcgen05`)."""
    widest = 0
    for step in section.steps:
        if step.lowering != 'tcgen05':
            continue
        widest = max(widest, _tcgen05_columns(section, step)[0])
    if not widest:
        return 0
    columns = max(32, 1 << (widest - 1).bit_length())
    if columns > 512:
        raise InternalError(f'a tcgen05 step needs {widest} columns of tensor '
                            f'memory, past its 512')
    return columns


def _module_table(instruction, prefix: str, numbers) -> str:
    """The name of a table of integers in the generated file, which says
    which numbers it holds (`multilinear._module_table_name`), kept by
    `instruction` for its `module_tables`; one zero where it holds none."""
    from tensorforge.backend.instructions.compute.multilinear import \
        _module_table_name
    numbers = tuple(int(n) for n in numbers) or (0,)
    name = _module_table_name(prefix, Datatype.I32, numbers)
    instruction._module_tables[name] = (Datatype.I32, numbers)
    instruction._context.module_tables[name] = numbers
    return name


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
        #: The block's tensor memory, where a step is tcgen05's
        #: (`tmem_columns`): allocated here, ahead of the loop, once -- an
        #: allocation gives up the block's permit to allocate again -- and
        #: freed behind the loop (`MultiElementRelease`).
        self.tmem = None
        #: The ring a producer fills the inputs through (`IRBuilder.
        #: mbarrier_ring`), where the plan has one, and by input the identity
        #: of its slots' windows (`GroupSection.ring`).
        self.ring = None
        self.ring_slots: Dict[int, object] = {}
        self._is_ready = True
        for symbol in self.symbols:
            symbol.add_user(self)

    def _budgets(self, builder) -> None:
        """The registers the producer's warpgroup gives the element warps
        (`element_group.register_budgets`): lowered first, behind the
        block's barrier raised -- a raise waits until the pool holds what it
        asks for -- each by the warpgroups of its role, whole, as the
        collective is (`engines.register_budget`)."""
        from tensorforge.backend.pir import engines
        from tensorforge.generators.element_group import register_budgets
        budgets = register_budgets(self._group)
        if budgets is None:
            return
        _, mine, theirs = budgets
        warp = builder.thread_id('y')
        elements = self._group.elements
        with builder.if_(builder.op('ge', BOOL, warp, elements,
                                    hint='producer')):
            builder.emit(engines.register_budget(mine, 'dec'))
        builder.barrier(Participants.BLOCK)
        with builder.if_(builder.op('lt', BOOL, warp, elements,
                                    hint='elements')):
            builder.emit(engines.register_budget(theirs, 'inc'))

    def _ring(self, builder, tid, threads) -> None:
        """The producer's ring, its slots' windows of every input zeroed:
        the copies write a box's rows, and the pitch past them is what the
        matrix tiling's padded steps read, zeros (`element_group.Buffer`).
        A slot's full side counts the producer's arrival, its empty side
        one of each element warp's."""
        section = self._group.section
        elements = self._group.elements
        fp = ScalarType(section.dtype)
        self.ring = builder.mbarrier_ring(
            'me', section.ring, full=1, empty=elements,
            leader='threadIdx.x == 0 && threadIdx.y == 0',
            arena=GeneralLexicon.TOTAL_SHR_MEM)
        if self.ring is None:
            raise InternalError('a producer\'s ring in a body without an '
                                'arena for its mbarriers')
        for buffer in section.inputs:
            key = self.ring_slots[id(buffer.tensor)] = object()
            for slot in range(section.ring):
                window = builder.alloc(
                    section.dtype, (elements * buffer.per_element,),
                    MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                    hint=_hint('r', buffer.tensor), stages=section.ring,
                    stage=slot, identity=key, lifetime='body', handed=True)
                with builder.for_(tid, elements * buffer.per_element,
                                  threads, hint='z',
                                  uniform=Uniformity.LANE) as loop:
                    builder.store(window, builder.const(0.0, fp),
                                  loop.induction)

    def module_tables(self):
        """The numbers this instruction reads by name at namespace scope:
        the nonzeros' cells in the operators, their contraction indices, and
        the row data (`_pack`), as `MultilinearInstruction.module_tables`
        hands its own out."""
        return dict(self._module_tables)

    def _table(self, prefix: str, numbers) -> str:
        return _module_table(self, prefix, numbers)

    def uses(self):
        return self.symbols

    def barrier_scope(self) -> Uniformity:
        return Uniformity.BLOCK

    def gen_ir(self, builder) -> None:
        section = self._group.section
        fp = ScalarType(section.dtype)
        # Every warp of the block, a producer's included.
        threads = self._group.threads * self._group.warps
        tid = builder.op('add', INDEX, builder.thread_id('x'),
                         builder.op('mul', INDEX, builder.thread_id('y'),
                                    self._group.threads, hint='w'),
                         hint='tid')
        symbols = {id(t): s for t, s in zip(section.operators, self.symbols)}
        buffers = {}
        for stage in section.stages:
            if stage.streamed:
                # Its step streams it through a ring, a trip at a time
                # (`_ring_fetch`, `_ring_put`).
                continue
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
            elif stage.order == 'canonical':
                self._canonical(builder, stage, source, shared, fp, tid,
                                threads)
            else:
                self._fragments(builder, stage, source, shared, fp, tid,
                                threads)
            buffers[(id(tensor), stage.window)] = shared
        tables = {id(step.terms): self._pack(builder, step, symbols, tid,
                                             threads)
                  for step in section.steps if step.terms is not None}
        if section.ring:
            self._ring(builder, tid, threads)
        columns = tmem_columns(section)
        # Its slot is the block's: the block's first warp writes the address
        # there, and every warp -- an element each, with a window of its own
        # of the multiplications' arena -- reads it.
        self.tmem = builder.tmem_alloc(
            columns, arena=GeneralLexicon.TOTAL_SHR_MEM) if columns else None
        if any(stage.order == 'canonical' and not stage.streamed
               for stage in section.stages):
            # The asynchronous engines read B through the async proxy, and
            # it was written through the generic one: each thread fences
            # its stores ahead of the barrier.
            from tensorforge.backend.pir.engines import FenceKind
            builder.fence(FenceKind.PROXY)
        builder.barrier(Participants.BLOCK)
        self._budgets(builder)
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

    def _fragments(self, builder, stage, source, shared, fp, tid, threads):
        """A matrix product's B in the order the instruction's fragments are
        read (`element_group.Stage`): cell after cell of its stored tiles --
        lane `l`'s register `j` of tile `(kt, nt)` is `B(kt k + l % 4 + 4 j,
        8 nt + l / 4)`, the operator's `(n0 + n, k0 + k)` inside the
        product's window and zero outside it -- written where the lane reads
        it, its two TF32 halves apart where the stage is split; a tile's
        `(kt, nt)` from the generated file's tables.  The operator is read
        at a clamped address, so that no lane reads past it."""
        from tensorforge.backend.instructions.compute.primitives import \
            nvidia
        b = builder
        (r0, r1), (d0, d1) = stage.window
        kregs = stage.k // 4
        cells = len(stage.tiles) * stage.k * 8
        kts = self._table('mbk', [kt for kt, _ in stage.tiles])
        nts = self._table('mbn', [nt for _, nt in stage.tiles])
        with b.for_(tid, cells, threads, hint='i',
                    uniform=Uniformity.LANE) as loop:
            fragment = b.op('div', INDEX, loop.induction, kregs, hint='f')
            j = b.op('rem', INDEX, loop.induction, kregs, hint='j')
            lane = b.op('rem', INDEX, fragment, 32, hint='l')
            tile = b.op('div', INDEX, fragment, 32, hint='t')
            k = b.op('add', INDEX, b.op('add', INDEX, b.op(
                'mul', INDEX, b.rawexpr(f'{kts}[{{0}}]', tile, type_=INDEX,
                                        pure=True, movable=True, hint='t'),
                stage.k, hint='k'), b.op('rem', INDEX, lane, 4, hint='k'),
                hint='k'), b.op('mul', INDEX, j, 4, hint='k'), hint='k')
            n = b.op('add', INDEX, b.op('mul', INDEX, b.rawexpr(
                f'{nts}[{{0}}]', tile, type_=INDEX, pure=True, movable=True,
                hint='t'), 8, hint='n'),
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

    @staticmethod
    def _canonical(builder, stage, source, shared, fp, tid, threads):
        """A matrix product's B in the canonical K-major layout an
        asynchronous engine reads through a descriptor
        (`element_group.Stage.canonical_offset`): `B(k, n)` the operator's
        `(n0 + n, k0 + k)` inside the product's window, zero outside it --
        past the window's contraction too, which the padded steps of K
        multiply an input's padding by (`element_group.Stage`) -- each cell
        written where its core matrix holds it, its two TF32 halves a layout
        apart, the upper ones first.  Read at a clamped address, as
        `_fragments` reads it."""
        from tensorforge.backend.instructions.compute.primitives import \
            nvidia
        b = builder
        (r0, r1), (d0, d1) = stage.window
        cells = stage.depth * stage.width
        with b.for_(tid, cells, threads, hint='i',
                    uniform=Uniformity.LANE) as loop:
            n = b.op('div', INDEX, loop.induction, stage.depth, hint='n')
            k = b.op('rem', INDEX, loop.induction, stage.depth, hint='k')
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
            # `(n / 8) depth 8 + (k / 4) 32 + (n % 8) 4 + k % 4`
            slot = b.op('add', INDEX, b.op('add', INDEX, b.op(
                'mul', INDEX, b.op('div', INDEX, n, 8, hint='s'),
                stage.depth * 8, hint='s'), b.op(
                'mul', INDEX, b.op('div', INDEX, k, 4, hint='s'), 32,
                hint='s'), hint='s'), b.op('add', INDEX, b.op(
                'mul', INDEX, b.op('rem', INDEX, n, 8, hint='s'), 4,
                hint='s'), b.op('rem', INDEX, k, 4, hint='s'), hint='s'),
                hint='s')
            if stage.split:
                (upper, lower), = nvidia.tfconvert(b, [value])
                b.store(shared, _as_float(b, upper), slot)
                b.store(shared, _as_float(b, lower),
                        b.op('add', INDEX, slot, cells, hint='s'))
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
        #: The block's tensor memory, where a step is tcgen05's
        #: (`MultiElementOperators.tmem`).
        self.tmem = None
        #: The trip's stage of D for its tcgen05 steps (`_d_stage`), once the
        #: first of them asks for it.
        self.dstage = None
        #: Whom the lowerings' barriers meet: the block, or the element
        #: warps' role where a producer fills the inputs (`_ring`).
        self.meet = Participants.BLOCK
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


def _mix(trip, steps) -> None:
    """Mixing steps of one group operand, block-wide (`element_group`'s
    `mix`): `X_d = G S_d` for each step `d` -- `G` the group operand, `S_d`
    an element's small operand, SeisSol's star for each direction -- a
    thread a row `m = e * width + p` of the products as the matrix tiling
    numbers the group's rows (`_matrix`): element `e`'s column `p`.

    A thread holds column `p` of every step's `S_d` in registers -- the
    star's 27 numbers -- and computes 16 bytes of rows of its column of
    every `X_d` at a time: `G`'s rows read 16 bytes for each `q` of the
    contraction, where its pitch allows (`element_group._pitched`), and each
    `X_d`'s written in one access where the consumer's pitch does -- rows of
    the group 16 bytes times an odd number apart, eight threads' rows on
    eight bank groups.  Rows past the window are written zero: the
    destination's padding, which the tiling's padded K reads.  Where the
    tiling reads a destination in TF32 halves (`element_group.Buffer.
    halves`), each value is split as the tiling would split it, and both
    halves are written, each 16 bytes at a time where the pitch allows.
    The block's threads past the group's rows take more of them: `sets`
    threads a row, each every `sets`-th group of rows."""
    b, fp = trip.builder, trip.fp
    section, group = trip.section, trip.group
    lead = steps[0].products[0]
    source = section.buffer(lead.first.tensor)
    rows, cols, depth = lead.rows, lead.cols, lead.depth
    width = cols[1] - cols[0]
    height = rows[1] - rows[0]
    threads = group.threads * group.elements
    count = group.elements * width
    sets = threads // count
    vec = 16 // section.dtype.size()
    blocks = -(-height // vec)
    rounds = -(-blocks // sets)
    vector = ScalarType(section.dtype, vec)

    def fits(buffer, first):
        # A block's rows one 16-byte access: aligned, and inside the column
        # -- past the window only into its padding.
        lo = first - buffer.rows[0]
        return (vec > 1 and buffer.pitch % vec == 0 and lo % vec == 0
                and lo + blocks * vec <= buffer.pitch
                and (rows[1] == buffer.rows[1] or height % vec == 0))

    tid = _index(b, 'add', b.op('mul', INDEX, b.thread_id('y'),
                                group.threads, hint='t'), trip.lane, 't')
    m = b.op('rem', INDEX, tid, count, hint='m')
    lot = b.op('div', INDEX, tid, count, hint='s')
    e = b.op('div', INDEX, m, width, hint='e')
    col = _index(b, 'add', b.op('rem', INDEX, m, width, hint='p'), cols[0],
                 'p')

    def index(buffer, row, column):
        # Element `e`'s `(row, column)` of `buffer`.
        base = b.op('mul', INDEX, e, buffer.per_element, hint='b')
        return _index(b, 'add', base, _box_index(
            b, row, column, (buffer.rows[0], buffer.cols[0]), buffer.pitch),
            'a')

    with b.if_(b.op('lt', BOOL, lot, sets, hint='in'),
               attrs=_lanes_active(sets * count, threads)):
        # The thread's column of every step's small operand.
        numbers = []
        for step in steps:
            small = section.buffer(step.products[0].second.tensor)
            numbers.append([b.load(trip.buffers[id(small.tensor)],
                                   index(small, q, col), type_=fp, hint='s')
                            for q in range(*depth)])
        loads, stores = fits(source, rows[0]), [
            fits(section.buffer(step.dest), rows[0]) for step in steps]
        last = height - (blocks - 1) * vec

        def block(g):
            first = b.op('mul', INDEX, g, vec, hint='l')
            # Which of the block's rows are the window's: all but in the
            # last block, where `last` are.
            live = [None if v < last else b.op(
                'lt', BOOL, first, height - v, hint='in')
                for v in range(vec)]
            acc = [[None] * vec for _ in steps]
            row = _index(b, 'add', first, rows[0], 'l')
            for k, q in enumerate(range(*depth)):
                at = index(source, row, q)
                if loads:
                    x = b.load(trip.buffers[id(source.tensor)], at,
                               type_=vector, align=16, hint='x')
                    xs = [b.extract(x, v, hint='x') for v in range(vec)]
                else:
                    xs = []
                    for v in range(vec):
                        cell = _index(b, 'add', at, v, 'a')
                        if live[v] is not None:
                            cell = b.op('select', INDEX, live[v], cell, at,
                                        hint='a')
                        xs.append(b.load(trip.buffers[id(source.tensor)],
                                         cell, type_=fp, hint='x'))
                for d in range(len(steps)):
                    w = numbers[d][k]
                    for v in range(vec):
                        acc[d][v] = (b.op('mul', fp, w, xs[v], hint='y')
                                     if acc[d][v] is None else
                                     b.op('fma', fp, w, xs[v], acc[d][v],
                                          hint='y'))
            for d, step in enumerate(steps):
                dest = section.buffer(step.dest)
                target = trip.buffers[id(step.dest)]
                at = index(dest, row, col)
                values = [acc[d][v] if live[v] is None else b.op(
                    'select', fp, live[v], acc[d][v], b.const(0.0, fp),
                    hint='y') for v in range(vec)]
                writes = [(at, values)]
                if dest.halves:
                    # The tiling reads the destination split
                    # (`element_group.Buffer.halves`): each value's TF32
                    # halves, the upper at its cell and the lower past the
                    # group's upper ones -- the split the tiling would make
                    # at every read of the value, made once.
                    from tensorforge.backend.instructions.compute.primitives \
                        import nvidia
                    split = nvidia.tfconvert(b, values)
                    writes = [(at, [_as_float(b, u) for u, _ in split]),
                              (_index(b, 'add', at,
                                      dest.half_offset(group.elements), 'a'),
                               [_as_float(b, l) for _, l in split])]
                for cell, data in writes:
                    if stores[d]:
                        b.store(target, b.pack(vector, *data, hint='y'),
                                cell, align=16)
                        continue
                    for v in range(vec):
                        b.store(target, data[v],
                                _index(b, 'add', cell, v, 'a'),
                                predicate=live[v])

        with b.for_(0, rounds, 1, hint='g') as loop:
            g = _index(b, 'add', b.op('mul', INDEX, loop.induction, sets,
                                      hint='g'), lot, 'g')
            if blocks % sets == 0:
                block(g)
            else:
                with b.if_(b.op('lt', BOOL, g, blocks, hint='in'),
                           attrs=_lanes_active(blocks, rounds * sets)):
                    block(g)


def _units(steps) -> list:
    """The section's steps as they are lowered: a run of `mix` steps of one
    group operand and window together (`_mix`) -- none reading what another
    writes -- every other step alone."""
    def key(step):
        p = step.products[0]
        return (id(p.first.tensor), p.rows, p.cols, p.depth)

    units: list = []
    for step in steps:
        run = units[-1] if units else None
        if (step.lowering == 'mix' and run and run[0].lowering == 'mix'
                and key(run[0]) == key(step)
                and not any(id(t) in {id(s.dest) for s in run}
                            for t in step.reads())):
            run.append(step)
        else:
            units.append([step])
    return units


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
    column, the contraction contiguous -- in F32 its values' TF32 halves
    where its producer wrote those (`element_group.Buffer.halves`), else
    split here; B the product's stage, zero outside its window and in the
    padding, split there or here.  A warp takes the tiles of D in turn, D
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
    if not ldmatrix and any(section.buffer(p.second.tensor).halves
                            for p in step.products):
        raise InternalError(f'{step.dest.alias}: A held in TF32 halves '
                            f'where it is read lane by lane')
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

    from tensorforge.generators.element_group import tile_units
    units = tile_units(section, step, trip.group.elements,
                       nvidia.sm_of(trip.context))
    slots, together = units.slots, units.group
    count = len(step.products)
    # The walk's tables (`element_group.TileUnits`): a warp's units, a
    # unit's M tiles and the N tiles of its slots, and by unit and product
    # the range of the entries -- a k-block, and a slot's stored tile.
    firsts, starts, entries = [0], [0], []
    for taken in units.warps:
        firsts.append(firsts[-1] + len(taken))
    for walked in units.entries:
        for listed in walked:
            entries += listed
            starts.append(starts[-1] + len(listed))
    t_warp = trip.table('mtw', firsts)
    t_unit = trip.table('mtu', [u for taken in units.warps for u in taken])
    t_range = trip.table('mtm', [x for lo, hi, _ in units.units
                                 for x in (lo, hi)])
    t_slot = trip.table('mtn', [nt for _, _, nts in units.units
                                for nt in nts])
    t_first = trip.table('mte', starts)
    t_block = trip.table('mtk', [kt for kt, _ in entries])
    t_tile = trip.table('mto', [o for _, offsets in entries
                                for o in offsets])
    # Where the N grid starts below the step's rows (`Stage.n0`): the rows
    # of a tile the step does not have are left out of D.
    grid = section.stage(step.products[0].first.tensor,
                         (step.products[0].rows, step.products[0].depth)).n0
    shift = rows[0] - grid

    def read(name, index, hint):
        return b.rawexpr(f'{name}[{{0}}]', index, type_=INDEX, pure=True,
                         movable=True, hint=hint)

    def at_(index, scale, offset, hint):
        return _index(b, 'add', b.op('mul', INDEX, index, scale, hint=hint),
                      offset, hint)

    def a_tile(mt, buffer, source, kt, k_lo, depth, offset=0):
        """A's registers of M tile `mt` at K `kt` of a product: by ldmatrix
        -- the padding and the clamped rows past the group read numbers,
        zeros or rows whose D is not written -- from the cells `offset`
        further on where that is given (a buffer's lower TF32 halves), or
        lane by lane, zero past the group and past the product's depth."""
        m0 = b.op('mul', INDEX, mt, m_, hint='m')
        if ldmatrix:
            lm = b.op('add', INDEX, m0, lrow, hint='m')
            if M % m_:
                lm = b.op('select', INDEX, b.op('lt', BOOL, lm, M, hint='in'),
                          lm, 0, hint='m')
            at = _index(b, 'add', rows_of(lm, buffer, cols[0], _index(
                b, 'add', b.op('add', INDEX, lcol, kt, hint='k'), k_lo,
                'k')), offset, 'a')
            return list(b.matrix_load(source, at, count=4, type_=fp,
                                      origins=[a_cells[slot] for slot
                                               in sorted(a_cells)],
                                      hint='a'))
        a = []
        for r0, c0 in a_cells.values():
            m = b.op('add', INDEX, m0, _index(b, 'add', quad, r0, 'm'),
                     hint='m')
            live = b.op('lt', BOOL, m, M, hint='in') if M % m_ else None
            kk = b.op('add', INDEX, _index(b, 'add', inquad, c0, 'k'), kt,
                      hint='k')
            if depth % k_:
                deep = b.op('lt', BOOL, kk, depth, hint='in')
                live = deep if live is None else b.op('and', BOOL, live,
                                                      deep, hint='in')
            at = rows_of(m, buffer, cols[0], _index(b, 'add', kk, k_lo, 'k'))
            if live is None:
                a.append(b.load(source, at, type_=fp, hint='a'))
                continue
            at = b.op('select', INDEX, live, at, 0, hint='a')
            value = b.load(source, at, type_=fp, hint='a')
            a.append(b.op('select', fp, live, value, b.const(0.0, fp),
                          hint='a'))
        return a

    def b_tile(staged, stage, place):
        """B's registers of the stored tile `place`, a lane's side by side:
        one load (`element_group.Stage`); its TF32 halves where split."""
        at = b.op('mul', INDEX, b.op('add', INDEX, b.op(
            'mul', INDEX, place, 32, hint='f'), trip.lane, hint='f'),
            stage.regs, hint='f')
        if stage.regs == 1:
            regs = [b.load(staged, at, type_=fp, hint='b')]
        else:
            vec = b.load(staged, at, type_=ScalarType(
                section.dtype, stage.regs), hint='b',
                align=stage.regs * section.dtype.size())
            regs = [b.extract(vec, r, hint='b') for r in range(stage.regs)]
        if stage.split:
            return None, [(nvidia._as_tf32(b, regs[j]),
                           nvidia._as_tf32(b, regs[kregs + j]))
                          for j in range(kregs)]
        return regs, None

    def epilogue(accs, m0, n0):
        """D, to the intermediate's buffer, or to the output from the
        accumulators: the cells of the group's rows and the step's rows,
        the padding's left out."""
        pointers = {}
        for slot, (r0, c0) in d_cells.items():
            m = b.op('add', INDEX, m0, _index(b, 'add', quad, r0, 'm'),
                     hint='m')
            n = b.op('add', INDEX, n0, _index(b, 'add', b.op(
                'mul', INDEX, inquad, 2, hint='n'), c0 - shift, 'n'),
                hint='n')
            live = b.op('lt', BOOL, n, height, hint='in')
            if shift:
                live = b.op('and', BOOL, live, b.op('ge', BOOL, n, 0,
                                                    hint='in'), hint='in')
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

    # A warp's units in turn; a unit's M tiles `together` at a time, D of
    # them and of its slots' N tiles in registers; a product's entries, A
    # read once an M tile and k-block, B once a slot -- a tile of zeros
    # neither read nor multiplied.
    w = trip.slot
    with b.for_(read(t_warp, w, 'u'), read(t_warp, _index(b, 'add', w, 1,
                                                          'u'), 'u'), 1,
                hint='u', uniform=Uniformity.MULT) as walk:
        unit = read(t_unit, walk.induction, 'u')
        lo = read(t_range, at_(unit, 2, 0, 'u'), 'm')
        hi = read(t_range, at_(unit, 2, 1, 'u'), 'm')
        nts = [read(t_slot, at_(unit, slots, s, 'u'), 'n')
               for s in range(slots)]
        last = _index(b, 'add', hi, -1, 'g')
        with b.for_(lo, hi, together, hint='g',
                    uniform=Uniformity.MULT) as tiles:
            g = tiles.induction
            # The group's M tiles, those past the unit clamped to its last:
            # what they multiply again, their D does not keep.
            mts = [g] + [b.op('select', INDEX, b.op(
                'lt', BOOL, _index(b, 'add', g, q, 'g'), hi, hint='in'),
                _index(b, 'add', g, q, 'g'), last, hint='g')
                for q in range(1, together)]
            accs = {(q, s): [b.declare(fp, hint='c', uniform=Uniformity.LANE)
                             for _ in d_cells]
                    for q in range(together) for s in range(slots)}
            for p, product in enumerate(step.products):
                buffer = section.buffer(product.second.tensor)
                source = trip.buffers[id(product.second.tensor)]
                stage = section.stage(product.first.tensor,
                                      (product.rows, product.depth))
                staged = trip.operators[(id(stage.tensor), stage.window)]
                k_lo = product.depth[0]
                depth = product.depth[1] - k_lo
                first = at_(unit, count, p, 'e')
                with b.for_(read(t_first, first, 'e'),
                            read(t_first, _index(b, 'add', first, 1, 'e'),
                                 'e'), 1, hint='e',
                            uniform=Uniformity.MULT) as walked:
                    e = walked.induction
                    kt = b.op('mul', INDEX, read(t_block, e, 'k'), k_,
                              hint='k')
                    a = [a_tile(mt, buffer, source, kt, k_lo, depth)
                         for mt in mts]
                    if buffer.halves:
                        # A as its producer split it (`element_group.
                        # Buffer.halves`): the upper halves where the values
                        # would be, the lower ones from their block, as the
                        # instruction's operands -- nothing converted.
                        lower = buffer.half_offset(trip.group.elements)
                        halves = [[(nvidia._as_tf32(b, u),
                                    nvidia._as_tf32(b, l))
                                   for u, l in zip(x, a_tile(
                                       mt, buffer, source, kt, k_lo, depth,
                                       lower))]
                                  for mt, x in zip(mts, a)]
                    else:
                        halves = ([nvidia.tfconvert(b, x) for x in a]
                                  if atom.mode == nvidia.MMAMode.TF32
                                  else None)
                    for s in range(slots):
                        place = read(t_tile, at_(e, slots, s, 'o'), 'o')
                        # A slot's tile where it holds one: always, with
                        # one slot, whose entries are its kept tiles.
                        with (contextlib.nullcontext() if slots == 1 else
                              b.if_(b.op('ge', BOOL, place, 0, hint='in'))):
                            regs, split = b_tile(staged, stage, place)
                            for q in range(together):
                                products = atom.products(
                                    b, a[q], regs, b_split=split,
                                    a_split=None if halves is None
                                    else halves[q])
                                for term, (fa, fb) in enumerate(products):
                                    atom.asmcall(b, accs[(q, s)], fa, fb,
                                                 accs[(q, s)],
                                                 term=atom.term(term))
            for q in range(together):
                mq = _index(b, 'add', g, q, 'g')
                for s in range(slots):
                    with b.if_(b.op('and', BOOL, b.op(
                            'lt', BOOL, mq, hi, hint='in'), b.op(
                            'ge', BOOL, nts[s], 0, hint='in'), hint='in')):
                        epilogue(accs[(q, s)],
                                 b.op('mul', INDEX, mq, m_, hint='m'),
                                 b.op('mul', INDEX, nts[s], n_, hint='n'))


def _group_index(b, m, width, buffer, col_lo, row):
    """The index in `buffer` of row `row` of the group's row `m`: element
    `m / width`, the element's column `col_lo + m % width` (`_matrix`'s
    rows)."""
    e = b.op('div', INDEX, m, width, hint='e')
    j = _index(b, 'add', b.op('rem', INDEX, m, width, hint='j'),
               col_lo - buffer.cols[0], 'j')
    base = b.op('mul', INDEX, e, buffer.per_element, hint='b')
    return _index(b, 'add', base, _box_index(
        b, row, j, (buffer.rows[0], 0), buffer.pitch), 'a')


def _d_writer(trip, step):
    """How the asynchronous lowerings hand a cell `(m, n)` of D to the
    step's destination, as `_matrix` hands its own: into the
    intermediate's buffer, added to it where the step accumulates; or to
    the output through the pointer of the row's element (`_Trip.pointer`),
    one reduction a cell where the step accumulates and a store where it
    assigns, nothing where the element is masked -- and nothing for a row
    past the group or a column past the step's rows.  `key` names a row
    whose pointer is read once for all its cells."""
    b = trip.builder
    section = trip.section
    fp = trip.fp
    rows, cols = step.rows, step.cols
    width = cols[1] - cols[0]
    height = rows[1] - rows[0]
    M = trip.group.elements * width
    target = trip.buffers[id(step.dest)] if step.dest.is_tmp else None
    dest = section.buffer(step.dest) if step.dest.is_tmp else None
    pointers = {}

    def write(m, n, value, key, padded: bool) -> None:
        live = None
        if not isinstance(n, int):
            live = b.op('lt', BOOL, n, height, hint='in')
        elif n >= height:
            return
        if padded:
            inside = b.op('lt', BOOL, m, M, hint='in')
            live = inside if live is None else b.op('and', BOOL, live,
                                                    inside, hint='in')
        if target is None:
            if key not in pointers:
                e = b.op('div', INDEX, m, width, hint='e')
                if padded:
                    e = b.op('select', INDEX, b.op('lt', BOOL, m, M,
                                                   hint='in'), e, 0,
                             hint='e')
                col = _index(b, 'add', b.op('rem', INDEX, m, width,
                                            hint='j'), cols[0], 'j')
                pointers[key] = trip.pointer(step.dest, e) + (col,)
            pointer, held, col = pointers[key]
            keep = held if live is None else b.op('and', BOOL, live, held,
                                                  hint='in')
            b.store(pointer, value, _address(
                b, step.dest, _index(b, 'add', n, rows[0], 'i'), col),
                predicate=keep, atomic=not step.assigns)
            return
        at = _group_index(b, m, width, dest, cols[0],
                          _index(b, 'add', n, rows[0], 'i'))
        if live is not None:
            at = b.op('select', INDEX, live, at, 0, hint='a')
        if not step.assigns:
            value = b.op('add', fp, b.load(target, at, type_=fp, hint='q'),
                         value, hint='q')
        b.store(target, value, at, predicate=live)
    return write


def _round_up(value: int, step: int) -> int:
    return -(-value // step) * step


def _terms(stage):
    """The products of a step of K on an asynchronous engine, as `(A's
    half, B's half)`: 3xTF32's three where B is split (`TF32_TERMS`); A's
    two halves by B's one where it is not, as mma_sync's tiling takes a B
    staged whole -- the operator then read in TF32, where its halves would
    not leave the group room."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    return nvidia.TF32_TERMS if stage.split else ((0, 0), (1, 0))


def _ring(trip, step, stages):
    """The step's ring in the block's shared memory (`element_group.
    RING_SLOTS` slots of `RING_CHUNK` rows of K of B, both halves where
    split), on 128 bytes, and the slot's size in elements."""
    from tensorforge.generators.element_group import (RING_CHUNK,
                                                      RING_SLOTS)
    width = max(st.width for st in stages)
    halves = 2 if stages[0].split else 1
    slot = RING_CHUNK * width * halves
    ring = trip.builder.alloc(trip.section.dtype, (RING_SLOTS * slot,),
                              MemSpace.SHARED,
                              arena=GeneralLexicon.TOTAL_SHR_MEM,
                              hint=_hint('r', step.dest), place_align=32)
    return ring, slot


def _ring_cells(trip, stage, k0, rows):
    """The hops of a chunk's cells over the block's threads (`_ring_fetch`,
    `_ring_put`): a hop's cell `(n, k)` -- row `n` of N, row `k` of the
    chunk's K -- the operator's `(row, col)` there and whether that is
    inside the stage's window, by hop; the last hop's lanes past the chunk
    take the first cell and `live` says which they are."""
    b = trip.builder
    (r0, r1), (d0, d1) = stage.window
    cells = rows * stage.width
    threads = 32 * trip.group.elements
    tid = b.op('add', INDEX, trip.lane, b.op('mul', INDEX, trip.slot, 32,
                                             hint='tid'), hint='tid')
    for hop in range(0, cells, threads):
        index = _index(b, 'add', tid, hop, 'i')
        live = None
        if hop + threads > cells:
            live = b.op('lt', BOOL, index, cells, hint='in')
            index = b.op('select', INDEX, live, index, 0, hint='i')
        n = b.op('div', INDEX, index, rows, hint='n')
        k = b.op('rem', INDEX, index, rows, hint='k')
        row = _index(b, 'add', n, stage.n0, 'r')
        col = _index(b, 'add', k, d0 + k0, 'c')
        inside = b.op('and', BOOL, b.op(
            'and', BOOL, b.op('ge', BOOL, row, r0, hint='in'),
            b.op('lt', BOOL, row, r1, hint='in'), hint='in'),
            b.op('lt', BOOL, col, d1, hint='in'), hint='in')
        yield n, k, row, col, inside, live


def _ring_fetch(trip, stage, k0, rows) -> list:
    """Rows `k0` to `k0 + rows` of K of the stage's B as the kernel is
    handed it, into registers: a value a hop of the block's threads over the
    chunk's cells (`_ring_cells`), read as `MultiElementOperators.
    _canonical` reads it -- a cell outside the window reads the operator's
    first, which `_ring_put` takes for a zero.  No load waits for another,
    and nothing reads them before the chunk is put: the loads are in flight
    while the engine multiplies the chunk before (`_pieces`)."""
    b = trip.builder
    fp = trip.fp
    source = _bound(b, trip.symbols[id(stage.tensor)])
    values = []
    for _, _, row, col, inside, _ in _ring_cells(trip, stage, k0, rows):
        address = b.op('select', INDEX, inside,
                       _address(b, stage.tensor, row, col), 0, hint='a')
        values.append(b.load(source, address, type_=fp, hint='k'))
    return values


def _ring_put(trip, stage, ring, at, k0, rows, values) -> None:
    """The chunk `_ring_fetch` read into the ring from element `at` on, in
    the canonical K-major order over `rows` -- core matrices of eight rows of
    N by four of K, 128 bytes apart along K and `rows` x 32 along N -- the
    upper halves, then the lower: every thread of the block its hops' cells,
    zeros outside the window -- past its contraction too
    (`element_group.Stage`).  The engine reads them through the async
    proxy: the caller fences and meets."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    b = trip.builder
    fp = trip.fp
    cells = rows * stage.width
    for (n, k, _, _, inside, live), value in zip(
            _ring_cells(trip, stage, k0, rows), values):
        value = b.op('select', fp, inside, value, b.const(0.0, fp),
                     hint='k')
        place = b.op('add', INDEX, b.op('add', INDEX, b.op(
            'mul', INDEX, b.op('div', INDEX, n, 8, hint='s'), rows * 8,
            hint='s'), b.op('mul', INDEX, b.op('div', INDEX, k, 4,
                                               hint='s'), 32, hint='s'),
            hint='s'), b.op('add', INDEX, b.op(
                'mul', INDEX, b.op('rem', INDEX, n, 8, hint='s'), 4,
                hint='s'), b.op('rem', INDEX, k, 4, hint='s'), hint='s'),
            hint='s')
        place = _index(b, 'add', place, at, 's')
        if stage.split:
            (upper, lower), = nvidia.tfconvert(b, [value])
            b.store(ring, _as_float(b, upper), place, predicate=live)
            b.store(ring, _as_float(b, lower),
                    b.op('add', INDEX, place, cells, hint='s'),
                    predicate=live)
        else:
            b.store(ring, value, place, predicate=live)


class _Piece:
    """A piece of a step's B the engine multiplies through descriptors:
    the rows `k0` to `k0 + rows` of K of a product's stage, in the
    canonical K-major order over `rows` from element `at` of `buffer` --
    the resident stage, whole (`rows` its depth), or a slot of the step's
    ring (`_pieces`)."""

    def __init__(self, product, stage, k0, rows, buffer, at):
        self.product, self.stage = product, stage
        self.k0, self.rows = k0, rows
        self.buffer, self.at = buffer, at

    def descriptors(self, b, kt, form):
        """The descriptors of the step of K at `kt` of the piece: the upper
        halves', and the lower halves' where the stage is split."""
        from tensorforge.backend.pir.engines import MatrixDescType
        desc = MatrixDescType(Datatype.TF32, lbo=self.stage.leading,
                              sbo=self.rows * 32, form=form)
        half = self.rows * self.stage.width
        return [b.matrix_desc(self.buffer, self.at + h * half + kt * 8,
                              desc=desc, hint='b')
                for h in range(1 + self.stage.split)]


def _pieces(trip, step):
    """A step's B as the engine takes it (`_Piece`), and how the next piece
    gets where it is read: the resident stages, a piece a product; or, where
    they are streamed (`Stage.streamed`), the chunks of the step's ring
    (`_ring`), each product's depth cut at `RING_CHUNK`, the slots in turn,
    and a callable that puts piece `i` into its slot (`_ring_put`) and
    reads piece `i + 1` ahead (`_ring_fetch`) -- None for the resident
    ones."""
    from tensorforge.generators.element_group import (RING_CHUNK,
                                                      RING_SLOTS)
    section = trip.section
    stages = [section.stage(p.first.tensor, (p.rows, p.depth))
              for p in step.products]
    if not stages[0].streamed:
        return [_Piece(p, st, 0, st.depth,
                       trip.operators[(id(st.tensor), st.window)], 0)
                for p, st in zip(step.products, stages)], None
    ring, slot = _ring(trip, step, stages)
    pieces = []
    for product, stage in zip(step.products, stages):
        for k0 in range(0, stage.depth, RING_CHUNK):
            at = (len(pieces) % RING_SLOTS) * slot
            pieces.append(_Piece(product, stage, k0,
                                 min(RING_CHUNK, stage.depth - k0), ring,
                                 at))

    # A chunk is read a piece ahead: putting piece `i` into its slot, the
    # block issues the loads of piece `i + 1`, which the engine's products of
    # piece `i` and the waits behind them hide, and puts them a piece later.
    # Read where it is put, the load was a latency a hop the block sat out
    # before every chunk (p5d's ncu: 46 % of tcgen05's samples at o8).
    ahead = {}

    def fetch(i):
        piece = pieces[i]
        return _ring_fetch(trip, piece.stage, piece.k0, piece.rows)

    def copy(i):
        piece = pieces[i]
        values = ahead.pop(i) if i in ahead else fetch(i)
        _ring_put(trip, piece.stage, ring, piece.at, piece.k0, piece.rows,
                  values)
        if i + 1 < len(pieces):
            ahead[i + 1] = fetch(i + 1)
    return pieces, copy


#: Where the four 8x8 matrices of a warp's A start in its 16 rows of a step
#: of K (`IRBuilder.matrix_load`'s `origins`): mma_sync's A of m16n8k8 over
#: TF32, which a warp of a warpgroup holds as wgmma's A from registers
#: (`nvidia.wgmma_a_fragment_bits`).
A_ORIGINS = ((0, 0), (8, 0), (0, 4), (8, 4))


def _wgmma(trip, step) -> None:
    """The tiling of an F32 operator step on wgmma (sm_90a), 3xTF32: a
    warpgroup takes tiles of 64 of the group's rows and the whole of N --
    the step's rows rounded to 8, one instruction -- D in its registers
    across the products (`wgmma_d_fragment_bits`), then handed to the
    destination as `_matrix` hands its own (`_d_writer`).

    A is wgmma's from registers, which a warp of the warpgroup holds as
    mma_sync's A, its 16 rows (`wgmma_a_fragment_bits`): ldmatrix over the
    pitched buffer, as `_matrix` loads it, then the two TF32 halves.  B is
    the product's stage in the canonical K-major order, its halves through
    two descriptors.  A step of K is the fence of the registers, the three
    products (`TF32_TERMS`), the commit and the wait for the group: the
    next step's A rewrites registers the products read.  Or, as
    `Options.wgmma_pending` asks, the wait leaves that many of the latest
    groups in flight: each step's A is in registers of its own, so the
    next step loads and splits its A while the last one's products run,
    and a wait retires the steps that many behind -- their A may be
    written again behind it -- and the last one, D's.

    Where B is streamed (`_pieces`), the first chunk is copied ahead; a
    chunk's steps of K take their A into registers of their own, and one
    commit closes them; while the engine multiplies, the block converts
    the next chunk into the other slot -- last read by the chunk before,
    which its wait retired -- from the registers it read it into a chunk
    before, and reads the one after it, then the wait, the fence of the
    async proxy and the block's barrier.  A chunk waits for itself whatever
    `wgmma_pending` says: the block's next copy writes the slot the chunk
    before read, which every warpgroup has to have retired.

    The tile loop runs as often in every warpgroup -- the collective's
    arrival is the block's, a multiplication being a warp -- so that a
    warpgroup past the last tile multiplies the first rows again and
    writes nothing."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    from tensorforge.backend.pir.engines import FenceKind
    from tensorforge.generators.element_group import (WARPGROUP_WARPS,
                                                      matrix_shape)
    b = trip.builder
    section = trip.section
    fp = trip.fp
    rows, cols = step.rows, step.cols
    width = cols[1] - cols[0]
    height = rows[1] - rows[0]
    m_, n_, k_ = matrix_shape('wgmma', 90)
    n = _round_up(height, n_)
    M = trip.group.elements * width
    groups = trip.group.elements // WARPGROUP_WARPS
    tiles = -(-M // m_)
    passes = -(-tiles // groups)
    padded = passes * groups * m_ > M
    lane = trip.lane
    wg = b.op('div', INDEX, trip.slot, WARPGROUP_WARPS, hint='wg')
    warp = b.op('rem', INDEX, trip.slot, WARPGROUP_WARPS, hint='w')
    quad = b.op('div', INDEX, lane, 4, hint='q')
    inquad = b.op('rem', INDEX, lane, 4, hint='q')
    # ldmatrix.x4 (`IRBuilder.matrix_load`): lane `l` names row `l % 8 + 8
    # ((l / 8) % 2)` of the warp's 16 at column `4 (l / 16)` (`_matrix`).
    lrow = b.op('add', INDEX, b.op('rem', INDEX, lane, 8, hint='l'),
                b.op('mul', INDEX, b.op('rem', INDEX, b.op(
                    'div', INDEX, lane, 8, hint='l'), 2, hint='l'), 8,
                    hint='l'), hint='l')
    lcol = b.op('mul', INDEX, b.op('div', INDEX, lane, 16, hint='l'), 4,
                hint='l')
    layouts = (nvidia.wgmma_a_fragment_bits(), None,
               nvidia.wgmma_d_fragment_bits(n))
    write = _d_writer(trip, step)
    with b.for_(0, passes, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
        tile = b.op('add', INDEX, wg, b.op('mul', INDEX, loop.induction,
                                           groups, hint='t'), hint='t')
        top = b.op('add', INDEX, b.op('mul', INDEX, tile, m_, hint='m'),
                   b.op('mul', INDEX, warp, 16, hint='m'), hint='m')
        lm = b.op('add', INDEX, top, lrow, hint='m')
        if padded:
            lm = b.op('select', INDEX, b.op('lt', BOOL, lm, M, hint='in'),
                      lm, 0, hint='m')
        accs = [b.declare(fp, hint='c', uniform=Uniformity.LANE)
                for _ in range(n // 2)]
        pieces, copy = _pieces(trip, step)
        # The groups committed and not retired, oldest first: the A
        # registers each read, and its descriptors.
        flight = []
        pending = (trip.context.get_user_options().wgmma_pending
                   if copy is None else 0)
        if copy is not None:
            copy(0)
            b.fence(FenceKind.PROXY)
            b.barrier(trip.meet)
        first = True
        for i, piece in enumerate(pieces):
            product, stage = piece.product, piece.stage
            group = section.buffer(product.second.tensor)
            source = trip.buffers[id(product.second.tensor)]
            terms = _terms(stage)
            k_lo = product.depth[0] + piece.k0
            registers, reads = list(accs), []
            for kt in range(0, piece.rows, k_):
                at = _group_index(b, lm, width, group, cols[0],
                                  _index(b, 'add', lcol, k_lo + kt, 'k'))
                a = list(b.matrix_load(source, at, count=4, type_=fp,
                                       origins=A_ORIGINS, hint='a'))
                split = nvidia.tfconvert(b, a)
                halves = ([u for u, _ in split], [lo for _, lo in split])
                bs = piece.descriptors(b, kt, 'wgmma')
                own = accs + halves[0] + halves[1]
                b.fence(FenceKind.REGISTERS, engine='wgmma', registers=own)
                for ia, ib in terms:
                    b.wgmma(n, halves[ia], bs[ib], accs,
                            scale_d=0 if first else 1, layouts=layouts)
                    first = False
                if copy is None:
                    b.commit_group('wgmma')
                    flight.append((halves[0] + halves[1], bs))
                    if len(flight) > pending:
                        flight = _wgmma_retire(b, flight, pending, accs)
                    continue
                registers += halves[0] + halves[1]
                reads += bs
            if copy is None:
                continue
            b.commit_group('wgmma')
            if i + 1 < len(pieces):
                copy(i + 1)
            b.matrix_wait('wgmma', 0, registers=registers, reads=reads)
            b.fence(FenceKind.PROXY)
            b.barrier(trip.meet)
        if flight:
            _wgmma_retire(b, flight, 0, accs)
        # D: register `i` of the lane is row `16 w + g + 8 ((i % 4) / 2)`,
        # column `8 (i / 4) + 2 t + i % 2` (PTX's D of m64nN).
        for i in range(n // 2):
            r = 8 * ((i % 4) // 2)
            m = b.op('add', INDEX, top, _index(b, 'add', quad, r, 'm'),
                     hint='m')
            col = b.op('add', INDEX, b.op('mul', INDEX, inquad, 2,
                                          hint='n'), 8 * (i // 4) + i % 2,
                       hint='n')
            write(m, col, accs[i], ('w', r), padded)


#: Columns a warp loads of tensor memory at once in the epilogue: D's row
#: a thread, a few of its columns live at a time.
TMEM_CHUNK = 32


def _wgmma_retire(b, flight, keep: int, accs) -> list:
    """The wait that leaves the `keep` latest groups of `flight` in flight
    (`wgmma.wait_group`): the A registers and the descriptors of the ones
    it retires may be written again behind it, and where it retires all of
    them, D's accumulators are valid.  The groups still in flight."""
    done, rest = flight[:len(flight) - keep], flight[len(flight) - keep:]
    registers = [r for regs, _ in done for r in regs]
    if not keep:
        registers = list(accs) + registers
    b.matrix_wait('wgmma', keep, registers=registers,
                  reads=[x for _, reads in done for x in reads])
    return rest


def _tcgen05(trip, step) -> None:
    """The tiling of an F32 operator step on tcgen05 (sm_100a), 3xTF32:
    the group's rows -- at most 128 (`element_group.engine_elements`) -- are
    the instruction's M, the lanes of tensor memory, which the block's
    first four warps hold, a thread its row; N is the step's rows rounded
    to 16, D's columns in tensor memory from 0.

    A product at a time: those four warps store their rows of A into
    tensor memory, both TF32 halves, behind D's columns (`tmem_st`); the
    engine's fences around a block barrier show them to the thread that
    issues; it issues the three products of each step of K (`TF32_TERMS`),
    B the stage in the canonical K-major order through descriptors of
    tcgen05's format, and commits them onto the block's mbarrier
    (`bulk_channel`); every thread waits for the phase.  Where B is
    streamed (`_pieces`), a product's chunks are committed one at a time:
    the first copied ahead, and while the engine multiplies one, the block
    converts the next into the other slot -- read into registers a chunk
    before -- and reads the one after it, the fence of the async proxy
    and a barrier closing each.  Then the four warps load D's columns
    (`tmem_ld`), a thread its row, into a stage in shared memory
    (`d_stage_stride`), and every element warp hands a share of it to the
    destination as `_matrix` hands its own (`_d_writer`).  The tensor
    memory is the
    block's, allocated ahead of the batch loop and freed behind it
    (`MultiElementOperators.tmem`): an allocation gives up the block's
    permit to allocate again (`tmemAlloc`), so a trip may not allocate."""
    from tensorforge.backend.pir.engines import FenceKind
    from tensorforge.generators.element_group import (TCGEN05_ROWS,
                                                      matrix_shape)
    b = trip.builder
    section = trip.section
    fp = trip.fp
    rows, cols = step.rows, step.cols
    width = cols[1] - cols[0]
    height = rows[1] - rows[0]
    _, n_, k_ = matrix_shape('tcgen05', 100)
    n = _round_up(height, n_)
    M = trip.group.elements * width
    if M > TCGEN05_ROWS:
        raise InternalError(f'a tcgen05 step over {M} rows of the group, '
                            f'past its {TCGEN05_ROWS} lanes')
    padded = M < TCGEN05_ROWS
    tmem = trip.tmem
    if tmem is None:
        raise InternalError('a tcgen05 step without the block\'s tensor '
                            'memory (`MultiElementOperators.tmem`)')
    # The block's one mbarrier, which every warp waits on: in the block's
    # arena, not in a warp's window of the multiplications'.
    cursor = b.bulk_channel('tcmma', 'threadIdx.x == 0 && threadIdx.y == 0',
                            participants=trip.meet,
                            arena=GeneralLexicon.TOTAL_SHR_MEM)
    if cursor is None:
        raise InternalError('a tcgen05 step in a body without an arena for '
                            'its mbarrier')
    issue = b.op('eq', BOOL, b.op('add', INDEX, trip.lane, b.op(
        'mul', INDEX, trip.slot, 32, hint='tid'), hint='tid'), 0,
        hint='issue')
    holds = b.op('lt', BOOL, trip.slot, 4, hint='lanes')
    m = b.op('add', INDEX, b.op('mul', INDEX, trip.slot, 32, hint='m'),
             trip.lane, hint='m')
    live = b.op('lt', BOOL, m, M, hint='in') if padded else None
    row = b.op('select', INDEX, live, m, 0, hint='m') if padded else m
    pieces, copy = _pieces(trip, step)
    if copy is not None:
        copy(0)
        b.fence(FenceKind.PROXY)
        b.barrier(trip.meet)
    first = True
    _, columns, apart = _tcgen05_columns(section, step)
    if apart:
        # Every product's A ahead of all of them, in columns of its own,
        # behind one barrier: one round of products, one commit and one
        # wait for the step.
        for piece, at in zip(pieces, columns):
            _tcgen05_a(trip, piece.product, piece.stage, tmem, at, M)
        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')
        b.barrier(trip.meet)
        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')
    reads = []
    for i, piece in enumerate(pieces):
        product, stage = piece.product, piece.stage
        upper_at = columns[step.products.index(product)]
        lower_at = upper_at + stage.depth
        if piece.k0 == 0 and not apart:
            # A's rows into tensor memory, eight columns at a time.
            _tcgen05_a(trip, product, stage, tmem, upper_at, M)
            b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')
            b.barrier(trip.meet)
            b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')
        if not apart:
            reads = []
        for kt in range(0, piece.rows, k_):
            bs = piece.descriptors(b, kt, 'tcgen05')
            for ia, ib in _terms(stage):
                b.tcgen05(n, tmem, bs[ib], tmem, issue=issue,
                          enable_input_d=0 if first else 1,
                          a_column=(upper_at, lower_at)[ia] + piece.k0 + kt)
                first = False
            reads += bs
        if apart and i + 1 < len(pieces):
            continue
        b.commit_group('tcgen05', object=cursor, issue=issue)
        if copy is not None and i + 1 < len(pieces):
            copy(i + 1)
        b.matrix_wait('tcgen05', object=cursor, reads=reads, tmem=[tmem])
        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')
        if copy is not None:
            b.fence(FenceKind.PROXY)
            b.barrier(trip.meet)
    write = _d_writer(trip, step)
    # D through shared memory, a row a lane-holding thread, so that every
    # element warp writes a share of it: the writes are reductions through
    # the elements' pointers, which the four warps holding the lanes --
    # two at o8 -- would otherwise issue alone (`D_STAGE`).
    stride = d_stage_stride(height)
    staged = _d_stage(trip)
    with b.if_(holds):
        # The step's rows of D, to the next eight, in widths the load has:
        # N's padding past them holds zeros nobody reads.
        c0, end = 0, _round_up(height, 8)
        while c0 < end:
            count = 1 << (min(TMEM_CHUNK, end - c0).bit_length() - 1)
            for j, value in enumerate(b.tmem_ld(tmem, c0, count)):
                if c0 + j < height:
                    b.store(staged, value, _index(
                        b, 'add', b.op('mul', INDEX, row, stride, hint='d'),
                        c0 + j, 'd'), predicate=live)
            c0 += count
    # The block's next use of the tensor memory, the next step's or the
    # next trip's, is behind a barrier between the engine's fences; the
    # staged rows behind the same barrier for every element warp.
    b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')
    b.barrier(trip.meet)
    threads = trip.group.threads * trip.group.elements
    tid = b.op('add', INDEX, trip.lane, b.op('mul', INDEX, trip.slot,
                                             trip.group.threads, hint='t'),
               hint='t')
    # A cell a thread a hop, the step's rows of a row in turn: the
    # consecutive threads' writes are one element's column, contiguous.
    with b.for_(tid, M * height, threads, hint='c',
                uniform=Uniformity.LANE) as loop:
        row = b.op('div', INDEX, loop.induction, height, hint='m')
        col = b.op('rem', INDEX, loop.induction, height, hint='n')
        value = b.load(staged, _index(b, 'add', b.op(
            'mul', INDEX, row, stride, hint='d'), col, 'd'), type_=fp,
            hint='d')
        write(row, col, value, 'cell', False)


def d_stage_stride(height: int) -> int:
    """The floats between two rows of D staged in shared memory (`_tcgen05`):
    the step's rows, odd, so that the lane-holding threads' stores of one
    column -- a row each -- fall into distinct banks."""
    return height | 1


def _d_stage(trip):
    """The trip's stage of D in shared memory (`_tcgen05`): one buffer for
    all of its tcgen05 steps, the widest step's rows of the group -- what
    `element_group.shared_bytes` counts -- which each step writes in turn.
    A step writes it behind the barrier ahead of its products, which every
    element warp reaches done reading the step before's.  A buffer a step,
    the chain's two steps took 27 kB more than the plan had, and on a GB200
    the block outgrew its 227 kB with the producer's ring."""
    if trip.dstage is None:
        size = max(trip.group.elements * (s.cols[1] - s.cols[0])
                   * d_stage_stride(s.rows[1] - s.rows[0])
                   for s in trip.section.steps if s.lowering == 'tcgen05')
        trip.dstage = trip.builder.alloc(
            trip.fp.base, (size,), MemSpace.SHARED,
            arena=GeneralLexicon.TOTAL_SHR_MEM, hint='dstage')
    return trip.dstage


def _tcgen05_a(trip, product, stage, tmem, at, M) -> None:
    """The product's A into tensor memory, its upper halves from column
    `at` on and its lower halves the depth behind, eight columns at a time
    -- by every band of four warps the group holds whole, a band a chunk
    of eight in turn: a warp reaches the lanes 32 (w % 4) on whatever its
    band (PTX ISA, "Tensor Memory"), the thread its row (`_tcgen05`)."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    b = trip.builder
    fp = trip.fp
    section = trip.section
    group = section.buffer(product.second.tensor)
    source = trip.buffers[id(product.second.tensor)]
    width = product.cols[1] - product.cols[0]
    k_lo, k_hi = product.depth
    bands = max(1, trip.group.elements // 4)
    band = b.op('div', INDEX, trip.slot, 4, hint='band')
    m = b.op('add', INDEX, b.op('mul', INDEX, b.op(
        'rem', INDEX, trip.slot, 4, hint='q'), 32, hint='m'), trip.lane,
        hint='m')
    from tensorforge.generators.element_group import TCGEN05_ROWS
    live = b.op('lt', BOOL, m, M, hint='in') if M < TCGEN05_ROWS else None
    row = b.op('select', INDEX, live, m, 0, hint='m') if live is not None \
        else m
    for kt in range(0, stage.depth, 8):
        mine = b.op('eq', BOOL, band, (kt // 8) % bands, hint='mine')
        with b.if_(mine):
            values = []
            for x in range(kt, kt + 8):
                if k_lo + x >= k_hi:
                    values.append(b.const(0.0, fp))
                    continue
                value = b.load(source, _group_index(
                    b, row, width, group, product.cols[0], k_lo + x),
                    type_=fp, hint='a')
                if live is not None:
                    value = b.op('select', fp, live, value, b.const(0.0, fp),
                                 hint='a')
                values.append(value)
            split = nvidia.tfconvert(b, values)
            b.tmem_st(tmem, at + kt, [_as_float(b, u) for u, _ in split])
            b.tmem_st(tmem, at + stage.depth + kt,
                      [_as_float(b, lo) for _, lo in split])


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
        #: The walk's anchor (`element_group.lane_anchors`): a zero the
        #: compiler cannot see through, added into the addresses the walk
        #: reads, or None; what a new one is made of (`reanchor`): the pass's
        #: column, None where the target takes none, and a count.
        self.anchor = None
        self.anchor_seed = None
        self.anchor_count = 0
        #: Where a walk holds a column an operator step computes -- the flux
        #: kernels' faces -- each tile of rows it reads is read at an anchor
        #: of the arithmetic before it, as the target spells one
        #: (`Lexic.chained_zero`, `rechain`), and `last` the arithmetic's
        #: latest value.  Not where no column is held: the volume took 60
        #: VGPRs more at o4 on gfx942 with its tiles chained.
        held = any(self.materialized(p) for step in self.section.steps
                   for p in step.products)
        self.chain = (context.target.lexic.chained_zero('{0}') if held
                      else None)
        self.last = None
        #: A chained section's phase (`_chained`): the scaled copies after
        #: each column (`_scaled`), whether the pass computes an output first
        #: (`_column`) -- not in a phase of copies alone -- and the output
        #: whose column `acc` then holds.
        self.fused: tuple = ()
        self.computes = any(s.kind is not self.kinds.SCALE
                            for s in self.section.steps)
        self.holds = (self.cones.outputs[0].dest if self.computes else None)
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
        """`index` plus the walk's anchor, where there is one."""
        if self.anchor is None:
            return index
        return _index(self.builder, 'add', index, self.anchor, 'a')

    def rechain(self) -> None:
        """A new anchor for the next tile of rows the walk reads, formed after
        the arithmetic so far (`chain`, `last`): the AMDGPU backend's
        instruction selection otherwise puts every read of a block ahead of
        its FMAs -- the walk's rows all at once -- and no scheduling barrier
        moves them back."""
        if self.chain is None or self.last is None or self.anchor is None:
            return
        # The zero is the same in every lane; the IR sees it as varying as
        # the value it is formed after, which is what its addresses are.
        self.anchor = self.builder.rawexpr(
            self.chain, self.last, type_=INDEX, hint='anc', pure=False,
            movable=False)

    def reanchor(self) -> None:
        """A new anchor for what follows (`element_group.lane_anchors`): a
        zero the compiler cannot see through, of the pass's column and of how
        many anchors the pass has had, so that no two are one value.  At every
        walk: two walks reading the same rows of the element -- the faces of
        `localFluxAll` -- read them at addresses the compiler cannot tell are
        the same, and reads them in each walk again rather than once for all
        of them, held from the first to the last."""
        if self.anchor_seed is None:
            return
        b = self.builder
        spelled = self.context.target.lexic.opaque_zero('{0}')
        seed = _index(b, 'add', self.anchor_seed, self.anchor_count, 'sd')
        self.anchor_count += 1
        self.anchor = b.rawexpr(
            spelled, *(() if '{0}' not in spelled else (seed,)),
            type_=INDEX, hint='anc', pure=False, movable=False,
            crosslane='{0}' in spelled, uniform=Uniformity.MULT)

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
            if p.kind is self.kinds.OPERATOR)
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
        if not self.sequence:
            # A range that read no number -- a phase of copies alone.
            self.table, self.block = None, None
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
        self.reanchor()

        def add(row, v, x, product, j):
            if x is None:
                return
            x = self.windowed(x, product, step)
            old = acc.get((row, j))
            acc[(row, j)] = (b.op('mul', fp, v, x, hint='x') if old is None
                             else b.op('fma', fp, v, x, old, hint='x',
                                       attrs=self.fma_attrs))
            self.last = acc[(row, j)]

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
            self.reanchor()
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
                # A tile of rows read is held while its rows are used, and
                # read after the tile before is computed (`rechain`).
                if tile is not None:
                    self.rechain()
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
    b = trip.builder
    cones = trip.cones
    steps = cones.outputs
    tensor = steps[0].dest
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
    # What is written under the lane's mask, held where it is computed, where
    # the target says how (`Lexic.opaque_value`), a walk holds a column an
    # operator step computes -- the flux kernels' faces -- and the numbers
    # are literals: AMD's compiler otherwise sinks each value, and the FMAs
    # it takes, into its write's guard, and the column's reads of the element
    # came out every row at once.  Where no step's column is held -- the
    # volume -- the sunk sums read registers that hold anyway, and held they
    # cost: 248 VGPRs and no AGPRs for the o6 F32 volume on gfx942 as they
    # are, 256 and 176 held.  Numbers read from memory are scalar loads,
    # which the backend then gathers ahead of the held arithmetic: the o4 F64
    # localFluxAll on gfx942 traded 3.7 kB of VGPR spill for 1228 SGPRs.
    held_columns = any(trip.materialized(p) for step in trip.section.steps
                       for p in step.products)
    spelled = (trip.context.target.lexic.opaque_value('{0}')
               if (trip.mask is not None and held_columns
                   and trip.values == 'literal') else None)

    def held(value, type_):
        if spelled is None:
            return value
        return b.rawexpr(spelled, value, type_=type_, hint='w', pure=False,
                         movable=False)

    _write(trip, tensor, write, trip.acc, held)


def _write(trip, tensor, write, values, held=None) -> None:
    """`values` -- `(row, slot)` -> a value, None for zero -- at rows
    `write` of the pass's columns of `tensor`, through the lane's pointers
    under its mask: a whole tile one access where it is one (`claim`).
    `held` hands a value through what keeps it where it is computed."""
    b, fp = trip.builder, trip.fp
    target = _bound(b, trip.symbols[id(tensor)])
    width = trip.width
    lower = int(tensor.bbox.lower()[0])
    held = held or (lambda value, type_: value)
    for j in range(len(trip.cols)):
        col = trip.cols[j]
        done = set()
        for i in write:
            if i in done or (i, j) not in values:
                continue
            first = i - (i - lower) % width
            tile = list(range(first, first + width))
            claim = trip.claim(tensor, first)
            if claim is not None and all((r, j) in values for r in tile):
                # A whole tile written: one access.
                parts = [values[(r, j)] if values[(r, j)] is not None
                         else b.const(0.0, fp) for r in tile]
                vector = ScalarType(trip.section.dtype, width)
                b.store(target, held(b.pack(vector, *parts, hint='q'),
                                     vector),
                        _address(b, tensor, first, col),
                        predicate=trip.mask, align=claim)
                done.update(tile)
                continue
            value = (values[(i, j)] if values[(i, j)] is not None
                     else b.const(0.0, fp))
            b.store(target, held(value, fp), _address(b, tensor, i, col),
                    predicate=trip.mask)
            done.add(i)


def _scaled(trip, step) -> None:
    """A scaled copy at the pass's columns (`element_group.Kind.SCALE`):
    `dest (+)= first * s` at each row of `first`'s window -- the phase's
    output where the pass holds it (`acc`), else read from the element's
    memory -- and written back as the pass's output is (`_write`).  Rows of
    an accumulation whose part is a known zero are not touched."""
    b, fp = trip.builder, trip.fp
    for p in step.products:
        source, dest = p.first.tensor, p.dest.tensor
        scale = None if p.second is None else _scalar(trip, p.second.tensor)
        rows = range(int(p.first.bbox.lower()[0]),
                     int(p.first.bbox.upper()[0]))
        held = trip.holds is not None and trip.holds is source
        values = {}
        trip.tiles = {}
        for j in range(len(trip.cols)):
            trip.slot = j
            for i in rows:
                x = (trip.acc.get((i, j)) if held
                     else trip.own(source, i, trip.col))
                if x is None:
                    if not p.add:
                        values[(i, j)] = None
                    continue
                if p.add:
                    old = trip.own(dest, i, trip.col)
                    values[(i, j)] = (
                        b.op('add', fp, old, x, hint='y') if scale is None
                        else b.op('fma', fp, x, scale, old, hint='y',
                                  attrs=trip.fma_attrs))
                else:
                    values[(i, j)] = (x if scale is None else
                                      b.op('mul', fp, x, scale, hint='y'))
        trip.tiles, trip.slot = {}, 0
        _write(trip, dest, list(rows), values)


def _scalar(trip, tensor):
    """A scaled copy's factor: its number where the description has it,
    else the value the kernel is handed (`element_group.Role.SCALAR`)."""
    b = trip.builder
    if tensor.has_values():
        import numpy as np
        return b.const(float(np.asarray(tensor.get_values()).reshape(-1)[0]),
                       trip.fp)
    # As `ScalarContractionInstruction` reads one: the number its binding
    # published in this body, else the parameter by its name.
    symbol = trip.symbols[id(tensor)]
    bound = symbol.pir_scalar(b)
    if bound is not None:
        return bound
    return b.rawexpr(symbol.name, type_=trip.fp, hint='s', pure=True,
                     movable=True)


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
            trip.anchor = None
            trip.anchor_seed = (None if spelled is None
                                else loop.induction)
            trip.anchor_count = 0
            trip.reanchor()

            def column():
                trip.cols, trip.slot = cols, 0
                trip.small = {}
                if trip.computes:
                    _column(trip)
                for step in trip.fused:
                    _scaled(trip, step)

            if width % classes == 0:
                column()
            else:
                with b.if_(b.op('lt', BOOL, cols[0], c1, hint='in'),
                           attrs=_lanes_active(width, classes * trips)):
                    column()
        trip.anchor, trip.anchor_seed = None, None
        trip.end_table()

    if len(plan.ranges) == 1:
        range_body(plan.ranges[0])
        return
    for n, rows in enumerate(plan.ranges):
        with b.if_(b.op('eq', BOOL, part, n, hint='part'),
                   attrs=_lanes_active(1, len(plan.ranges))):
            range_body(rows)


def _chained(builder, context, group, symbols) -> list:
    """A trip of a chained section in the lanes form
    (`element_group.Lanes.phases`): its phases one after another, each over
    all its output's columns in the one warp -- a lane reads back what it
    wrote in the phases before -- the scaled copies of a phase's output in
    its column pass, from the registers that hold it, and a phase of copies
    alone a pass of its own.  The phases' trips, for their tables."""
    from tensorforge.generators import element_group
    section = group.section
    trips = []
    for phase, plan in zip(section.phases(), group.lanes.phases):
        if phase.main:
            sub = dataclasses.replace(
                group, section=section.phase_section(phase), lanes=plan)
        else:
            outputs = []
            for step in phase.scales:
                if not any(o is step.dest for o in outputs):
                    outputs.append(step.dest)
            copies = dataclasses.replace(section, steps=phase.scales,
                                         outputs=tuple(outputs))
            rows = element_group.Cones(copies).rows
            sub = dataclasses.replace(
                group, section=copies,
                lanes=element_group.Lanes(1, (rows,), 0))
        trip = _LaneTrip(builder, context, sub, symbols)
        trip.fused = phase.scales
        _lanes(trip)
        trips.append(trip)
    return trips


#: Each step's lowering (`element_group.Step.lowering`).
LOWERINGS = {'fma': _fma, 'rows': _per_output,
             'mix': lambda trip, step: _mix(trip, [step]),
             'mma_sync': _matrix, 'dmma': _matrix, 'lane': _lane,
             'wgmma': _wgmma, 'tcgen05': _tcgen05}


class MultiElementRelease(AbstractInstruction):
    """Behind the batch loop: the block's tensor memory freed
    (`MultiElementOperators.tmem`), as a block frees what it allocated
    before it exits -- behind a block barrier between the engine's fences,
    which the free brings (`engines.tmem_dealloc`)."""

    def __init__(self, context: Context, operators):
        super().__init__(context)
        self._operators = operators
        self._is_ready = True

    def uses(self):
        return ()

    def barrier_scope(self) -> Uniformity:
        return Uniformity.BLOCK

    def gen_ir(self, builder) -> None:
        if self._operators.tmem is not None:
            builder.tmem_dealloc(self._operators.tmem)

    def __str__(self) -> str:
        return 'multi-element release of the tensor memory'


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
        self._inputs += tuple(self._symbols[id(t)] for t in section.scalars
                              if not t.has_values())
        self._outputs = tuple(self._symbols[id(t)] for t in section.outputs)
        #: The generated file's tables a trip reads, by name: the matrix
        #: tiling's walk (`_matrix`).
        self._module_tables: Dict[str, tuple] = {}
        for symbol in self._inputs + self._outputs:
            symbol.add_user(self)

    def defs(self):
        return self._outputs

    def uses(self):
        return self._inputs + self._outputs

    def module_tables(self):
        """The numbers the section reads by name at namespace scope -- the
        lanes form's `table` (`_LaneTrip.number`), the places of a sparse
        operand's cells (`_sparse_read`), the matrix tiling's walk
        (`_matrix`) -- as `MultiElementOperators.module_tables` hands its
        own out."""
        return {**self._tables, **self._module_tables}

    def barrier_scope(self) -> Uniformity:
        return Uniformity.BLOCK

    def gen_ir(self, builder) -> None:
        section = self._group.section
        if self._group.lanes is not None:
            # The lanes form: nothing staged, a lane its element's
            # (`_lanes`) -- a chained section's phases one after another
            # (`_chained`).
            if self._group.lanes.phases:
                trips = _chained(builder, self._context, self._group,
                                 self._symbols)
            else:
                trips = [_LaneTrip(builder, self._context, self._group,
                                   self._symbols)]
                _lanes(trips[0])
            dtype = section.dtype
            self._tables = {}
            for trip in trips:
                self._tables.update({name: (dtype, numbers)
                                     for name, numbers in trip.tables.items()})
                self._tables.update(trip.int_tables)
            return
        self._tables = {}
        trip = _Trip(builder, self._context, self._group, self._symbols,
                     self._operators.buffers, self._operators.tables)
        trip.table = lambda prefix, numbers: _module_table(self, prefix,
                                                           numbers)
        trip.tmem = self._operators.tmem
        if section.ring:
            self._ring(trip)
            return
        elements = self._group.elements
        for buffer in section.inputs + section.intermediates:
            trip.buffers[id(buffer.tensor)] = builder.alloc(
                section.dtype, (elements * buffer.room,),
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
        for buffer in section.intermediates:
            if buffer.pitch > buffer.height:
                self._pad(trip, buffer)
        if tokens:
            builder.wait(*tokens)
        for tensor in section.pointer_tables:
            trip.publish(tensor)
        # What was written and read since the block last met: a step that
        # reads what is written, or writes what is written or read, waits --
        # a run of `mix` steps as one (`_units`).
        written = {id(b.tensor) for b in section.inputs}
        read = set()
        for unit in _units(section.steps):
            reads = {id(t) for step in unit for t in step.reads()}
            dests = {id(step.dest) for step in unit}
            if written & reads or dests & (written | read):
                builder.barrier(Participants.BLOCK)
                written, read = set(), set()
            if unit[0].lowering == 'mix':
                _mix(trip, unit)
            else:
                LOWERINGS[unit[0].lowering](trip, unit[0])
            written |= dests
            read |= reads

    def _ring(self, trip) -> None:
        """A trip with the inputs through the producer's ring: the block in
        two roles (`IRBuilder.roles`) -- the producer's warp behind the
        element warps, and theirs -- which meet on the ring's barriers alone
        and each on a named barrier of its own.

        The trip's slot and the parity of its round are the trip's count's
        (`IRBuilder.ring_round`), which both roles read off the group's first
        element: the block's trips are its blocks' in turn.  The slot's
        windows are the inputs' buffers for the trip (`MultiElementOperators.
        ring_slots`); the intermediates and the pointer tables stay the
        element warps', a trip each."""
        from tensorforge.backend.instructions.batch_loop import (
            BatchLoop, GROUP_KEY, MASK_KEY)
        b = trip.builder
        section = self._group.section
        elements = self._group.elements
        ring = self._operators.ring
        bound = BatchLoop.indices_in(b)
        if not bound or GROUP_KEY not in bound:
            raise InternalError('a producer\'s trip outside the element '
                                'group\'s loop')
        group = bound[GROUP_KEY]
        block = b.extern_value('blockIdx.x', INDEX, uniform=Uniformity.BLOCK,
                               hint='bid')
        grid = b.extern_value('gridDim.x', INDEX, uniform=Uniformity.GRID,
                              hint='grid')
        count = b.op('div', INDEX, b.op('sub', INDEX, b.op(
            'div', INDEX, group, elements, hint='k'), block, hint='k'),
            grid, hint='k')
        slot, parity = b.ring_round(ring, count)
        for buffer in section.inputs:
            trip.buffers[id(buffer.tensor)] = b.alloc(
                section.dtype, (elements * buffer.per_element,),
                MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                hint=_hint('r', buffer.tensor), stages=section.ring,
                stage=slot, identity=self._operators.ring_slots[
                    id(buffer.tensor)], handed=True)
        for buffer in section.intermediates:
            trip.buffers[id(buffer.tensor)] = b.alloc(
                section.dtype, (elements * buffer.per_element,),
                MemSpace.SHARED, arena=GeneralLexicon.TOTAL_SHR_MEM,
                hint=_hint('g', buffer.tensor))
        unit = max(1, WORD.base.size() // self._context.fp_type.size())
        for tensor in section.pointer_tables:
            trip.pointers[id(tensor)] = b.alloc(
                Datatype.I64, (elements,), MemSpace.SHARED,
                arena=GeneralLexicon.TOTAL_SHR_MEM, hint=_hint('p', tensor),
                place_align=unit if unit > 1 else None)
        trip.meet = Participants.ROLE
        roles = _role_split(b, self._context, self._group)
        with roles.then():
            self._produce(trip, ring, slot, parity, group, bound[MASK_KEY])
        with roles.otherwise():
            self._consume(trip, ring, slot, parity)

    def _produce(self, trip, ring, slot, parity, group, active_of) -> None:
        """The producer's trip: the slot back from the element warps' last
        round of it, then lane `e` takes element `e` of the group -- the
        group's first where `e` is none of the batch, so that every copy
        reads an element's memory and the slot's bytes stay what its full
        side announces -- a bulk copy a column of an input the copy takes
        (`element_group.bulk_input`); the others a cell an asynchronous copy
        of the warp's lanes, each lane a share of its element's
        (`_produce_copies`); the warp meets, and its first lane announces
        the bulk copies' bytes on the full side, their tokens with them."""
        from tensorforge.backend.pir.build import RingSlot
        from tensorforge.backend.pir.core import SIZE
        from tensorforge.generators.element_group import (bulk_input,
                                                          bulk_rows)
        b = trip.builder
        section = self._group.section
        elements = self._group.elements
        lane = trip.lane
        windows = [trip.buffers[id(buf.tensor)] for buf in section.inputs]
        b.ring_wait(ring, 'empty', slot, b.op('sub', INDEX, 1, parity,
                                              hint='free'), writes=windows)
        # The producer's first warp: where the producer is a warpgroup
        # (`element_group.register_budgets`), the others keep its trips and
        # do nothing else.
        first_warp = b.op('eq', BOOL, b.thread_id('y'), elements,
                          hint='pw')
        mine = b.op('and', BOOL, first_warp,
                    b.op('lt', BOOL, lane, elements, hint='mine'),
                    hint='mine')
        element = b.op('add', SIZE, group, lane, hint='e')
        source = b.op('select', SIZE, active_of(b, element), element, group,
                      hint='e')
        tokens, nbytes, rest = [], 0, []
        for buffer, symbol in zip(section.inputs, self._inputs):
            window = trip.buffers[id(buffer.tensor)]
            if not bulk_input(buffer):
                rest.append((buffer, symbol, window))
                continue
            pointer = _element_pointer(b, self._context, symbol, source,
                                       section.dtype)
            base = b.op('mul', INDEX, lane, buffer.per_element, hint='b')
            width = buffer.cols[1] - buffer.cols[0]
            rows = bulk_rows(buffer)
            for c in range(width):
                tokens.append(b.copy_bulk(
                    window, pointer, barrier=RingSlot(ring, slot),
                    dst_index=(_index(b, 'add', base, c * buffer.pitch,
                                      'd'),),
                    src_index=(_address(b, buffer.tensor, buffer.rows[0],
                                        buffer.cols[0] + c),),
                    elems=rows, predicate=mine))
            nbytes += elements * width * rows * section.dtype.size()
        if rest:
            with b.if_(first_warp):
                self._produce_copies(trip, rest, group, active_of)
        # The lanes' copies, each lane's waited for, ahead of the first
        # lane's arrival, which hands them over with the bulk copies' bytes.
        b.barrier(Participants.WAVE)
        b.ring_arrive(ring, 'full', slot, b.op('and', BOOL, first_warp,
                                               b.op('eq', BOOL, lane, 0),
                                               hint='first'),
                      tx=nbytes, tokens=tokens)

    def _produce_copies(self, trip, buffers, group, active_of) -> None:
        """The inputs no bulk copy takes, by the producer's first warp, each
        cell an asynchronous copy (`IRBuilder.copy_async`), every copy of the
        trip issued before the warp waits for any.

        Lane `l` takes element `l % E` of the group's `E`, and with the other
        `32 // E` lanes of that element every `32 // E`-th cell of its box:
        at o6's sixteen, two lanes an element, 41 hops a star matrix of 81
        cells.  A lane holds its own element's pointers and nothing else --
        the warp over one element at a time would hold the group's 48 of
        them, and ptxas spilled them -- and a hop's lanes are pairs of
        adjacent cells of sixteen elements, the sectors the hops after it
        read again.

        Not a lane's loads and stores: a load through the element's pointer
        may alias the window the store before it wrote, so the compiler
        issued them a load and a store at a time, one latency each -- 243 of
        them a trip, which held the element warps at the ring for five times
        their own trip (p5d's ncu).  A copy has no value to wait for.

        An input stored sparse -- SeisSol's `star` under its GPU memory
        layout, 24 of 81 cells -- is copied a stored cell a copy, the cells
        `_sparse_read` stages without the ring: each from its place among
        the stored entries to where the box holds it (`_stored_cells`), the
        lanes of an element every `32 // E`-th of them, a branch a share of
        the lanes.  The cells it does not store are the slot's zeros, which
        nothing writes after the prologue (`MultiElementOperators._ring`)."""
        from tensorforge.backend.pir.core import SIZE
        b = trip.builder
        section = self._group.section
        elements = self._group.elements
        lanes = trip.group.threads
        per = max(1, lanes // elements)
        lane = trip.lane
        mine = b.op('rem', INDEX, lane, elements, hint='me')
        share = b.op('div', INDEX, lane, elements, hint='sh')
        element = b.op('add', SIZE, group, mine, hint='e')
        source = b.op('select', SIZE, active_of(b, element), element, group,
                      hint='e')
        copies = []
        guard = (b.if_(b.op('lt', BOOL, lane, per * elements, hint='in'))
                 if per * elements < lanes else contextlib.nullcontext())
        with guard:
            for buffer, symbol, window in buffers:
                pointer = _element_pointer(b, self._context, symbol, source,
                                           section.dtype)
                base = b.op('mul', INDEX, mine, buffer.per_element, hint='b')
                if not buffer.tensor.is_dense():
                    cells = _stored_cells(buffer)
                    for k in range(per):
                        ours = cells[k::per]
                        if not ours:
                            continue
                        with b.if_(b.op('eq', BOOL, share, k, hint='sh')):
                            for at, place in ours:
                                copies.append(b.copy_async(
                                    window, pointer,
                                    dst_index=(_index(b, 'add', base, at,
                                                      'd'),),
                                    src_index=(place,), elems=1,
                                    threads=lanes))
                    continue
                height = buffer.height
                total = height * (buffer.cols[1] - buffer.cols[0])
                for hop in range(0, total, per):
                    cell = _index(b, 'add', share, hop, 'j')
                    inside = (contextlib.nullcontext() if hop + per <= total
                              else b.if_(b.op('lt', BOOL, cell, total,
                                              hint='in')))
                    with inside:
                        row = _index(b, 'add', b.op('rem', INDEX, cell, height,
                                                    hint='r'),
                                     buffer.rows[0], 'r')
                        col = _index(b, 'add', b.op('div', INDEX, cell, height,
                                                    hint='c'),
                                     buffer.cols[0], 'c')
                        copies.append(b.copy_async(
                            window, pointer,
                            dst_index=(_index(b, 'add', base, _box_index(
                                b, row, col,
                                (buffer.rows[0], buffer.cols[0]),
                                buffer.pitch), 'd'),),
                            src_index=(_address(b, buffer.tensor, row, col),),
                            elems=1, threads=lanes))
            b.wait(*copies)

    def _consume(self, trip, ring, slot, parity) -> None:
        """The element warps' trip: the slot as its bytes land, the steps
        as without the ring -- meeting on their role's barrier, the inputs
        already the slot's -- and each warp's first lane gives the slot
        back behind the warp's barrier, done with it."""
        b = trip.builder
        section = self._group.section
        windows = [trip.buffers[id(buf.tensor)] for buf in section.inputs]
        b.ring_wait(ring, 'full', slot, parity, writes=windows)
        if not self._context.get_user_options().enable_sync_block_opt:
            # Nothing places the role's barriers: the last trip's reads of
            # the intermediates and the pointer tables are met here, as the
            # block's barrier meets them without the ring.
            b.barrier(trip.meet)
        # Zeros past the rows of the intermediates, as without the ring: the
        # matrix tiling's padded steps of K read them, and nothing else
        # writes them -- shared memory as a kernel finds it may hold NaN,
        # which no zero of B takes out.  The inputs' slots are zeroed once,
        # ahead of the loop (`MultiElementOperators._ring`).
        padded = [buffer for buffer in section.intermediates
                  if buffer.pitch > buffer.height]
        for buffer in padded:
            self._pad(trip, buffer)
        for tensor in section.pointer_tables:
            trip.publish(tensor)
        # The steps as without the ring (`gen_ir`), a run of `mix` steps as
        # one (`_units`) -- `mix` takes the element warps' threads, a row of
        # A each -- meeting on the role's barrier.  The inputs are the
        # slot's, which its full side showed every element warp; the zeros
        # each warp wrote into its element's intermediates are met as the
        # staging is without the ring: `mix` writes the rows past every
        # element's window up to its 16 bytes.
        written, read = {id(buffer.tensor) for buffer in padded}, set()
        for unit in _units(section.steps):
            reads = {id(t) for step in unit for t in step.reads()}
            dests = {id(step.dest) for step in unit}
            if written & reads or dests & (written | read):
                b.barrier(trip.meet)
                written, read = set(), set()
            if unit[0].lowering == 'mix':
                _mix(trip, unit)
            else:
                LOWERINGS[unit[0].lowering](trip, unit[0])
            written |= dests
            read |= reads
        b.barrier(Participants.WAVE)
        b.ring_arrive(ring, 'empty', slot, b.op('eq', BOOL, trip.lane, 0,
                                                hint='first'))

    def _stage(self, trip, buffer, symbol) -> list:
        """The warp's element's box of an operand, from its own memory into
        the group's buffer -- over its pitch where it has one: the lanes
        write consecutive cells, where a column's padding would leave a gap,
        and two lanes of a warp a bank.  Their tokens, where the copies are
        asynchronous.

        A dense operand where the target's copies are
        (`element_group.async_staging`) goes by asynchronous copies
        (`IRBuilder.copy_async`), a cell a lane a hop, every hop issued
        before the trip waits for any, so that `pir.wrap` can take them a
        trip ahead.  A cell of the padding copies its column's first row:
        a number, which the zeros of B's padding take out of the tiling's
        padded steps of K (`_matrix`) and `mix` reads past and leaves out
        -- a zero stored there would be a second writer of the buffer, which
        the pass does not take a trip ahead.  The last hop's lanes past the
        box stand outside a guard block, as a loader's ragged tail does: a
        copy predicated itself is one the pass does not take either.
        Otherwise by loads and stores, the padding written zero in the same
        pass, and a sparse operand's cells read through its table
        (`_sparse_read`)."""
        from tensorforge.generators.element_group import async_staging
        b = trip.builder
        source = _bound(b, symbol)
        height, pitch = buffer.height, buffer.pitch
        width = buffer.cols[1] - buffer.cols[0]
        if buffer.tensor.is_dense() and async_staging(self._context):
            total = pitch * width
            lanes = trip.group.threads
            tokens = []
            for hop in range(0, total, lanes):
                index = _index(b, 'add', trip.lane, hop, 'j')
                guard = (contextlib.nullcontext() if hop + lanes <= total
                         else b.if_(b.op('lt', BOOL, index, total,
                                         hint='in')))
                with guard:
                    r = b.op('rem', INDEX, index, pitch, hint='r')
                    row = _index(b, 'add', r, buffer.rows[0], 'r')
                    col = _index(b, 'add', b.op('div', INDEX, index, pitch,
                                                hint='c'), buffer.cols[0], 'c')
                    # A cell of the padding copies its column's first row,
                    # and is never zeroed: B's padding is zero, and `mix`
                    # selects zero past the rows, so the number reaches no
                    # result.  Even an Inf or a NaN there could only reach
                    # its own element's row, whose data is bad already.
                    read = row if pitch == height else _index(b, 'add', b.op(
                        'select', INDEX, b.op('lt', BOOL, r, height,
                                              hint='in'), r, 0, hint='r'),
                        buffer.rows[0], 'r')
                    tokens.append(b.copy_async(
                        trip.buffers[id(buffer.tensor)], source,
                        dst_index=(trip.at(buffer, row, col),),
                        src_index=(_address(b, buffer.tensor, read, col),),
                        elems=1, threads=lanes))
            return tokens
        with b.for_(trip.lane, pitch * width, trip.group.threads,
                    hint='j', uniform=Uniformity.LANE) as loop:
            r = b.op('rem', INDEX, loop.induction, pitch, hint='r')
            row = _index(b, 'add', r, buffer.rows[0], 'r')
            col = _index(b, 'add', b.op('div', INDEX, loop.induction,
                                        pitch, hint='c'),
                         buffer.cols[0], 'c')
            # A cell of the padding reads the column's first row, and writes
            # a zero.
            inside = (None if pitch == height else
                      b.op('lt', BOOL, r, height, hint='in'))
            read = row if inside is None else _index(b, 'add', b.op(
                'select', INDEX, inside, r, 0, hint='r'), buffer.rows[0],
                'r')
            if buffer.tensor.is_dense():
                value = b.load(source, _address(b, buffer.tensor, read, col),
                               type_=trip.fp, hint='s')
            else:
                value = _sparse_read(b, self._context, self._tables, source,
                                     buffer.tensor,
                                     _address(b, buffer.tensor, read, col),
                                     trip.fp)
            if inside is not None:
                value = b.op('select', trip.fp, inside, value,
                             b.const(0.0, trip.fp), hint='s')
            b.store(trip.buffers[id(buffer.tensor)], value,
                    trip.at(buffer, row, col))
        return []

    @staticmethod
    def _pad(trip, buffer) -> None:
        """Zeros past the rows of every column of the warp's element, where
        an intermediate's buffer has a pitch (`element_group.Buffer`): what
        the matrix tiling's padded steps of K read, to multiply zeros -- of
        both TF32 halves, where the buffer holds them.  An input's are
        staged (`_stage`)."""
        b = trip.builder
        pad = buffer.pitch - buffer.height
        width = buffer.cols[1] - buffer.cols[0]
        target = trip.buffers[id(buffer.tensor)]
        with b.for_(trip.lane, pad * width, trip.group.threads, hint='z',
                    uniform=Uniformity.LANE) as loop:
            row = _index(b, 'add', b.op('rem', INDEX, loop.induction, pad,
                                        hint='r'), buffer.rows[1], 'r')
            col = _index(b, 'add', b.op('div', INDEX, loop.induction, pad,
                                        hint='c'), buffer.cols[0], 'c')
            zero = b.const(0.0, trip.fp)
            at = trip.at(buffer, row, col)
            b.store(target, zero, at)
            if buffer.halves:
                b.store(target, zero, _index(
                    b, 'add', at, buffer.half_offset(trip.group.elements),
                    'a'))

    def __str__(self) -> str:
        section = self._group.section
        return (f'multi-element section, {len(section.steps)} steps into '
                f'{", ".join(t.alias for t in section.outputs)}, '
                f'{self._group.elements} elements a block')


__all__ = ['LOWERINGS', 'MultiElementOperators', 'MultiElementSection',
           'UNROLL']
