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

The lowerings are the FMA forms, `fma` for an operator product and `rows`
for a mixing one -- the warp of an element computes that element's outputs
of the step, a lane an output, each product of the step adding only into
its own window of the destination -- and the matrix tiling of an operator
step (`mma_sync`, `dmma`; `_matrix`), which reads the same buffers: A from
the group operand's (by ldmatrix in F32, where the buffer is pitched for
it), B from the operator's stage, and D reaching its consumer's buffer, or
the output through a result buffer.

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


class MultiElementOperators(AbstractInstruction):
    """The section's operators, copied into the block's shared memory once,
    ahead of the batch loop, as the section stages them
    (`element_group.Stage`): as stored, or as a matrix product's B -- its
    window, zero elsewhere.  Every thread of the block a share of the
    entries, then the block's barrier.

    The copies are handed to the section (`buffers`, by tensor and window),
    which reads them in every trip."""

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
        builder.barrier(Participants.BLOCK)
        self.buffers = buffers

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

    def __init__(self, builder, context, group, symbols, operators):
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
LOWERINGS = {'fma': _per_output, 'rows': _per_output,
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
                     self._operators.buffers)
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
        for buffer in section.inputs + section.intermediates:
            if buffer.pitch > buffer.height:
                self._pad(trip, buffer)
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
        height = buffer.height
        width = buffer.cols[1] - buffer.cols[0]
        with b.for_(trip.lane, height * width, trip.group.threads,
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
                    trip.at(buffer, row, col))

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
