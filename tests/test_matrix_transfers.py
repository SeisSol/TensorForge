# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""ldmatrix and stmatrix as transfer engines (`Op.MATRIX_LOAD`,
`Op.MATRIX_STORE`): a warp's 8x8 matrices of 16-byte rows between shared
memory and the registers of its lanes, as one collective.

What is checked here: the vocabulary, the fragment layout the op carries
against the instruction's statement and against `mma.sync`'s A fragment, the
statements and what the verifier holds them to, the text and the targets
that have it, the figures (ldmatrix counted as the call it was), the host
oracle (`reference.kernel_eval`) on loads, stores and both, the SASS of a
build for sm_80 and sm_90, and on the local GPU the bodies against what the
PTX statement says they move.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Tuple

import pytest

from tensorforge import toolchain
from tensorforge.backend.instructions.compute.bitlayout import BitLayout
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.backend.pir import engines, transfers
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, Access, Effect,
                                          IRError, MemSpace, Op, ScalarType,
                                          Stmt, Uniformity, XorSwizzle,
                                          walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import Completion
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'

#: The A fragment of `mma.sync`'s m16n8k8 over TF32, as four matrices: the
#: tile's rows 0-7 and 8-15 at columns 0-3, then at 4-7.
A_TILE = ((0, 0), (8, 0), (0, 4), (8, 4))


def _ctx(arch='sm_90', backend='cuda'):
    return Context(arch=arch, backend=backend, fp_type=Datatype.F32)


class _Sink:
    def __init__(self):
        self.lines = []

    def __call__(self, text, *args, **kwargs):
        self.lines.append(text)

    def __getattr__(self, _):
        return lambda *a, **k: None

    @contextlib.contextmanager
    def _block(self, head):
        self.lines.append(f'{head} {{')
        yield
        self.lines.append('}')

    def If(self, cond):
        return self._block(f'if ({cond})')

    def Else(self):
        return self._block('else')

    def Block(self, head=''):
        return self._block(head)


class _Metrics:
    """What the emitter records of the code and the issue mix."""

    def __init__(self):
        self.code = 0
        self.mix = {}

    def record_code(self, units):
        self.code += units

    def record_mix(self, category, issued, code):
        self.mix[category] = self.mix.get(category, 0) + issued


def _emit(body, ctx=None, metrics=None):
    sink = _Sink()
    Emitter(sink, ctx if ctx is not None else _ctx(), metrics).run(body)
    return sink.lines


def _stmts(body, op):
    return [s for s, _ in walk(body) if s.op == op]


def _atom(m, n, k):
    return next(a for a in nvidia.INSTRS
                if (a.m, a.n, a.k) == (m, n, k)
                and a.mode == nvidia.MMAMode.TF32)


# -- the vocabulary ----------------------------------------------------------- #

def test_the_engines_are_transfer_engines_that_complete_on_issue():
    """Each engine is the transfer of its own op, which the registry
    describes, and completes on issue: there is nothing to wait for."""
    for name, op in (('ldmatrix', Op.MATRIX_LOAD),
                     ('stmatrix', Op.MATRIX_STORE)):
        engine = engines.TRANSFER_ENGINES[name]
        assert engine.ops == (op,) and engine.completes is Completion.SYNC
        assert engines.MATRIX_TRANSFER_OPS[name] == op
        assert op in Op.KNOWN and op in engines.REGISTRY
        spec = engines.REGISTRY[op]
        assert spec.completes is Completion.SYNC and not spec.movable
    assert engines.REGISTRY[Op.MATRIX_LOAD].effect == Effect.READ
    assert engines.REGISTRY[Op.MATRIX_STORE].effect == Effect.WRITE
    assert engines.MATRIX_TRANSFER_SM == {'ldmatrix': 75, 'stmatrix': 90}


# -- the fragment layout ------------------------------------------------------ #

def test_four_matrices_are_the_a_fragment_of_m16n8k8():
    """The placement the multi-element tiling loads A by is the A layout of
    the instruction that reads it -- stated once by the transfer, once by
    the matrix engine, and the same."""
    assert engines.matrix_transfer_bits(A_TILE) == nvidia.a_fragment_bits(
        _atom(16, 8, 8))


PLACEMENTS = {
    'x1': ((0, 0),),
    'x2-rows': ((0, 0), (8, 0)),
    'x2-cols': ((0, 0), (0, 4)),
    'x4-a': A_TILE,
    'x4-rows': ((0, 0), (8, 0), (16, 0), (24, 0)),
    'x4-cols': ((0, 0), (0, 4), (0, 8), (0, 12)),
    'x4-swapped': ((0, 0), (16, 0), (8, 0), (24, 0)),
}


@pytest.mark.parametrize('name', sorted(PLACEMENTS))
def test_the_layout_puts_each_element_where_the_ptx_says(name):
    """Every element of the tile, located by the layout, is where the PTX
    statement puts it: register `i` of lane `t` holds the word `t % 4` of
    row `t / 4` of matrix `i` -- and no two elements share a place."""
    origins = PLACEMENTS[name]
    bits = engines.matrix_transfer_bits(origins)
    assert isinstance(bits, BitLayout)
    seen = set()
    for i, (r0, c0) in enumerate(origins):
        for t in range(32):
            at = bits.locate(r0 + t // 4, c0 + t % 4)
            assert (at.slot, at.lane, at.element) == (i, t, 0)
            seen.add((at.slot, at.lane))
    assert len(seen) == 32 * len(origins)


@pytest.mark.parametrize('origins', [
    ((0, 0), (0, 0)),                       # two matrices in one place
    ((0, 0), (8, 0), (0, 4), (8, 8)),       # the fourth off the sum
    ((0, 0), (16, 0)),                      # a gap of eight rows
    ((0, 0), (8, 4)),                       # a step of two axes
    ((0, 0), (0, 2)),                       # half a row across
    ((8, 0),),                              # not from the tile's start
    ((0, 0), (8, 0), (16, 0)),              # three matrices
])
def test_a_placement_of_no_bits_has_no_layout(origins):
    assert engines.matrix_transfer_bits(origins) is None


# -- statements --------------------------------------------------------------- #

def _load_body(count=4, origins=None, type_=None):
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=_ctx())
    s = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='s', extern='s',
                offset=0)
    lane = b.thread_id('x')
    at = b.op('mul', INDEX, lane, 4, hint='at')
    regs = b.matrix_load(s, at, count=count, origins=origins, type_=type_,
                         hint='r')
    return b, s, at, regs


def test_a_load_is_a_declared_read_of_its_buffer_and_pinned():
    """What every pass reads off it: one read of the shared buffer, the
    registers the lanes' own, no pure function a pass may merge or move."""
    b, s, at, regs = _load_body(origins=A_TILE)
    load, = _stmts(b.finish(), Op.MATRIX_LOAD)
    assert load.target == regs and load.args == (s, at)
    assert not load.pure and not load.movable and load.effect == Effect.READ
    assert [(a.kind, a.space, a.base) for a in load.accesses] == [
        (Effect.READ, MemSpace.SHARED, s)]
    assert all(v.uniformity is Uniformity.LANE for v in regs)
    assert all(v.type == ScalarType(Datatype.F32) for v in regs)
    assert load.attr('engine') == 'ldmatrix' and load.attr('count') == 4
    assert load.attr('shape') == (8, 8) and load.attr('width') == 16
    assert load.attr('layout') == engines.matrix_transfer_bits(A_TILE)
    assert engines.verify_engine(load) == []
    verify(b.finish())


def test_a_store_is_a_declared_write_of_its_buffer_and_pinned():
    b, s, at, regs = _load_body(count=2)
    store = b.matrix_store(s, at, regs)
    assert store.op == Op.MATRIX_STORE and store.args == (s, at) + regs
    assert not store.pure and not store.movable
    assert store.effect == Effect.WRITE
    assert [(a.kind, a.space) for a in store.accesses] == [
        (Effect.WRITE, MemSpace.SHARED)]
    assert store.attr('engine') == 'stmatrix' and store.attr('count') == 2
    assert store.attr('layout') is None
    assert engines.verify_engine(store) == []
    verify(b.finish())


def test_a_placement_of_no_bits_is_refused_where_the_load_is_built():
    with pytest.raises(IRError, match='no layout of bits'):
        _load_body(count=2, origins=((0, 0), (16, 0)))


def _swizzled_body(granule):
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=_ctx())
    s = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='s', extern='s',
                offset=0, swizzle=XorSwizzle(8, granule))
    lane = b.thread_id('x')
    at = b.op('mul', INDEX, lane, 4, hint='at')
    regs = b.matrix_load(s, at, count=4, hint='r')
    b.matrix_store(s, at, regs)
    return b, at


def test_a_row_of_a_swizzled_buffer_moves_as_one_granule():
    """A row is 16 bytes that move as one: through a buffer permuted in
    granules of four floats the lane's row is where the permutation puts
    it, for the load and the store alike; a permutation that splits a row
    is refused -- its other words would be read from where they are not."""
    b, at = _swizzled_body(4)
    body = b.finish()
    for op in (Op.MATRIX_LOAD, Op.MATRIX_STORE):
        stmt, = _stmts(body, op)
        index = stmt.args[1]
        assert index is not at
        xor = next(s for s, _ in walk(body) if s.target == (index,))
        assert xor.op == 'bitxor' and xor.args[0] is at
    with pytest.raises(IRError, match='permuted by'):
        _swizzled_body(1)


def _changed(stmt, **kw):
    return dataclasses.replace(stmt, **kw)


@pytest.mark.parametrize('change,why', [
    (lambda s: _changed(s, attrs=tuple((k, 3 if k == 'count' else v)
                                       for k, v in s.attrs)),
     'the instruction moves one of'),
    (lambda s: _changed(s, attrs=tuple((k, (16, 16) if k == 'shape' else v)
                                       for k, v in s.attrs)),
     'matrices of'),
    (lambda s: _changed(s, attrs=tuple((k, 'stmatrix' if k == 'engine'
                                        else v) for k, v in s.attrs)),
     'ldmatrix loads and stmatrix stores'),
    (lambda s: _changed(s, target=s.target[:3]), 'registers for 4 matrices'),
    (lambda s: _changed(s, accesses=()), 'one read of shared memory'),
    (lambda s: _changed(s, accesses=(Access(Effect.WRITE, MemSpace.SHARED,
                                            s.args[0]),)),
     'one read of shared memory'),
    (lambda s: _changed(s, attrs=s.attrs + (('layout', 'rows'),)),
     'which is no BitLayout'),
])
def test_the_verifier_holds_a_transfer_to_what_the_instruction_is(change,
                                                                  why):
    b, *_ = _load_body()
    load, = _stmts(b.finish(), Op.MATRIX_LOAD)
    diag = engines.verify_engine(change(load))
    assert any(why in d for d in diag), diag


def test_a_register_is_a_word_of_32_bits():
    """A row is four words, and a register one of them: an element of 64
    bits would be half a register too many -- refused where the load is
    built, and by the verifier."""
    with pytest.raises(IRError, match='no register of a 32-bit element'):
        _load_body(type_=ScalarType(Datatype.F64))
    b, *_ = _load_body()
    load, = _stmts(b.finish(), Op.MATRIX_LOAD)
    wide = tuple(dataclasses.replace(v, type=ScalarType(Datatype.F64))
                 for v in load.target)
    assert any('no register of a 32-bit element' in d
               for d in engines.verify_engine(_changed(load, target=wide)))


def test_a_store_of_what_is_no_register_is_refused_where_it_is_built():
    b, s, at, regs = _load_body(count=2)
    with pytest.raises(IRError, match='3 matrices; the instruction moves'):
        b.matrix_store(s, at, regs[:1] + regs)
    with pytest.raises(IRError, match='no register of a 32-bit element'):
        b.matrix_store(s, at, (regs[0], 1.0))


def test_every_lane_takes_part():
    """`.aligned`: a predicate would leave lanes of the warp out."""
    b, *_ = _load_body()
    load, = _stmts(b.finish(), Op.MATRIX_LOAD)
    lane = next(s for s, _ in walk(b.finish()) if s.op == 'mul').target[0]
    guarded = _changed(load, predicate=b.op('lt', BOOL, lane, 8))
    assert any('every lane of the warp takes part' in d
               for d in engines.verify_engine(guarded))


# -- the text and the targets ------------------------------------------------- #

def _as_the_call_it_was(body):
    """`body` with its load as the multi-element tiling wrote it before it
    was an op: a split with a callee, declared a read of the buffer and
    pinned (`IRBuilder.split_op`'s `reads`, as it was)."""
    def call(s):
        if s.op != Op.MATRIX_LOAD:
            return s
        return Stmt(op=Op.SPLIT, target=s.target, args=s.args, pure=False,
                    movable=False, effect=Effect.READ, accesses=s.accesses,
                    attrs=(('callee', 'tensorforge::ldmatrixX4'),))
    return tuple(call(s) for s in body)


def test_a_load_is_the_call_it_was_to_the_byte_and_the_figure():
    """The registers declared, then the call writing them through
    references: the text the multi-element tiling wrote when the load was a
    call (`tensorforge::ldmatrixX4`), to the byte -- the index inlined as it
    was -- and the code and the issue mix the emitter counted for it."""
    b, s, at, regs = _load_body(origins=A_TILE)
    body = b.finish()
    old = _as_the_call_it_was(body)
    now, then = _Metrics(), _Metrics()
    lines = _emit(body, metrics=now)
    assert lines == _emit(old, metrics=then)
    assert (now.code, now.mix) == (then.code, then.mix)
    names = ', '.join(str(v) for v in regs)
    assert lines[-1] == (f'tensorforge::ldmatrixX4({names}, s, '
                         f'(threadIdx.x * 4));')


@pytest.mark.parametrize('count', engines.MATRIX_TRANSFER_COUNTS)
def test_the_text_of_each_count(count):
    b, s, at, regs = _load_body(count=count)
    b.matrix_store(s, at, regs)
    lines = _emit(b.finish())
    names = ', '.join(str(v) for v in regs)
    assert f'tensorforge::ldmatrixX{count}({names}, s, {at});' in lines
    assert f'tensorforge::stmatrixX{count}(s, {at}, {names});' in lines


@pytest.mark.parametrize('arch,backend,load,store', [
    ('sm_70', 'cuda', False, False), ('sm_75', 'cuda', True, False),
    ('sm_86', 'cuda', True, False), ('sm_90', 'cuda', True, True),
    ('sm_120', 'cuda', True, True), ('gfx942', 'hip', False, False),
    ('pvc', 'oneapi', False, False)])
def test_the_targets_that_have_each(arch, backend, load, store):
    """ldmatrix from sm_75, stmatrix from sm_90, and CUDA alone."""
    b, s, at, regs = _load_body()
    body = b.finish()
    ld, = _stmts(body, Op.MATRIX_LOAD)
    st = engines.matrix_store(s, at, regs, access=Access(
        Effect.WRITE, MemSpace.SHARED, s))
    for stmt, has in ((ld, load), (st, store)):
        ctx = _ctx(arch, backend)
        if has:
            assert engines.lower(stmt, ctx.target, name=str).lines
        else:
            with pytest.raises(IRError):
                engines.lower(stmt, ctx.target, name=str)


def test_the_helpers_move_floats():
    b, *_ = _load_body(type_=ScalarType(Datatype.I32))
    with pytest.raises(IRError, match='the helpers move floats'):
        _emit(b.finish())


# -- the figures -------------------------------------------------------------- #

def test_ldmatrix_counts_as_the_call_it_was_and_stmatrix_as_a_store():
    """ldmatrix adds no code and no issue, as the call it was written as
    before it was an op, so that no figure of the kernels that load A by it
    moves -- in the emitter's count and in the cover the transfers are
    placed by (`transfers.executed`), as the call's; a store of shared
    memory by stmatrix is one unit and one issue of the shared memory
    pipe."""
    b, s, at, regs = _load_body()
    body = b.finish()
    load, = _stmts(body, Op.MATRIX_LOAD)
    store = engines.matrix_store(s, at, regs, access=Access(
        Effect.WRITE, MemSpace.SHARED, s))
    assert transfers.executed((load,)) == transfers.executed(
        _as_the_call_it_was((load,))) == 0
    assert transfers.executed((store,)) == 1
    assert engines.pipe_of(load) is None
    assert engines.pipe_of(store) == 'shared.store'
    without, with_store = _Metrics(), _Metrics()
    _emit(body, metrics=without)
    _emit(body + (store,), metrics=with_store)
    assert with_store.code == without.code + 1
    assert with_store.mix.get('shared.store', 0) == without.mix.get(
        'shared.store', 0) + 1


# -- bodies ------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Case:
    """A warp's tile of `ROWS` x `COLS` floats in shared memory, rows a
    pitch of `PITCH` apart, and `count` matrices at `origins` in it.

    `kind` `load`: the tile staged from `m1`, loaded by ldmatrix, each
    lane's registers written to `m0` at `lane * count + i`.  `store`: each
    lane's registers read from `m1` there, stored by stmatrix into a tile of
    zeros, the tile written to `m0`.  `both`: the tile loaded by ldmatrix
    from one buffer and stored by stmatrix into another, written to `m0`."""
    kind: str
    origins: Tuple[Tuple[int, int], ...]

    ROWS = 32
    COLS = 16
    PITCH = 20

    @property
    def count(self):
        return len(self.origins)


CASES = {f'{kind}-{name}': Case(kind, PLACEMENTS[name])
         for kind in ('load', 'store', 'both')
         for name in ('x1', 'x2-cols', 'x4-a', 'x4-rows')}


def _lane_row(b, case, lane):
    """The element the lane's row starts at in the tile: row `lane % 8` of
    matrix `(lane / 8) % count` -- every lane names one, those past the
    matrices one the instruction does not read."""
    m = b.op('rem', INDEX, b.op('div', INDEX, lane, 8), case.count,
             hint='m') if case.count > 1 else 0
    row = b.op('rem', INDEX, lane, 8, hint='r')
    col = 0
    first = case.origins[0]
    for bit in range((case.count - 1).bit_length()):
        step = case.origins[1 << bit]
        on = b.op('bitand', INDEX, b.op('shr', INDEX, m, bit), 1, hint='o')
        if step[0] - first[0]:
            row = b.op('add', INDEX, row,
                       b.op('mul', INDEX, on, step[0] - first[0]), hint='r')
        if step[1] - first[1]:
            col = b.op('add', INDEX, col,
                       b.op('mul', INDEX, on, step[1] - first[1]), hint='c')
    return b.op('add', INDEX, b.op('mul', INDEX, row, Case.PITCH), col,
                hint='at')


def _fill(b, dst, value_of, lane):
    """Every element of the tile `dst`, its value `value_of(element)`, the
    lanes taking turns."""
    total = Case.ROWS * Case.COLS
    for k in range(total // 32):
        idx = b.op('add', INDEX, lane, 32 * k, hint='i')
        at = b.op('add', INDEX, b.op('mul', INDEX, b.op('div', INDEX, idx,
                                                         Case.COLS),
                                     Case.PITCH),
                  b.op('rem', INDEX, idx, Case.COLS), hint='t')
        b.store(dst, value_of(idx), at)


def _drain(b, src, out, lane):
    """The tile `src` into `out`, element by element."""
    total = Case.ROWS * Case.COLS
    for k in range(total // 32):
        idx = b.op('add', INDEX, lane, 32 * k, hint='i')
        at = b.op('add', INDEX, b.op('mul', INDEX, b.op('div', INDEX, idx,
                                                         Case.COLS),
                                     Case.PITCH),
                  b.op('rem', INDEX, idx, Case.COLS), hint='t')
        b.store(out, b.load(src, at), idx)


def _body(case: Case, arch='sm_90'):
    """`case` built and placed: (context, body, the arena in floats)."""
    ctx = _ctx(arch)
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=ctx)
    size = Case.ROWS * Case.PITCH
    tile = b.alloc(Datatype.F32, (size,), MemSpace.SHARED, hint='tile',
                   extern='tile')
    other = b.alloc(Datatype.F32, (size,), MemSpace.SHARED, hint='other',
                    extern='other') if case.kind == 'both' else None
    gin = b.alloc(Datatype.F32, (Case.ROWS * Case.COLS,), MemSpace.GLOBAL,
                  hint='gin', extern='m1')
    out = b.alloc(Datatype.F32, (Case.ROWS * Case.COLS,), MemSpace.GLOBAL,
                  hint='o', extern='m0')
    lane = b.thread_id('x')
    at = None
    if case.kind in ('load', 'both'):
        _fill(b, tile, lambda idx: b.load(gin, idx), lane)
        if other is not None:
            # Zeros ahead of the load: written behind it, they would race
            # with its reads where the placement lets the dead tile and
            # this buffer share bytes.
            _fill(b, other, lambda idx: b.const(0.0), lane)
        b.barrier('block')
        at = _lane_row(b, case, lane)
        regs = b.matrix_load(tile, at, count=case.count,
                             origins=case.origins, hint='r')
        if case.kind == 'load':
            for i, v in enumerate(regs):
                b.store(out, v, b.op('add', INDEX,
                                     b.op('mul', INDEX, lane, case.count), i))
        else:
            b.matrix_store(other, at, regs, origins=case.origins)
            b.barrier('block')
            _drain(b, other, out, lane)
    else:
        _fill(b, tile, lambda idx: b.const(0.0), lane)
        b.barrier('block')
        regs = [b.load(gin, b.op('add', INDEX,
                                 b.op('mul', INDEX, lane, case.count), i),
                       hint='v') for i in range(case.count)]
        at = _lane_row(b, case, lane)
        b.matrix_store(tile, at, regs, origins=case.origins)
        b.barrier('block')
        _drain(b, tile, out, lane)
    body, layout = allocate(b.finish(), arenas={'shrMem': MULT})
    verify(body)
    return ctx, body, layout.per_mult


def _expected(case, inputs):
    """What `m0` holds behind the body, by the PTX statement alone."""
    n = case.count
    if case.kind == 'load':
        out = [0.0] * (Case.ROWS * Case.COLS)
        for t in range(32):
            for i, (r0, c0) in enumerate(case.origins):
                out[t * n + i] = inputs[(r0 + t // 4) * Case.COLS
                                        + c0 + t % 4]
        return out
    out = [0.0] * (Case.ROWS * Case.COLS)
    for i, (r0, c0) in enumerate(case.origins):
        for r in range(8):
            for w in range(4):
                cell = (r0 + r) * Case.COLS + c0 + w
                out[cell] = (inputs[cell] if case.kind == 'both'
                             else inputs[(4 * r + w) * n + i])
    return out


def _oracle_source(lines):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    return ('void k() {\n'
            '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + ''.join(f'  {line}\n' for line in keep) + '}')


@pytest.mark.parametrize('name', sorted(CASES))
def test_the_oracle_moves_what_the_ptx_says(name):
    """The host oracle runs each body for one warp: what lands in `m0` is
    what the PTX statement says the instructions move, element for element
    -- and no access to shared memory races."""
    case = CASES[name]
    ctx, body, _ = _body(case)
    src = _oracle_source(_emit(body, ctx))
    races = []
    seed = 5
    mem = kernel_eval.evaluate_wave(src, 32, seed=seed, globals_only=True,
                                    races=races)
    slot = kernel_eval.Slot(seed)
    inputs = [slot.read('m1', i) for i in range(Case.ROWS * Case.COLS)]
    got = [mem.get(('m0', i), 0.0) for i in range(Case.ROWS * Case.COLS)]
    want = _expected(case, inputs)
    if case.kind == 'load':
        used = 32 * case.count
        assert got[:used] == want[:used]
    else:
        assert got == want
    assert not races, [str(r) for r in races[:4]]


def test_the_oracle_sees_a_read_of_another_lanes_row_unordered():
    """A row stmatrix wrote is the write of the lane that named it: read by
    another lane with no barrier between, it races."""
    case = CASES['store-x4-a']
    ctx, body, _ = _body(case)
    lines = _emit(body, ctx)
    store = next(i for i, line in enumerate(lines)
                 if line.startswith('tensorforge::stmatrixX4('))
    after = next(i for i in range(store, len(lines))
                 if lines[i].strip() == '__syncthreads();')
    races = []
    kernel_eval.evaluate_wave(_oracle_source(lines[:after] + lines[after + 1:]),
                              32, seed=5, globals_only=True, races=races)
    assert any(r.kind == 'RAW' for r in races)


def test_the_oracle_refuses_a_warp_that_is_not_whole():
    case = CASES['load-x4-a']
    ctx, body, _ = _body(case)
    lines = _emit(body, ctx)
    at = next(i for i, line in enumerate(lines)
              if line.startswith('tensorforge::ldmatrixX4('))
    guarded = lines[:at] + ['if (threadIdx.x < 16) {', lines[at], '}'] + \
        lines[at + 1:]
    with pytest.raises(kernel_eval.Abort, match='16 lanes of the warp'):
        kernel_eval.evaluate_wave(_oracle_source(guarded), 32, seed=5,
                                  globals_only=True)


# -- the build ---------------------------------------------------------------- #

def _cuda_source(lines, arena):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    n = Case.ROWS * Case.COLS
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda.h"

__global__ void __launch_bounds__(32) transfer_case(float *__restrict__ m0,
    const float *__restrict__ m1) {{
  extern __shared__ __align__(128) float shrMem[];
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main(int argc, char **argv) {{
  const size_t n = {n};
  std::vector<float> h(n), d(n);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 4, n, in) != n) return 2;
  std::fclose(in);
  float *m0, *m1;
  CHECK(cudaMalloc(&m0, n * 4));
  CHECK(cudaMalloc(&m1, n * 4));
  CHECK(cudaMemcpy(m1, h.data(), n * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(m0, 0, n * 4));
  transfer_case<<<1, 32, {arena * 4}>>>(m0, m1);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d.data(), m0, n * 4, cudaMemcpyDeviceToHost));
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 4, n, out) != n) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


def _build(case, tmp_path, arch, product='cubin'):
    ctx, body, arena = _body(case, arch)
    src = tmp_path / 'transfer.cu'
    src.write_text(_cuda_source(_emit(body, ctx), arena))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / {'cubin': 'k.cubin', 'ptx': 'k.ptx'}.get(product,
                                                            'transfer')
    cmd = [_nvcc(), *nvcc.language_flags(), *nvcc.target_flags(arch),
           '-I', str(INCLUDE), '-o', str(out), str(src)]
    if product in ('cubin', 'ptx'):
        cmd.insert(1, f'-{product}')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


def _sass(cubin):
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    return subprocess.run([tool, '-sass', str(cubin)], capture_output=True,
                          text=True, timeout=600).stdout


@pytest.mark.parametrize('name', ['load-x1', 'load-x2-cols', 'load-x4-a'])
def test_compiled_for_sm80_a_load_is_one_ldsm(name, tmp_path):
    """One LDSM of the count the body states, and no trap: sm_80 has it."""
    case = CASES[name]
    sass = _sass(_build(case, tmp_path, 'sm_80'))
    assert len(re.findall(rf'\bLDSM\.16\.M88(?:\.{case.count})?\b',
                          sass)) == 1, sass[:3000]
    assert 'BPT.TRAP' not in sass


@pytest.mark.parametrize('name', ['store-x1', 'store-x2-cols', 'both-x4-a'])
def test_compiled_for_sm90_a_store_is_one_stsm(name, tmp_path):
    case = CASES[name]
    sass = _sass(_build(case, tmp_path, 'sm_90'))
    assert len(re.findall(rf'\bSTSM\.16\.M88(?:\.{case.count})?\b',
                          sass)) == 1, sass[:3000]
    assert 'BPT.TRAP' not in sass


@pytest.mark.parametrize('name', ['load-x2-cols', 'store-x4-a'])
def test_the_ptx_issues_what_the_body_states(name, tmp_path):
    """One instruction of the body's count, its registers one each: a
    helper that named one register twice would move one row twice and
    another never, which only a GPU of sm_90 would otherwise show."""
    case = CASES[name]
    ptx = _build(case, tmp_path, 'sm_90', product='ptx').read_text()
    verb = 'ldmatrix' if case.kind == 'load' else 'stmatrix'
    assert re.findall(rf'{verb}\.sync\.aligned\.m8n8\.x(\d)\.shared\.b16',
                      ptx) == [str(case.count)]
    (registers,) = re.findall(rf'{verb}\.sync\.aligned\.m8n8\.x\d\.shared'
                              r'\.b16[^{]*\{([^}]*)\}', ptx)
    names = [r.strip() for r in registers.split(',')]
    assert len(names) == case.count == len(set(names))
    assert 'trap;' not in ptx


# -- on the hardware ---------------------------------------------------------- #

def _gpu():
    tool = shutil.which('nvidia-smi')
    if tool is None:
        return None
    try:
        run = subprocess.run([tool, '--query-gpu=compute_cap',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    first = run.stdout.strip().splitlines()[:1]
    if run.returncode or not first:
        return None
    return 'sm_' + first[0].strip().replace('.', '')


@pytest.mark.parametrize('name', sorted(CASES))
def test_gpu_moves_what_the_ptx_says(name, tmp_path):
    """The bodies on the local GPU (sm_90 or newer, which has both), built
    for it: `m0` holds what the PTX statement says, to the bit."""
    arch = _gpu()
    if arch is None or int(arch[3:]) < 90:
        pytest.skip('no GPU of sm_90 or newer')
    case = CASES[name]
    exe = _build(case, tmp_path, arch, product='binary')
    n = Case.ROWS * Case.COLS
    values = [float((i * 2654435761) % 1000003) / 7.0 for i in range(n)]
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{n}f', *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    got = struct.unpack(f'<{n}f', (tmp_path / 'out.bin').read_bytes())
    inputs = list(struct.unpack(f'<{n}f', struct.pack(f'<{n}f', *values)))
    want = _expected(case, inputs)
    if case.kind == 'load':
        used = 32 * case.count
        assert list(got[:used]) == want[:used]
    else:
        assert list(got) == want
