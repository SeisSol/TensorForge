# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""tcgen05 on the matrix op (`Op.MATRIX_MAC`, engine `tcgen05`): NVIDIA's
5th generation tensor core with tensor memory, sm_100a.

A and B are read from shared memory through descriptors of tcgen05's
format, A also from tensor memory; D lives in tensor memory
(`MemSpace.TMEM`), which the block allocates and frees (`Op.TMEM_ALLOC`,
`Op.TMEM_DEALLOC`) and a warp loads and stores its quarter of the lanes of
(`Op.TMEM_LD`, `Op.TMEM_ST`).  One thread issues a product and commits it
onto an mbarrier (`Op.COMMIT_GROUP` with an object), every thread waits for
the phase (`Op.WAIT_PHASE`), and the engine's fences order its operations
against the threads' synchronization (`FenceKind.THREAD_SYNC`).  What is
checked here: the vocabulary, the instruction descriptor, the statements
and what the verifier holds them to, the text, the host oracle
(`reference.kernel_eval`) on bodies of every form, the allocation, the
build for sm_100a with the SASS and the PTX read back, and on an sm_100 GPU
the bodies run against the products computed here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, BufferType, Effect,
                                          IRError, MemSpace, Op,
                                          Participants, Uniformity, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import (Completion, FenceKind,
                                             MatrixDescType, Source)
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'

#: tcgen05's format of the no-swizzle K-major layout: the K halves of a row
#: 128 bytes apart, the groups of eight rows 256 apart (GB200's reading,
#: `~/tf-probe/tcgen05/gemm56.cu`).
PLAIN = MatrixDescType(Datatype.TF32, lbo=128, sbo=256, form='tcgen05')
#: The 128-byte swizzle, rows of 32 TF32 of K, atoms 1024 apart (`volume7`).
SW128 = MatrixDescType(Datatype.TF32, swizzle=128, lbo=16, sbo=1024,
                       form='tcgen05')


def _ctx(arch='sm_100a', backend='cuda'):
    return Context(arch=arch, backend=backend, fp_type=Datatype.F32)


def _stmts(body, op):
    return [s for s, _ in walk(body) if s.op == op]


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


def _emit(body, ctx=None, metrics=None):
    sink = _Sink()
    Emitter(sink, ctx if ctx is not None else _ctx(), metrics).run(body)
    return sink.lines


@dataclasses.dataclass
class Case:
    """A body of one block of four warps -- 128 threads, one multiplication
    -- that stages A (128 x 8K) and B (n x 8K) from global memory, A into
    shared memory or into tensor memory, multiplies them on tcgen05 into
    tensor memory, and stores D back, each thread its row.

    `products` K steps of 8; `commits` 1 commits them all at once and waits
    for the one phase, more commits and waits for each product, the
    mbarrier's phases one after another.  `enable` a value: enable-input-d
    computed rather than the literal."""
    n: int = 16
    tmem_a: bool = False
    desc: MatrixDescType = PLAIN
    products: int = 1
    commits: int = 1
    enable: bool = False

    @property
    def k(self):
        return 8 * self.products

    @property
    def columns(self):
        """Tensor memory the block allocates: D's n columns, then A's."""
        need = self.n + (self.k if self.tmem_a else 0)
        return max(32, 1 << (need - 1).bit_length())


def _stage(b, case, dst, src, rows, lane):
    """Every lane copies its elements of the `rows` x `case.k` operand
    `src` (row-major) to where the descriptor's layout puts them in `dst`:
    the K steps one block of rows x 8 after the other where not swizzled,
    the whole K in a row of the 128-byte atoms where it is."""
    total = rows * case.k
    for i in range((total + 127) // 128):
        g = b.op('add', INDEX, lane, 128 * i) if i else lane
        r, k = b.op('div', INDEX, g, case.k), b.op('rem', INDEX, g, case.k)
        if case.desc.swizzle:
            byte = b.op('add', INDEX, b.op('mul', INDEX,
                                           b.op('rem', INDEX, r, 8), 128),
                        b.op('mul', INDEX, k, 4))
            chunk = b.op('bitxor', INDEX, b.op('div', INDEX, byte, 16),
                         b.op('rem', INDEX, r, 8))
            at = b.op('add', INDEX,
                      b.op('mul', INDEX, b.op('div', INDEX, r, 8), 256),
                      b.op('div', INDEX, b.op('add', INDEX,
                                              b.op('mul', INDEX, chunk, 16),
                                              b.op('rem', INDEX, byte, 16)),
                           4), hint='at')
        else:
            step, kk = b.op('div', INDEX, k, 8), b.op('rem', INDEX, k, 8)
            at = b.op('add', INDEX, b.op('mul', INDEX, step, 8 * rows),
                      b.op('add', INDEX,
                           b.op('add', INDEX,
                                b.op('mul', INDEX, b.op('div', INDEX, r, 8),
                                     64),
                                b.op('mul', INDEX, b.op('div', INDEX, kk, 4),
                                     32)),
                           b.op('add', INDEX,
                                b.op('mul', INDEX, b.op('rem', INDEX, r, 8),
                                     4),
                                b.op('rem', INDEX, kk, 4))), hint='at')
        if total % 128 and i == total // 128:
            with b.if_(b.op('lt', BOOL, g, total)):
                b.store(dst, b.load(src, g), at)
        else:
            b.store(dst, b.load(src, g), at)


def _step(case, rows, step):
    """Where step `step` of an operand of `rows` rows starts, in elements."""
    return 8 * step if case.desc.swizzle else step * 8 * rows


def _loads(n):
    """D's n columns as the loads that move them: widths the instruction
    has (`engines.TMEM_COUNTS`), the widest first."""
    out, first = [], 0
    while first < n:
        count = max(c for c in engines.TMEM_COUNTS if c <= n - first)
        out.append((first, count))
        first += count
    return out


def _tcgen05_body(case: Case):
    """`case` built and placed: (context, body, the arena in floats)."""
    ctx = _ctx()
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=ctx)
    align = engines.swizzle_alignment(case.desc.swizzle) // 4
    place = align if case.desc.swizzle else None
    sa = None if case.tmem_a else b.alloc(
        Datatype.F32, (128 * case.k,), MemSpace.SHARED, hint='sa',
        extern='sa', place_align=place)
    sb = b.alloc(Datatype.F32, (case.n * case.k,), MemSpace.SHARED,
                 hint='sb', extern='sb', place_align=place)
    ga = b.alloc(Datatype.F32, (128 * case.k,), MemSpace.GLOBAL, hint='ga',
                 extern='m1')
    gb = b.alloc(Datatype.F32, (case.n * case.k,), MemSpace.GLOBAL,
                 hint='gb', extern='m2')
    out = b.alloc(Datatype.F32, (128 * case.n,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    if sa is not None:
        _stage(b, case, sa, ga, 128, lane)
    _stage(b, case, sb, gb, case.n, lane)
    tmem = b.tmem_alloc(case.columns)
    cursor = b.bulk_channel('mma', 'threadIdx.x == 0',
                            participants=Participants.BLOCK)
    if case.tmem_a:
        # Each thread its row of A, a step's eight columns at a time, into
        # its warp's quarter of the lanes past D's columns; then made
        # visible to the thread that issues.
        for step in range(case.products):
            values = [b.load(ga, b.op('add', INDEX,
                                      b.op('mul', INDEX, lane, case.k),
                                      8 * step + x)) for x in range(8)]
            b.tmem_st(tmem, case.n + 8 * step, values)
        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')
    b.fence(FenceKind.PROXY)
    b.barrier('block')
    if case.tmem_a:
        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')
    issue = b.op('eq', BOOL, lane, 0, hint='issue')
    reads = []
    for step in range(case.products):
        bd = b.matrix_desc(sb, _step(case, case.n, step), desc=case.desc,
                           hint='bd')
        if case.tmem_a:
            a, a_column = tmem, case.n + 8 * step
        else:
            a = b.matrix_desc(sa, _step(case, 128, step), desc=case.desc,
                              hint='ad')
            a_column = 0
            reads.append(a)
        reads.append(bd)
        if case.enable:
            enable = b.op('lt', BOOL, b.thread_id('y'), step, hint='en')
        else:
            enable = 0 if step == 0 else 1
        b.tcgen05(case.n, a, bd, tmem, issue=issue, enable_input_d=enable,
                  a_column=a_column)
        if case.commits > 1 or step == case.products - 1:
            b.commit_group('tcgen05', object=cursor, issue=issue)
            b.matrix_wait('tcgen05', object=cursor, reads=reads,
                          tmem=[tmem])
            reads = []
    b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')
    for first, count in _loads(case.n):
        values = b.tmem_ld(tmem, first, count)
        for j, v in enumerate(values):
            b.store(out, v, b.op('add', INDEX, b.op('mul', INDEX, lane,
                                                    case.n), first + j))
    b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')
    b.tmem_dealloc(tmem)
    body, layout = allocate(b.finish(), arenas={'shrMem': MULT})
    return ctx, body, layout.per_mult


CASES = {
    'ss16': Case(16),
    'ss64': Case(64),
    'ss256': Case(256),
    'k2': Case(32, products=2),
    'k2-commits': Case(32, products=2, commits=2),
    'k3-enable': Case(16, products=3, enable=True),
    'ts16': Case(16, tmem_a=True),
    'ts-k2-commits': Case(48, tmem_a=True, products=2, commits=2),
    'sw128-k4': Case(16, desc=SW128, products=4),
}


def _oracle_source(lines):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    return ('void k() {\n'
            '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + ''.join(f'  {line}\n' for line in keep) + '}')


def _expected(case, a, b):
    """D = A B^T over the TF32 operands, summed in double."""
    tf = kernel_eval._tf32
    return [sum(tf(a[r * case.k + x]) * tf(b[c * case.k + x])
                for x in range(case.k))
            for r in range(128) for c in range(case.n)]


def _run_oracle(case, src=None, seed=3):
    ctx, body, _ = _tcgen05_body(case)
    if src is None:
        src = _oracle_source(_emit(body, ctx))
    mem = kernel_eval.evaluate_wave(src, 128, seed=seed, globals_only=True)
    slot = kernel_eval.Slot(seed)
    a = [slot.read('m1', i) for i in range(128 * case.k)]
    b = [slot.read('m2', i) for i in range(case.n * case.k)]
    return [mem[('m0', i)] for i in range(128 * case.n)], _expected(case, a, b)


# -- the vocabulary ------------------------------------------------------------ #

def test_the_engine_is_one_threads_and_completes_on_a_barrier():
    engine = engines.MATRIX_ENGINES['tcgen05']
    assert engine.collective is None and engine.work
    assert engine.completes is Completion.BARRIER
    assert engine.sources == ((Source.SHARED, Source.TMEM), (Source.SHARED,),
                              (Source.TMEM,))
    assert engines.COMMIT_ENGINES == {'wgmma': Completion.GROUP,
                                      'tcgen05': Completion.BARRIER}


def test_tensor_memory_is_a_space_and_its_ops_are_entries():
    assert MemSpace.TMEM not in (MemSpace.SHARED, MemSpace.REGISTER)
    for op in (Op.TMEM_ALLOC, Op.TMEM_DEALLOC):
        spec = engines.spec_of(op)
        assert spec.effect == Effect.BARRIER | Effect.WRITE
        assert spec.arrival is Uniformity.BLOCK and spec.settles
    assert engines.spec_of(Op.TMEM_LD).effect == Effect.READ
    assert engines.spec_of(Op.TMEM_ST).effect == Effect.WRITE
    assert not engines.spec_of(Op.TMEM_LD).movable


def test_the_instruction_descriptor():
    """PTX's fields for `kind::tf32`: D FP32, A and B TF32, K-major, N >> 3
    from bit 17, M >> 4 from bit 24 -- the prototypes' on GB200."""
    assert engines.tcgen05_idesc((128, 64, 8)) == (
        (1 << 4) | (2 << 7) | (2 << 10) | (8 << 17) | (8 << 24))
    assert engines.tcgen05_idesc((128, 128, 8)) == 0x08200910


def test_the_descriptor_of_tcgen05s_format():
    assert engines.matrix_descriptor_bits(PLAIN) == (
        (8 << 16) | (16 << 32) | (1 << 46))
    assert engines.matrix_descriptor_bits(SW128) == (
        (1 << 16) | (64 << 32) | (1 << 46) | (2 << 61))


# -- the statements ------------------------------------------------------------ #

def _products(case):
    return _stmts(_tcgen05_body(case)[1], Op.MATRIX_MAC)


def test_a_product_is_in_place_in_tensor_memory():
    (s,) = _products(CASES['ss16'])
    (a,), (b,), (c,) = engines.fragments(s)
    assert s.attr('engine') == 'tcgen05' and s.attr('shape') == (128, 16, 8)
    assert s.attr('sources') == (Source.SHARED, Source.SHARED, Source.TMEM)
    assert s.attr('scalars') == ('enable_input_d', 'issue')
    assert s.attr('d_column') == 0 and s.attr('collective') is None
    assert isinstance(c.type, BufferType) and c.type.space is MemSpace.TMEM
    assert s.effect == Effect.WRITE | Effect.READ | Effect.ASYNC
    assert not s.movable
    assert [(x.space, x.kind) for x in s.accesses] == [
        (MemSpace.TMEM, Effect.READ | Effect.WRITE),
        (MemSpace.SHARED, Effect.READ), (MemSpace.SHARED, Effect.READ)]
    assert s.accesses[0].base is c
    assert engines.verify_engine(s) == []


def test_a_from_tensor_memory_is_read_there():
    first, _ = _products(CASES['ts-k2-commits'])
    (a,), _, (c,) = engines.fragments(first)
    assert a is c and first.attr('a_column') == 48
    assert [(x.space, x.kind) for x in first.accesses] == [
        (MemSpace.TMEM, Effect.READ | Effect.WRITE),
        (MemSpace.TMEM, Effect.READ), (MemSpace.SHARED, Effect.READ)]
    assert engines.verify_engine(first) == []


def test_every_body_verifies():
    for case in CASES.values():
        body = _tcgen05_body(case)[1]
        assert verify(body, strict=False) == [], case


@pytest.mark.parametrize('change,why', [
    (dict(shape=(64, 16, 8)), 'shape'),
    (dict(shape=(128, 24, 8)), 'shape'),
    (dict(types=(Datatype.F64,) * 4), 'TF32 operands'),
    (dict(d_column=24), 'columns past'),
])
def test_a_product_this_lowering_does_not_write_is_refused(change, why):
    (s,) = _products(CASES['ss16'])
    for k, v in change.items():
        s = s.with_attr(k, v)
    with pytest.raises(IRError, match=why):
        engines.lower(s, _ctx().target, name=str)


def test_every_thread_issuing_is_refused():
    (s,) = _products(CASES['ss16'])
    s = dataclasses.replace(s, args=s.args[:-1] + (1,))
    with pytest.raises(IRError, match='every thread would issue'):
        engines.lower(s, _ctx().target, name=str)


def test_a_wgmma_descriptor_is_no_tcgen05_operand():
    (s,) = _products(CASES['ss16'])
    b = IRBuilder(fptype=Datatype.F32, context=_ctx())
    sa = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='s', extern='s',
                 arena='blk', offset=0)
    wg = b.matrix_desc(sa, 0, desc=dataclasses.replace(PLAIN, form='wgmma'))
    bad = dataclasses.replace(s, args=(wg,) + s.args[1:])
    with pytest.raises(IRError, match="tcgen05's format"):
        engines.lower(bad, _ctx().target, name=str)


@pytest.mark.parametrize('arch,backend', [('sm_100', 'cuda'),
                                          ('sm_90a', 'cuda'),
                                          ('gfx942', 'hip')])
def test_only_sm100a_writes_one(arch, backend):
    body = _tcgen05_body(CASES['ss16'])[1]
    target = _ctx(arch, backend).target
    for op in (Op.MATRIX_MAC, Op.TMEM_ALLOC, Op.TMEM_LD, Op.TMEM_DEALLOC):
        with pytest.raises(IRError):
            engines.lower(_stmts(body, op)[0], target, name=str)


def test_a_commit_onto_an_object_names_it_and_the_issuer():
    body = _tcgen05_body(CASES['ss16'])[1]
    (commit,) = _stmts(body, Op.COMMIT_GROUP)
    (product,) = _stmts(body, Op.MATRIX_MAC)
    assert commit.args == (product.args[-1],)
    assert engines.object_of(commit).kind is engines.SyncKind.MBARRIER
    assert engines.verify_engine(commit) == []
    with pytest.raises(IRError, match='mbarrier'):
        engines.commit_group('tcgen05')
    with pytest.raises(IRError, match='counter'):
        engines.commit_group('wgmma', engines.object_value(commit),
                             commit.args[0])


def test_the_wait_is_the_phase_and_names_what_it_makes_valid():
    body = _tcgen05_body(CASES['ss16'])[1]
    (wait,) = _stmts(body, Op.WAIT_PHASE)
    (alloc,) = _stmts(body, Op.TMEM_ALLOC)
    tmem = alloc.target[0]
    assert [(a.space, a.kind) for a in wait.accesses] == [
        (MemSpace.TMEM, Effect.READ | Effect.WRITE),
        (MemSpace.SHARED, Effect.READ), (MemSpace.SHARED, Effect.READ)]
    assert wait.accesses[0].base is tmem
    assert wait.effect == Effect.READ | Effect.WRITE | Effect.ASYNC


def test_the_allocation_slot_is_occupied_where_it_is_written():
    """The slot the address comes through is a buffer the allocation writes
    and reads: placed apart from the operands that are live across it."""
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    at = {m.group(1): int(m.group(2)) for m in (
        re.search(r'(\w+) = (?:reinterpret_cast<\w+\*>\()?&shrMem\[(\d+)\]', l)
        for l in lines) if m}
    slot = next(v for k, v in at.items() if 'tmemSlot' in k)
    assert not 0 <= slot < 128 * 8 and not 1024 <= slot < 1024 + 16 * 8


# -- the text ------------------------------------------------------------------ #

def test_the_text_of_a_product():
    """Inside the guard of the targets that have it: the issue predicate and
    the literal enable-input-d, D's address in tensor memory, the
    descriptors, the instruction descriptor an immediate; volatile and
    clobbering memory."""
    (s,) = _products(CASES['ss16'])
    (a,), (b,), (c,) = engines.fragments(s)
    issue = s.args[-1]
    lowered = engines.lower(s, _ctx().target, name=str)
    assert lowered.lines == (
        '#if defined(TENSORFORGE_TCGEN05)',
        'asm volatile("{\\n.reg .pred e, p;\\nsetp.ne.b32 e, %4, 0;\\n"\n'
        '"@e tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, 0;\\n}"\n'
        ': \n'
        f': "r"({c}), "l"({a}), "l"({b}), "n"(0x08040910u), '
        f'"r"(static_cast<uint32_t>({issue}))\n'
        ': "memory"\n);',
        '#endif')


def test_the_text_of_a_product_from_tensor_memory_with_a_value_enable():
    body = _tcgen05_body(Case(16, tmem_a=True, enable=True))[1]
    (s,) = _stmts(body, Op.MATRIX_MAC)
    (a,), (b,), (c,) = engines.fragments(s)
    enable, issue = s.args[-2:]
    text = engines.lower(s, _ctx().target, name=str).lines[1]
    assert text == (
        'asm volatile("{\\n.reg .pred e, p;\\nsetp.ne.b32 e, %4, 0;\\n'
        'setp.ne.b32 p, %5, 0;\\n"\n'
        '"@e tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\\n}"\n'
        ': \n'
        f': "r"({c}), "r"({a} + 16u), "l"({b}), "n"(0x08040910u), '
        f'"r"(static_cast<uint32_t>({issue})), '
        f'"r"(static_cast<uint32_t>({enable}))\n'
        ': "memory"\n);')


def test_the_tensor_memory_ops_and_fences():
    body = _tcgen05_body(CASES['ts16'])[1]
    target = _ctx().target
    (alloc,) = _stmts(body, Op.TMEM_ALLOC)
    slot = alloc.args[0]
    assert engines.lower(alloc, target, name=str).result == (
        f'tensorforge::tmemAlloc<32>(&{slot}[0])')
    (dealloc,) = _stmts(body, Op.TMEM_DEALLOC)
    assert engines.lower(dealloc, target, name=str).lines == (
        f'tensorforge::tmemDealloc<32>({alloc.target[0]});',)
    (ld,) = _stmts(body, Op.TMEM_LD)
    lines = engines.lower(ld, target, name=str).lines
    assert lines[:16] == tuple(f'float {v}{{}};' for v in ld.target)
    assert lines[16] == '#if defined(TENSORFORGE_TCGEN05)'
    assert lines[17].startswith(
        'asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, ')
    assert lines[17].endswith(
        f': "r"(tensorforge::tmemWarpAddress({alloc.target[0]}, 0u))\n);')
    assert lines[18:] == ('#endif', 'tensorforge::tmemWaitLd();')
    (st,) = _stmts(body, Op.TMEM_ST)
    lines = engines.lower(st, target, name=str).lines
    assert lines[1].startswith(
        'asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, ')
    assert lines[-1] == 'tensorforge::tmemWaitSt();'
    fences = [x for x in _stmts(body, Op.FENCE)
              if x.attr('kind') is FenceKind.THREAD_SYNC]
    assert [engines.lower(x, target).lines for x in fences] == [
        ('tensorforge::tcgen05FenceBefore();',),
        ('tensorforge::tcgen05FenceAfter();',),
        ('tensorforge::tcgen05FenceAfter();',),
        ('tensorforge::tcgen05FenceBefore();',)]
    (commit,) = _stmts(body, Op.COMMIT_GROUP)
    assert engines.lower(commit, target, name=str).lines == (
        f'tensorforge::tcgen05Commit({engines.object_value(commit)}, '
        f'{commit.args[0]});',)
    assert 'tensorforge::tcgen05Commit(mma_bulk, ' in '\n'.join(_emit(body))


def test_a_load_or_store_the_instruction_has_no_width_for_is_refused():
    body = _tcgen05_body(CASES['ss16'])[1]
    (ld,) = _stmts(body, Op.TMEM_LD)
    bad = dataclasses.replace(ld, target=ld.target[:3])
    with pytest.raises(IRError, match='columns of tensor memory'):
        engines.lower(bad, _ctx().target, name=str)
    assert any('columns a thread' in d for d in engines.verify_engine(bad))


# -- the host oracle ------------------------------------------------------------ #

@pytest.mark.parametrize('name', sorted(CASES))
def test_the_oracle_multiplies_as_the_ptx_lays_the_operands_out(name):
    """D in tensor memory, row r lane r, as the oracle reads the operands
    on its own (`Interp._tcgen05_mma`); the bodies stage by the layouts the
    descriptors state and load D by the warps' quarters."""
    got, want = _run_oracle(CASES[name])
    worst = max(abs(g - w) / max(1.0, abs(w)) for g, w in zip(got, want))
    assert worst < 1e-12, worst


def test_d_read_ahead_of_its_phase_is_refused():
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    wait = lines.index('tensorforge::bulk_wait(mma_bulk);')
    early = lines[:wait] + lines[wait + 1:]
    with pytest.raises(kernel_eval.Abort, match='product writes it'):
        _run_oracle(CASES['ss16'], _oracle_source(early))


def test_an_operand_written_in_flight_is_a_race():
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    wait = lines.index('tensorforge::bulk_wait(mma_bulk);')
    racy = lines[:wait] + ['sa[0] = 0.5f;'] + lines[wait:]
    with pytest.raises(kernel_eval.Abort, match='in flight'):
        _run_oracle(CASES['ss16'], _oracle_source(racy))


def test_tensor_memory_freed_in_flight_is_refused():
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    wait = lines.index('tensorforge::bulk_wait(mma_bulk);')
    free = next(i for i, l in enumerate(lines) if 'tmemDealloc' in l)
    early = lines[:wait] + [lines[free]]
    with pytest.raises(kernel_eval.Abort, match='freed with products'):
        _run_oracle(CASES['ss16'], _oracle_source(early))


def test_a_warp_reaches_its_quarter_alone():
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    # Every warp at lane 0, the first warp's quarter.
    bad = [re.sub(r'tensorforge::tmemWarpAddress\(([^,]+),\s*([^)]+)\)',
                  r'(\1 + \2)', l) for l in lines]
    with pytest.raises(kernel_eval.Abort, match='reaches the lanes'):
        _run_oracle(CASES['ss16'], _oracle_source(bad))


def test_a_second_allocation_of_the_block_is_refused():
    """The allocation gives up the block's permit to allocate
    (`tmemAlloc`): another behind it, as a trip of a loop around it would
    make, is undefined on the hardware."""
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    alloc = next(line for line in lines if 'tmemAlloc' in line)
    again = re.sub(r'uint32_t (\w+) =', r'uint32_t \1Again =', alloc)
    with pytest.raises(kernel_eval.Abort,
                       match='allocates tensor memory again'):
        _run_oracle(CASES['ss16'], _oracle_source(lines + [again]))


def test_a_slot_of_each_warps_own_is_refused():
    """The block's first warp writes the address into the slot, and every
    thread reads its own: a slot each warp has a window of its own for --
    the multiplications' arena in a layout of a warp each -- holds no
    address for the others."""
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    i = next(i for i, line in enumerate(lines) if 'tmemAlloc' in line)
    slot = re.search(r'&(\w+)\[0\]', lines[i]).group(1)
    own = [f'int32_t * warpSlot = &{slot}[threadIdx.x / 32];',
           lines[i].replace(f'&{slot}[0]', '&warpSlot[0]')]
    with pytest.raises(kernel_eval.Abort, match="not the block's"):
        _run_oracle(CASES['ss16'],
                    _oracle_source(lines[:i] + own + lines[i + 1:]))


def test_a_wait_on_an_mbarrier_nobody_armed_is_refused():
    """The leader arms the mbarrier its cursor names, and the commit
    arrives there: a warp whose cursor names a barrier of its own waits for
    a phase nothing completes -- on the hardware, forever."""
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    i = next(i for i, line in enumerate(lines) if 'bulk_arm' in line)
    barrier = re.search(r'bulk_arm\((\w+),', lines[i]).group(1)
    own = [f'float * warpBarrier = &{barrier}[(threadIdx.x / 32) * 2];',
           lines[i].replace(f'bulk_arm({barrier},', 'bulk_arm(warpBarrier,')]
    with pytest.raises(kernel_eval.Abort, match='nobody armed'):
        _run_oracle(CASES['ss16'],
                    _oracle_source(lines[:i] + own + lines[i + 1:]))


def test_a_wait_a_phase_behind_is_refused():
    """Two commits onto the barrier and one wait: the next wait is for the
    parity of the phase under way, which it waits out."""
    lines = _emit(_tcgen05_body(CASES['ss16'])[1])
    i = next(i for i, line in enumerate(lines) if 'tcgen05Commit' in line)
    twice = lines[:i + 1] + [lines[i]] + lines[i + 1:]
    with pytest.raises(kernel_eval.Abort, match='never complete'):
        _run_oracle(CASES['ss16'], _oracle_source(twice))


# -- the target and the build -------------------------------------------------- #

def test_sm100a_is_sm100_with_tcgen05():
    a, plain = _ctx('sm_100a').target, _ctx('sm_100').target
    assert a.tcgen05() and not plain.tcgen05()
    assert not _ctx('sm_90a').target.tcgen05()
    assert a.hw.max_local_mem_size_per_block == \
        plain.hw.max_local_mem_size_per_block


def _cuda_source(case, lines, arena):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda.h"

__global__ void __launch_bounds__(128) tcgen05_case(float *__restrict__ m0,
    const float *__restrict__ m1, const float *__restrict__ m2) {{
  extern __shared__ __align__(1024) unsigned char tfRaw[];
  float *shrMem = reinterpret_cast<float *>(tfRaw + ((1024u -
      (static_cast<unsigned>(__cvta_generic_to_shared(tfRaw)) & 1023u)) & 1023u));
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main(int argc, char **argv) {{
  const size_t na = {128 * case.k}, nb = {case.n * case.k}, nd = {128 * case.n};
  std::vector<float> h(na + nb), d(nd);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 4, na + nb, in) != na + nb) return 2;
  std::fclose(in);
  float *m0, *m1, *m2;
  CHECK(cudaMalloc(&m0, nd * 4));
  CHECK(cudaMalloc(&m1, na * 4));
  CHECK(cudaMalloc(&m2, nb * 4));
  CHECK(cudaMemcpy(m1, h.data(), na * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(m2, h.data() + na, nb * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(m0, 0, nd * 4));
  const int smem = {arena * 4} + 1024;
  CHECK(cudaFuncSetAttribute(tcgen05_case,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  tcgen05_case<<<1, 128, smem>>>(m0, m1, m2);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d.data(), m0, nd * 4, cudaMemcpyDeviceToHost));
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 4, nd, out) != nd) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


#: The first CUDA release that builds each arch-specific target: an older
#: toolkit refuses it ("Unsupported gpu architecture"), as x-wing0's CUDA
#: 12.4 refuses sm_100a.
ARCH_RELEASE = {'sm_90a': (12, 0), 'sm_100a': (12, 8)}


def _nvcc_builds(arch):
    """The toolkit's nvcc, or a skip where it builds no `arch`: the base
    target missing from `nvcc --list-gpu-code`, or a release older than the
    one that brought the arch-specific target (`ARCH_RELEASE`)."""
    nvcc = _nvcc()
    listed = subprocess.run([nvcc, '--list-gpu-code'], capture_output=True,
                            text=True, timeout=60).stdout.split()
    found = re.search(r'release (\d+)\.(\d+)', subprocess.run(
        [nvcc, '--version'], capture_output=True, text=True,
        timeout=60).stdout)
    release = (int(found.group(1)), int(found.group(2))) if found else (0, 0)
    if (arch.rstrip('af') not in listed
            or release < ARCH_RELEASE.get(arch, (0, 0))):
        pytest.skip(f'nvcc {release[0]}.{release[1]} builds no {arch}')
    return nvcc


def _build(case, tmp_path, flags, product='cubin'):
    ctx, body, arena = _tcgen05_body(case)
    src = tmp_path / 'tcgen05.cu'
    src.write_text(_cuda_source(case, _emit(body, ctx), arena))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / {'cubin': 'k.cubin', 'ptx': 'k.ptx'}.get(product,
                                                            'tcgen05')
    cmd = [_nvcc_builds('sm_100a'), *nvcc.language_flags(), *flags,
           '-I', str(INCLUDE), '-o', str(out), str(src)]
    if product in ('cubin', 'ptx'):
        cmd.insert(1, f'-{product}')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


@pytest.mark.parametrize('name', sorted(CASES))
def test_compiled_for_sm100a_each_product_is_one_utchmma(name, tmp_path):
    """Built as the toolchain builds an sm_100a target: one UTCHMMA a
    product, one UTCBAR a commit, one LDTM a load of D, the allocation and
    the free."""
    case = CASES[name]
    cubin = _build(case, tmp_path,
                   toolchain.COMPILERS['cuda'].target_flags('sm_100a'))
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(cubin)], capture_output=True,
                          text=True, timeout=600).stdout
    count = lambda pattern: len(re.findall(pattern, sass))
    assert count(r'\bUTCHMMA\b') == case.products
    assert count(r'\bUTCBAR\b') == case.commits
    assert count(r'\bLDTM\.x\d+') == len(_loads(case.n))
    assert count(r'\bSTTM\.x8\b') == (case.products if case.tmem_a else 0)
    assert count(r'UTCATOMSWS\.FIND_AND_SET') >= 1
    assert count(r'UTCATOMSWS\.AND') == 1


@pytest.mark.parametrize('name', sorted(CASES))
def test_compiled_for_sm100a_the_ptx_issues_what_the_body_states(name,
                                                                  tmp_path):
    """The PTX: as many products, commits, loads and stores of tensor
    memory, and fences as the body states, the allocation and the free once
    each, the products in their order between the commits."""
    case = CASES[name]
    ctx, body, _ = _tcgen05_body(case)
    ptx = _build(case, tmp_path,
                 toolchain.COMPILERS['cuda'].target_flags('sm_100a'),
                 product='ptx').read_text()
    order = re.findall(r'tcgen05\.(mma|commit|ld|st)\b', ptx)
    want = []
    for s, _ in walk(body):
        if s.op == Op.MATRIX_MAC:
            want.append('mma')
        elif s.op == Op.COMMIT_GROUP:
            want.append('commit')
        elif s.op == Op.TMEM_LD:
            want.append('ld')
        elif s.op == Op.TMEM_ST:
            want.append('st')
    assert order == want
    assert len(re.findall(r'tcgen05\.alloc\b', ptx)) == 1
    assert len(re.findall(r'tcgen05\.dealloc\b', ptx)) == 1
    assert len(re.findall(r'tcgen05\.wait::ld', ptx)) == len(
        _stmts(body, Op.TMEM_LD))
    # No helper traps: their guard holds for sm_100a.  (The SASS has traps
    # of ptxas's own, in its expansion of the allocation and the free.)
    assert 'trap;' not in ptx


def test_built_with_arch_sm100a_the_generic_ptx_compiles(tmp_path):
    _build(CASES['ss16'], tmp_path, ['-arch=sm_100a'], product='binary')


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


def _inputs(case, kind):
    out = []
    for i in range(128 * case.k + case.n * case.k):
        h = (i * 2654435761) & 0xFFFFFFFF
        if kind == 'int':
            out.append(float((h >> 7) % 7 - 3))
        else:
            out.append(((h >> 8) % 4096 - 2048) / 977.0)
    return out


@pytest.mark.parametrize('kind', ['int', 'real'])
@pytest.mark.parametrize('name', sorted(CASES))
def test_gpu_on_sm100(name, kind, tmp_path):
    """The bodies run on an sm_100 part, built for sm_100a, against the
    products computed here: to the bit with integer operands, within FP32's
    rounding of the sums with the others (TF32 operands cut to 19 bits)."""
    if _gpu() != 'sm_100':
        pytest.skip('no sm_100 GPU')
    case = CASES[name]
    exe = _build(case, tmp_path,
                 toolchain.COMPILERS['cuda'].target_flags('sm_100a'),
                 product='binary')
    values = _inputs(case, kind)
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{len(values)}f', *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    got = struct.unpack(f'<{128 * case.n}f',
                        (tmp_path / 'out.bin').read_bytes())
    want = _expected(case, values[:128 * case.k], values[128 * case.k:])
    scale = max(abs(w) for w in want) or 1.0
    bound = 0.0 if kind == 'int' else 1e-5 * scale
    bad = [(i // case.n, i % case.n, g, w)
           for i, (g, w) in enumerate(zip(got, want)) if abs(g - w) > bound]
    assert not bad, (f'{len(bad)} of {len(want)} elements of D differ '
                     f'(row, column, got, want): {bad[:6]}')
