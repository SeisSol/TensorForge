# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""gfx1250's asynchronous copy into the LDS (`pir.engines`, `async_lds`),
under `Options.amd_async_copies`.

`global_load_async_to_lds` moves 1, 4, 8 or 16 bytes a lane from global
memory into the LDS, each lane to an address of its own -- unlike CDNA's
LDS-DMA, which takes one base a wave -- and counts on `asynccnt`, which
counts these copies and nothing else.  Neither the compiler nor a block
barrier waits for one: each wait is written out on that counter, its count
the copies issued behind the one it waits for.  Compile-only: no device
here has the instruction, so what runs is the reference interpreter.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from tensorforge.backend import pir
from tensorforge.backend.pir import INDEX, IRBuilder, MemSpace, Op
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import Completion
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.vm.lexic.hip_lexic import HipLexic

from test_pir_engines import _Commented, _assemble, _generated

F32 = Datatype.F32


def _target(arch, **options):
    return Context(arch=arch, backend='hip', fp_type=F32,
                   options=Options(**options) if options else None)


# -- the engine and the target ------------------------------------------------ #

def test_the_engine_counts_its_copies_on_asynccnt():
    engine = engines.TRANSFER_ENGINES['async_lds']
    assert engine.ops == (Op.COPY_ASYNC,)
    assert engine.completes is Completion.COUNTER
    assert engines.counter_of('async_lds') == 'load'
    assert engines.waited_on('async_lds') == 'asynccnt'
    assert engines.one_to_one('async_lds')


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_gfx125x_copies_through_it(arch):
    target = _target(arch).target
    assert target.async_copy_path() and target.async_lds()
    assert target.copy_async_sizes() == (1, 4, 8, 16)
    assert engines.copy_engine(target) == 'async_lds'
    assert not target.one_wait_counter('async_lds')


def test_cdna_keeps_its_dma_and_rdna_has_none():
    cdna = _target('gfx942').target
    assert engines.copy_engine(cdna) == 'lds_dma' and not cdna.async_lds()
    assert cdna.copy_async_sizes() == (1, 2, 4)
    assert cdna.one_wait_counter('lds_dma') and cdna.one_wait_counter()
    rdna = _target('gfx1150').target
    assert not rdna.async_copy_path() and rdna.copy_async_sizes() == ()
    assert engines.copy_engine(rdna) == 'cp_async'


def test_staging_takes_it_where_asked_whatever_the_lanes():
    """No base a wave to share: a multiplication of any width, elements of
    four or eight bytes, and only where asked."""
    def staging(threads, elem=4, on=True, arch='gfx1250'):
        ctx = _target(arch, amd_async_copies=on)
        return ctx.target.async_staging(ctx.get_user_options(), threads, elem)
    assert staging(16) and staging(32) and staging(64, elem=8)
    assert staging(32, arch='gfx1251')
    assert not staging(32, on=False) and not staging(32, elem=2)


@pytest.mark.parametrize('nbytes,bits,elem', [
    (1, 8, 'char'), (4, 32, 'int'), (8, 64, 'tensorforge::VectorT<int, 2>'),
    (16, 128, 'tensorforge::VectorT<int, 4>')])
def test_the_lexic_spells_one_builtin_a_size(nbytes, bits, elem):
    lex = HipLexic('hip', 'amd')
    assert lex.copy_async('&s0[i]', '&m1[j]', nbytes, engine='async_lds') == (
        f'__builtin_amdgcn_global_load_async_to_lds_b{bits}('
        f'(__attribute__((address_space(1))) {elem}*)(&m1[j]), '
        f'(__attribute__((address_space(3))) {elem}*)(&s0[i]), 0, 0);')


def test_the_lexic_refuses_what_the_copy_cannot_move():
    lex = HipLexic('hip', 'amd')
    assert lex.copy_async('d', 's', 2, engine='async_lds') is None
    assert lex.copy_async('d', 's', 12, engine='async_lds') is None
    assert lex.copy_async('d', 's', 16, 4, engine='async_lds') is None
    assert 'global_load_lds(' in lex.copy_async('d', 's', 4)


# -- waits on asynccnt --------------------------------------------------------- #

def _body(copies=3, arch='gfx1250', extra=None):
    """`copies` copies of one wave each, the lane past 32 elements a copy,
    each waited for and read back reversed -- every lane reading another's
    element within the copy's own 32.  `extra` is called between the copies
    and the waits, and what it returns is waited for behind them."""
    ctx = _target(arch)
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (32 * copies,), MemSpace.SHARED, hint='s0',
                 extern='s0', arena='blk', offset=0)
    g = b.alloc(F32, (32 * copies,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (32 * copies,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    ats = [lane] + [b.op('add', INDEX, lane, 32 * k, hint='at')
                    for k in range(1, copies)]
    toks = [b.copy_async(s0, g, dst_index=(at,), src_index=(at,), threads=32)
            for at in ats]
    later = extra(b, g, lane) if extra is not None else None
    for k, (tok, at) in enumerate(zip(toks, ats)):
        b.wait(tok)
        back = b.op('sub', INDEX, 32 * k + 31, lane, hint='bk')
        b.store(out, b.load(s0, back, hint='u'), at)
    if later is not None:
        b.store(out, b.wait(later), lane)
    return ctx, b.finish()


def _emit(ctx, body):
    out, diag = pir.schedule_async(body)
    assert not diag, diag
    sink = _Commented()
    Emitter(sink, ctx).run(out)
    return out, sink.lines


def _kernel(lines):
    keep = [l for l in lines if not re.match(r'^float m\d+\[', l)]
    return ('void k() {\n'
            '  auto* blk = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + ''.join(f'  {l}\n' for l in keep) + '}')


def test_a_wait_is_asynccnt_s_and_counts_the_copies_behind_it():
    ctx, body = _body()
    waits = [s for s in body if s.op == Op.WAIT_COUNTER]
    assert [(w.attr('counter'), w.attr('engine')) for w in waits] == [
        ('asynccnt', 'async_lds')] * 3
    out, lines = _emit(ctx, body)
    assert [s.attr('count') for s in out if s.op == Op.WAIT_COUNTER] == [
        2, 1, 0]
    copies = [i for i, l in enumerate(lines) if 'async_to_lds_b32(' in l]
    waited = [l for l in lines if l.startswith('__builtin_amdgcn_s_wait_')]
    assert len(copies) == 3 and waited == [
        '__builtin_amdgcn_s_wait_asynccnt(2);',
        '__builtin_amdgcn_s_wait_asynccnt(1);',
        '__builtin_amdgcn_s_wait_asynccnt(0);']


def test_a_load_into_registers_is_not_counted():
    """It goes to `loadcnt`, which the wait on `asynccnt` does not see."""
    ctx, body = _body(extra=lambda b, g, lane: b.load_async(g, lane))
    out, _ = _emit(ctx, body)
    first = next(s for s in out if s.op == Op.WAIT_COUNTER)
    assert first.attr('count') == 2 and first.attr('prior_unified') == 3


def test_a_copy_another_counter_counts_is_not_counted():
    """A wait on one counter counts what that counter counts: a copy of the
    same class whose engine is waited on another counter is none of its
    business."""
    ctx, body = _body()
    copies = [s for s in body if s.op == Op.COPY_ASYNC]
    other = replace(copies[1], attrs=tuple(
        (k, 'lds_dma' if k == 'engine' else v) for k, v in copies[1].attrs))
    body = tuple(other if s is copies[1] else s for s in body)
    out, _ = pir.schedule_async(body)
    first = next(s for s in out if s.op == Op.WAIT_COUNTER)
    assert first.attr('count') == 1


def test_a_drain_is_asynccnt_at_zero():
    """Nothing else waits for the copies in front of a read -- neither the
    compiler nor a block barrier."""
    ctx = _target('gfx1250')
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s0', extern='s0',
                 arena='blk', offset=0)
    g = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    at = b.op('add', INDEX, lane, 32, hint='at')
    for where in (lane, at):
        b.copy_async(s0, g, dst_index=(where,), src_index=(where,),
                     threads=32)
    b.wait()
    for where in (lane, at):
        b.store(out, b.load(s0, where, hint='u'), where)
    body, _ = pir.schedule_async(b.finish())
    sink = _Commented()
    Emitter(sink, ctx).run(body)
    lines = sink.lines
    last = max(i for i, l in enumerate(lines) if 'async_to_lds' in l)
    read = next(i for i, l in enumerate(lines) if '= s0[' in l)
    assert [l for l in lines[last:read] if 's_wait_asynccnt' in l] == [
        '__builtin_amdgcn_s_wait_asynccnt(0);']


# -- the reference interpreter ------------------------------------------------- #

def test_the_interpreter_lands_a_copy_at_its_wait():
    """Every lane reads another's element of each copy behind its wait:
    the copy's value and no race.  A wait one copy laxer reads the first in
    flight, and so does no wait."""
    from tensorforge.reference import kernel_eval
    ctx, body = _body()
    _, lines = _emit(ctx, body)
    src = _kernel(lines)

    def run(text):
        races = []
        mem = kernel_eval.evaluate_wave(text, 32, seed=5, globals_only=True,
                                        races=races)
        return mem, sorted({r.kind for r in races})
    mem, races = run(src)
    seed = kernel_eval.Slot(5)
    assert races == []
    assert all(mem[('m0', 32 * k + t)] == seed.read('m1', 32 * k + 31 - t)
               for k in range(3) for t in range(32))
    assert run(src.replace('s_wait_asynccnt(2)', 's_wait_asynccnt(3)',
                           1))[1] == ['FLIGHT']
    assert run(src.replace('__builtin_amdgcn_s_wait_asynccnt(2);', '',
                           1))[1] == ['FLIGHT']


def test_the_interpreter_writes_each_lane_where_it_says_and_no_barrier_lands_it():
    """A lane's own address, wherever it points -- not one base a wave; and
    a block barrier lands nothing, since its fence waits for no
    `asynccnt`: only the wait does, a wave at a time."""
    from tensorforge.reference import kernel_eval
    copy = ('__builtin_amdgcn_global_load_async_to_lds_b{bits}('
            '(__attribute__((address_space(1))) {elem}*)(&m1[{si}]), '
            '(__attribute__((address_space(3))) {elem}*)(&s0[{di}]), 0, 0);')

    def run(*stmts, lanes=32):
        src = ('void k() {\n'
               '  auto* blk = reinterpret_cast<float*>(totalShrMemPtr);\n'
               '  float * __restrict__ s0 = &blk[0];\n'
               + ''.join(f'  {s}\n' for s in stmts) + '}')
        races = []
        mem = kernel_eval.evaluate_wave(src, lanes, seed=7, globals_only=True,
                                        races=races)
        return mem, sorted({r.kind for r in races})
    seed = kernel_eval.Slot(7)
    one = copy.format(bits=32, elem='int', si='threadIdx.x',
                      di='31 - threadIdx.x')
    mem, races = run(one, '__builtin_amdgcn_s_wait_asynccnt(0);',
                     'm0[threadIdx.x] = s0[threadIdx.x];')
    assert races == []
    assert all(mem[('m0', t)] == seed.read('m1', 31 - t) for t in range(32))
    four = copy.format(bits=128, elem='tensorforge::VectorT<int, 4>',
                       si='4 * threadIdx.x', di='4 * threadIdx.x')
    mem, races = run(four, '__builtin_amdgcn_s_wait_asynccnt(0);',
                     'm0[threadIdx.x] = s0[4 * threadIdx.x + 3];')
    assert races == []
    assert all(mem[('m0', t)] == seed.read('m1', 4 * t + 3)
               for t in range(32))
    behind_barrier = run(copy.format(bits=32, elem='int', si='threadIdx.x',
                                     di='threadIdx.x'),
                         '__syncthreads();',
                         'm0[threadIdx.x] = s0[(threadIdx.x + 32) % 64];',
                         lanes=64)
    assert behind_barrier[1] == ['FLIGHT']
    waited = run(copy.format(bits=32, elem='int', si='threadIdx.x',
                             di='threadIdx.x'),
                 '__builtin_amdgcn_s_wait_asynccnt(0);', '__syncthreads();',
                 'm0[threadIdx.x] = s0[(threadIdx.x + 32) % 64];', lanes=64)
    assert waited[1] == []
    assert all(waited[0][('m0', t)] == seed.read('m1', (t + 32) % 64)
               for t in range(64))


# -- generated kernels ---------------------------------------------------------- #

@pytest.mark.parametrize('name', ['trip_prepared', 'k_unroll_tail'])
def test_a_preloaded_operator_goes_through_the_copy_where_asked(name):
    """At any lanes: the operator the block preloads into the LDS moves by
    the copy, waited for on `asynccnt` in front of the block barrier.  Off,
    it moves as before."""
    src = _generated(name, 'gfx1250', amd_async_copies=True)
    lines = src.splitlines()
    copies = [i for i, l in enumerate(lines) if 'async_to_lds_b' in l]
    wait = next(i for i, l in enumerate(lines)
                if '__builtin_amdgcn_s_wait_asynccnt(0);' in l)
    barrier = next(i for i in range(wait, len(lines))
                   if '__syncthreads();' in lines[i])
    assert copies and max(copies) < wait < barrier
    plain = _generated(name, 'gfx1250')
    assert 'async_to_lds' not in plain and 's_wait_asynccnt' not in plain


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_the_preload_compiles_to_the_copy_and_its_wait(arch, tmp_path):
    """Read off the instructions: the copies are the async ones, and the
    wait in front of the barrier is `asynccnt`'s at zero."""
    src = _generated('trip_prepared', arch, amd_async_copies=True)
    asm = _assemble('#include "tensorforge_device/hip.h"\n' + src, arch,
                    tmp_path)
    copies = [i for i, ins in enumerate(asm)
              if ins.startswith('global_load_async_to_lds_b')]
    barrier = next(i for i, ins in enumerate(asm)
                   if ins.startswith('s_barrier'))
    assert copies and max(copies) < barrier
    assert any(re.match(r's_wait_asynccnt (0x)?0$', ins)
               for ins in asm[max(copies):barrier])


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_the_compiler_keeps_the_copies_and_the_counts(arch, tmp_path):
    """Three copies and their waits, compiled: three copies, and each wait
    kept as written -- the compiler places none of these itself."""
    ctx, body = _body(arch=arch)
    _, lines = _emit(ctx, body)
    keep = [l for l in lines if not re.match(r'^float m\d+\[', l)]
    src = ('#include "tensorforge_device/hip.h"\n'
           '__global__ void __launch_bounds__(32) k(float *__restrict__ m0, '
           'float *__restrict__ m1) {\n  extern __shared__ float blk[];\n'
           + ''.join(f'  {l}\n' for l in keep) + '}\n')
    asm = _assemble(src, arch, tmp_path)
    assert sum(ins.startswith('global_load_async_to_lds_b32')
               for ins in asm) == 3
    counts = [int(m.group(1), 0) for m in
              (re.match(r's_wait_asynccnt ((0x)?[0-9a-f]+)$', ins)
               for ins in asm) if m]
    assert counts == [2, 1, 0]
