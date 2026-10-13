# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A window copied into shared memory by the tensor memory accelerator.

`Options.bulk_copies`: one `cp.async.bulk` for the whole window, issued by one
thread of the multiplication and completing on an mbarrier every thread waits
on (`tensorforge::bulk_copy`, `bulk_wait` in `cuda.h`), where `cp.async` has
every thread copy sixteen bytes and count groups.  No GPU here, so what is
held is what the generated code says and what the host oracle computes from
it: the product, and -- through `kernel_eval.Races` -- that every thread
waits for the copy before it reads, that the leader arms the barrier before
anyone waits on it, and that the next copy waits for the reads of the last.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backend.pir import walk
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (SIZE, BufferType, Effect, IRError,
                                          MemSpace, Op, Uniformity, XorSwizzle)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.options import registry
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

CASES = Path(__file__).resolve().parent / "cases"
F32 = Datatype.F32

#: Cases whose operands promise sixteen-byte aligned strides, which is what a
#: bulk copy needs of its source.
ALIGNED = ['aligned_operands', 'wide_cascade']


def _module(name):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_bulk__' + name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _build(name, arch='sm_90', **options):
    mod = _module(name)
    gen = Generator(mod.descr_list(),
                    Context(arch=arch, backend='cuda', fp_type=mod.DTYPE,
                            options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return mod, gen.get_kernel(), kernel_eval.launch_geometry(
        gen.get_launcher())


def _product(mod, mem, elements):
    """`D = A B` against numpy for each element, over the oracle's seed."""
    seed = kernel_eval.Slot(11)
    m, n, k = mod.M, mod.N, mod.K
    worst = 0.0
    for e in range(elements):
        a = np.array([[seed.read('m1', e * m * k + i + j * m)
                       for j in range(k)] for i in range(m)])
        b = np.array([[seed.read('m2', e * k * n + i + j * k)
                       for j in range(n)] for i in range(k)])
        got = np.array([[mem.get(('m0', e * m * n + i + j * m), 0.0) or 0.0
                         for j in range(n)] for i in range(m)])
        worst = max(worst, float(np.max(np.abs(a @ b - got))))
    return worst


# --------------------------------------------------------------------------- #
# the generated code
# --------------------------------------------------------------------------- #

def test_off_by_default():
    assert registry()['bulk_copies'].default is False
    _, src, _ = _build('aligned_operands')
    assert 'bulk_copy' not in src and '__pipeline_memcpy_async' in src


@pytest.mark.parametrize('name', ALIGNED)
def test_the_window_is_one_copy_its_threads_wait_for(name):
    """The whole window in one copy of its bytes, the cursor armed ahead of
    everything with a barrier behind it, and every wait one on the cursor:
    no group, no commit."""
    _, src, _ = _build(name, bulk_copies=True)
    lines = src.splitlines()
    copies = [l for l in lines if 'tensorforge::bulk_copy(' in l]
    assert copies and all(re.search(r', (\d+)\);$', l) for l in copies)
    assert all(int(re.search(r', (\d+)\);$', l).group(1)) % 16 == 0
               for l in copies)
    arm = next(i for i, l in enumerate(lines) if 'bulk_arm(' in l)
    assert arm < min(i for i, l in enumerate(lines) if 'bulk_copy(' in l)
    assert '__syncwarp' in lines[arm + 1] or '__syncthreads' in lines[arm + 1]
    # At the read, in the masked element's drain, and behind the loop.
    assert src.count('tensorforge::bulk_wait(') == 3
    assert '__pipeline_commit' not in src
    assert '__pipeline_wait_prior' not in src


def test_a_wait_for_the_copy_needs_no_barrier_behind_it():
    """Every thread waits for the copy itself: nothing stands between the
    wait and the first read of the window, where `cp.async` needs a barrier
    for the other threads' parts."""
    _, bulk, _ = _build('aligned_operands', bulk_copies=True)
    _, plain, _ = _build('aligned_operands')

    def barrier_behind(src, wait):
        lines = src.splitlines()
        at = next(i for i, l in enumerate(lines) if wait in l)
        read = next(i for i in range(at + 1, len(lines))
                    if re.search(r'=\s*s0\[', lines[i]))
        return any(re.search(r'__syncwarp|__syncthreads', l)
                   for l in lines[at + 1:read])
    assert barrier_behind(plain, '__pipeline_wait_prior')
    assert not barrier_behind(bulk, 'tensorforge::bulk_wait')


@pytest.mark.parametrize('name,arch,options', [
    ('aligned_operands', 'sm_86', {}),
    ('chain_five', 'sm_90', {}),
], ids=['no-tma', 'unaligned'])
def test_a_copy_that_cannot_be_one_keeps_cp_async(name, arch, options):
    """sm_86 has no tensor memory accelerator; `chain_five`'s operands promise
    no alignment beyond the element."""
    _, src, _ = _build(name, arch=arch, bulk_copies=True, **options)
    assert 'bulk_copy' not in src and 'bulk_arm' not in src
    assert '__pipeline_memcpy_async' in src


def test_a_bulk_copy_keeps_one_stage_where_two_are_asked_for():
    """A second stage would put the next copy in flight on the barrier the
    one before it has not completed."""
    _, src, _ = _build('aligned_operands', bulk_copies=True,
                       enable_multibuffer=True)
    copies = [l for l in src.splitlines() if 'tensorforge::bulk_copy(' in l]
    assert copies and all('&s0[0 + 0]' in l for l in copies)


# --------------------------------------------------------------------------- #
# the host oracle
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ALIGNED)
def test_the_bulk_copied_kernel_computes_the_product(name):
    """Three elements through one multiplication: the copy ahead of the loop,
    the ones at its tail, the waits and the drain behind it -- and no access
    the hardware would not have ordered."""
    mod, src, (lanes, _) = _build(name, bulk_copies=True)
    races = []
    mem = kernel_eval.evaluate_wave(src, lanes, seed=11, globals_only=True,
                                    races=races, elements=3)
    assert _product(mod, mem, 3) < 1e-6
    assert not races, races


def _without(src, pattern, which):
    lines = src.splitlines()
    hits = [i for i, l in enumerate(lines) if re.search(pattern, l)]
    drop = hits[which]
    return '\n'.join(lines[:drop] + lines[drop + 1:])


@pytest.mark.parametrize('pattern,which,kind', [
    (r'__syncwarp', 0, 'RAW'),
    (r'__syncwarp', 1, 'WAR'),
    (r'bulk_wait', 0, 'FLIGHT'),
], ids=['arming', 'reads-before-the-next-copy', 'wait'])
def test_what_the_bulk_copy_needs_is_there_for_a_reason(pattern, which, kind):
    """Taken out, each is the race it is there to prevent: the barrier behind
    the arming, a thread waiting on a barrier the leader has not armed; the
    one at the end of an element, the next copy over a window still read;
    the wait, a read of the window in flight."""
    _, src, (lanes, _) = _build('aligned_operands', bulk_copies=True)
    races = []
    kernel_eval.evaluate_wave(_without(src, pattern, which), lanes, seed=3,
                              races=races, elements=3)
    assert kind in {r.kind for r in races}, races


def test_the_oracle_copies_through_the_leader_alone():
    """Only the thread the cursor names issues; the others note the copy."""
    mem = kernel_eval.Slot(3)
    for lane, expect in ((1, None), (0, 2.5)):
        env = {'dst': kernel_eval.Ptr(mem, f'dst{lane}'),
               'src': kernel_eval.Ptr(mem, 'src'),
               'bar': kernel_eval.Ptr(mem, f'bar{lane}'),
               'threadIdx': type('T', (), {'x': lane})()}
        it = kernel_eval.Interp(mem, env)
        it.widths['dst'] = 4
        for i in range(4):
            mem.write('src', i, float(i) + 0.5)
        it.assign('tensorforge::BulkCursor c = tensorforge::bulk_arm('
                  'bar, threadIdx.x == 0)')
        it.assign('tensorforge::bulk_copy(c, &dst[0], &src[0], 16)')
        it.assign('tensorforge::bulk_wait(c)')
        got = mem.data.get((f'dst{lane}', 2))
        assert got == expect


# --------------------------------------------------------------------------- #
# the passes
# --------------------------------------------------------------------------- #

def _channel_body():
    """A body with a window, its bulk copy and the wait for it."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    src = b.decl_expr(
        'const float *const __restrict__ glb_m0', '&m0[{0} * 64]',
        BufferType(F32, (64,), MemSpace.GLOBAL, readonly=True), 'm0',
        args=(k,), kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
        extern='glb_m0')
    s0 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s0', extern='s0')
    cursor = b.bulk_channel('s0', 'threadIdx.x == 0', threads=32)
    tok = b.copy_bulk(s0, src, cursor=cursor, dst_index=(0,), src_index=(0,),
                      elems=64)
    b.wait(tok)
    out = b.decl_expr(
        'float *const __restrict__ glb_m9', '&m9[{0} * 64]',
        BufferType(F32, (64,), MemSpace.GLOBAL), 'm9', args=(k,),
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m9',
        extern='glb_m9')
    b.store(out, b.load(s0, b.thread_id('x'), hint='u'), b.thread_id('x'))
    return b, cursor, b.finish()


def test_a_channel_is_set_up_once_ahead_of_everything():
    b, cursor, body = _channel_body()
    assert b.bulk_channel('s0', 'threadIdx.x == 0') == cursor
    alloc, arm, barrier = body[:3]
    assert alloc.op == Op.ALLOC and alloc.attr('lifetime') == 'body'
    assert arm.op == Op.RAWSTMT and 'bulk_arm' in arm.text
    assert barrier.op == Op.BARRIER


def test_a_body_without_an_arena_has_no_channel():
    assert IRBuilder(fptype=F32).bulk_channel('s0', 'true') is None


def test_a_permuted_window_takes_no_bulk_copy():
    b = IRBuilder(fptype=F32, arena='shrMem')
    s0 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s0', extern='s0',
                 swizzle=XorSwizzle(8))
    with pytest.raises(IRError, match='permuted'):
        b.copy_bulk(s0, 'm0', cursor='c', dst_index=(0,), src_index=(0,),
                    elems=64)


def test_the_barrier_shares_bytes_with_nothing():
    """Armed once and counting phases from then on: occupied for the whole
    body, whatever its accesses say."""
    _, _, body = _channel_body()
    placed, layout = allocate(body, arenas={'shrMem': MULT})
    offsets = {s.attr('extern'): s.attr('offset') for s, _ in walk(placed)
               if s.op == Op.ALLOC and s.attr('offset') is not None}
    assert offsets['s0_barrier'] != offsets['s0']
    assert layout.per_mult >= 64 + 2


def test_a_bulk_copy_joins_no_group():
    _, _, body = _channel_body()
    out, diag = schedule_async(body)
    assert not diag
    assert not any(s.op == Op.COMMIT_ASYNC for s, _ in walk(out))
    wait = next(s for s, _ in walk(out) if s.op == Op.WAIT)
    assert wait.attr('counter') == 'bulk'
    assert wait.attr('bulk') == 's0_bulk'


def test_a_bulk_copy_is_a_pipeline_of_its_own():
    """Beside a copy into shared memory by `cp.async`: neither counts towards
    how far the other is issued ahead (`transfers.memory_path`)."""
    from tensorforge.backend.pir.transfers import memory_path, transfers
    _, _, body = _channel_body()
    b = IRBuilder(fptype=F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    src = b.decl_expr(
        'const float *const __restrict__ glb_m1', '&m1[{0} * 64]',
        BufferType(F32, (64,), MemSpace.GLOBAL, readonly=True), 'm1',
        args=(k,), kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
        extern='glb_m1')
    s1 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s1', extern='s1')
    lane = b.thread_id('x')
    b.copy_async(s1, src, dst_index=(lane,), src_index=(lane,))
    bulk = transfers(body)
    copy = transfers(b.finish())
    assert len(bulk) == 1 and len(copy) == 1
    assert memory_path(bulk[0]) != memory_path(copy[0])
