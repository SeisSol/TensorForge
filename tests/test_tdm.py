# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""gfx1250's tensor data mover (`pir.engines`, engine `tdm`), under
`Options.amd_tensor_copies`.

`tensor_load_to_lds` copies a tile of global memory into the LDS in one
instruction a wave, through a descriptor in scalar registers
(`Op.TENSOR_DESC`, `pir.tensors`), and counts on `tensorcnt`, which counts
these copies and nothing else.  The first wave of the lanes a tile is for
issues it; neither the compiler nor a block barrier waits for it, so each wait
is written out on that wave's counter, and the barrier behind the wait shows
the tile to the other waves.  Compile-only: no device here has the
instruction, so what runs is the reference interpreter, and the descriptor's
layout is checked against the one LLVM's lowering, Triton and
composable_kernel agree on.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import replace

import pytest

from tensorforge.backend import pir
from tensorforge.backend.pir import INDEX, IRBuilder, IRError, MemSpace, Op
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import Completion
from tensorforge.backend.pir.tensors import tensor_descriptors, verify_tensor
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options

from test_pir_engines import _Commented, _assemble, _generated

F32 = Datatype.F32
ON = {'amd_async_copies': True, 'amd_tensor_copies': True}


def _target(arch, backend='hip', **options):
    return Context(arch=arch, backend=backend, fp_type=F32,
                   options=Options(**options) if options else None)


# -- the engine and the target ------------------------------------------------ #

def test_the_engine_counts_its_tiles_on_tensorcnt():
    engine = engines.TRANSFER_ENGINES['tdm']
    assert engine.ops == (Op.COPY_ASYNC,)
    assert engine.completes is Completion.COUNTER
    assert engines.counter_of('tdm') == 'load'
    assert engines.waited_on('tdm') == 'tensorcnt'
    assert engines.one_to_one('tdm')


@pytest.mark.parametrize('arch,backend,has', [
    ('gfx1250', 'hip', True), ('gfx1251', 'hip', True),
    ('gfx942', 'hip', False), ('gfx950', 'hip', False),
    ('gfx1150', 'hip', False), ('sm_90', 'cuda', False),
    ('pvc', 'oneapi', False)])
def test_gfx125x_alone_has_the_tensor_data_mover(arch, backend, has):
    target = _target(arch, backend).target
    assert target.tensor_copies() is has
    if has:
        assert not target.one_wait_counter('tdm')
        # Its per-lane copy stays what it was.
        assert engines.copy_engine(target) == 'async_lds'


# -- the descriptor's layout ---------------------------------------------------- #

def _bits(words, at, width, value):
    """`value` into bits `at` .. `at + width` of the 32-bit `words`, counted
    from the first word's lowest bit."""
    for k in range(width):
        if value >> k & 1:
            words[(at + k) // 32] |= 1 << ((at + k) % 32)


def _reference(lds, glob, rows, cols, pitch, size_log2):
    """The two groups of a 2D tile, from the fields at the bit offsets the
    sources state -- LLVM's `AMDGPULowerDescriptor` (48, 80, 112, 128, 144,
    160, 208), Triton's tables and composable_kernel's bitfields: group 0
    its count (0), the LDS address (32), the global one (64, 57 bits) and
    its type 2 (126); group 1 the data size (16), the extents of a row and
    of the rows in global memory (48, 80), the tile's (112, 128), and the
    pitch (160, 48 bits)."""
    g0, g1 = [0] * 4, [0] * 8
    _bits(g0, 0, 2, 1)
    _bits(g0, 32, 32, lds)
    _bits(g0, 64, 57, glob)
    _bits(g0, 126, 2, 2)
    _bits(g1, 16, 2, size_log2)
    _bits(g1, 48, 32, cols)
    _bits(g1, 80, 32, rows)
    _bits(g1, 112, 16, cols)
    _bits(g1, 128, 16, rows)
    _bits(g1, 160, 48, pitch)
    return g0, g1


#: Tiles to hold the packing to: LLVM's test of a 2D tile (i32, 128 rows of
#: 64, a pitch of 64), a narrow one of doubles, and pitches and addresses
#: whose high halves are not zero.
_TILES = [
    (0x0, 0x7f0012345678, 128, 64, 64, 2),
    (0x1230, 0x01abcdef12345678, 9, 56, 56, 3),
    (0xfffc, 0x0100000000000004, 1, 65535, 65535, 2),
    (0x40, 0x10, 65535, 1, (1 << 40) + 7, 0),
]


def test_the_reference_is_llvm_s_test_of_a_2d_tile():
    """The words LLVM's AMDGPU dialect test expects of `globalSize [128,
    64] globalStride [64, 1] sharedSize [128, 64]` over i32."""
    _, g1 = _reference(0, 0, 128, 64, 64, 2)
    assert g1 == [2 << 16, 64 << 16, (64 >> 16) | (128 << 16),
                  (128 >> 16) | (64 << 16), 128, 64, 0, 0]


def test_the_header_packs_the_words_the_sources_agree_on(tmp_path):
    """`tensorDescriptorWords` as compiled for gfx1250: every word of both
    groups against the reference, in static assertions."""
    asserts = []
    for n, (lds, glob, rows, cols, pitch, size) in enumerate(_TILES):
        g0, g1 = _reference(lds, glob, rows, cols, pitch, size)
        asserts.append(f'constexpr auto w{n} = tensorforge::tensorDescriptorWords('
                       f'{lds}u, {glob}ull, {rows}u, {cols}u, {pitch}ull, '
                       f'{size}u);')
        for k, word in enumerate(g0):
            asserts.append(f'static_assert(w{n}.group0[{k}] == {word}u, '
                           f'"group 0 word {k} of tile {n}");')
        for k, word in enumerate(g1):
            asserts.append(f'static_assert(w{n}.group1[{k}] == {word}u, '
                           f'"group 1 word {k} of tile {n}");')
    src = '\n'.join(['__global__ void k() {}'] + asserts) + '\n'
    _assemble('#include "tensorforge_device/hip.h"\n' + src, 'gfx1250',
              tmp_path)


# -- the copy, its descriptor and its waits ------------------------------------ #

def _body(copies=3, arch='gfx1250', threads=64, shape=(4, 8), pitch=16,
          team='mult', extra=None):
    """`copies` tiles of `shape`, rows `pitch` apart in `m1`, each into its
    own part of `s0` and waited for in turn; `extra` is called between the
    copies and the waits, and what it returns is waited for behind them."""
    ctx = _target(arch)
    b = IRBuilder(fptype=F32, context=ctx)
    rows, cols = shape
    tile = rows * cols
    s0 = b.alloc(F32, (tile * copies,), MemSpace.SHARED, hint='s0',
                 extern='s0', arena='blk', offset=0)
    g = b.alloc(F32, (rows * pitch * copies,), MemSpace.GLOBAL, hint='g',
                extern='m1')
    out = b.alloc(F32, (tile * copies,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    toks = [b.copy_tensor(s0, g, shape=shape, pitch=pitch, threads=threads,
                          team=team, dst_index=(tile * k,),
                          src_index=(rows * pitch * k,))
            for k in range(copies)]
    later = extra(b, g) if extra is not None else None
    lane = b.thread_id('x')
    for k, tok in enumerate(toks):
        b.wait(tok)
        at = b.op('add', INDEX, lane, tile * k, hint='at')
        b.store(out, b.load(s0, at, hint='u'), at)
    if later is not None:
        b.wait(later)
    return ctx, b, b.finish()


def _emit(ctx, b, body):
    out, diag = pir.schedule_async(body)
    assert not diag, diag
    out = tensor_descriptors(out, b.scratch)
    sink = _Commented()
    Emitter(sink, ctx).run(out)
    return out, sink.lines


def test_a_tensor_copy_says_its_tile_and_is_issued_by_its_first_wave():
    _, _, body = _body(copies=1)
    copy = next(s for s in body if s.op == Op.COPY_ASYNC)
    assert copy.attr('engine') == 'tdm' and copy.attr('leader') is True
    assert copy.attr('shape') == (4, 8) and copy.attr('pitch') == 16
    assert copy.attr('threads') == 64 and copy.attr('team') == 'mult'
    assert copy.attr('elems') == 32 and copy.attr('width') == 128
    lead = next(s for s in body if s.target and s.target[0] is copy.predicate)
    assert lead.op == 'lt' and lead.args[1] == 32
    assert verify_tensor(copy) == []
    pir.verify(body)
    wait = next(s for s in body if s.op == Op.WAIT_COUNTER)
    assert (wait.attr('counter'), wait.attr('engine')) == ('tensorcnt', 'tdm')


def test_a_permuted_window_or_another_team_is_refused_by_the_builder():
    ctx = _target('gfx1250')
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (32,), MemSpace.SHARED, hint='s0', extern='s0',
                 arena='blk', offset=0)
    g = b.alloc(F32, (32,), MemSpace.GLOBAL, hint='g', extern='m1')
    with pytest.raises(IRError):
        b.copy_tensor(s0, g, shape=(4, 8), pitch=8, threads=64, team='wave')


def test_a_wait_counts_the_tiles_behind_it_and_nothing_else():
    """Its count is the tiles issued behind the one it waits for -- under
    their first wave's predicate, which issues all of them -- and not a load
    into registers, which `tensorcnt` does not count."""
    ctx, b, body = _body(extra=lambda b, g: b.load_async(g, 0))
    out, lines = _emit(ctx, b, body)
    waits = [s for s in out if s.op == Op.WAIT_COUNTER]
    assert [w.attr('count') for w in waits] == [2, 1, 0]
    assert waits[0].attr('prior_unified') == 3
    assert [l for l in lines if 's_wait_tensorcnt' in l] == [
        '__builtin_amdgcn_s_wait_tensorcnt(2);',
        '__builtin_amdgcn_s_wait_tensorcnt(1);',
        '__builtin_amdgcn_s_wait_tensorcnt(0);']


def test_a_copy_of_another_engine_is_not_counted():
    ctx, b, body = _body()
    copies = [s for s in body if s.op == Op.COPY_ASYNC]
    other = replace(copies[1], attrs=tuple(
        (k, 'async_lds' if k == 'engine' else v) for k, v in copies[1].attrs))
    body = tuple(other if s is copies[1] else s for s in body)
    out, _ = pir.schedule_async(body)
    first = next(s for s in out if s.op == Op.WAIT_COUNTER)
    assert first.attr('count') == 1


def test_the_descriptor_stands_ahead_of_its_copy_and_the_copy_names_it():
    ctx, b, body = _body(copies=2)
    out = tensor_descriptors(body, b.scratch)
    descs = [k for k, s in enumerate(out) if s.op == Op.TENSOR_DESC]
    assert len(descs) == 2
    for k in descs:
        desc, copy = out[k], out[k + 1]
        assert copy.op == Op.COPY_ASYNC and copy.attr('desc') is desc.target[0]
        assert desc.args == copy.args and desc.pure and desc.movable
        assert desc.predicate is None and not desc.accesses
        assert (desc.attr('shape'), desc.attr('pitch')) == ((4, 8), 16)
        assert isinstance(desc.target[0].type, engines.TensorDescType)
        assert verify_tensor(desc) == []
    pir.verify(out)
    assert tensor_descriptors(out, b.scratch) == out
    assert tensor_descriptors(body[:2], b.scratch) == body[:2]


def test_the_emitter_writes_the_descriptor_the_copy_and_the_wait():
    ctx, b, body = _body(copies=1)
    _, lines = _emit(ctx, b, body)
    text = '\n'.join(lines)
    assert re.search(r'^tensorforge::TensorDescriptor (\w+) = '
                     r'tensorforge::tensorDescriptor<float>\(&s0\[0\], '
                     r'&m1\[0\], 4, 8, 16\);$', text, re.M), text
    name = re.search(r'TensorDescriptor (\w+) =', text).group(1)
    at = lines.index(f'tensorforge::tensorLoad({name});')
    assert re.match(r'^if \(.*threadIdx\.x.*< 32\)* \{$', lines[at - 1])
    wait = lines.index('__builtin_amdgcn_s_wait_tensorcnt(0);')
    assert wait > at and lines[wait + 1] == engines.TENSOR_FENCE


def test_a_drain_in_a_body_of_tensor_copies_is_tensorcnt_at_zero():
    ctx = _target('gfx1250')
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s0', extern='s0',
                 arena='blk', offset=0)
    g = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='o', extern='m0')
    b.copy_tensor(s0, g, shape=(2, 32), pitch=32, threads=32)
    b.wait()
    lane = b.thread_id('x')
    b.store(out, b.load(s0, lane, hint='u'), lane)
    _, lines = _emit(ctx, b, b.finish())
    load = next(i for i, l in enumerate(lines) if 'tensorLoad' in l)
    read = next(i for i, l in enumerate(lines) if '= s0[' in l)
    assert [l for l in lines[load:read] if 'wait_tensorcnt' in l
            or 'signal_fence' in l] == ['__builtin_amdgcn_s_wait_tensorcnt(0);',
                                         engines.TENSOR_FENCE]


def _refused(body, ctx, b):
    out, _ = pir.schedule_async(body)
    out = tensor_descriptors(out, b.scratch)
    with pytest.raises(IRError, match='a tensor copy cannot be issued'):
        Emitter(_Commented(), ctx).run(out)


def _with(body, **attrs):
    copy = next(s for s in body if s.op == Op.COPY_ASYNC)
    changed = replace(copy, attrs=tuple(
        (k, attrs.get(k, v)) for k, v in copy.attrs))
    return tuple(changed if s is copy else s for s in body)


@pytest.mark.parametrize('change', [
    {'threads': 48}, {'threads': None}, {'team': 'wave'},
    {'shape': (4, 65536)}, {'shape': (0, 8)}, {'pitch': 7},
    {'width': 12 * 32}, {'leader': False}])
def test_a_copy_the_mover_cannot_issue_is_refused(change):
    """Lanes that are no whole number of waves, a tile past the descriptor's
    fields, a pitch shorter than a row, an element of a size it does not
    spell, a copy not issued by its first wave: refused, not lowered as
    loads and stores a lane of that wave would make over the whole tile."""
    ctx, b, body = _body(copies=1)
    _refused(_with(body, **change), ctx, b)


def test_a_copy_under_another_wave_s_predicate_or_a_lane_guard_is_refused():
    ctx, b, body = _body(copies=1)
    lead = next(s for s in body if s.op == 'lt')
    narrow = replace(lead, args=(lead.args[0], 16))
    _refused(tuple(narrow if s is lead else s for s in body), ctx, b)
    # Under a guard of the lanes: the first wave may not reach it whole.
    b2 = IRBuilder(fptype=F32, context=ctx)
    s0 = b2.alloc(F32, (32,), MemSpace.SHARED, hint='s0', extern='s0',
                  arena='blk', offset=0)
    g = b2.alloc(F32, (32,), MemSpace.GLOBAL, hint='g', extern='m1')
    guard = b2.op('lt', pir.BOOL, b2.thread_id('x'), 48, hint='g')
    with b2.if_(guard):
        tok = b2.copy_tensor(s0, g, shape=(4, 8), pitch=8, threads=64)
    b2.wait(tok)
    _refused(b2.finish(), ctx, b2)


def test_a_block_s_copy_needs_a_guard_the_block_shares():
    """A tile for the block's lanes under a guard of one multiplication's:
    the block's first wave might not reach it."""
    ctx = _target('gfx1250')
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (32,), MemSpace.SHARED, hint='s0', extern='s0',
                 arena='blk', offset=0)
    g = b.alloc(F32, (32,), MemSpace.GLOBAL, hint='g', extern='m1')
    mult = b.op('lt', pir.BOOL, b.thread_id('y'), 2, hint='m')
    with b.if_(mult):
        tok = b.copy_tensor(s0, g, shape=(4, 8), pitch=8, threads=64,
                            team='block')
    b.wait(tok)
    _refused(b.finish(), ctx, b)


def test_copies_for_the_lanes_of_two_teams_are_refused():
    """A multiplication's tile, then the block's, and a wait for the first:
    its count is the block's tile behind it, which the block's first wave
    issues -- and the first wave of another multiplication does not, so the
    wait would return there with its own tile in flight."""
    ctx = _target('gfx1250')
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s0', extern='s0',
                 arena='blk', offset=0)
    g = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    mine = b.copy_tensor(s0, g, shape=(4, 8), pitch=8, threads=32)
    block = b.copy_tensor(s0, g, shape=(4, 8), pitch=8, threads=64,
                          team='block', dst_index=(32,), src_index=(32,))
    b.wait(mine)
    b.wait(block)
    out, diag = pir.schedule_async(b.finish())
    assert not diag, diag
    assert [s.attr('count') for s in out if s.op == Op.WAIT_COUNTER] == [1, 0]
    out = tensor_descriptors(out, b.scratch)
    with pytest.raises(IRError, match="'block' \\(64\\) and of 'mult' "
                                      "\\(32\\), which different waves"):
        Emitter(_Commented(), ctx).run(out)


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1150'])
def test_a_target_without_the_mover_refuses_the_copy(arch):
    ctx, b, body = _body(copies=1, arch=arch)
    _refused(body, ctx, b)


def test_a_copy_without_a_descriptor_is_refused_where_it_is_written():
    ctx, b, body = _body(copies=1)
    out, _ = pir.schedule_async(body)
    with pytest.raises(IRError, match='names no descriptor'):
        Emitter(_Commented(), ctx).run(out)


def test_the_verifier_holds_a_copy_and_a_descriptor_to_their_tile():
    ctx, b, body = _body(copies=1)
    out = tensor_descriptors(body, b.scratch)
    desc = next(s for s in out if s.op == Op.TENSOR_DESC)
    copy = next(s for s in out if s.op == Op.COPY_ASYNC)
    assert verify_tensor(replace(desc, attrs=(('shape', (4,)),
                                              ('pitch', 16))))
    assert verify_tensor(replace(copy, predicate=None))
    assert verify_tensor(replace(copy, attrs=tuple(
        (k, 'nothing' if k == 'desc' else v) for k, v in copy.attrs)))
    with pytest.raises(IRError):
        pir.verify(tuple(replace(s, predicate=None) if s is copy else s
                         for s in out))


# -- the reference interpreter ------------------------------------------------- #

_SEED = 7

#: A tile of 4 rows of 8 elements, its rows 16 apart from `m1[3]` on, into
#: `s0`; the copy under its first wave's predicate, then the wait and the
#: block barrier, then every lane of two waves reads an element of it.
_ISSUE = ['tensorforge::TensorDescriptor td = '
          'tensorforge::tensorDescriptor<float>(&s0[0], &m1[3], 4, 8, 16);',
          'if (threadIdx.x < 32) {', 'tensorforge::tensorLoad(td);', '}']
_WAIT = ['__builtin_amdgcn_s_wait_tensorcnt(0);',
         '__atomic_signal_fence(__ATOMIC_SEQ_CST);']
_BARRIER = ['__syncthreads();']
_READ = ['m0[threadIdx.x] = s0[(threadIdx.x + 3) % 32];']


def _run(*parts, lanes=64):
    from tensorforge.reference import kernel_eval
    src = ('void k() {\n'
           '  auto* blk = reinterpret_cast<float*>(totalShrMemPtr);\n'
           '  float * __restrict__ s0 = &blk[0];\n'
           + ''.join(f'  {s}\n' for part in parts for s in part) + '}')
    races = []
    mem = kernel_eval.evaluate_wave(src, lanes, seed=_SEED, globals_only=True,
                                    races=races)
    return mem, sorted({r.kind for r in races})


def test_the_interpreter_copies_the_tile_once_and_lands_it_at_its_wait():
    from tensorforge.reference import kernel_eval
    seed = kernel_eval.Slot(_SEED)
    mem, races = _run(_ISSUE, _WAIT, _BARRIER, _READ)
    assert races == []
    for t in range(64):
        k = (t + 3) % 32
        assert mem[('m0', t)] == seed.read('m1', 3 + (k // 8) * 16 + k % 8)


@pytest.mark.parametrize('parts,kind', [
    ((_ISSUE, _BARRIER, _READ), 'FLIGHT'),
    ((_ISSUE, _BARRIER, _WAIT, _READ), 'RAW'),
    ((_ISSUE, _WAIT, _READ), 'RAW'),
    ((_ISSUE[:1] + _ISSUE[2:3], _WAIT, _BARRIER, _READ), 'FLIGHT'),
], ids=['no-wait', 'wait-behind-the-barrier', 'no-barrier', 'every-wave'])
def test_what_the_copy_needs_is_there_for_a_reason(parts, kind):
    """No wait: the tile is read in flight -- the barrier's fence waits for
    no `tensorcnt`.  The wait behind the barrier, or no barrier: the other
    wave reads what only the first one waited for.  Issued by every wave: a
    second copy over the first one in flight."""
    _, races = _run(*parts)
    assert kind in races, races


def test_the_first_wave_alone_needs_no_barrier():
    """Its own lanes read behind its own wait."""
    _, races = _run(_ISSUE, _WAIT, _READ, lanes=32)
    assert races == []


def test_a_wait_one_tile_laxer_reads_the_second_in_flight():
    second = [_ISSUE[0].replace('td =', 'tu =').replace('&s0[0], &m1[3]',
                                                       '&s0[32], &m1[67]'),
              'if (threadIdx.x < 32) {', 'tensorforge::tensorLoad(tu);', '}']
    reads = ['m0[threadIdx.x] = s0[32 + (threadIdx.x + 3) % 32];']
    lax = ['__builtin_amdgcn_s_wait_tensorcnt(1);']
    _, races = _run(_ISSUE, second, lax, _BARRIER, reads)
    assert 'FLIGHT' in races
    _, races = _run(_ISSUE, second, _WAIT, _BARRIER, reads)
    assert races == []


# -- generated kernels ---------------------------------------------------------- #

def _code(text):
    """The kernel without its comments, which name the options, and without
    its name, which is a hash of them."""
    return re.sub(r'kernel_\w+', 'kernel', re.sub(r'//[^\n]*', '', text))


def test_a_preloaded_operator_goes_through_one_tensor_copy_where_asked():
    """The operator the block preloads into the LDS moves as one tile by the
    block's first wave, waited for on `tensorcnt` in front of the block
    barrier.  Without the switch it moves by a copy a lane, as before, and
    without `amd_async_copies` by loads and stores."""
    src = _generated('trip_prepared', 'gfx1250', **ON)
    lines = src.splitlines()
    loads = [i for i, l in enumerate(lines) if 'tensorforge::tensorLoad(' in l]
    assert len(loads) == 1
    assert re.search(r'tensorDescriptor<float>\(&glb_m1\[0 \+ 0\], '
                     r'&ptr_glb_m1\[0 \+ 0\], 16, 56, 56\)', src), src
    assert re.search(r'if \(\(\(threadIdx\.x \+ \(tfMult \* 32\)\) < 32\)\)',
                     lines[loads[0] - 1])
    wait = lines.index('      __builtin_amdgcn_s_wait_tensorcnt(0);')
    assert wait > loads[0]
    assert lines[wait + 1].strip() == engines.TENSOR_FENCE
    assert lines[wait + 2].strip() == '__syncthreads();'
    assert 'async_to_lds' not in src
    per_lane = _generated('trip_prepared', 'gfx1250', amd_async_copies=True)
    assert 'tensorLoad' not in per_lane and 'async_to_lds' in per_lane
    plain = _generated('trip_prepared', 'gfx1250', amd_tensor_copies=True)
    assert _code(plain) == _code(_generated('trip_prepared', 'gfx1250'))


def _built(name, arch, **options):
    """The kernel of the corpus case `name` for HIP on `arch`, and the
    lanes and multiplications of its block (`kernel_eval.launch_geometry`)."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path
    from tensorforge.generators.generator import Generator
    from tensorforge.reference import kernel_eval
    path = next((Path(__file__).resolve().parent / 'cases').rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_tdm__' + name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    gen = Generator(mod.descr_list(), Context(
        arch=arch, backend='hip', fp_type=mod.DTYPE,
        options=Options(**{**getattr(mod, 'OPTIONS', {}), **options})))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), kernel_eval.launch_geometry(gen.get_launcher())


def test_a_kernel_computes_with_its_tile_what_it_does_with_copies_a_lane():
    """The whole kernel through the interpreter, every multiplication of a
    block over two elements: what the block computes with its operator
    copied as one tile by its first wave is what it computes with a copy a
    lane -- and nothing is read in flight, or behind a barrier missing."""
    from tensorforge.reference import kernel_eval
    src, geometry = _built('packed_bits', 'gfx1250', **ON)
    lanes, mults = geometry[:2]
    assert src.count('tensorforge::tensorLoad(') == 1
    plain, _ = _built('packed_bits', 'gfx1250', amd_async_copies=True)

    def run(text):
        races = []
        mem = kernel_eval.evaluate_wave(text, lanes, seed=11,
                                        globals_only=True, races=races,
                                        elements=2, mults=mults, block=True)
        return mem, sorted({r.kind for r in races})
    mem, races = run(src)
    ref, _ = run(plain)
    assert races == []
    keys = {k for k in set(mem) | set(ref) if str(k[0]).startswith('m')}
    assert keys and all(mem.get(k) == ref.get(k) for k in keys)


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1150'])
def test_the_switch_changes_nothing_where_there_is_no_mover(arch):
    assert (_code(_generated('trip_prepared', arch, **ON))
            == _code(_generated('trip_prepared', arch,
                                amd_async_copies=True)))


def _isa(src, arch, tmp_path):
    return _assemble('#include "tensorforge_device/hip.h"\n' + src, arch,
                     tmp_path)


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_the_preload_compiles_to_one_tensor_load_its_wait_and_the_barrier(
        arch, tmp_path):
    """Read off the instructions: one tensor load from scalar registers,
    behind a branch on the execution mask -- the first wave alone -- then
    `tensorcnt` at zero, then the barrier, and no LDS read in between."""
    asm = _isa(_generated('trip_prepared', arch, **ON), arch, tmp_path)
    loads = [i for i, ins in enumerate(asm)
             if ins.startswith('tensor_load_to_lds')]
    assert len(loads) == 1
    assert re.match(r'tensor_load_to_lds s\[\d+:\d+\], s\[\d+:\d+\]$',
                    asm[loads[0]])
    branch = max(i for i in range(loads[0])
                 if asm[i].startswith('s_cbranch_execz'))
    assert asm[branch - 1].startswith('v_cmpx_gt_u32')
    wait = next(i for i in range(loads[0], len(asm))
                if asm[i].startswith('s_wait_tensorcnt'))
    assert re.match(r's_wait_tensorcnt (0x)?0$', asm[wait])
    barrier = next(i for i in range(wait, len(asm))
                   if asm[i].startswith('s_barrier'))
    assert not any(ins.startswith(('ds_load', 'ds_read'))
                   for ins in asm[loads[0]:barrier])


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_the_compiler_keeps_the_copies_and_the_counts(arch, tmp_path):
    """Three tiles and their waits, compiled: three tensor loads, and each
    wait kept as written -- the compiler places none of these itself."""
    ctx, b, body = _body(arch=arch)
    _, lines = _emit(ctx, b, body)
    keep = [l for l in lines if not re.match(r'^float m\d+\[', l)]
    src = ('__global__ void __launch_bounds__(64) k(float *__restrict__ m0, '
           'float *__restrict__ m1) {\n  extern __shared__ float blk[];\n'
           + ''.join(f'  {l}\n' for l in keep) + '}\n')
    asm = _isa(src, arch, tmp_path)
    assert sum(ins.startswith('tensor_load_to_lds') for ins in asm) == 3
    counts = [int(m.group(1), 0) for m in
              (re.match(r's_wait_tensorcnt ((0x)?[0-9a-f]+)$', ins)
               for ins in asm) if m]
    assert counts == [2, 1, 0]


def test_a_read_of_the_tile_is_not_answered_by_an_earlier_one(tmp_path):
    """clang 22 declares the copy and its wait to touch no memory a program
    names: without the fences, a read of an LDS address behind a copy into
    it and its wait came out as the value read before the copy."""
    head = ('__global__ void __launch_bounds__(32) k(float *__restrict__ out, '
            'const float *__restrict__ in) {\n'
            '  extern __shared__ float s0[];\n'
            '  float a = s0[threadIdx.x];\n'
            '  tensorforge::TensorDescriptor td = '
            'tensorforge::tensorDescriptor<float>(&s0[0], &in[0], 1, 32, 32);\n'
            '  tensorforge::tensorLoad(td);\n'
            '  __builtin_amdgcn_s_wait_tensorcnt(0);\n')
    tail = '  out[threadIdx.x] = a + s0[threadIdx.x];\n}\n'
    asm = _isa(head + f'  {engines.TENSOR_FENCE}\n' + tail, 'gfx1250',
               tmp_path)
    wait = next(i for i, ins in enumerate(asm)
                if ins.startswith('s_wait_tensorcnt'))
    assert any(ins.startswith('ds_load') for ins in asm[wait:])


def test_the_host_shim_takes_the_helpers():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    src = _generated('trip_prepared', 'gfx1250', **ON)
    assert 'tensorforge::tensorLoad' in src
    result = syntax.check_source(src, shim=syntax.shim_for('hip'))
    assert result.ok, result.errors()


def test_clang_23_writes_the_copy_in_its_own_form(tmp_path):
    """clang 23 has no `_d2` form: one builtin of five groups, the first of
    unsigned words.  Compiled where a clang that knows gfx1250 is on PATH."""
    from tensorforge import toolchain
    clang = shutil.which('clang++')
    if clang is None:
        pytest.skip('no clang++')
    path = tmp_path / 'k.hip'
    path.write_text('#include "tensorforge_device/hip.h"\n'
                    + _generated('trip_prepared', 'gfx1250', **ON))
    run = subprocess.run([clang, '-x', 'hip', '--rocm-path=/opt/rocm',
                          '-nogpulib', '-O3', '-std=c++17',
                          '--offload-arch=gfx1250', '--cuda-device-only',
                          '-I', toolchain.include_dir(), '-S', '-o', '-',
                          str(path)], capture_output=True, text=True,
                         timeout=600)
    if run.returncode and re.search(r'unsupported|invalid offload arch|'
                                    r'cannot find ROCm|cannot find HIP runtime|'
                                    r'hip_runtime',
                                    run.stderr):
        pytest.skip('this clang++ does not compile HIP for gfx1250')
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.stdout.count('tensor_load_to_lds') == 1
