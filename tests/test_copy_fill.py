# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Asynchronous copies that fill, and copies under a predicate of their own.

A copy that fills (`IRBuilder.copy_async`'s `valid`, `Stmt.copy_valid`) is
issued by every lane it is for, moves its bytes where its condition holds,
and where it does not reads nothing and writes its engine's fill -- zeros,
`cp.async`'s source size of zero (`engines.TransferEngine.fill`).  So the
destination is written whole by the one transfer, and to the passes the
copy is none under a predicate: `pir.wrap` takes it a trip ahead as any
other.  A copy under a predicate of its own is taken too now, its predicate
held together with the element's flag where the copy follows the element's
pointer (`_Rewrite._guarded`).

What is checked here: the statement and the verifier, the engines that
fill, the text on CUDA and the loads and stores where the body is
synchronous, the host oracle on the four-argument zero-fill of the pipeline
primitives and on the copy that fills, `pir.wrap` on both kinds of copy,
the SASS of a build for sm_80, and on the local GPU the copies against what
the PTX statement says they write.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.backend.pir import engines, verify, walk
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, Access, Effect,
                                          IRError, MemSpace, Op,
                                          Participants, Uniformity)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.wrap import wrap_loads
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval
from test_pir_wrap import DST, PTRS, SRC, _binding, _defs, _loop, \
    _ops_before_loop

F32 = Datatype.F32
INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'


def _ctx(arch='sm_86', backend='cuda'):
    return Context(arch=arch, backend=backend, fp_type=F32)


class _Sink:
    def __init__(self):
        self.lines = []

    def __call__(self, text, *args, **kwargs):
        self.lines.append(text)

    def __getattr__(self, _):
        return lambda *a, **k: None


def _emit(body, ctx):
    sink = _Sink()
    Emitter(sink, ctx).run(body)
    return sink.lines


def _copies(body):
    return [s for s, _ in walk(body) if s.op is Op.COPY_ASYNC]


# -- the statement ------------------------------------------------------------ #

def _one_copy(arch='sm_86', backend='cuda', elems=4, zfill=0, fill=True):
    """One lane's copy of `elems` floats, which fills where the lane is
    below 13: (context, builder, the buffers, the copy's token)."""
    ctx = _ctx(arch, backend)
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    s = b.alloc(F32, (256,), MemSpace.SHARED, hint='s', extern='s')
    g = b.alloc(F32, (256,), MemSpace.GLOBAL, hint='g', extern='m1')
    lane = b.thread_id('x')
    at = b.op('mul', INDEX, lane, elems, hint='at')
    ok = b.op('lt', BOOL, lane, 13, hint='ok') if fill else None
    tok = b.copy_async(s, g, dst_index=(at,), src_index=(at,), elems=elems,
                       zfill=zfill, valid=ok)
    return ctx, b, (s, g, at, ok), tok


def test_a_copy_that_fills_ends_in_its_condition():
    """The engine's fill, said on the copy, and the condition its last
    operand -- which the index accessors leave out, and which the copy reads
    as it reads every operand."""
    ctx, b, (s, g, at, ok), tok = _one_copy()
    b.wait(tok)
    body = b.finish()
    copy, = _copies(body)
    assert copy.attr('fill') == 'zero' and copy.copy_valid is ok
    assert copy.copy_dst_index == (at,) and copy.copy_src_index == (at,)
    assert copy.args[-1] is ok and ok in copy.operands()
    assert copy.predicate is None
    verify(body)
    plain = _copies(_one_copy(fill=False)[1].finish())[0]
    assert plain.attr('fill') is None and plain.copy_valid is None
    assert plain.copy_src_index == (plain.args[-1],)


def test_the_engines_that_fill():
    """cp.async fills with zeros; the engines that read every byte they
    move do not."""
    fills = {name: e.fill for name, e in engines.TRANSFER_ENGINES.items()}
    assert fills.pop('cp_async') == 'zero'
    assert set(fills.values()) == {None}
    assert engines.fill_of('cp_async') == 'zero'
    assert engines.fill_of('lds_dma') is None and engines.fill_of(None) is None


@pytest.mark.parametrize('arch', ['gfx942', 'gfx90a', 'gfx1250'])
def test_an_engine_that_reads_every_byte_is_refused_a_fill(arch):
    """AMD's copies into the LDS have no source size: refused where the
    copy is built, rather than reading what the fill stands for."""
    with pytest.raises(IRError, match='reads every byte it moves'):
        _one_copy(arch, 'hip')


def test_the_verifier_holds_a_copy_that_fills():
    import dataclasses
    ctx, b, (s, g, at, ok), tok = _one_copy()
    copy, = _copies(b.finish())
    assert engines.fill_refusal(copy) is None
    wrong = dataclasses.replace(copy, attrs=tuple(
        (k, 'nan' if k == 'fill' else v) for k, v in copy.attrs))
    assert 'where cp_async writes' in engines.fill_refusal(wrong)
    unbool = dataclasses.replace(copy, args=copy.args[:-1] + (at,))
    with pytest.raises(IRError, match='no boolean value'):
        verify((unbool,))


# -- the text ----------------------------------------------------------------- #

@pytest.mark.parametrize('elems,zfill,read', [
    (4, 0, 16), (2, 0, 8), (1, 0, 4), (4, 1, 12)])
def test_the_text_on_cuda_is_one_copy_of_a_source_size(elems, zfill, read):
    """One copy with the source size a value: all of what it reads where the
    condition holds, zero where not (`tensorforge::copyAsyncFill`)."""
    ctx, b, (s, g, at, ok), tok = _one_copy(elems=elems, zfill=zfill)
    b.wait(tok)
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    lines = _emit(body, ctx)
    want = (f'tensorforge::copyAsyncFill<{4 * elems}>(&s[v3_at], '
            f'&m1[v3_at], (threadIdx.x < 13) ? {read}u : 0u);')
    assert want in lines


def test_the_copies_that_do_not_fill_are_spelled_as_they_were():
    """A copy without a condition, and one with trailing zeros alone, keep
    the pipeline primitive's three- and four-argument forms."""
    ctx, b, (s, g, at, _), tok = _one_copy(fill=False)
    tok2 = b.copy_async(s, g, dst_index=(at,), src_index=(at,), elems=4,
                        zfill=1)
    b.wait(tok, tok2)
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    lines = _emit(body, ctx)
    assert '__pipeline_memcpy_async(&s[v3_at], &m1[v3_at], 16);' in lines
    assert '__pipeline_memcpy_async(&s[v3_at], &m1[v3_at], 16, 4);' in lines


@pytest.mark.parametrize('elems', [1, 4])
def test_a_synchronous_body_reads_where_it_holds_and_writes_zeros(elems):
    """Where the body is synchronous -- sm_75 has no asynchronous copy --
    the copy is a load and a store: the source read under the condition,
    whose other arm reads nothing, and zero where it fails."""
    ctx, b, (s, g, at, ok), tok = _one_copy(elems=elems)
    b.wait(tok)
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    lines = _emit(body, _ctx('sm_75'))
    if elems == 1:
        assert ('s[v3_at] = (threadIdx.x < 13) ? m1[v3_at] : 0.0f;'
                in lines), lines
    else:
        assert any(re.fullmatch(r'for \(int (c\d+) = 0; \1 < 4; \+\+\1\) '
                                r'\{ s\[\(v3_at\) \+ \1\] = \(threadIdx\.x < 13\) \? '
                                r'm1\[\(v3_at\) \+ \1\] : 0\.0f; \}', line)
                   for line in lines), lines


def test_the_lexic_of_amd_spells_no_copy_that_fills():
    lex = _ctx('gfx942', 'hip').target.lexic
    assert lex.copy_async('&d[0]', '&s[0]', 4, valid='ok') is None
    assert lex.copy_async('&d[0]', '&s[0]', 4) is not None


# -- the oracle --------------------------------------------------------------- #

def _source(lines, decls=''):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    return ('void k() {\n'
            '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + decls + ''.join(f'  {line}\n' for line in keep) + '}')


def _fill_body(elems, zfill, arch='sm_86', read_early=False):
    """32 lanes, each copying `elems` floats of `m1` into `s` -- the lanes
    from 13 on with their source index past `m1`'s end, where the copy
    reads nothing -- and `s` written back to `m0`."""
    ctx = _ctx(arch)
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    s = b.alloc(F32, (32 * elems,), MemSpace.SHARED, hint='s', extern='s')
    g = b.alloc(F32, (13 * elems,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (32 * elems,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    at = b.op('mul', INDEX, lane, elems, hint='at')
    ok = b.op('lt', BOOL, lane, 13, hint='ok')
    far = b.op('add', INDEX, at, 1 << 20, hint='far')
    src = b.op('select', INDEX, ok, at, far, hint='src')
    tok = b.copy_async(s, g, dst_index=(at,), src_index=(src,), elems=elems,
                       zfill=zfill, valid=ok)
    if read_early:
        b.store(out, b.load(s, at, hint='early'), at)
    b.wait(tok)
    b.barrier(Participants.BLOCK)
    for e in range(elems):
        k = b.op('add', INDEX, at, e, hint='k')
        b.store(out, b.load(s, k, hint='v'), k)
    body, diag = schedule_async(b.finish())
    assert not diag, diag
    body, layout = allocate(body, arenas={'shrMem': MULT})
    verify(body)
    return ctx, body, layout.per_mult


def _filled(elems, zfill, inputs):
    out = []
    for lane in range(32):
        for e in range(elems):
            read = lane < 13 and e < elems - zfill
            out.append(inputs[lane * elems + e] if read else 0.0)
    return out


@pytest.mark.parametrize('elems,zfill', [(4, 0), (2, 0), (1, 0), (4, 1),
                                         (4, 3)])
def test_the_oracle_reads_where_the_copy_holds_and_zeros_elsewhere(elems,
                                                                   zfill):
    """The copy that fills on the host oracle: each lane below 13 its
    elements, less the zero-filled tail; the others zeros, their source --
    far past the operand -- read nowhere; and no access races."""
    ctx, body, _ = _fill_body(elems, zfill)
    src = _source(_emit(body, ctx))
    races = []
    seed = 7
    mem = kernel_eval.evaluate_wave(src, 32, seed=seed, globals_only=True,
                                    races=races)
    slot = kernel_eval.Slot(seed)
    inputs = [slot.read('m1', i) for i in range(32 * elems)]
    got = [mem[('m0', i)] for i in range(32 * elems)]
    assert got == _filled(elems, zfill, inputs)
    assert not races, [str(r) for r in races[:4]]


def test_the_oracle_takes_the_four_argument_form_of_the_pipeline():
    """`__pipeline_memcpy_async(dst, src, N, Z)`: `N - Z` bytes read, the
    last `Z` zero -- the form a ragged staging run ends in, which the oracle
    refused before."""
    src = ('void k() {\n'
           '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
           '  float * s = &shrMem[0];\n'
           '  __pipeline_memcpy_async(&s[4 * threadIdx.x], '
           '&m1[4 * threadIdx.x], 16, 4);\n'
           '  __pipeline_commit();\n'
           '  __pipeline_wait_prior(0);\n'
           + ''.join(f'  m0[4 * threadIdx.x + {i}] = '
                     f's[4 * threadIdx.x + {i}];\n' for i in range(4))
           + '}')
    mem = kernel_eval.evaluate_wave(src, 32, seed=3, globals_only=True)
    slot = kernel_eval.Slot(3)
    got = [mem[('m0', i)] for i in range(128)]
    want = [slot.read('m1', i) if i % 4 < 3 else 0.0 for i in range(128)]
    assert got == want


def test_the_oracle_holds_a_fill_in_flight_until_the_wait():
    """The zeros land with the copy: read ahead of its wait, they race."""
    ctx, body, _ = _fill_body(4, 0, read_early=True)
    races = []
    kernel_eval.evaluate_wave(_source(_emit(body, ctx)), 32, seed=7,
                              globals_only=True, races=races)
    assert any(r.kind == 'FLIGHT' for r in races)


# -- pir.wrap ----------------------------------------------------------------- #

def _section(*, valid=False, held=False, flags=False, element_pointer=False):
    """A batch loop over elements whose body copies its element's window
    into a shared buffer, waits, and reads it back -- the copy one that
    fills (`valid`), or one under a predicate of its own (`held`), the lane
    below 13 either way.  (`test_pir_wrap.Section`'s shared form.)"""
    b = IRBuilder(fptype=F32, arena='shrMem')
    count = b.extern_value('numElements0', SIZE, hint='count')
    start = b.extern_value('start', SIZE, uniform=Uniformity.MULT,
                           hint='start')
    inside = b.op('lt', BOOL, start, count, hint='inside')
    first = b.op('select', SIZE, inside, start, 0, hint='batchId1',
                 escapes=True)
    window = b.alloc(F32, (64,), MemSpace.SHARED, hint='s', extern='s0')
    with b.for_('start', count, 'stride', hint='batchId0', index_type=SIZE,
                peel_index=first, uniform=Uniformity.MULT,
                flag_word='flags0[{0}]' if flags else None) as f:
        k = f.induction
        ahead = b.op('add', SIZE, k, 'stride', hint='ahead1')
        fits = b.op('lt', BOOL, ahead, count, hint='inbatch1')
        f._next_index = b.op('select', SIZE, fits, ahead, k,
                             hint='batchId1', escapes=True)
        src = _binding(b, k, base=PTRS if element_pointer else SRC,
                       element_pointer=element_pointer)
        out = _binding(b, k, base=DST, name='glb_m1', writable=True)
        if flags:
            allowed = b.decl_expr(
                'const bool allowed', 'static_cast<bool>(flags0[{0}])',
                BOOL, None, args=(k,), hint='allowed', extern='allowed')
            guard = b.if_(allowed, attrs=(('guard', 'element'),))
        else:
            from contextlib import nullcontext
            guard = nullcontext()
        with guard:
            b.mark('defines', window)
            lane = b.thread_id('x')
            lanes = b.op('lt', BOOL, lane, 13, hint='lanes')
            tok = b.copy_async(window, src, dst_index=(lane,),
                               src_index=(lane,),
                               valid=lanes if valid else None,
                               predicate=lanes if held else None)
            b.store(out, 0.0, 1)
            b.wait(tok)
            b.store(out, b.load(window, lane, hint='u'), lane)
        b.barrier(Participants.MULT, threads=32)
    body = b.finish()
    verify(body)
    report = []
    out = wrap_loads(body, b.scratch, distance=1, stages=1, report=report)
    verify(out)
    return out, report


@pytest.mark.parametrize('kind', ['valid', 'held'])
def test_wrap_takes_the_copy_a_trip_ahead(kind):
    """Both copies move: the peel issues the first element's ahead of the
    loop and the tail the next one's behind the last read, each carrying
    its condition -- the one that fills as its last operand, the other as
    its predicate -- and the schedule counts them as any copy."""
    out, report = _section(**{kind: True})
    assert report == ['+ s0 [shr]'], report
    peel = [s for s in _ops_before_loop(out) if s.op is Op.COPY_ASYNC]
    tail = [s for s in _loop(out).regions[0].body if s.op is Op.COPY_ASYNC]
    assert len(peel) == len(tail) == 1
    defs = _defs(out)
    for copy in peel + tail:
        held = copy.copy_valid if kind == 'valid' else copy.predicate
        assert defs[held.id].op == 'lt'
        assert (copy.attr('fill') == 'zero') is (kind == 'valid')
        assert (copy.predicate is None) is (kind == 'valid')
    scheduled, diag = schedule_async(out)
    assert not diag, diag


def test_wrap_holds_a_copys_own_predicate_with_the_elements_flag():
    """A copy that follows the element's pointer takes the element's flag
    as its predicate, and one with a predicate of its own takes both: their
    conjunction, right ahead of it."""
    out, report = _section(held=True, flags=True, element_pointer=True)
    assert report == ['+ s0 [shr]'], report
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    defs = _defs(out)
    for stmts, flag in ((region[guard + 1:], 'allowed_next'),
                        (_ops_before_loop(out), 'allowed_peel')):
        copies = [i for i, s in enumerate(stmts) if s.op is Op.COPY_ASYNC]
        assert len(copies) == 1
        copy = stmts[copies[0]]
        both = stmts[copies[0] - 1]
        assert both.op == 'and' and both.target == (copy.predicate,)
        flag_value, lanes = both.args
        assert defs[flag_value.id].attr('extern') == flag
        assert defs[lanes.id].op == 'lt'


def test_wrap_takes_a_fill_with_the_elements_flag_as_its_predicate():
    """A copy that fills is no predicated one: the flag is its predicate,
    and its condition stays its last operand."""
    out, report = _section(valid=True, flags=True, element_pointer=True)
    assert report == ['+ s0 [shr]'], report
    region = _loop(out).regions[0].body
    defs = _defs(out)
    tail = [s for s in region if s.op is Op.COPY_ASYNC]
    assert len(tail) == 1
    assert defs[tail[0].predicate.id].attr('extern') == 'allowed_next'
    assert defs[tail[0].copy_valid.id].op == 'lt'


# -- the build ---------------------------------------------------------------- #

def _cuda_source(lines, arena, elems):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    n_in, n_out = 13 * elems, 32 * elems
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda.h"
#include <cuda_pipeline.h>

__global__ void __launch_bounds__(32) fill_case(float *__restrict__ m0,
    const float *__restrict__ m1) {{
  extern __shared__ __align__(16) float shrMem[];
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main(int argc, char **argv) {{
  std::vector<float> h({n_in}), d({n_out});
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 4, {n_in}, in) != {n_in}) return 2;
  std::fclose(in);
  float *m0, *m1;
  CHECK(cudaMalloc(&m0, {n_out} * 4));
  CHECK(cudaMalloc(&m1, {n_in} * 4));
  CHECK(cudaMemcpy(m1, h.data(), {n_in} * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(m0, 0xff, {n_out} * 4));
  fill_case<<<1, 32, {arena * 4}>>>(m0, m1);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d.data(), m0, {n_out} * 4, cudaMemcpyDeviceToHost));
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 4, {n_out}, out) != {n_out}) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


def _build(elems, zfill, tmp_path, arch, product='cubin'):
    ctx, body, arena = _fill_body(elems, zfill)
    src = tmp_path / 'fill.cu'
    src.write_text(_cuda_source(_emit(body, ctx), arena, elems))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / ('k.cubin' if product == 'cubin' else 'fill')
    cmd = [_nvcc(), *nvcc.language_flags(), *nvcc.target_flags(arch),
           '-I', str(INCLUDE), '-o', str(out), str(src)]
    if product == 'cubin':
        cmd.insert(1, '-cubin')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


@pytest.mark.parametrize('elems', [4, 1])
def test_compiled_for_sm80_a_copy_that_fills_is_one_ldgsts(elems, tmp_path):
    """One `LDGSTS` a copy, the source ignored under a predicate, and no
    branch around it: the source size a register (`copyAsyncFill`)."""
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(_build(elems, 0, tmp_path,
                                                     'sm_80'))],
                          capture_output=True, text=True,
                          timeout=600).stdout
    copies = re.findall(r'\bLDGSTS\.[^\n]*', sass)
    assert len(copies) == 1, copies
    assert re.search(r'\],\s*!?P\d\s*;', copies[0]), copies[0]
    assert 'BPT.TRAP' not in sass


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


@pytest.mark.parametrize('elems,zfill', [(4, 0), (2, 0), (1, 0), (4, 1),
                                         (4, 3)])
def test_gpu_reads_where_the_copy_holds_and_zeros_elsewhere(elems, zfill,
                                                            tmp_path):
    """On the local GPU (sm_80 or newer): each lane below 13 its elements,
    less the zero-filled tail, and zeros for the others, whose source
    addresses lie a million elements past the operand -- read nowhere, or
    the run would fault."""
    arch = _gpu()
    if arch is None or int(arch[3:]) < 80:
        pytest.skip('no GPU of sm_80 or newer')
    exe = _build(elems, zfill, tmp_path, arch, product='binary')
    values = [float(i + 1) for i in range(13 * elems)]
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{len(values)}f',
                                                  *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    got = list(struct.unpack(f'<{32 * elems}f',
                             (tmp_path / 'out.bin').read_bytes()))
    assert got == _filled(elems, zfill, values + [0.0] * (19 * elems))
