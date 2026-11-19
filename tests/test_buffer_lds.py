# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The LDS form of the buffer engine (`pir.buffers`, `form='buffer'`): an
LDS-DMA copy (`engines`, `lds_dma`, gfx90a, gfx94x and gfx950) whose source
the wave holds alike reads through that source's buffer resource,
`buffer_load_dword ... lds` (`tensorforge::bufferLoadLds`), under
`buffer_transfers` and `amd_async_copies`.

It stays the LDS-DMA: one base a wave, lane t's four bytes t places past
it, its waits written on `vmcnt`.  A read past the resource gives zero, and
the zero is written; so a guard stays the branch it is -- a lane held off by
its offset would write zero into its place in the LDS, past the window's
end -- and a predicated copy keeps the global form.  No device here has the
instruction (the local gfx1150 has no LDS-DMA): what runs is the compiler,
its instructions, and the reference interpreter.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace

import pytest

from tensorforge.backend import pir
from tensorforge.backend.pir import INDEX, IRBuilder, MemSpace, Op
from tensorforge.backend.pir.buffers import buffer_base, verify_buffer
from tensorforge.backend.pir.core import walk_stmts
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import ResourceType
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options

from test_pir_engines import _Commented, _dma_kernel, _generated

F32 = Datatype.F32
ON = {'amd_async_copies': True, 'buffer_transfers': True,
      'lanes_per_mult': 64}

#: The builder of the body `_dma_body` built last, whose values the pass
#: numbers its resources past.
_BUILDER = []


def _dma_body(copies=2, threads=64, dst=None):
    """`copies` LDS-DMA copies of one wave each from `m1`, the lane past 64
    elements a copy, each waited for and read back reversed; `dst` replaces
    the destination of the first (`test_pir_engines._dma_body`)."""
    ctx = Context(arch='gfx942', backend='hip', fp_type=F32)
    b = IRBuilder(fptype=F32, context=ctx)
    s0 = b.alloc(F32, (64 * copies,), MemSpace.SHARED, hint='s0',
                 extern='s0', arena='blk', offset=0)
    g = b.alloc(F32, (64 * copies,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (64 * copies,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    ats = [lane] + [b.op('add', INDEX, lane, 64 * k, hint='at')
                    for k in range(1, copies)]
    toks = [b.copy_async(s0, g, dst_index=(dst(b, lane) if k == 0 and dst
                                            else at,),
                         src_index=(at,), threads=threads)
            for k, at in enumerate(ats)]
    for k, (tok, at) in enumerate(zip(toks, ats)):
        b.wait(tok)
        back = b.op('sub', INDEX, 64 * k + 63, lane, hint='bk')
        b.store(out, b.load(s0, back, hint='u'), at)
    _BUILDER[:] = [b]
    return ctx, b.finish()


def _buffered(body, threads=64, wave=64):
    return pir.buffer_transfers(body, _BUILDER[0].scratch, threads=threads,
                                wave=wave)


def _copies(body):
    return [s for s in walk_stmts(body) if s.op == Op.COPY_ASYNC]


# -- what the pass takes --------------------------------------------------------- #

def test_an_lds_dma_copy_of_a_wave_s_base_reads_through_its_resource():
    _, body = _dma_body(copies=2)
    out = _buffered(body)
    copies = _copies(out)
    assert len(copies) == 2
    # One over the copies' source, `m1`, and one over `m0`, which the
    # stores of the body go through.
    resources = {s.args[0].id: s for s in out if s.op == Op.RESOURCE}
    source = copies[0].copy_src
    assert len(resources) == 2 and source.id in resources
    rsrc = resources[source.id].target[0]
    assert isinstance(rsrc.type, ResourceType) and rsrc.type.elem is F32
    assert resources[source.id].args[1] == 128 * 4
    for copy in copies:
        assert copy.attr('engine') == 'lds_dma'
        assert copy.attr('form') == 'buffer'
        assert copy.attr('resource') is rsrc
        assert verify_buffer(copy) == []
    pir.verify(out)
    # Still the LDS-DMA's: one wait each on vmcnt.
    waits = [s for s in out if s.op == Op.WAIT_COUNTER]
    assert [(w.attr('counter'), w.attr('engine')) for w in waits] == [
        ('vmcnt', 'lds_dma')] * 2


def test_a_predicated_copy_keeps_the_global_form():
    """A lane its predicate holds off would write zero into the LDS."""
    _, body = _dma_body(copies=1)
    copy = _copies(body)[0]
    lead = next(s for s in body if s.op == Op.CALL)
    guarded = replace(copy, predicate=lead.target[0])
    assert buffer_base(guarded) is None
    assert buffer_base(copy) is not None


@pytest.mark.parametrize('change', [
    {'engine': 'async_lds'}, {'engine': 'cp_async'}, {'width': 8},
    {'zfill': 1}, {'elems': 2}, {'form': 'global'}])
def test_another_copy_keeps_its_form(change):
    _, body = _dma_body(copies=1)
    copy = _copies(body)[0]
    attrs = dict(copy.attrs)
    attrs.update(change)
    assert buffer_base(replace(copy, attrs=tuple(attrs.items()))) is None


def test_the_verifier_holds_the_lds_form_to_its_resource():
    _, body = _dma_body(copies=1)
    copy = _copies(_buffered(body))[0]
    assert verify_buffer(copy) == []
    assert verify_buffer(replace(copy, attrs=tuple(
        (k, None if k == 'resource' else v) for k, v in copy.attrs)))
    lead = next(s for s in body if s.op == Op.CALL)
    assert verify_buffer(replace(copy, predicate=lead.target[0]))
    assert verify_buffer(replace(copy, attrs=tuple(
        (k, 'async_lds' if k == 'engine' else v) for k, v in copy.attrs)))


# -- what it is written as -------------------------------------------------------- #

def _emit(ctx, body):
    out, diag = pir.schedule_async(body)
    assert not diag, diag
    out = _buffered(out)
    sink = _Commented()
    Emitter(sink, ctx).run(out)
    return out, sink.lines


def test_the_copy_is_written_through_the_resource_and_waited_for_on_vmcnt():
    ctx, body = _dma_body(copies=2)
    _, lines = _emit(ctx, body)
    text = '\n'.join(lines)
    name = re.search(r'tensorforge::BufferResource (\w+) = '
                     r'tensorforge::bufferResource\(m1, 512\);', text)
    assert name, text
    copies = [l for l in lines if 'bufferLoadLds' in l]
    assert len(copies) == 2 and 'global_load_lds' not in text
    assert all(re.match(rf'tensorforge::bufferLoadLds<float>\({name.group(1)}, '
                        rf'&s0\[[^\]]+\], [^;]+\);$', l) for l in copies)
    assert [l for l in lines if 's_waitcnt' in l] == [
        '__builtin_amdgcn_s_waitcnt(3953);',      # vmcnt(1)
        '__builtin_amdgcn_s_waitcnt(3952);']      # vmcnt(0)


def test_without_the_dma_the_body_lowers_as_before():
    """Where the body's copies take the synchronous fallback, the resource
    stands unused and the copy is a load and a store, as it was."""
    ctx, body = _dma_body(copies=1, dst=lambda b, lane: b.op(
        'add', INDEX, lane, lane, hint='twice'))
    _, lines = _emit(ctx, body)
    assert not any('bufferLoadLds' in l for l in lines)
    assert any(re.match(r'^s0\[.*\] = m1\[.*\];$', l) for l in lines)


# -- the reference interpreter ---------------------------------------------------- #

_SEED = 9

#: A resource over the first 40 elements of `m1`; every lane of a wave of 64
#: copies its own element through it into the LDS, behind one base a wave.
_COPY = ['tensorforge::BufferResource r = tensorforge::bufferResource(m1, '
         '160);',
         'tensorforge::bufferLoadLds<float>(r, &s0[threadIdx.x], '
         'threadIdx.x);']
_WAIT = ['__builtin_amdgcn_s_waitcnt(3952);']
_READ = ['m0[threadIdx.x] = s0[63 - threadIdx.x];']


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


def test_the_interpreter_reads_past_the_resource_as_zero_and_writes_it():
    from tensorforge.reference import kernel_eval
    seed = kernel_eval.Slot(_SEED)
    mem, races = _run(_COPY, _WAIT, _READ)
    assert races == []
    for t in range(64):
        k = 63 - t
        assert mem[('m0', t)] == (seed.read('m1', k) if k < 40 else 0.0)


def test_the_interpreter_writes_one_base_a_wave_and_lands_it_at_vmcnt():
    """The destination is the wave's first lane's, whatever a lane names --
    and a read with the copy in flight is reported."""
    from tensorforge.reference import kernel_eval
    seed = kernel_eval.Slot(_SEED)
    shifted = [_COPY[0], _COPY[1].replace('&s0[threadIdx.x]',
                                          '&s0[2 * threadIdx.x]')]
    mem, races = _run(shifted, _WAIT, _READ)
    assert races == []
    assert all(mem[('m0', t)] == (seed.read('m1', 63 - t) if 63 - t < 40
                                  else 0.0) for t in range(64))
    _, races = _run(_COPY, _READ)
    assert 'FLIGHT' in races


def test_the_interpreter_reads_and_writes_through_a_resource():
    """The engine's loads and stores: zero and nowhere past the resource
    or under a guard that fails."""
    from tensorforge.reference import kernel_eval
    seed = kernel_eval.Slot(_SEED)
    src = ['tensorforge::BufferResource r = tensorforge::bufferResource(m1, '
           '160);',
           'tensorforge::BufferResource w = tensorforge::bufferResource(m0, '
           '128);',
           'float v = tensorforge::bufferLoad<float>(r, threadIdx.x, '
           'threadIdx.x != 3);',
           'tensorforge::bufferStore<float>(w, threadIdx.x, v + 1.0f, '
           'threadIdx.x != 5);']
    mem, _ = _run(src)
    for t in range(64):
        want = None if t == 5 or t >= 32 else (
            (seed.read('m1', t) if t < 40 and t != 3 else 0.0) + 1.0)
        got = mem.get(('m0', t))
        if want is None:
            assert got is None or got == seed.read('m0', t)
        else:
            assert got == want


# -- generated kernels -------------------------------------------------------------- #

def _code(text):
    return re.sub(r'kernel_\w+', 'kernel', re.sub(r'//[^\n]*', '', text))


def test_a_preloaded_operator_goes_through_the_lds_form_where_asked():
    """The operator the block preloads by the LDS-DMA reads through its
    resource, waited for as before; with either switch alone, as before."""
    src = _generated('trip_prepared', 'gfx942', **ON)
    assert re.search(r'tensorforge::BufferResource (\w+) = '
                     r'tensorforge::bufferResource\(ptr_glb_m1, \d+\);', src)
    assert 'bufferLoadLds<float>' in src and 'global_load_lds' not in src
    lines = src.splitlines()
    copy = max(i for i, l in enumerate(lines) if 'bufferLoadLds' in l)
    wait = next(i for i in range(copy, len(lines))
                if '__builtin_amdgcn_s_waitcnt(3952);' in lines[i])
    assert '__syncthreads();' in lines[wait + 1]
    dma = _generated('trip_prepared', 'gfx942', amd_async_copies=True,
                     lanes_per_mult=64)
    assert 'global_load_lds' in dma and 'bufferLoadLds' not in dma
    loads = _generated('trip_prepared', 'gfx942', buffer_transfers=True,
                       lanes_per_mult=64)
    assert 'bufferLoadLds' not in loads and 'global_load_lds' not in loads


def test_gfx1250_s_copy_into_the_lds_is_no_lds_dma():
    src = _generated('trip_prepared', 'gfx1250', amd_async_copies=True,
                     buffer_transfers=True)
    assert 'bufferLoadLds' not in src and 'async_to_lds' in src


def _assemble(source, arch, tmp_path):
    from harness import syntax
    from tensorforge import toolchain
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    path = tmp_path / f'{arch}.hip'
    path.write_text('#include "tensorforge_device/hip.h"\n' + source)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '-I', toolchain.include_dir(), '--cuda-device-only',
                          '-S', '-o', '-', str(path)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and re.search(r'unsupported .*architecture|'
                                    r'invalid offload arch', run.stderr):
        pytest.skip(f'hipcc does not know {arch}')
    assert run.returncode == 0, run.stderr[-3000:]
    return [line.strip() for line in run.stdout.splitlines()
            if line.startswith('\t') and not line.strip().startswith(';')]


@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942', 'gfx950'])
def test_the_preload_compiles_to_buffer_loads_into_the_lds(arch, tmp_path):
    """Read off the instructions: the copies are `buffer_load_dword ...
    lds` on a resource in scalar registers, behind the base in M0, and
    `vmcnt(0)` stands ahead of the barrier with no LDS read before it."""
    asm = _assemble(_generated('trip_prepared', arch, **ON), arch, tmp_path)
    copies = [i for i, ins in enumerate(asm)
              if re.match(r'buffer_load_dword v\d+, s\[\d+:\d+\], 0 offen '
                          r'lds$', ins)]
    assert copies and not any('global_load_lds' in ins for ins in asm)
    for i in copies:
        assert any(asm[j].startswith('s_mov_b32 m0') for j in
                   range(max(0, i - 4), i))
    barrier = next(i for i, ins in enumerate(asm)
                   if ins.startswith('s_barrier'))
    wait = max(i for i in range(barrier) if asm[i].startswith('s_waitcnt'))
    assert re.match(r's_waitcnt vmcnt\(0\)', asm[wait]) and wait > max(copies)
    assert not any(ins.startswith('ds_read') for ins in asm[copies[0]:barrier])


def test_the_compiler_keeps_the_copies_and_the_counts(tmp_path):
    """Two copies and their waits, compiled for gfx942: two buffer loads
    into the LDS, and each wait kept as written -- the compiler places none
    in front of a read of what such a copy wrote, as for the global form."""
    ctx, body = _dma_body(copies=2)
    _, lines = _emit(ctx, body)
    keep = [l for l in _dma_kernel(lines).splitlines()[2:-1]]
    src = ('__global__ void __launch_bounds__(64) k(float *__restrict__ m0, '
           'float *__restrict__ m1) {\n  extern __shared__ float blk[];\n'
           + '\n'.join(keep) + '\n}\n')
    asm = _assemble(src, 'gfx942', tmp_path)
    assert sum(bool(re.match(r'buffer_load_dword .* lds$', ins))
               for ins in asm) == 2
    counts = [int(m.group(1)) for m in
              (re.match(r's_waitcnt vmcnt\((\d+)\)$', ins) for ins in asm)
              if m]
    assert counts[:2] == [1, 0]


def test_a_kernel_computes_with_the_lds_form_what_it_does_without():
    """The whole kernel through the interpreter, every multiplication of a
    block over two elements, gfx942 at 64 lanes: the same values through the
    resource as through the global pointer, and no race."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path
    from tensorforge.generators.generator import Generator
    from tensorforge.reference import kernel_eval

    def build(**options):
        path = next((Path(__file__).resolve().parent / 'cases').rglob(
            'packed_bits.py'))
        spec = importlib.util.spec_from_file_location('tf_buflds', path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
        gen = Generator(mod.descr_list(), Context(
            arch='gfx942', backend='hip', fp_type=mod.DTYPE,
            options=Options(**{**getattr(mod, 'OPTIONS', {}), **options})))
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        return (gen.get_kernel(),
                kernel_eval.launch_geometry(gen.get_launcher())[:2])

    src, (lanes, mults) = build(**ON)
    assert 'bufferLoadLds' in src
    plain, _ = build(amd_async_copies=True, lanes_per_mult=64)

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


def test_the_host_shim_takes_the_helpers():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    src = _generated('trip_prepared', 'gfx942', **ON)
    assert 'tensorforge::bufferLoadLds' in src
    result = syntax.check_source(src, shim=syntax.shim_for('hip'))
    assert result.ok, result.errors()
