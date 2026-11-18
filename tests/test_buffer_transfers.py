# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Global transfers through AMD's buffer instructions (`pir.buffers`,
`Options.buffer_transfers`).

An access the engine takes names a resource over its binding and the engine
(`resource`, `engine='buffer'`), and its predicate becomes an offset past
the resource: a load under one is no select, a store under one no branch.
Which accesses: 4, 8 or 16 bytes a lane of 32- or 64-bit numbers, in global
memory, with no cache hint, through a binding the wave holds alike -- the
rest stays what it was, and with the switch off everything does.
"""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
from types import SimpleNamespace

import pytest

from tensorforge.backend import pir
from tensorforge.backend.pir import (BOOL, INDEX, SIZE, BufferType, IRBuilder,
                                     MemSpace, Op, ScalarType)
from tensorforge.backend.pir.buffers import buffer_base, wave_level
from tensorforge.backend.pir.core import Uniformity, walk_stmts
from tensorforge.backend.pir.engines import (TRANSFER_ENGINES, Completion,
                                             ResourceType, kept_facts)
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Addressing, Datatype, Residence
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

F32 = Datatype.F32


def _context(backend='hip', arch='gfx942', **options):
    return Context(arch=arch, backend=backend, fp_type=F32,
                   options=Options(**options) if options else None)


def _operand(name, by_value=False):
    """What the accesses through a binding are recorded against: the
    operand's symbol, of which the pass reads only whether the kernel takes
    its numbers rather than its address."""
    return SimpleNamespace(name=name, obj=SimpleNamespace(
        passed_by_value=by_value))


def _binding(builder, name, index, extent=64, writable=False, by_value=False,
             space=MemSpace.GLOBAL):
    """`glb_<name>`: one batch element of the operand, its pointer computed
    from the element index -- declared, as `ptr_manip` declares it, with no
    level of its own."""
    return builder.decl_expr(
        f'const float *const glb_{name}', f'&{name}[{{0}} * {extent}]',
        BufferType(F32, (extent,), space, readonly=not writable),
        _operand(name, by_value), space=space, args=(index,),
        extern=f'glb_{name}')


def _body(level=Uniformity.MULT, **binding):
    """A guarded load, a wide load, a hinted load and a guarded store,
    through bindings computed from an element index of `level`."""
    b = IRBuilder(fptype=F32, context=_context())
    element = b.extern_value('batchId0', SIZE, uniform=level)
    src = _binding(b, 'm1', element, **binding)
    dst = _binding(b, 'm0', element, writable=True)
    lane = b.thread_id('x')
    guard = b.op('lt', BOOL, lane, 9, hint='g')
    value = b.load(src, lane, predicate=guard, hint='v')
    wide = b.load(src, b.op('mul', INDEX, lane, 4), type_=ScalarType(F32, 4),
                  align=16, hint='w')
    hinted = b.load(src, b.op('add', INDEX, lane, 32), nontemporal='cs',
                    hint='h')
    total = b.op('add', ScalarType(F32), value, hinted)
    b.store(dst, total, lane, predicate=guard)
    b.store(dst, wide, b.op('add', INDEX, lane, 16), align=16)
    return b, b.finish()


def _source(body, context=None) -> str:
    writer = Writer()
    pir.emit(body, writer, context or _context())
    return writer.get_src()


def _through(body):
    return [s for s in walk_stmts(body) if s.attr('engine') == 'buffer']


# -- the engine -------------------------------------------------------------- #

def test_the_engine_is_a_synchronous_one_of_loads_and_stores():
    engine = TRANSFER_ENGINES['buffer']
    assert engine.ops == (Op.LOAD, Op.STORE)
    assert engine.completes is Completion.SYNC
    assert Op.RESOURCE in Op.KNOWN


def test_only_amd_has_buffer_resources():
    assert _context('hip', 'gfx942').target.buffer_resources()
    assert _context('hip', 'gfx1150').target.buffer_resources()
    assert not _context('cuda', 'sm_86').target.buffer_resources()
    assert not _context('oneapi', 'pvc').target.buffer_resources()
    assert _context('cuda', 'sm_86').target.lexic.buffer_resource_type() is None


def test_the_wave_level_is_the_multiplication_where_it_fills_waves():
    assert wave_level(64, 64) is Uniformity.MULT
    assert wave_level(128, 64) is Uniformity.MULT
    assert wave_level(32, 32) is Uniformity.MULT
    assert wave_level(32, 64) is Uniformity.MULTGROUP
    assert wave_level(48, 32) is Uniformity.MULTGROUP
    assert wave_level(None, 64) is Uniformity.MULTGROUP


# -- the pass ---------------------------------------------------------------- #

def test_accesses_go_through_a_resource_behind_their_binding():
    b, body = _body()
    out = pir.buffer_transfers(body, b.scratch, threads=64, wave=64)
    pir.verify(out)
    resources = {}
    for at, r in enumerate(out):
        if r.op != Op.RESOURCE:
            continue
        # Right behind the declaration of what it is made over.
        assert out[at - 1].target == (r.args[0],)
        assert r.args[1] == 64 * 4
        assert isinstance(r.target[0].type, ResourceType)
        assert r.target[0].type.elem is F32
        assert r.pure and r.movable and not r.accesses
        resources[r.target[0].id] = r.args[0]
    assert len(resources) == 2
    through = _through(out)
    assert [s.op for s in through] == [Op.LOAD, Op.LOAD, Op.STORE, Op.STORE]
    for s in through:
        assert resources[s.attr('resource').id] is s.args[0]
    # The hinted load stays a global one.
    hinted = [s for s in walk_stmts(out) if s.attr('nontemporal')]
    assert len(hinted) == 1 and hinted[0].attr('engine') is None


def test_a_guard_is_an_offset_and_not_a_select_or_a_branch():
    b, body = _body()
    src = _source(pir.buffer_transfers(body, b.scratch, threads=64, wave=64))
    rsrc = re.findall(r'tensorforge::BufferResource (\w+) = '
                      r'tensorforge::bufferResource\(glb_m(\d), 256\);', src)
    assert sorted(m for _, m in rsrc) == ['0', '1'], src
    names = {m: name for name, m in rsrc}
    guard = re.search(r'bool (\w+) = ', src).group(1)
    assert re.search(rf'float \w+ = tensorforge::bufferLoad<float>\('
                     rf'{names["1"]}, [^;]*, {guard}\);', src), src
    assert re.search(rf'tensorforge::VectorT<float, 4> \w+ = tensorforge::'
                     rf'bufferLoad<float, 4>\({names["1"]}, [^;]*\);', src), src
    assert re.search(rf'tensorforge::bufferStore<float>\({names["0"]}, '
                     rf'[^;]*, {guard}\);', src), src
    assert re.search(rf'tensorforge::bufferStore<float, 4>\({names["0"]}, ',
                     src), src
    assert 'if (' not in src and '?' not in src, src
    assert '__builtin_nontemporal_load(&glb_m1[' in src, src


def test_a_binding_the_wave_does_not_hold_alike_stays_global():
    """32 lanes on a wave of 64: two elements share the wave, and their
    bindings differ in it.  A batch-constant operator does not."""
    b, body = _body()
    assert pir.buffer_transfers(body, b.scratch, threads=32, wave=64) is body
    b, body = _body(level=Uniformity.GRID)
    out = pir.buffer_transfers(body, b.scratch, threads=32, wave=64)
    assert len(_through(out)) == 4


def test_a_binding_is_as_uniform_as_what_it_is_computed_from():
    """The declaration states the grid's level; the lane index it reads
    makes it the lane's."""
    b = IRBuilder(fptype=F32, context=_context())
    src = _binding(b, 'm1', b.thread_id('x'))
    assert src.uniformity is Uniformity.GRID
    b.load(src, 0)
    body = b.finish()
    assert pir.buffer_transfers(body, b.scratch, threads=64, wave=64) is body


def test_what_the_engine_does_not_spell_stays_global():
    b = IRBuilder(fptype=F32, context=_context())
    element = b.extern_value('batchId0', SIZE, uniform=Uniformity.GRID)
    src = _binding(b, 'm1', element)
    lane = b.thread_id('x')
    guard = b.op('lt', BOOL, lane, 9)
    b.load(src, lane, predicate=guard, other=b.const(1.0))  # its own zero
    b.load(src, lane, nontemporal='cg')                    # a hint
    b.load(src, lane, type_=ScalarType(Datatype.F64))     # converted
    half = BufferType(Datatype.F16, (64,), MemSpace.GLOBAL, readonly=True)
    small = b.decl_expr('const half *const glb_h', '&h[{0}]', half,
                        _operand('h'), args=(element,), extern='glb_h')
    b.load(small, lane)                                    # 2 bytes
    shared = _binding(b, 's', element, space=MemSpace.SHARED)
    b.load(shared, lane)                                   # not global
    by_value = _binding(b, 'a', element, by_value=True)
    b.load(by_value, lane)                                 # an argument
    body = b.finish()
    out = pir.buffer_transfers(body, b.scratch, threads=64, wave=64)
    taken = _through(out)
    # A scalar of another type converts, as an assignment does.
    assert len(taken) == 1 and taken[0].target[0].type.base is Datatype.F64


def test_a_vector_of_another_element_is_not_reinterpreted():
    b = IRBuilder(fptype=F32, context=_context())
    element = b.extern_value('batchId0', SIZE)
    src = _binding(b, 'm1', element)
    b.load(src, 0, type_=ScalarType(Datatype.I32, 2), align=8)
    body = b.finish()
    load = next(x for x in body if x.op == Op.LOAD)
    assert buffer_base(load) is None


def test_an_access_through_the_symbol_takes_the_binding_of_its_name():
    """Written `glb_m0[...]`, the store reaches what `glb_m0` is declared as
    where it stands -- and not where a window of the same name in a region
    hides it, which it does in that region only."""
    b = IRBuilder(fptype=F32, context=_context())
    element = b.extern_value('batchId0', SIZE)
    _binding(b, 'm0', element, writable=True)
    symbol = SimpleNamespace(name='glb_m0', obj=SimpleNamespace(
        passed_by_value=False), stype=None, datatype=F32)
    lane = b.thread_id('x')
    b.store(symbol, b.const(1.0), lane, space=MemSpace.GLOBAL)
    with b.If(b.op('lt', BOOL, lane, 9)):
        _binding(b, 'm0', element, space=MemSpace.SHARED)
        b.store(symbol, b.const(2.0), lane, space=MemSpace.GLOBAL)
    b.store(symbol, b.const(3.0), lane, space=MemSpace.GLOBAL)
    body = b.finish()
    out = pir.buffer_transfers(body, b.scratch, threads=64, wave=64)
    stores = [s for s in walk_stmts(out) if s.op == Op.STORE]
    assert [s.attr('engine') for s in stores] == ['buffer', None, 'buffer']
    src = _source(out)
    assert re.search(r'tensorforge::bufferStore<float>\(\w+, threadIdx.x, '
                     r'1.0f\);', src), src
    assert 'glb_m0[threadIdx.x] = 2.0f;' in src, src


def test_a_resource_is_no_register_and_keeps_no_fact_of_a_redirect():
    b, body = _body()
    out = pir.buffer_transfers(body, b.scratch, threads=64, wave=64)
    assert pir.pressure(out) == pir.pressure(body)
    for s in _through(out):
        assert kept_facts(s) == tuple((k, v) for k, v in s.attrs
                                      if k == 'width')


# -- generated kernels ------------------------------------------------------- #

def _gemm(backend='hip', arch='gfx942', beta=0.0, **options):
    """`C = A B (+ C)`, `B` an operator of the batch: its reads are the
    operator's, and with no cache hint asked for (`cache_hints`) the
    element's reads and its store are plain ones."""
    def tensor(alias, rows, cols, addressing=Addressing.STRIDED):
        return Tensor([rows, cols], addressing,
                      BoundingBox([0, 0], [rows, cols]), alias=alias,
                      datatype=F32)
    a = tensor('A', 56, 9)
    b = tensor('B', 9, 9, Addressing.NONE)
    c = tensor('C', 56, 9)
    generator = Generator([GemmDescr(False, False, SubTensor(a), SubTensor(b),
                                     SubTensor(c), alpha=1.0, beta=beta)],
                          _context(backend, arch, **options))
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    return generator.get_kernel()


def _code(text):
    """The kernel without its comments, which name the options, and without
    its name, which is a hash of them."""
    return re.sub(r'kernel_\w+', 'kernel', re.sub(r'//[^\n]*', '', text))


def test_the_switch_off_or_off_amd_changes_nothing():
    assert 'buffer' not in _code(_gemm(lanes_per_mult=64))
    for backend, arch in (('cuda', 'sm_86'), ('oneapi', 'pvc')):
        assert (_code(_gemm(backend, arch, buffer_transfers=True))
                == _code(_gemm(backend, arch)))


@pytest.mark.parametrize('arch,lanes', [('gfx942', 64), ('gfx1150', 32)])
def test_a_kernel_a_wave_wide_reads_and_writes_through_resources(arch, lanes):
    src = _gemm('hip', arch, buffer_transfers=True, lanes_per_mult=lanes,
                cache_hints='none')
    assert re.search(r'tensorforge::bufferResource\(glb_m\d+, \d+\)', src), src
    assert 'tensorforge::bufferStore<float>' in src, src
    assert 'tensorforge::bufferLoad<float>' in src, src


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
    return run.stdout


@pytest.mark.parametrize('arch,lanes', [('gfx942', 64), ('gfx950', 64),
                                        ('gfx1150', 32), ('gfx1250', 32)])
def test_the_resource_is_a_scalar_and_the_guard_no_branch(arch, lanes,
                                                         tmp_path):
    """Compiled: buffer instructions on a resource in scalar registers, with
    no loop around them that reads it a lane at a time -- no more
    `v_readfirstlane` than the kernel has without them -- and nothing in
    `soffset`, which the range check leaves out."""
    isa = _assemble(_gemm('hip', arch, buffer_transfers=True,
                          lanes_per_mult=lanes, cache_hints='none'), arch,
                    tmp_path)
    plain = _assemble(_gemm('hip', arch, lanes_per_mult=lanes,
                            cache_hints='none'), arch, tmp_path)
    ops = re.findall(r'^\s+(buffer_(?:load|store)_\w+)\s+([^\n]*)', isa, re.M)
    assert ops, isa[-2000:]
    for name, operands in ops:
        parts = [p.strip() for p in operands.split(',')]
        assert parts[2].startswith('s['), (name, operands)
        assert parts[3].split()[0] in ('0', 'null'), (name, operands)
    assert isa.count('v_readfirstlane') == plain.count('v_readfirstlane')


def test_the_host_shim_takes_the_helpers():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    src = _gemm('hip', 'gfx942', buffer_transfers=True, lanes_per_mult=64,
                cache_hints='none')
    assert 'tensorforge::bufferStore' in src
    result = syntax.check_source(src, shim=syntax.shim_for('hip'))
    assert result.ok, result.errors()


#: Through the helpers on a device: a guarded lane and one past the resource
#: read zero and write nothing, and the words are the numbers -- read back as
#: they were stored, not converted from the integers the builtins return.
_DEVICE_CHECK = r"""
#include "tensorforge_device/hip.h"
#include <cstdio>

__global__ void k(const float *in, float *out, const double *din,
                  double *dout, int n) {
  const auto ri = tensorforge::bufferResource(in, 256 * 4);
  const auto ro = tensorforge::bufferResource(out, 256 * 4);
  const auto rdi = tensorforge::bufferResource(din, 128 * 8);
  const auto rdo = tensorforge::bufferResource(dout, 128 * 8);
  const int t = threadIdx.x;
  const bool g = t < n;
  const float a = tensorforge::bufferLoad<float>(ri, t, g);
  const float past = tensorforge::bufferLoad<float>(ri, t + 256);
  const auto b = tensorforge::bufferLoad<float, 4>(ri, 4 * t);
  tensorforge::bufferStore<float>(ro, t, a + past, g);
  tensorforge::bufferStore<float>(ro, t + 64, b[0] + b[1] + b[2] + b[3]);
  tensorforge::bufferStore<float>(ro, t + 1000, 99.0f);
  const double e = tensorforge::bufferLoad<double>(rdi, t, g);
  const auto f = tensorforge::bufferLoad<double, 2>(rdi, 2 * (t % 32));
  tensorforge::bufferStore<double>(rdo, t, e, g);
  tensorforge::bufferStore<double, 2>(rdo, 64 + 2 * (t % 32), f, t < 32);
}

int main() {
  const int n = 40;
  float in[256], out[256 + 1100];
  double din[128], dout[128];
  for (int i = 0; i < 256; ++i) in[i] = i + 1.5f;
  for (int i = 0; i < 256 + 1100; ++i) out[i] = -1.0f;
  for (int i = 0; i < 128; ++i) { din[i] = 0.5 * i + 1; dout[i] = -1.0; }
  float *dIn, *dOut;
  double *dDin, *dDout;
  if (hipMalloc(&dIn, sizeof in) || hipMalloc(&dOut, sizeof out) ||
      hipMalloc(&dDin, sizeof din) || hipMalloc(&dDout, sizeof dout) ||
      hipMemcpy(dIn, in, sizeof in, hipMemcpyHostToDevice) ||
      hipMemcpy(dOut, out, sizeof out, hipMemcpyHostToDevice) ||
      hipMemcpy(dDin, din, sizeof din, hipMemcpyHostToDevice) ||
      hipMemcpy(dDout, dout, sizeof dout, hipMemcpyHostToDevice)) {
    printf("SKIP no device\n");
    return 0;
  }
  k<<<1, 64>>>(dIn, dOut, dDin, dDout, n);
  if (hipMemcpy(out, dOut, sizeof out, hipMemcpyDeviceToHost) ||
      hipMemcpy(dout, dDout, sizeof dout, hipMemcpyDeviceToHost)) {
    printf("FAIL copy\n");
    return 1;
  }
  int bad = 0;
  for (int t = 0; t < 64; ++t) {
    bad += out[t] != (t < n ? in[t] : -1.0f);
    bad += out[t + 64] != in[4 * t] + in[4 * t + 1] + in[4 * t + 2] +
                              in[4 * t + 3];
    bad += out[t + 1000] != -1.0f;
    bad += dout[t] != (t < n ? din[t] : -1.0);
  }
  for (int t = 0; t < 32; ++t) {
    bad += dout[64 + 2 * t] != din[2 * t];
    bad += dout[65 + 2 * t] != din[2 * t + 1];
  }
  printf("%s %d\n", bad ? "FAIL" : "OK", bad);
  return bad != 0;
}
"""


def test_the_helpers_on_a_device(tmp_path):
    from harness import syntax
    from harness.gpu_detect import detect_amd
    from tensorforge import toolchain
    gpus = detect_amd()
    if not gpus:
        pytest.skip('no AMD device')
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    path = tmp_path / 'check.hip'
    path.write_text(_DEVICE_CHECK)
    exe = tmp_path / 'check'
    build = subprocess.run([hipcc, '-O3', '-std=c++17',
                            f'--offload-arch={gpus[0].arch}', '-I',
                            toolchain.include_dir(), '-o', str(exe), str(path)],
                           capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, build.stderr[-3000:]
    run = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=120)
    if run.stdout.startswith('SKIP'):
        pytest.skip(run.stdout.strip())
    assert run.returncode == 0 and run.stdout.startswith('OK'), run.stdout
