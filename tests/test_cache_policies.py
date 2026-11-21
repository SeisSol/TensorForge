# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Cache policies on transfers (`engines.CACHE_POLICIES`): what the caches
are told about the lines a transfer touches, a transfer's `cache`.

The kinds `Options.cache_hints` asks for are the policies `cg` -- the L1
bypass, `__ldcg`/`__stcg` -- and `cs`, streamed, spelled as they were.  The
others are L2's: `evict_first` and `evict_last`, through a policy
`createpolicy` makes and the access's `.L2::cache_hint` (`tensorforge::ldL2`,
`stL2`), and a box through a tensor map under one (`tmaLoad<P>`).  Nothing
asks for those yet, so no kernel changes.  Checked here: which target spells
which, what the verifier refuses, the text, the oracle, the PTX, and on a
GPU of sm_80 or newer the values, which a policy leaves alone.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import test_tma as tma
from tensorforge import toolchain
from tensorforge.backend.pir import engines, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, IRError, MemSpace, Op,
                                          ScalarType)
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

F32, F64, I32 = Datatype.F32, Datatype.F64, Datatype.I32
LAST = 'tensorforge::L2Evict::Last'
FIRST = 'tensorforge::L2Evict::First'


def _target(arch, backend='cuda'):
    return Context(arch=arch, backend=backend, fp_type=F32).target


# -- the registry ------------------------------------------------------------ #

def test_the_policies():
    assert set(engines.CACHE_POLICIES) == {'ca', 'cg', 'cs', 'evict_first',
                                           'evict_last'}
    cg = engines.CACHE_POLICIES['cg']
    assert (cg.l1, cg.l2, cg.l2_evict) == ('bypass', 'normal', None)
    for name, evict in (('evict_first', 'First'), ('evict_last', 'Last')):
        policy = engines.CACHE_POLICIES[name]
        assert policy.l2 == name and policy.l2_evict == evict
        assert policy.sm == 80 and 'tma' in policy.engines
    assert engines.CACHE_POLICIES['ca'].engines == ('cp_async',)


@pytest.mark.parametrize('request_,want', [('cg', 'cg'), ('cs', 'cs'),
                                           (True, 'cg')])
def test_a_kind_of_hint_resolves_as_it_did(request_, want):
    """The kinds of `Options.cache_hints` where `__ldcg` has an overload,
    a bare request the L1 bypass; none for a wide value on CUDA, any on
    AMD."""
    sm90, gfx = _target('sm_90'), _target('gfx942', 'hip')
    assert engines.cache_policy(request_, sm90, F32) == want
    assert engines.cache_policy(request_, sm90, F32, 2) is None
    assert engines.cache_policy(request_, gfx, F32, 4) == want
    assert engines.cache_policy(False, sm90, F32) is None
    assert engines.cache_policy(request_, None, F32) is engines.UNRESOLVED


@pytest.mark.parametrize('name', ['evict_first', 'evict_last'])
def test_an_l2_policy_where_cuda_has_it(name):
    for arch in ('sm_80', 'sm_90', 'sm_120'):
        for elem in (F32, F64):
            assert engines.cache_policy(name, _target(arch), elem) == name
    assert engines.cache_policy(name, _target('sm_75'), F32) is None
    assert engines.cache_policy(name, _target('sm_90'), I32) is None
    assert engines.cache_policy(name, _target('sm_90'), F32, 4) is None
    assert engines.cache_policy(name, _target('gfx942', 'hip'), F32) is None


def test_a_request_for_no_policy_of_a_load_is_refused():
    with pytest.raises(IRError, match='they take'):
        engines.cache_policy('ca', _target('sm_90'), F32)
    with pytest.raises(IRError, match='they take'):
        engines.cache_policy('evict_middle', _target('sm_90'), F32)


# -- a load and a store under a policy ---------------------------------------- #

def _body(arch='sm_90', load='evict_last', store='evict_first', elem=F32):
    """A warp reading `m1[t]` under `load` and `m1[t + 32]` plainly, and
    storing their sum to `m0[t]` under `store`."""
    ctx = Context(arch=arch, backend='cuda', fp_type=elem)
    b = IRBuilder(fptype=elem, context=ctx)
    src = b.alloc(elem, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(elem, (32,), MemSpace.GLOBAL, hint='o', extern='m0')
    t = b.thread_id('x')
    x = b.load(src, t, nontemporal=load)
    y = b.load(src, b.op('add', INDEX, t, 32))
    b.store(out, b.op('add', ScalarType(elem), x, y), t, nontemporal=store)
    return ctx, b.finish()


def _lines(arch='sm_90', **kw):
    """The body's text, without the declarations of the global buffers,
    which are a kernel's parameters."""
    ctx, body = _body(arch, **kw)
    return [line for line in tma._emit(body, ctx)
            if not re.match(r'^(float|double) m\d+\[', line)]


def test_the_policy_is_the_transfers():
    _, body = _body()
    load, plain = [s for s, _ in walk(body) if s.op == Op.LOAD]
    (store,) = [s for s, _ in walk(body) if s.op == Op.STORE]
    assert load.attr('cache') == 'evict_last' and plain.attr('cache') is None
    assert store.attr('cache') == 'evict_first'
    verify(body)


def test_the_text_of_an_l2_policy():
    text = '\n'.join(_lines())
    assert f'tensorforge::ldL2<{LAST}>(&m1[threadIdx.x])' in text
    assert (f'tensorforge::stL2<{FIRST}>(&m0[threadIdx.x], (v3_ld + v5_ld));'
            in text)
    assert text.count('tensorforge::ldL2') == 1


def test_a_kind_of_hint_is_spelled_as_it_was():
    text = '\n'.join(_lines(load='cg', store='cs'))
    assert '__ldcg(&m1[threadIdx.x])' in text and '__stcs(&m0[' in text
    assert 'L2Evict' not in text


def test_where_the_target_has_no_l2_policy_the_access_is_plain():
    text = '\n'.join(_lines('sm_75'))
    assert 'L2Evict' not in text and '__ld' not in text and '__st' not in text


def test_a_policy_its_engine_does_not_take_is_refused():
    _, body = _body()
    load = next(s for s, _ in walk(body) if s.op == Op.LOAD)
    assert engines.cache_refusal(load) is None
    unknown = load.with_attr('cache', 'evict_middle')
    assert 'which is none' in engines.cache_refusal(unknown)
    with pytest.raises(IRError, match='which is none'):
        verify((unknown,))
    copy = load.with_attr('cache', 'ca')
    assert 'which cp_async take' in engines.cache_refusal(copy)


def test_a_box_under_a_policy_of_no_box_is_refused():
    case = tma.Case((64, 64), (8, 32), ((3, 4),), cache='cg')
    with pytest.raises(IRError, match='which scalar, vector, cp_async take'):
        tma._body(case)


def _oracle_source(lines):
    return 'void k() {\n' + ''.join(f'  {line}\n' for line in lines) + '}'


@pytest.mark.parametrize('elem', [F32, F64])
def test_the_oracle_reads_and_writes_under_a_policy_as_without(elem):
    lines = _lines(elem=elem)
    mem = kernel_eval.evaluate_wave(_oracle_source(lines), 32, seed=5,
                                    globals_only=True)
    slot = kernel_eval.Slot(5)
    for t in range(32):
        assert mem[('m0', t)] == slot.read('m1', t) + slot.read('m1', t + 32)


# -- a box under a policy ------------------------------------------------------ #

BOXES = {
    '2d-last': tma.Case((64, 64), (8, 32), ((3, 4),), swizzle=128,
                        cache='evict_last'),
    '3d-first': tma.Case((16, 16, 64), (3, 4, 16), ((2, 4, 16),),
                         cache='evict_first'),
}


def test_the_text_of_a_box_under_a_policy():
    lines = tma._emit(tma._body(BOXES['3d-first'])[1])
    assert [line for line in lines if 'tmaLoad' in line] == [
        f'tensorforge::tmaLoad<{FIRST}>(tma_bulk, &sa[0], &tmA, 768, 16, 4, '
        f'2);']
    plain = tma._emit(tma._body(tma.CASES['3d'])[1])
    assert [line for line in plain if 'tmaLoad' in line] == [
        'tensorforge::tmaLoad(tma_bulk, &sa[0], &tmA, 768, 16, 4, 2);']


@pytest.mark.parametrize('name', sorted(BOXES))
def test_the_oracle_copies_a_box_under_a_policy_as_without(name):
    got, want = tma._run_oracle(BOXES[name])
    assert not tma._same(got, want)


# -- the builds ---------------------------------------------------------------- #

def _cuda_source(lines, elem):
    ctype = 'float' if elem == F32 else 'double'
    body = ''.join(f'  {line}\n' for line in lines)
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda.h"

__global__ void policies({ctype} *__restrict__ m0,
                         const {ctype} *__restrict__ m1) {{
{body}}}

int main(int argc, char **argv) {{
  std::vector<{ctype}> h(64), d(32);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), sizeof({ctype}), 64, in) != 64) return 2;
  std::fclose(in);
  {ctype} *m0, *m1;
  if (cudaMalloc(&m0, 32 * sizeof({ctype})) ||
      cudaMalloc(&m1, 64 * sizeof({ctype})))
    return 4;
  cudaMemcpy(m1, h.data(), 64 * sizeof({ctype}), cudaMemcpyHostToDevice);
  policies<<<1, 32>>>(m0, m1);
  if (cudaDeviceSynchronize() != cudaSuccess) return 5;
  cudaMemcpy(d.data(), m0, 32 * sizeof({ctype}), cudaMemcpyDeviceToHost);
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), sizeof({ctype}), 32, out) != 32) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _build(tmp_path, arch, elem, product):
    nvcc = tma._nvcc_builds(arch)
    src = tmp_path / 'policies.cu'
    src.write_text(_cuda_source(_lines(arch, elem=elem), elem))
    out = tmp_path / {'ptx': 'k.ptx'}.get(product, 'k')
    cmd = [nvcc, *toolchain.COMPILERS['cuda'].language_flags(),
           *toolchain.COMPILERS['cuda'].target_flags(arch), '-I',
           str(tma.INCLUDE), '-o', str(out), str(src)]
    if product == 'ptx':
        cmd.insert(1, '-ptx')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


@pytest.mark.parametrize('arch', ['sm_80', 'sm_90', 'sm_120'])
@pytest.mark.parametrize('elem', [F32, F64])
def test_compiled_the_access_takes_its_policy(arch, elem, tmp_path):
    ptx = _build(tmp_path, arch, elem, 'ptx').read_text()
    kind = 'f32' if elem == F32 else 'f64'
    assert 'createpolicy.fractional.L2::evict_last.b64' in ptx
    assert 'createpolicy.fractional.L2::evict_first.b64' in ptx
    assert len(re.findall(rf'ld\.global\.L2::cache_hint\.{kind}\b', ptx)) == 1
    assert len(re.findall(rf'st\.global\.L2::cache_hint\.{kind}\b', ptx)) == 1
    assert 'trap;' not in ptx


@pytest.mark.parametrize('name', sorted(BOXES))
def test_compiled_the_box_takes_its_policy(name, tmp_path):
    case = BOXES[name]
    ptx = tma._build(case, tmp_path, toolchain.COMPILERS['cuda'].target_flags(
        'sm_90'), product='ptx').read_text()
    rank = len(case.box)
    assert len(re.findall(
        rf'cp\.async\.bulk\.tensor\.{rank}d\.shared::cluster\.global\.tile\.'
        r'mbarrier::complete_tx::bytes\.L2::cache_hint', ptx)) == 1
    assert f'createpolicy.fractional.L2::{case.cache}.b64' in ptx
    assert 'trap;' not in ptx


def _gpu_arch():
    found = tma._gpu()
    return found if found is not None and int(found[3:]) >= 80 else None


@pytest.mark.parametrize('elem', [F32, F64])
def test_gpu_a_policy_moves_the_same_numbers(elem, tmp_path):
    """Built for the GPU present, of sm_80 or newer: a load under
    `evict_last` and a store under `evict_first`, against the sums computed
    here, to the bit."""
    arch = _gpu_arch()
    if arch is None:
        pytest.skip('no GPU of sm_80 or newer')
    exe = _build(tmp_path, arch, elem, 'binary')
    fmt = 'f' if elem == F32 else 'd'
    values = [0.25 * ((7 * i) % 61) - 3.0 for i in range(64)]
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<64{fmt}', *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    got = struct.unpack(f'<32{fmt}', (tmp_path / 'out.bin').read_bytes())
    assert list(got) == [values[t] + values[t + 32] for t in range(32)]


@pytest.mark.parametrize('name', sorted(BOXES))
def test_gpu_a_box_under_a_policy_lands_as_without(name, tmp_path):
    arch = tma._gpu()
    if arch is None or int(arch[3:]) < 90:
        pytest.skip('no GPU of sm_90 or newer')
    case = BOXES[name]
    exe = tma._build(case, tmp_path,
                     toolchain.COMPILERS['cuda'].target_flags(arch),
                     product='binary', arch=arch)
    values = tma._inputs(case)
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{len(values)}f',
                                                  *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    count = case.volume * len(case.origins)
    got = struct.unpack(f'<{count}f', (tmp_path / 'out.bin').read_bytes())
    assert not tma._same(got, tma._expected(case, values))
