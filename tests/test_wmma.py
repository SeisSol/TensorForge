# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""AMD's WMMA on the matrix op (`Op.MATRIX_MAC`): engine `wmma` for RDNA 3
and 4 (`(A, B, C)`) and `wmma_gfx1250` for gfx1250 and gfx1251 (the
modifiers among the operands), issued from the catalog's rows as an MFMA is
(`amd.catalog.MatrixOp.mac`), with the layouts the vendor states.

Nothing emits a WMMA yet, so no kernel changes.  What is checked here: the
engines, which row takes which, the text of both forms, what is refused,
the builds for gfx1150, gfx1200 and gfx1250 with the instruction read back
from the ISA, and on the local gfx1150 an RDNA 3 product against NumPy,
the operands laid out as the catalog's layouts say.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backend.instructions.compute.primitives.amd import (
    catalog, layouts)
from tensorforge.backend.pir import engines, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import IRError, Op, Participants, ScalarType
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

F32 = Datatype.F32
INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'


def _op(builtin):
    return next(o for o in catalog.MATRIX_OPS if o.builtin == builtin)


def _ctx(arch):
    return Context(arch=arch, backend='hip', fp_type=F32)


def _product(builtin, arch, **modifiers):
    """One product of `builtin` on operands named `a`, `bv` and `c`: the
    builder and the statement."""
    op = _op(builtin)
    b = IRBuilder(fptype=F32, context=_ctx(arch))
    a = b.extern_value('a', ScalarType(op.a.dtype, op.a.per_lane))
    bv = b.extern_value('bv', ScalarType(op.b.dtype, op.b.per_lane))
    c = b.extern_value('c', ScalarType(op.d.dtype, op.d.per_lane))
    op.mac(b, ScalarType(op.d.dtype, op.d.per_lane), a, bv, c, **modifiers)
    body = b.finish()
    return body, next(s for s, _ in walk(body) if s.op == Op.MATRIX_MAC)


def _line(body, arch):
    lines = []
    Emitter(lines.append, _ctx(arch)).run(body)
    return next(line for line in lines if '__builtin_amdgcn_wmma' in line)


# -- the engines and the rows ------------------------------------------------- #

def test_the_wmma_engines():
    for name in ('wmma', 'wmma_gfx1250'):
        engine = engines.MATRIX_ENGINES[name]
        assert engine.collective is Participants.WAVE and engine.work
        assert engine.completes is engines.Completion.SYNC


@pytest.mark.parametrize('op', [o for o in catalog.MATRIX_OPS
                                if o.builtin.startswith('wmma')],
                         ids=lambda o: o.builtin)
def test_each_wmma_row_is_its_engine_with_the_vendor_s_layouts(op):
    _, s = _product(op.builtin, 'gfx1250')
    want = 'wmma' if op.call is catalog.Call.WMMA else 'wmma_gfx1250'
    assert s.attr('engine') == want and s.attr('instruction') == op.builtin
    assert s.attr('shape') == (op.m, op.n, op.k)
    assert s.attr('layouts') == catalog.vendor_layouts(op)
    assert engines.verify_engine(s) == []
    if want == 'wmma_gfx1250':
        assert dict(s.attr('modifiers')) == dict.fromkeys(
            engines.WMMA_MODIFIERS, 0)
    else:
        assert not s.attr('modifiers')


def test_an_mfma_is_issued_as_it_was():
    _, s = _product('mfma_f32_4x4x1f32', 'gfx942')
    assert s.attr('engine') == 'mfma'
    assert s.attr('modifiers') == (('cbsz', 0), ('abid', 0), ('blgp', 0))


# -- the text ----------------------------------------------------------------- #

def test_the_text_of_rdna_s_wmma():
    body, _ = _product('wmma_f32_16x16x16_f16_w32', 'gfx1150')
    assert _line(body, 'gfx1150').endswith(
        '= __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, c);')


def test_the_text_of_gfx1250_s_wmma_with_its_modifiers():
    body, _ = _product('wmma_f32_16x16x4_f32', 'gfx1250', reuse_a=1,
                       a_neg=1)
    assert _line(body, 'gfx1250').endswith(
        '= __builtin_amdgcn_wmma_f32_16x16x4_f32(true, a, false, bv, 0, c, '
        'true, false);')


def test_what_is_refused():
    with pytest.raises(ValueError, match='takes no modifiers'):
        _product('wmma_f32_16x16x16_f16_w32', 'gfx1150', reuse_a=1)
    with pytest.raises(ValueError, match='has no modifier'):
        _product('wmma_f32_16x16x4_f32', 'gfx1250', reuse_c=1)
    with pytest.raises(ValueError, match='takes no modifiers'):
        _product('mfma_f32_4x4x1f32', 'gfx942', reuse_a=1)
    body, s = _product('wmma_f32_16x16x4_f32', 'gfx1250')
    cuda = Context(arch='sm_90', backend='cuda', fp_type=F32)
    with pytest.raises(IRError, match="AMD's, on HIP or SYCL for AMD"):
        engines.lower(s, cuda.target, name=str)
    acpp = Context(arch='gfx1250', backend='acpp', fp_type=F32)
    assert engines.lower(s, acpp.target, name=str).result.startswith(
        '__builtin_amdgcn_wmma_f32_16x16x4_f32(')


# -- the builds --------------------------------------------------------------- #

def _hipcc():
    found = shutil.which('hipcc')
    if found is None:
        pytest.skip('hipcc not found')
    return found


def _source(line, a, b, d, out_regs):
    """A kernel of one wave that loads each lane's registers of A and B from
    `af` and `bf`, a register after the other, takes the product of the
    emitted line with C zero, and stores D's registers to `dout`."""
    return f'''#include <hip/hip_runtime.h>
#include "tensorforge_device/hip.h"

typedef {a[0]} va __attribute__((ext_vector_type({a[1]})));
typedef {b[0]} vb __attribute__((ext_vector_type({b[1]})));
typedef {d[0]} vd __attribute__((ext_vector_type({d[1]})));

__global__ void __launch_bounds__(32) product(const {a[0]} *af,
    const {b[0]} *bf, {d[0]} *dout) {{
  const int t = threadIdx.x;
  va a;
  vb bv;
  vd c = {{}};
  for (int s = 0; s < {a[1]}; ++s) a[s] = af[t * {a[1]} + s];
  for (int s = 0; s < {b[1]}; ++s) bv[s] = bf[t * {b[1]} + s];
  {line}
  for (int s = 0; s < {out_regs}; ++s) dout[t * {out_regs} + s] = v3_acc[s];
}}
'''


#: builtin -> (target, A's, B's and D's element type and count a lane): one
#: row of each family, RDNA 3's, RDNA 4's and both of gfx1250's.
CASES = {
    'wmma_f32_16x16x16_f16_w32': ('gfx1150', ('_Float16', 16),
                                  ('_Float16', 16), ('float', 8)),
    'wmma_f32_16x16x16_f16_w32_gfx12': ('gfx1200', ('_Float16', 8),
                                        ('_Float16', 8), ('float', 8)),
    'wmma_f32_16x16x32_f16': ('gfx1250', ('_Float16', 16), ('_Float16', 16),
                              ('float', 8)),
    'wmma_f32_16x16x4_f32': ('gfx1250', ('float', 2), ('float', 2),
                             ('float', 8)),
}


@pytest.mark.parametrize('builtin', sorted(CASES))
def test_built_for_its_target_the_product_is_one_v_wmma(builtin, tmp_path):
    arch, a, b, d = CASES[builtin]
    body, _ = _product(builtin, arch)
    src = tmp_path / 'k.hip'
    src.write_text(_source(_line(body, arch), a, b, d, d[1]))
    asm = tmp_path / 'k.s'
    run = subprocess.run([_hipcc(), '-x', 'hip', '-std=c++17',
                          f'--offload-arch={arch}', '--cuda-device-only',
                          '-O3', '-S', '-I', str(INCLUDE), '-o', str(asm),
                          str(src)], capture_output=True, text=True,
                         timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    # The mnemonic drops the wave size and the generation the builtin names.
    mnemonic = 'v_' + re.sub(r'_(w32|w64|gfx12)', '', builtin)
    text = asm.read_text()
    assert re.findall(r'v_wmma\w*', text) == [mnemonic]


def _rocm_gpu():
    tool = shutil.which('rocminfo') or '/opt/rocm/bin/rocminfo'
    if not Path(tool).exists():
        return None
    try:
        out = subprocess.run([tool], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r'Name:\s+(gfx\d+\w*)', out)
    return found.group(1) if found else None


def test_gpu_rdna3_s_wmma_multiplies_as_the_layouts_say(tmp_path):
    """On the local gfx1150: each lane's registers of A and B filled as the
    catalog's layouts say they hold the operands (`layouts.element_at`),
    duplicates included, and D read back the same way, against A @ B.

    Small integers, so a misplaced element is off by an integer.  Not to
    the bit: RDNA 3's WMMA does not round as IEEE does.  Measured here, with
    a negative product among the terms the sum comes out an f32 ulp low --
    1*1 + 0*(-0) gives 0.99999994, and -1 gives -1.0000001 -- with the
    kernel one load, one `v_wmma_f32_16x16x16_f16` and one store a lane."""
    if _rocm_gpu() != 'gfx1150':
        pytest.skip('no gfx1150 GPU')
    builtin = 'wmma_f32_16x16x16_f16_w32'
    _, a_t, b_t, d_t = CASES[builtin]
    op = _op(builtin)
    body, _ = _product(builtin, 'gfx1150')
    src = tmp_path / 'k.hip'
    kernel = _source(_line(body, 'gfx1150'), a_t, b_t, d_t, d_t[1])
    rng = np.random.default_rng(7)
    am = rng.integers(-3, 4, (16, 16)).astype(np.float32)
    bm = rng.integers(-3, 4, (16, 16)).astype(np.float32)
    afrag = np.zeros((32, 16), np.float16)
    bfrag = np.zeros((32, 16), np.float16)
    for t in range(32):
        for s in range(16):
            _, m, k = layouts.element_at(op, 'A', s, t)
            afrag[t, s] = am[m, k]
            _, k, n = layouts.element_at(op, 'B', s, t)
            bfrag[t, s] = bm[k, n]
    main = '''
#include <cstdio>
#include <vector>
int main(int argc, char **argv) {
  std::vector<_Float16> h(2 * 512);
  std::vector<float> d(256);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 2, 1024, in) != 1024) return 2;
  std::fclose(in);
  _Float16 *af, *bf;
  float *dout;
  if (hipMalloc(&af, 1024) || hipMalloc(&bf, 1024) || hipMalloc(&dout, 1024))
    return 4;
  hipMemcpy(af, h.data(), 1024, hipMemcpyHostToDevice);
  hipMemcpy(bf, h.data() + 512, 1024, hipMemcpyHostToDevice);
  product<<<1, 32>>>(af, bf, dout);
  if (hipDeviceSynchronize() != hipSuccess) return 5;
  hipMemcpy(d.data(), dout, 1024, hipMemcpyDeviceToHost);
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 4, 256, out) != 256) return 3;
  std::fclose(out);
  return 0;
}
'''
    src.write_text(kernel + main)
    exe = tmp_path / 'k'
    run = subprocess.run([_hipcc(), '-x', 'hip', '-std=c++17',
                          '--offload-arch=gfx1150', '-O2', '-I',
                          str(INCLUDE), '-o', str(exe), str(src)],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    (tmp_path / 'in.bin').write_bytes(afrag.tobytes() + bfrag.tobytes())
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    dregs = np.frombuffer((tmp_path / 'out.bin').read_bytes(),
                          np.float32).reshape(32, 8)
    got = np.full((16, 16), np.nan, np.float32)
    for t in range(32):
        for s in range(8):
            _, m, n = layouts.element_at(op, 'D', s, t)
            got[m, n] = dregs[t, s]
    assert np.abs(got - am @ bm).max() < 1e-3
