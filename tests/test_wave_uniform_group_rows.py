# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Each multiplication's row of a group, read as the wave's
(`wave_uniform_group_rows`, off by default).

Where the multiplications of one wave take their trips through the batch
loop together, the group's first element is the same on every lane of the
wave, and each multiplication's row is that plus its place in the group.
Read as the wave's, the loop counts on the scalar unit; and what a row reads
-- its flag word and the pointers of its element -- is loaded once per
multiplication, from a row the whole wave holds, every lane picking its own
(`pir.rows.group_rows`, `tensorforge::pick`).

The host oracle checks the numbers under flags that hold rows back, and it
checks the claim the loop rests on as well: a value read as the wave's has
to be the same on every lane of the wave that executes it
(`Interp._wave_uniform`).
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.strategy import Strategy
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

INCLUDE = Path(__file__).resolve().parent.parent / 'src' / 'tensorforge' / 'include'

#: The group traversal, asked for: the products are the generic ones, which
#: would not group the wave on their own.
GROUPED = {'wave_group_traversal': True}


@pytest.fixture(autouse=True)
def _generic_products(monkeypatch):
    monkeypatch.setattr(amd, 'ORDER', (Strategy.GENERIC,))


def _gemm(backend, arch, **options):
    """`C[b] = A[b] B` over arrays of pointers, with flags: 32 rows, half a
    wave of 64 on gfx942."""
    rows, k = 32, 9
    a, c = (Tensor([rows, k], Addressing.PTR_BASED,
                   BoundingBox([0, 0], [rows, k]), alias=alias,
                   datatype=Datatype.F32) for alias in 'AC')
    b = Tensor([k, k], Addressing.NONE, BoundingBox([0, 0], [k, k]),
               alias='B', datatype=Datatype.F32)
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                      options=Options(**options))
    generator = Generator([GemmDescr(False, False, SubTensor(a), SubTensor(b),
                                     SubTensor(c), alpha=1.0, beta=0.0)],
                          context)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        generator.generate()
    return generator


def _unnamed(source):
    """The source without the kernel's name and the options it spells."""
    source = re.sub(r'\bkernel_\w+', 'kernel', source)
    return re.sub(r'//[^\n]*', '', source)


#: Two multiplications to a wave of 64, and four.
SHARED = [(0, 2), (16, 4)]


def test_off_by_default():
    assert (_gemm('hip', 'gfx942', lanes_per_mult=16, **GROUPED).get_kernel()
            == _gemm('hip', 'gfx942', lanes_per_mult=16,
                     wave_uniform_group_rows=False, **GROUPED).get_kernel())


@pytest.mark.parametrize('lanes,group', SHARED, ids=['32-lanes', '16-lanes'])
def test_each_multiplication_reads_its_row_as_the_waves(lanes, group):
    """The loop from the group's first element as the wave's; a flag read
    per multiplication; and every mask and element pointer a pick among one
    per multiplication."""
    code = _unnamed(_gemm('hip', 'gfx942', lanes_per_mult=lanes,
                          wave_uniform_group_rows=True, **GROUPED)
                    .get_kernel())
    assert re.search(r'for \(size_t \w*batchIdGroup0 = '
                     r'tensorforge::wave_uniform<64, unsigned>\(', code)
    # Once per multiplication for each row the kernel binds: the first
    # group's, the next group's and the body's own.
    reads = re.findall(r'static_cast<bool>\(flags0\[', code)
    assert len(reads) == 3 * group
    mults = ', '.join(f'mult{m}_batchIdActive0' for m in range(group))
    assert re.search(rf'const bool batchIdActive0 = tensorforge::pick<{group}>'
                     rf'\(\w+, {mults}\);', code)
    # The first multiplication's row is the group's element itself, and the
    # first group's and the next group's rows are no lane's own any more.
    assert re.search(r'const bool mult0_batchIdActive0 = \w*batchIdGroup0 < '
                     r'numElements0', code)
    assert not re.search(r'\b(?:peel|wrap)_batchIdActive0 \?', code)
    for name in ('glb_m0', 'peel_glb_m1', 'wrap_glb_m1'):
        mults = ', '.join(f'mult{m}_{name}' for m in range(group))
        assert re.search(rf'const {name} = tensorforge::pick<{group}>'
                         rf'\(\w+, {mults}\);', code), name


@pytest.mark.parametrize('lanes,group', SHARED, ids=['32-lanes', '16-lanes'])
def test_every_element_pointer_is_a_multiplications_own(lanes, group):
    """`&m[batchId][off]` only in one multiplication's copy of a binding."""
    code = _unnamed(_gemm('hip', 'gfx942', lanes_per_mult=lanes,
                          wave_uniform_group_rows=True, **GROUPED)
                    .get_kernel())
    lines = [line for line in code.splitlines()
             if re.search(r'&(m\d+)\[[^\]]+\]\[0 \+ \1_extraOffset\]', line)]
    assert lines, 'no element pointer in the kernel'
    assert all(re.search(r' const mult\d+_\w*glb_m\d+ = ', line)
               for line in lines), lines
    assert len(lines) % group == 0


@pytest.mark.parametrize('backend,arch,options', [
    ('hip', 'gfx1150', {}),
    ('hip', 'gfx942', {'lanes_per_mult': 16}),
    ('hip', 'gfx942', {'lanes_per_mult': 16, 'stack_elements': 2, **GROUPED}),
    ('cuda', 'sm_86', {'lanes_per_mult': 16, **GROUPED}),
], ids=['a-wave-to-itself', 'rows-of-their-own', 'stacked', 'cuda'])
def test_where_it_does_not_apply(backend, arch, options):
    """A multiplication that fills its wave, which `wave_uniform_rows`
    serves; rows that take their own trips; a stack of elements a trip, whose
    masks are no row's; and CUDA, which has no first lane to read."""
    off = _gemm(backend, arch, **options).get_kernel()
    on = _gemm(backend, arch, wave_uniform_group_rows=True,
               **options).get_kernel()
    assert _unnamed(on) == _unnamed(off)
    assert 'tensorforge::pick' not in on


def _mask(mults: int):
    """A flag per element of three trips: row 1 held back on its first trip
    and not on its second, row 2 the other way round."""
    flags = [1] * (3 * mults)
    flags[1] = 0
    flags[2 + mults] = 0
    return flags


def _written(generator, races=None) -> dict:
    """What the kernel leaves in the operands it writes, over three trips of
    one block, under `_mask`: every multiplication of the block run, so that
    every wave is whole."""
    src = generator.get_kernel()
    lanes, mults = ke.launch_geometry(generator.get_launcher())
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    params = re.split(r',(?![^<]*>)', signature)
    outputs = {p.split()[-1] for p in params
               if ('*' in p or 'SpacePtr' in p)
               and not p.strip().startswith('const') and '<const ' not in p}
    mem = ke.evaluate_wave(src, lanes, seed=11, globals_only=True,
                           mults=mults, block=True, races=races,
                           elements=3 * mults,
                           preset={'flags0': _mask(mults)})
    written = {k: v for k, v in mem.items()
               if k[0].split('@')[0] in outputs}
    assert written, 'nothing written to compare'
    return written


@pytest.mark.parametrize('lanes,group', SHARED, ids=['32-lanes', '16-lanes'])
@pytest.mark.parametrize('wrap', [True, False], ids=['wrapped', 'unwrapped'])
def test_the_numbers_do_not_change(lanes, group, wrap):
    """Rows held back on one trip and not the next, the transfers issued a
    group ahead or not -- and the group's first element the same on each
    lane of the wave."""
    def build(**switch):
        return _gemm('hip', 'gfx942', lanes_per_mult=lanes,
                     enable_wrap_loads=wrap, **GROUPED, **switch)
    on = build(wave_uniform_group_rows=True)
    assert f'tensorforge::pick<{group}>(' in on.get_kernel()
    races = []
    assert _written(on, races) == _written(build())
    assert not races, races


# -- the oracle --------------------------------------------------------------- #

ORACLE_KERNEL = '''__global__ void kernel_rows(float * m0) {
  const unsigned r = tensorforge::wave_uniform<LANES>(threadIdx.y);
  const unsigned p = tensorforge::pick<2>(threadIdx.y % 2, 10, 20);
  m0[threadIdx.x + 32 * threadIdx.y] = r + p;
}
'''


def test_the_oracle_holds_a_wave_to_one_value():
    """Two multiplications of 32 lanes in a wave of 64 have two rows, and
    the first lane's is not the second multiplication's: the oracle stops."""
    with pytest.raises(ke.Abort, match='the lanes of the wave hold different'):
        ke.evaluate_wave(ORACLE_KERNEL.replace('LANES', '64'), 32, mults=2,
                         block=True)


def test_the_oracle_reads_a_value_of_one_multiplication():
    """Over the lanes of one multiplication the row is one, and `pick` takes
    the multiplication's own value."""
    mem = ke.evaluate_wave(ORACLE_KERNEL.replace('LANES', '32'), 32, mults=2,
                           block=True)
    assert mem[('m0', 0)] == 0 + 10
    assert mem[('m0', 32 + 5)] == 1 + 20


# -- the header ------------------------------------------------------------------ #

PICK = '''
#define __device__
#define __forceinline__ inline
#include <cstdio>
#include "tensorforge_device/hip_wave.h"
int main() {
  for (unsigned m = 0; m < 3; ++m) {
    std::printf("%d ", tensorforge::pick<3>(m, 10, 20, 30));
  }
  std::printf("%d\\n", tensorforge::pick<COUNT>(1u, 1.5, 2.5, 3.5) == 2.5);
}
'''


def _host_compiler():
    for name in ('c++', 'g++', 'clang++'):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def _host_run(source, tmp_path):
    compiler = _host_compiler()
    if compiler is None:
        pytest.skip('no host C++ compiler')
    path = tmp_path / 'pick.cpp'
    path.write_text(source)
    built = subprocess.run([compiler, '-std=c++17', f'-I{INCLUDE}', str(path),
                            '-o', str(tmp_path / 'pick')],
                           capture_output=True, text=True, timeout=120)
    if built.returncode != 0:
        return built, None
    return built, subprocess.run([str(tmp_path / 'pick')], capture_output=True,
                                 text=True, timeout=60)


def test_a_lane_picks_its_own_multiplications_value(tmp_path):
    """`pick` is plain C++, so the host runs it: each multiplication's value
    by its place."""
    built, ran = _host_run(PICK.replace('COUNT', '3'), tmp_path)
    assert ran is not None, built.stderr
    assert ran.stdout.split() == ['10', '20', '30', '1']


def test_one_value_per_multiplication(tmp_path):
    """A pick with a value too many does not compile."""
    built, _ = _host_run(PICK.replace('COUNT', '2'), tmp_path)
    assert built.returncode != 0
    assert 'one value per multiplication' in built.stderr


# -- what the compile makes of it -------------------------------------------- #

def _amdgpu_clang():
    """A clang that compiles HIP device code for AMDGPU, or None."""
    for name in ('clang++', 'clang'):
        path = shutil.which(name)
        if path is None:
            continue
        probe = subprocess.run(
            [path, '-x', 'hip', '--offload-arch=gfx942',
             '--offload-device-only', '-nogpulib', '-nogpuinc', '-S', '-o',
             '-', '-'], input='', capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            return path
    return None


#: The form the generator gives two multiplications of a wave of 64, with
#: the execution-space keywords the HIP runtime would supply.
KERNEL = '''
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __forceinline__ inline __attribute__((always_inline))
#include "tensorforge_device/hip_wave.h"
extern "C" __global__ void rows(const float *const *__restrict__ table,
                                const unsigned *__restrict__ flags,
                                float *out, unsigned n) {
  const unsigned y = __builtin_amdgcn_workitem_id_y();
  const unsigned x = __builtin_amdgcn_workitem_id_x();
  const unsigned place = y % 2;
  const unsigned group = tensorforge::wave_uniform<64, unsigned>(y - place);
  const unsigned row1 = group + 1;
  const bool active0 = group < n && static_cast<bool>(flags[group]);
  const bool active1 = row1 < n && static_cast<bool>(flags[row1]);
  const float *p0 = &table[group][0];
  const float *p1 = &table[active1 ? row1 : group][0];
  const float *p = tensorforge::pick<2>(place, p0, p1);
  if (tensorforge::pick<2>(place, active0, active1)) {
    out[x + 32 * y] = p[x];
  }
}
'''


def test_the_pointers_and_flags_are_scalar_loads(tmp_path):
    """From the group's first element as the wave's, each multiplication's
    pointer and flag word is loaded through the scalar cache, and a lane
    picks its own with a select."""
    clang = _amdgpu_clang()
    if clang is None:
        pytest.skip('no clang that compiles HIP device code for AMDGPU')
    source = tmp_path / 'rows.hip'
    source.write_text(KERNEL)
    built = subprocess.run(
        [clang, '-x', 'hip', '--offload-arch=gfx942', '--offload-device-only',
         '-nogpulib', '-nogpuinc', '-O3', '-std=c++17', '-S', f'-I{INCLUDE}',
         str(source), '-o', '-'], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stderr
    asm = built.stdout
    assert 's_load_dwordx2' in asm and 'global_load_dwordx2' not in asm
    assert re.search(r's_load_dword\s', asm)
    assert 'v_cndmask_b32' in asm
