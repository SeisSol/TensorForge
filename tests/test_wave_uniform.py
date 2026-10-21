# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The row of the block, read once as the wave's (`wave_uniform_rows`).

Where every multiplication is a whole number of waves, `threadIdx.y` is the
same on every lane of a wave -- and the compiler cannot see it, since it does
not know the block.  Read once through `readfirstlane`, the row is held in a
scalar register, and with it the element index, the flags read and the
address of every element pointer.  The pointer itself is loaded from memory,
so it is read as the wave's as well (`GetElementPtr._element_pointer`).

`tensorforge::wave_uniform<Lanes>` reads the first lane only where `Lanes` is
a whole number of waves, decided with `if constexpr` on `WaveSize`: 64 lanes
on gfx9 and earlier, 32 from gfx10 on, what HIP builds.  A compile for another
width -- `-mwavefrontsize64` on gfx11 -- stops where clang names its width at
compile time.  The instruction stream shows both, where a clang that targets
AMDGPU is at hand.
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

ROW = re.compile(r'const unsigned tfMult = '
                 r'tensorforge::wave_uniform<(\d+), unsigned>\(threadIdx\.y\);')


def _gemm(backend, arch, rows, **options):
    """`C[b] = A[b] B` over arrays of pointers, `rows` the lead extent, and `B`
    the batch's alike -- staged by the whole block, which numbers its threads
    by the row as well."""
    k = 9
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


def _code(generator):
    """The kernel without its comments, which spell the options."""
    return re.sub(r'//[^\n]*', '', generator.get_kernel())


#: A multiplication that is a whole number of waves: 32 lanes on a 32-wide
#: wave, and 64 lanes, asked for, on a 64-wide one.
WHOLE = [('gfx1150', 32, {}), ('gfx942', 64, {'lanes_per_mult': 64})]


@pytest.mark.parametrize('arch,rows,options', WHOLE, ids=['gfx1150', 'gfx942'])
def test_a_multiplication_of_whole_waves_reads_its_row_once(arch, rows,
                                                            options):
    """One declaration, the lanes of the multiplication with it, and no other
    `threadIdx.y` in the kernel: every reader takes the row as the wave's."""
    generator = _gemm('hip', arch, rows, **options)
    code = _code(generator)
    lanes, _ = ke.launch_geometry(generator.get_launcher())
    assert [int(n) for n in ROW.findall(code)] == [lanes]
    assert code.count('threadIdx.y') == 1


@pytest.mark.parametrize('arch,rows,options', WHOLE, ids=['gfx1150', 'gfx942'])
def test_the_element_pointers_are_read_as_the_waves(arch, rows, options):
    """Every `&m[batchId][off]` of an operand passed as an array of pointers
    goes through `wave_uniform`, and with the multiplication's lanes."""
    generator = _gemm('hip', arch, rows, **options)
    code = _code(generator)
    lanes, _ = ke.launch_geometry(generator.get_launcher())
    pointers = [m.start() for m in re.finditer(
        r'&(m\d+)\[[^\]]+\]\[0 \+ \1_extraOffset\]', code)]
    assert pointers, 'no element pointer in the kernel'
    wrap = f'tensorforge::wave_uniform<{lanes}>('
    assert all(code[:at].endswith(wrap) for at in pointers)


def test_the_row_comes_ahead_of_the_arena():
    """A multiplication's window into shared memory starts at the row times
    its share, so the row is declared before the first window."""
    code = _code(_gemm('hip', 'gfx1150', 32))
    declaration = ROW.search(code)
    named = declaration.start() + len('const unsigned ')
    uses = [m.start() for m in re.finditer(r'\btfMult\b', code)
            if m.start() != named]
    assert re.search(r'&totalShrMem\[\d+ \* tfMult\b', code), (
        'expected a multiplication\'s window addressed by the row')
    assert uses and min(uses) > declaration.end()


@pytest.mark.parametrize('backend,arch,rows,options', [
    ('hip', 'gfx942', 32, {}),
    ('cuda', 'sm_86', 32, {}),
    ('hip', 'gfx1150', 32, {'wave_uniform_rows': False}),
], ids=['rows-sharing-a-wave', 'cuda', 'switched-off'])
def test_the_row_stays_the_hardwares(backend, arch, rows, options):
    """Where several multiplications share a wave the row differs between
    them; CUDA has no `readfirstlane`; and the switch turns it off."""
    code = _code(_gemm(backend, arch, rows, **options))
    assert 'wave_uniform' not in code
    assert 'tfMult' not in code


@pytest.mark.parametrize('arch,rows,options', WHOLE, ids=['gfx1150', 'gfx942'])
def test_the_numbers_do_not_change(arch, rows, options, monkeypatch):
    """Through the host oracle, which runs no matrix fragment product and no
    DPP accumulation, so the products are the generic ones: the kernel
    computes with the row read as the wave's what it computes without."""
    monkeypatch.setattr(amd, 'ORDER', (Strategy.GENERIC,))

    def written(**switch):
        generator = _gemm('hip', arch, rows, **options, **switch)
        src = generator.get_kernel()
        lanes, mults = ke.launch_geometry(generator.get_launcher())
        mem = ke.evaluate_wave(src, lanes, seed=3, globals_only=True,
                               mults=mults)
        signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
        outputs = {p.split()[-1] for p in signature.split(',')
                   if '*' in p and not p.strip().startswith('const')
                   and '<const ' not in p}
        out = {key: v for key, v in mem.items()
               if key[0].split('@')[0] in outputs}
        assert out, 'nothing written to compare'
        return out

    assert written() == written(wave_uniform_rows=False)


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


#: The execution-space keywords the HIP runtime would supply; the header
#: needs nothing else of it.
KERNEL = '''
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __forceinline__ inline __attribute__((always_inline))
#include "tensorforge_device/hip_wave.h"
extern "C" __global__ void row(unsigned *out) {
  out[__builtin_amdgcn_workitem_id_x()] =
      tensorforge::wave_uniform<LANES>(__builtin_amdgcn_workitem_id_y());
}
'''


def _compile(clang, arch, lanes, tmp_path, flags=()):
    source = tmp_path / 'row.hip'
    source.write_text(KERNEL.replace('LANES', str(lanes)))
    return subprocess.run(
        [clang, '-x', 'hip', f'--offload-arch={arch}', *flags,
         '--offload-device-only', '-nogpulib', '-nogpuinc', '-O3',
         '-std=c++17', '-S', f'-I{INCLUDE}', str(source), '-o', '-'],
        capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('arch,lanes,first_lane', [
    ('gfx1150', 32, True),
    ('gfx1150', 16, False),
    ('gfx1150', 64, True),
    ('gfx942', 32, False),
    ('gfx942', 64, True),
], ids=['gfx1150-32', 'gfx1150-16', 'gfx1150-64', 'gfx942-32', 'gfx942-64'])
def test_the_compile_reads_the_first_lane_of_whole_waves_only(
        arch, lanes, first_lane, tmp_path):
    """`readfirstlane` where the lanes are a whole number of waves -- 64
    lanes are two on gfx1150 -- and nowhere else."""
    clang = _amdgpu_clang()
    if clang is None:
        pytest.skip('no clang that compiles HIP device code for AMDGPU')
    built = _compile(clang, arch, lanes, tmp_path)
    assert built.returncode == 0, built.stderr
    assert ('v_readfirstlane_b32' in built.stdout) == first_lane


def _names_its_width_in_the_macro_only(clang):
    """Whether this clang says its wave width as `__AMDGCN_WAVEFRONT_SIZE`
    and has no builtin for it, which is where `hip_wave.h` checks it."""
    probe = subprocess.run(
        [clang, '-x', 'hip', '--offload-arch=gfx1150', '--offload-device-only',
         '-nogpulib', '-nogpuinc', '-E', '-'],
        input='#if __has_builtin(__builtin_amdgcn_wavefrontsize)\nbuiltin\n'
              '#endif\n',
        capture_output=True, text=True, timeout=60)
    return probe.returncode == 0 and 'builtin' not in probe.stdout


def test_a_compile_for_another_width_stops(tmp_path):
    """`-mwavefrontsize64` on gfx11 is not what HIP builds, and `WaveSize`
    would take it for 32: the compile stops at the check instead."""
    clang = _amdgpu_clang()
    if clang is None:
        pytest.skip('no clang that compiles HIP device code for AMDGPU')
    if not _names_its_width_in_the_macro_only(clang):
        pytest.skip('this clang names its wave width through the builtin')
    built = _compile(clang, 'gfx1150', 32, tmp_path,
                     flags=['-mwavefrontsize64'])
    assert built.returncode != 0
    assert 'a wave of another width' in built.stderr
