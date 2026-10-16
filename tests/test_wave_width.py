# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The collectives of `hip.h` by the width of the wave (`WaveSize`).

A block that is the whole wave is broadcast with `readlane`, a scalar read; a
narrower one by the DPP, swizzle and permute forms, which work per row or per
32 lanes alike in a wave of either width.  Nothing reaches past the wave:
the instructions that move a value between lanes end there, and a block wider
than the wave does not compile.  A boolean reduction counts a lane's bit in
the ballot by its place in the wave -- with two multiplications in one wave,
not by `threadIdx.x`.

What the compiler makes of it is in the instruction stream, so these compile
`hip.h` for the device: with a clang that targets AMDGPU and the headers of a
HIP installation (`TF_HIP_INCLUDE`, a path list, or `$ROCM_PATH/include`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

INCLUDE = Path(__file__).resolve().parent.parent / 'src' / 'tensorforge' / 'include'


def _hip_includes():
    """The include directories of a HIP installation, or None."""
    listed = os.environ.get('TF_HIP_INCLUDE')
    candidates = ([Path(p) for p in listed.split(os.pathsep) if p] if listed
                  else [Path(os.environ.get('ROCM_PATH', '/opt/rocm')) / 'include'])
    if any((d / 'hip' / 'hip_runtime.h').exists() for d in candidates):
        return candidates
    return None


def _clang():
    for name in ('clang++', 'clang'):
        path = shutil.which(name)
        if path is None:
            continue
        probe = subprocess.run(
            [path, '-x', 'hip', '--offload-arch=gfx942', '--offload-device-only',
             '-nogpulib', '-nogpuinc', '-S', '-o', '-', '-'],
            input='', capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            return path
    return None


def _wrapper(clang):
    """clang's own HIP runtime wrapper, which `-nogpuinc` leaves out."""
    res = subprocess.run([clang, '-print-resource-dir'], capture_output=True,
                         text=True, timeout=60)
    wrapper = Path(res.stdout.strip()) / 'include' / '__clang_hip_runtime_wrapper.h'
    return wrapper if wrapper.exists() else None


@pytest.fixture(scope='module')
def compile_hip(tmp_path_factory):
    clang = _clang()
    if clang is None:
        pytest.skip('no clang that compiles HIP device code for AMDGPU')
    includes = _hip_includes()
    if includes is None:
        pytest.skip('no HIP headers (TF_HIP_INCLUDE or $ROCM_PATH/include)')
    wrapper = _wrapper(clang)
    if wrapper is None:
        pytest.skip('no __clang_hip_runtime_wrapper.h in clang\'s resource dir')
    root = tmp_path_factory.mktemp('wave_width')

    def run(arch, body):
        source = root / f'{arch}_{abs(hash(body))}.hip'
        source.write_text('#include "tensorforge_device/hip.h"\n'
                          'extern "C" __global__ void k(float *out, '
                          'const float *in, bool *flag) {\n'
                          '  const unsigned i = threadIdx.x + blockDim.x * '
                          'threadIdx.y;\n' + body + '\n}\n')
        return subprocess.run(
            [clang, '-x', 'hip', f'--offload-arch={arch}',
             '--offload-device-only', '-nogpulib', '-nogpuinc', '-O3',
             '-std=c++17', '-S', '-D__HIP_PLATFORM_AMD__', '-include',
             str(wrapper), *[f'-I{d}' for d in includes], f'-I{INCLUDE}',
             str(source), '-o', '-'],
            capture_output=True, text=True, timeout=300)
    return run


BROADCAST = '  out[i] = tensorforge::broadcast<{block}, 1, 3>(in[i]);'


@pytest.mark.parametrize('arch,block,expect,absent', [
    ('gfx1150', 32, 'v_readlane_b32', 'ds_swizzle'),
    ('gfx942', 64, 'v_readlane_b32', 'ds_swizzle'),
    ('gfx942', 32, 'ds_swizzle_b32', 'v_readlane'),
], ids=['gfx1150-wave', 'gfx942-wave', 'gfx942-half'])
def test_a_block_that_is_the_wave_reads_its_lane(compile_hip, arch, block,
                                                 expect, absent):
    """`readlane` for the whole wave, and only there: half a wave of 64 is
    two blocks, each with its own lane, which one scalar read cannot serve."""
    built = compile_hip(arch, BROADCAST.format(block=block))
    assert built.returncode == 0, built.stderr
    assert expect in built.stdout
    assert absent not in built.stdout


@pytest.mark.parametrize('body,what', [
    (BROADCAST.format(block=64), 'a broadcast reaches the lanes of one wave'),
    ('  out[i] = tensorforge::swap<64>(in[i]);',
     'a swap reaches the lanes of one wave'),
], ids=['broadcast', 'swap'])
def test_a_block_wider_than_the_wave_does_not_compile(compile_hip, body, what):
    """64 lanes are two waves on gfx1150, and no lane instruction crosses."""
    built = compile_hip('gfx1150', body)
    assert built.returncode != 0
    assert what in built.stderr


def test_a_ballot_bit_is_the_lanes_place_in_the_wave(compile_hip):
    """Two 32-lane multiplications in a wave of 64 count `threadIdx.x` from 0
    alike; the lane's bit in the ballot is its place in the wave, the lane id
    `v_mbcnt` counts."""
    built = compile_hip(
        'gfx942',
        '  using And = tensorforge::ReductionOperation<bool, '
        'tensorforge::Operation::And>;\n'
        '  flag[i] = tensorforge::reduction<And, 32, 1, bool>(in[i] > 0.0f);')
    assert built.returncode == 0, built.stderr
    assert 'v_mbcnt_hi_u32_b32' in built.stdout
