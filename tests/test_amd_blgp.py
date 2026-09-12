# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lead operand handed between the multiplications of a wave (`blgp`).

At 32 or 16 lanes a wave holds two or four multiplications, and a
batch-constant lead operand gives each of them the same rows.  So each reads a
different contraction step -- the `p`-th reads `k + p` -- and every MFMA of the
group takes its step from those lanes through `blgp`.  Nothing here runs an
MFMA; the arrangement is checked, the numbers come from a CDNA package.
"""
import contextlib
import importlib.util
import io
import re
import warnings

import pytest

from test_amd_packed_dpp import CASES, _kernel

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.strategy import (ComputeShape,
                                                               Strategy)
from tensorforge.backend.pir.core import Uniformity
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

#: One lane-batched MFMA and its `cbsz`, `abid` and `blgp`.
MFMA = re.compile(r'__builtin_amdgcn_mfma_f32_4x4x1f32\([^;]*?, (\d+), (\d+), '
                  r'(\d+)\);')


def _blgps(src):
    return [int(m.group(3)) for m in MFMA.finditer(src)]


def _kernel_at(name, arch, threads):
    """`_kernel` at a lane count of the caller's rather than the deduced one."""
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_blgp__' + name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        ctx = Context(arch=arch, backend='hip', fp_type=case.DTYPE,
                      options=Options())
        gen = Generator(case.descr_list(), ctx,
                        lanes=LaneConfig(num_threads=threads,
                                         num_active_threads=threads,
                                         lead_width=1))
        gen.generate()
    return gen.get_kernel()


def test_two_or_four_multiplications_share_a_wave_of_the_same_rows():
    assert amd.wave_mults(32, True) == 2
    assert amd.wave_mults(16, True) == 4
    # A whole wave, and eight to one, which `blgp` has no pattern for.
    assert amd.wave_mults(64, True) == 1
    assert amd.wave_mults(8, True) == 1
    # Rows that differ between the multiplications are not the same rows.
    assert amd.wave_mults(32, False) == 1
    # A packed lead operand, and F64, whose field is the negation (CDNA3).
    assert amd.wave_mults(32, True, width=2) == 1
    assert amd.wave_mults(32, True, dtype=Datatype.F64) == 1


def test_the_multiplications_handing_it_on_run_in_step():
    """Every multiplication of the wave has to have read its step when the
    MFMA issues, so the batch loop drives the wave together."""
    shape = ComputeShape(threads=32, accumulator=Datatype.F32, sparse=False,
                         explicit_simd=False, a_uniform=True)
    assert amd.convergence(Strategy.MATRIX, shape) is Uniformity.MULTGROUP
    assert amd.convergence(Strategy.DPP, shape) is None
    per_element = ComputeShape(threads=32, accumulator=Datatype.F32,
                               sparse=False, explicit_simd=False)
    assert amd.convergence(Strategy.MATRIX, per_element) is None


@pytest.mark.parametrize('arch', ['gfx942', 'gfx90a'])
def test_a_batch_constant_operator_is_read_once_per_wave(arch):
    """`local_flux`: 32 lanes, so two multiplications a wave, each step read
    by one of them and handed to both -- `blgp` 1 and 2 alike.  The steps a
    contraction has left over read as before (`blgp` 0).  The loop is the
    wave group's, fenced at its head: it has no guard around the body, and
    without one LLVM held the staged operators across it."""
    src = _kernel('local_flux', arch)
    blgps = _blgps(src)
    assert blgps.count(1) == blgps.count(2) > 0
    assert set(blgps) <= {0, 1, 2}
    assert '(threadIdx.y % 2) * 56' in src, 'the p-th reads step k + p'
    assert 'batchIdGroup0' in src
    assert '__builtin_amdgcn_sched_barrier(0);' in src


def test_four_multiplications_take_a_quarter_of_the_wave_each():
    """At 16 lanes, four multiplications a wave: `blgp` 4 to 7, one group of
    16 lanes broadcast to all 64."""
    src = _kernel_at('local_flux', 'gfx942', 16)
    blgps = _blgps(src)
    assert blgps.count(4) == blgps.count(5) == blgps.count(6) \
        == blgps.count(7) > 0
    assert set(blgps) <= {0, 4, 5, 6, 7}
    assert '(threadIdx.y % 4) * 56' in src
    assert 'batchIdGroup0' in src


def test_without_the_switch_every_multiplication_reads_every_step(monkeypatch):
    monkeypatch.setattr(amd, 'B_DUPLICATION', False)
    src = _kernel('local_flux', 'gfx942')
    assert _blgps(src) and set(_blgps(src)) == {0}
    assert 'batchIdGroup0' not in src
