# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the kernel tells the compiler about its block.

With only the thread count in `__launch_bounds__`, ptxas -O3 sizes registers
for an occupancy it guesses: SeisSol's viscoelastic time derivative got 40 of
255, spilled 58 KB and ran five times slower than with one block per SM
stated.  The second argument is CUDA's; HIP reads it as warps per execution
unit, so it stays out there.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator


def _bounds(arch, backend, **options):
    path = next((Path(__file__).parent / 'cases').rglob('local_flux.py'))
    spec = importlib.util.spec_from_file_location('tf_bounds__local_flux', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    gen = Generator(case.descr_list(), Context(
        arch=arch, backend=backend, fp_type=case.DTYPE,
        options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    threads = (gen.launch_config().threads_per_mult
               * gen.launch_config().mults_per_block)
    found = re.findall(r'__launch_bounds__\(([^)]*)\)', gen.get_kernel())
    assert found
    return threads, found


def test_cuda_states_one_block_per_sm():
    threads, found = _bounds('sm_120', 'cuda')
    assert set(found) == {f'{threads}, 1'}


def test_zero_leaves_the_thread_count_alone():
    threads, found = _bounds('sm_120', 'cuda', min_blocks_per_sm=0)
    assert set(found) == {f'{threads}'}


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1150'])
def test_hip_keeps_its_own_reading_of_the_second_argument(arch):
    threads, found = _bounds(arch, 'hip')
    assert set(found) == {f'{threads}'}
