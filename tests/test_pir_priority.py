# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Wave priorities on AMD (`Options.amd_setprio`, `pir.priority`): where the
markers stand in a kernel, and that nothing else changes."""
from __future__ import annotations

import contextlib
import io
import re

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

from test_rolling import gemm, make

SETPRIO = re.compile(r'__builtin_amdgcn_s_setprio\((\d)\);')


def _kernel(backend='hip', arch='gfx942', **opts):
    descrs = [gemm(make('A', [24, 24]), make('B', [24, 5]), make('C', [24, 5])),
              gemm(make('C', [24, 5]), make('D', [5, 5]), make('E', [24, 5]))]
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32,
                                    options=Options(**opts)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _without_priorities(src):
    # The kernel's name carries a digest of its options.
    lines = [re.sub(r'kernel_\w+', 'kernel', ln) for ln in src.splitlines()
             if 'setprio' not in ln and 'tensorforge-meta' not in ln
             and '// options:' not in ln]
    return '\n'.join(lines)


def test_off_by_default():
    assert not SETPRIO.search(_kernel())


def test_loads_raise_a_wave_around_its_transfers():
    """High ahead of a transfer and low behind it, the two alternating; the
    kernel otherwise the same statement for statement."""
    src = _kernel(amd_setprio='loads')
    levels = [int(m.group(1)) for m in SETPRIO.finditer(src)]
    assert levels and levels[0] == 2
    assert all(a != b for a, b in zip(levels, levels[1:]))
    # Every raised stretch holds a load of global memory.  (A load the
    # forward pass took from a register keeps its comment and is no
    # transfer, so it stands unraised.)
    for stretch in re.findall(r's_setprio\(2\);(.*?)s_setprio\(0\);', src,
                              re.S):
        assert 'load{g>' in stretch and ('load' in stretch.split('\n', 2)[-1])
    assert _without_priorities(src) == _without_priorities(_kernel())


def test_compute_is_the_reverse():
    levels = [int(m.group(1)) for m in
              SETPRIO.finditer(_kernel(amd_setprio='compute'))]
    assert levels[0] == 2 and levels[1] == 0


def test_alternate_raises_every_other_wave_once():
    src = _kernel(amd_setprio='alternate')
    assert len(SETPRIO.findall(src)) == 1
    assert 'readfirstlane' in src


@pytest.mark.parametrize('mode', ['loads', 'alternate'])
def test_not_on_nvidia(mode):
    assert not SETPRIO.search(_kernel(backend='cuda', arch='sm_90',
                                      amd_setprio=mode))


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        _kernel(amd_setprio='sideways')
