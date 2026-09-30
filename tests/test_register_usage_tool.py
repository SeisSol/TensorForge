# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`tools/register_usage.py` without a toolchain: its translation units, its
command lines, and the comparison its report makes.

What the compilers print is read by `tensorforge.toolchain`, and pinned in
`test_toolchain.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'register_usage', ROOT / 'tools' / 'register_usage.py')
ru = importlib.util.module_from_spec(_spec)
# Registered before execution: the module defines a dataclass, and
# `dataclasses` resolves annotations through `sys.modules[cls.__module__]`,
# which is not there for a module loaded by path alone.
sys.modules[_spec.name] = ru
_spec.loader.exec_module(ru)


def test_the_translation_unit_uses_the_real_device_header():
    """Not the host shim `tests/harness/syntax.py` puts on top.

    That shim exists so a host compiler accepts device code, which is right
    for a syntax check and wrong here: a register count only means something
    if the device compiler saw what it will actually see.
    """
    tu = ru.BACKENDS['hip'].translation_unit('__global__ void k() {}')
    assert 'hip/hip_runtime.h' in tu
    assert 'tensorforge_device/hip.h' in tu
    assert 'shim' not in tu


def test_the_comparison_counts_both_register_files(capsys):
    """CDNA has two, and a kernel can be tight in either.

    Comparing on VGPRs alone would call a configuration cheaper for having
    moved its accumulator into AGPRs, which is not a saving.  Here the model
    prefers the 16-lane build; on VGPRs alone the compiler would too, and
    with the AGPRs counted it prefers the 32-lane one.
    """
    rows = [ru.Measurement('k', 16, 100, 100, 1, vgprs=100, agprs=50),
            ru.Measurement('k', 32, 200, 200, 1, vgprs=120, agprs=0)]
    ru.report(rows, verbose=False)
    assert 'rank agreement model/compiler: 0/1' in capsys.readouterr().out


@pytest.mark.parametrize("name,arch", [("hip", "gfx90a"), ("cuda", "sm_80"),
                                       ("sycl", "pvc")])
def test_each_backend_builds_a_command_and_its_own_headers(name, arch):
    """And each asks its own compiler for its own flag.

    Held together because the three differ in every part -- the flag, the
    header, the way the architecture is named -- and a shared function with
    three branches inside it is where those quietly drift into each other.
    """
    b = ru.BACKENDS[name]
    cmd = b.command('cc', arch, Path('k.cpp'), Path('k.o'), Path('/inc'), [])
    assert cmd[0] == 'cc' and str(Path('k.cpp')) in cmd
    assert any(arch in part for part in cmd)
    assert 'tensorforge_device' in b.translation_unit('void k() {}')
    assert 'shim' not in b.translation_unit('void k() {}')
