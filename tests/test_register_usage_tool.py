# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The one part of `tools/register_usage.py` that cannot be tried out here.

Everything else in that tool runs without a toolchain: generation, the model
figure, the report.  Reading hipcc's resource remarks cannot, and a parser
written against a remembered format is a parser that silently returns nothing.
Silently, because the tool's failure mode is an empty field rather than an
error -- it would print a table of dashes and conclude that nothing correlates.

So the format is pinned here, from the shape AMDGPU actually emits, with the
three traps that a name list walks into: the unit in brackets *before* the
colon, the two separate spill fields, and the non-numeric remarks in between.
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


#: One kernel's worth, as `-Rpass-analysis=kernel-resource-usage` prints it.
REMARKS = """\
k.hip.cpp:12:1: remark: Function Name: _Z13kernel_abc123Pf [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     SGPRs: 34 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     VGPRs: 148 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     Occupancy [waves/SIMD]: 3 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     VGPRs Spill: 12 [-Rpass-analysis=kernel-resource-usage]
k.hip.cpp:12:1: remark:     LDS Size [bytes/block]: 1792 [-Rpass-analysis=kernel-resource-usage]
"""


def test_every_numeric_field_is_read():
    f = ru.parse_remarks(REMARKS)
    assert f == {
        'sgprs': 34,
        'vgprs': 148,
        'agprs': 0,
        'scratchsize': 0,
        'occupancy': 3,
        'sgprsspill': 0,
        'vgprsspill': 12,
        'ldssize': 1792,
    }


def test_the_bracketed_unit_does_not_hide_the_number():
    """`ScratchSize [bytes/lane]: 0` -- the colon is not next to the name.

    A pattern listing field names followed by a colon misses exactly the three
    fields that carry units, which are scratch, occupancy and LDS.  Scratch is
    the one that says a kernel spilled.
    """
    f = ru.parse_remarks(REMARKS)
    assert 'scratchsize' in f and 'occupancy' in f and 'ldssize' in f


def test_the_two_spill_fields_are_kept_apart():
    """There is no `SpillCount`; there are `SGPRs Spill` and `VGPRs Spill`.

    And `SGPRs Spill:` must not be read as `SGPRs:` -- doing so would report
    the spill count as the register count, which is a number in the right
    range and the wrong meaning.
    """
    f = ru.parse_remarks(REMARKS)
    assert f['sgprs'] == 34 and f['sgprsspill'] == 0
    assert f['vgprs'] == 148 and f['vgprsspill'] == 12


def test_non_numeric_remarks_are_skipped():
    """`Function Name:` and `Dynamic Stack: False` are not measurements."""
    f = ru.parse_remarks(REMARKS)
    assert 'functionname' not in f
    assert 'dynamicstack' not in f


def test_several_kernels_report_the_largest():
    """A translation unit may hold more than one, and the budget is per kernel.

    The maximum is the only reading that cannot understate what the hardware
    has to fit.
    """
    two = REMARKS + REMARKS.replace('VGPRs: 148', 'VGPRs: 96')
    assert ru.parse_remarks(two)['vgprs'] == 148


def test_nothing_parsed_is_distinguishable_from_zero():
    """Which is what the caller turns into a diagnosable error.

    An empty dict means the format was not recognised; a dict of zeros means
    the kernel is free.  Conflating them is how a tool reports that nothing
    correlates when in fact nothing was measured.
    """
    assert ru.parse_remarks('') == {}
    assert ru.parse_remarks('k.cpp:1:1: error: no such file') == {}


def test_the_translation_unit_uses_the_real_device_header():
    """Not the host shim `tests/harness/syntax.py` puts on top.

    That shim exists so a host compiler accepts device code, which is right
    for a syntax check and wrong here: a register count only means something
    if the device compiler saw what it will actually see.
    """
    tu = ru.translation_unit('__global__ void k() {}')
    assert 'hip/hip_runtime.h' in tu
    assert 'tensorforge_device/hip.h' in tu
    assert 'shim' not in tu


@pytest.mark.parametrize("field", ["vgprs", "agprs"])
def test_the_comparison_counts_both_register_files(field):
    """CDNA has two, and a kernel can be tight in either.

    Comparing on VGPRs alone would call a configuration cheaper for having
    moved its accumulator into AGPRs, which is not a saving.
    """
    import inspect
    src = inspect.getsource(ru.report)
    assert f'.{field}' in src
