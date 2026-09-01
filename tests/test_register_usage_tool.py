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
    tu = ru.BACKENDS['hip'].translation_unit('__global__ void k() {}')
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


# ----------------------------------------------------------------------
# nvcc, whose format is different and equally unguessable
# ----------------------------------------------------------------------

PTXAS = """\
ptxas info    : 218125 bytes gmem, 920 bytes cmem[3]
ptxas info    : Compiling entry function '_Z6kernelPf' for 'sm_80'
ptxas info    : Function properties for _Z6kernelPf
    0 bytes stack frame, 12 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 93 registers, 7136 bytes smem, 432 bytes cmem[0], 64 bytes cmem[2]
"""


def test_the_register_count_is_written_the_other_way_round():
    """`Used 93 registers`, where everything else is `<n> bytes <what>`.

    Two patterns, not one, and a single pattern over `<n> <unit> <what>` would
    silently drop the field the whole tool is about.
    """
    assert ru.parse_ptxas(PTXAS)['vgprs'] == 93


def test_the_registers_land_under_the_amd_key():
    """NVIDIA has one register file where CDNA has two.

    The report compares on `vgprs + agprs`, so putting the count under `vgprs`
    and leaving `agprs` absent makes that read correctly with no per-vendor
    case at the one place the numbers are used.
    """
    f = ru.parse_ptxas(PTXAS)
    assert 'vgprs' in f and 'agprs' not in f


def test_spill_stores_and_loads_collapse_to_one_figure():
    """They are two views of the same traffic; the larger is the honest one."""
    assert ru.parse_ptxas(PTXAS)['vgprsspill'] == 12


def test_the_cmem_numbers_are_not_mistaken_for_shared_memory():
    """`432 bytes cmem[0]` is constant memory, and is not a resource here."""
    f = ru.parse_ptxas(PTXAS)
    assert f['ldssize'] == 7136


def test_a_clean_ptxas_build_still_reports_the_registers():
    clean = PTXAS.replace('12 bytes spill stores, 12 bytes spill loads',
                          '0 bytes spill stores, 0 bytes spill loads')
    f = ru.parse_ptxas(clean)
    assert f['vgprs'] == 93 and 'vgprsspill' not in f


# ----------------------------------------------------------------------
# Intel, which answers less, and has to say so rather than say zero
# ----------------------------------------------------------------------

def test_an_intel_build_reports_spills_and_no_register_count():
    """IGC puts the count in a shader dump, not on the command line.

    `IGC_ShaderDumpEnable=1` writes `.asm` files under `/tmp/IntelIGC`, which
    is a directory to scrape rather than a stream to read, and whose format
    moves with the driver.  What reaches the command line is the spill
    warning, and that is the signal that decides whether a configuration blew
    the register file.
    """
    err = ("warning: kernel _ZTS6kernel  compiled SIMD16 allocated 128 regs "
           "and spilled around 384 bytes\n")
    f = ru.parse_igc(err)
    assert f.get('vgprsspill') == 384
    assert 'vgprs' not in f, (
        'a register count that was never reported must stay absent, not '
        'become zero -- a caller comparing configurations on a missing '
        'number would rank them equal instead of declining to rank them')


def test_a_silent_intel_build_is_not_an_error():
    """Nothing to say means it did not spill, which is the good case."""
    assert ru.parse_igc('') == {}


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
