# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the vendor compilers report, and where they are found.

`tensorforge.toolchain` reads the compilers' own account of a kernel for the
autotuner, the benchmark builds and `tools/register_usage.py` alike.  None of
that can be tried out here, and a parser written against a remembered format
is a parser that silently returns nothing -- an empty field rather than an
error.  So each format is pinned from the shape the compiler actually emits,
with the traps a name list walks into.
"""

from __future__ import annotations

import pytest

from tensorforge import toolchain


# ----------------------------------------------------------------------
# AMDGPU resource remarks
# ----------------------------------------------------------------------

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
    f = toolchain.remark_fields(REMARKS)
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
    f = toolchain.remark_fields(REMARKS)
    assert 'scratchsize' in f and 'occupancy' in f and 'ldssize' in f


def test_the_two_spill_fields_are_kept_apart():
    """There is no `SpillCount`; there are `SGPRs Spill` and `VGPRs Spill`.

    And `SGPRs Spill:` must not be read as `SGPRs:` -- doing so would report
    the spill count as the register count, which is a number in the right
    range and the wrong meaning.
    """
    f = toolchain.remark_fields(REMARKS)
    assert f['sgprs'] == 34 and f['sgprsspill'] == 0
    assert f['vgprs'] == 148 and f['vgprsspill'] == 12


def test_non_numeric_remarks_are_skipped():
    """`Function Name:` and `Dynamic Stack: False` are not measurements."""
    f = toolchain.remark_fields(REMARKS)
    assert 'functionname' not in f
    assert 'dynamicstack' not in f


def test_several_kernels_report_the_largest():
    """A translation unit may hold more than one, and the budget is per kernel.

    The maximum is the only reading that cannot understate what the hardware
    has to fit.
    """
    two = REMARKS + REMARKS.replace('VGPRs: 148', 'VGPRs: 96')
    assert toolchain.remark_fields(two)['vgprs'] == 148


def test_nothing_parsed_is_distinguishable_from_zero():
    """Which is what the caller turns into a diagnosable error.

    An empty dict means the format was not recognized; a dict of zeros means
    the kernel is free.  Conflating them is how a tool reports that nothing
    correlates when in fact nothing was measured.
    """
    assert toolchain.remark_fields('') == {}
    assert toolchain.remark_fields('k.cpp:1:1: error: no such file') == {}


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
    assert toolchain.ptxas_fields(PTXAS)['vgprs'] == 93


def test_the_registers_land_under_the_amd_key():
    """NVIDIA has one register file where CDNA has two.

    The report compares on `vgprs + agprs`, so putting the count under `vgprs`
    and leaving `agprs` absent makes that read correctly with no per-vendor
    case at the one place the numbers are used.
    """
    f = toolchain.ptxas_fields(PTXAS)
    assert 'vgprs' in f and 'agprs' not in f


def test_spill_stores_and_loads_collapse_to_one_figure():
    """They are two views of the same traffic; the larger is the honest one."""
    assert toolchain.ptxas_fields(PTXAS)['vgprsspill'] == 12


def test_the_cmem_numbers_are_not_mistaken_for_shared_memory():
    """`432 bytes cmem[0]` is constant memory, and is not a resource here."""
    f = toolchain.ptxas_fields(PTXAS)
    assert f['ldssize'] == 7136


def test_a_clean_ptxas_build_still_reports_the_registers():
    clean = PTXAS.replace('12 bytes spill stores, 12 bytes spill loads',
                          '0 bytes spill stores, 0 bytes spill loads')
    f = toolchain.ptxas_fields(clean)
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
    f = toolchain.igc_fields(err)
    assert f.get('vgprsspill') == 384
    assert 'vgprs' not in f, (
        'a register count that was never reported must stay absent, not '
        'become zero -- a caller comparing configurations on a missing '
        'number would rank them equal instead of declining to rank them')


def test_a_silent_intel_build_is_not_an_error():
    """Nothing to say means it did not spill, which is the good case."""
    assert toolchain.igc_fields('') == {}


# ----------------------------------------------------------------------
# what a ranking reads
# ----------------------------------------------------------------------

def test_ptxas_output_is_read():
    log = ("ptxas info    : Used 168 registers, used 1 barriers, 384 bytes cmem[0]\n"
           "    328 bytes stack frame, 328 bytes spill stores, 196 bytes spill loads\n")
    r = toolchain.resources('cuda', log)
    assert (r.registers, r.spill_bytes) == (168, 524)


def test_amdgpu_resource_remarks_are_read():
    log = ("remark:     VGPRs: 110 [-Rpass-analysis=kernel-resource-usage]\n"
           "remark:     AGPRs: 16 [-Rpass-analysis=kernel-resource-usage]\n"
           "remark:     ScratchSize [bytes/lane]: 124 [-Rpass-analysis=kernel-resource-usage]\n"
           "remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]\n")
    r = toolchain.resources('hip', log)
    assert (r.registers, r.spill_bytes, r.register_blocks) == (126, 124, 4)


def test_the_compiler_is_the_callers_then_the_environments_then_the_paths(monkeypatch):
    monkeypatch.setenv('TF_NVCC', '/from/env/nvcc')
    monkeypatch.setattr(toolchain.shutil, 'which', lambda name: '/on/path/' + name)
    assert toolchain.Toolchain(nvcc='/given/nvcc').compiler('nvidia') == '/given/nvcc'
    assert toolchain.Toolchain().compiler('nvidia') == '/from/env/nvcc'
    monkeypatch.delenv('TF_NVCC')
    assert toolchain.Toolchain().compiler('nvidia') == '/on/path/nvcc'
    assert toolchain.Toolchain().compiler('intel') == '/on/path/icpx'
    assert toolchain.Toolchain(icpx='/given/icpx').compiler('intel') == '/given/icpx'
    assert toolchain.Toolchain().compiler('other') is None


def test_igc_reports_a_retry_as_a_spill_and_silence_as_none():
    assert toolchain.resources('oneapi', '').spill_bytes == 0
    retry = ("[pvc] warning: in kernel 'k': [RetryManager] Start recompilation "
             "of the kernel")
    assert toolchain.resources('oneapi', retry).spill_bytes == 1
    assert toolchain.resources('oneapi', 'kernel k spilled 384 bytes').spill_bytes == 384
    assert toolchain.resources('oneapi', retry).registers is None


def test_igc_reads_the_spill_memory_it_reports():
    """IGC 2026, ESIMD local_flux on pvc: the figure is on its own line and
    says nothing of a retry."""
    log = ("Spill memory used = 33088 bytes for kernel _ZTSZZ30kernel_kernel\n"
           " Compiling kernel with spill code may degrade performance.")
    assert toolchain.resources('oneapi', log).spill_bytes == 33088


def test_intel_states_its_spilling_in_the_binary_and_not_on_the_console(tmp_path):
    """IGC prints nothing about a SPMD kernel that spills.

    `elastic-o6s:neighboringFlux` at sixteen lanes carries 13888 bytes of
    spilling -- 55 spill and 72 fill messages in its ISA -- and its build
    prints not one word; at 32 lanes the same kernel has none and is 1.5x
    faster.  So the figure that decides between them is in the object's
    `.ze_info` note, which is where this looks.
    """
    obj = tmp_path / 'k.so'
    obj.write_bytes(b'\x7fELF' + b'...' + b'  spill_size:      13888\n' + b'...')
    assert toolchain.zeinfo_spill(str(obj)) == 13888
    quiet = tmp_path / 'q.so'
    quiet.write_bytes(b'\x7fELF ze_info: kernels: nothing to say')
    assert toolchain.zeinfo_spill(str(quiet)) == 0, (
        'a note that does not mention spilling is a note saying there is none')
    bare = tmp_path / 'b.so'
    bare.write_bytes(b'\x7fELF nothing to say')
    assert toolchain.zeinfo_spill(str(bare)) is None, (
        'no note is not the same answer as a note saying nothing')
    # The vector backend writes no `spill_size:` line, ever.  What it spills
    # is the scratch buffer it asks the runtime for, and the 32-lane
    # explicit-SIMD `elastic-o6d:localFluxAll` -- 1376 spill messages in its
    # ISA -- states it only this way.
    vc = tmp_path / 'vc.so'
    vc.write_bytes(b'\x7fELF' + b'''
  execution_env:
    grf_count:       256
  per_thread_memory_buffers:
    - type:            scratch
      usage:           single_space
      size:            10688
''')
    assert toolchain.zeinfo_spill(str(vc)) == 10688, (
        'reading only the first spelling would make every explicit-SIMD '
        'candidate come back spill-free, and the two lane counts '
        'indistinguishable to the one scorer able to tell them apart')
    clean_vc = tmp_path / 'cvc.so'
    clean_vc.write_bytes(b'\x7fELF  execution_env:\n    grf_count:       256\n')
    assert toolchain.zeinfo_spill(str(clean_vc)) == 0, (
        'and a vector build that spills nothing asks for no such buffer')
    assert toolchain.zeinfo_spill(str(tmp_path / 'missing.so')) is None, (
        'no object is not the same answer as no spilling')
    # and the console parser still answers for what it can see
    assert toolchain.resources('oneapi', 'spill memory used = 96 bytes').spill_bytes == 96


def test_the_igc_spelling_of_today_is_read_by_every_caller():
    """`Spill memory used = ... bytes for kernel ...`, one line per kernel.

    The field vocabulary and the ranking both come from `igc_fields`, so a
    spelling only one caller knows cannot happen: here the report fields a
    benchmark records and the figure a ranking reads agree.
    """
    log = "Spill memory used = 33088 bytes for kernel _ZTS3foo\n"
    assert toolchain.igc_fields(log) == {'vgprsspill': 33088}
    assert toolchain.report_fields('esimd', log) == {'vgprsspill': 33088}
    assert toolchain.resources('oneapi', log).spill_bytes == 33088


def test_the_stack_frame_is_the_scratch_size_on_nvidia_too():
    """One key for per-lane scratch, whichever vendor states it."""
    f = toolchain.ptxas_fields(
        "ptxas info    : Used 40 registers\n"
        "    96 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n")
    assert f['scratchsize'] == 96 and 'scratch' not in f


# ----------------------------------------------------------------------
# the compilers
# ----------------------------------------------------------------------

@pytest.mark.parametrize("backend,arch,flag", [
    ("cuda", "sm_90", "-arch=sm_90"),
    ("hip", "gfx942", "--offload-arch=gfx942"),
    ("acpp", "generic", "--acpp-targets=generic"),
])
def test_each_compiler_names_the_target_its_own_way(backend, arch, flag):
    assert toolchain.COMPILERS[backend].target_flags(arch) == [flag]


def test_an_icpx_target_is_an_ahead_of_time_build(monkeypatch):
    """A JIT build never reaches IGC, so a target means a device build, and
    the device compiler's own options ride along."""
    monkeypatch.delenv('TF_ICPX_DEVICE_OPTIONS', raising=False)
    assert toolchain.COMPILERS['oneapi'].target_flags('pvc') == [
        '-fsycl-targets=spir64_gen', '-Xsycl-target-backend', '-device pvc']
    monkeypatch.setenv('TF_ICPX_DEVICE_OPTIONS', '-internal_options -x')
    assert toolchain.COMPILERS['esimd'].target_flags('pvc')[-1] == (
        '-device pvc -internal_options -x')


def test_the_package_variable_wins_over_the_vendor_one(monkeypatch):
    monkeypatch.setattr(toolchain.shutil, 'which', lambda name: None)
    monkeypatch.setenv('NVCC', '/vendor/nvcc')
    monkeypatch.delenv('TF_NVCC', raising=False)
    assert toolchain.COMPILERS['cuda'].find() == '/vendor/nvcc'
    monkeypatch.setenv('TF_NVCC', '/package/nvcc')
    assert toolchain.COMPILERS['cuda'].find() == '/package/nvcc'
    assert toolchain.COMPILERS['cuda'].find('/given') == '/given'


def test_every_backend_links_its_runtime_translation_unit():
    """The launcher calls into it, so a linked binary needs it, and it has to
    exist in the headers this package ships."""
    import os
    for entry in toolchain.COMPILERS.values():
        assert os.path.exists(os.path.join(toolchain.include_dir(), entry.aux))
