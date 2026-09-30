# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The vendor compilers: where they are, how a target is named to them, and
what they report about a kernel.

Everything that compiles generated code asks here -- the autotuner's
`CompiledScore`, the test harness, the benchmark builds and
`tools/register_usage.py`.  So a compiler is found the same way wherever it is
needed, an architecture is spelled the same way on every command line, and a
compiler's account of a kernel is read by one parser per vendor rather than by
one per caller.

What stays with the caller is what it builds -- an object, a cubin, a linked
test binary -- and at which optimization level: those answer the caller's
question, not the compiler's.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -- the compilers ----------------------------------------------------------- #

@dataclass(frozen=True)
class Compiler:
    """One vendor compiler, as the generator backend it builds for sees it.

    `env` names the environment variables that point at the binary, looked up
    in order: the `TF_` spelling first, which is the package's convention, then
    the plain one other tooling tends to set.
    """
    backend: str
    binary: str
    env: Tuple[str, ...]
    #: The runtime translation unit for this language.  A linked binary needs
    #: it even though nothing calls it directly: the generated launcher does,
    #: through `CHECK_ERR`.
    aux: str
    suffix: str = '.cpp'

    def find(self, path: Optional[str] = None) -> Optional[str]:
        """The binary: `path` where given, else the first environment variable
        that is set, else the default name on `PATH`; None where none has one."""
        if path:
            return path
        for name in self.env:
            value = os.environ.get(name)
            if value:
                return value
        return shutil.which(self.binary)

    def language_flags(self) -> List[str]:
        """What every compilation of generated code needs, whatever it builds."""
        return ['-std=c++17']

    def target_flags(self, arch: str) -> List[str]:
        """The flags that make the build one for `arch`."""
        raise NotImplementedError

    def report_flags(self) -> List[str]:
        """The flags that make the compiler say what a kernel costs it."""
        return []


@dataclass(frozen=True)
class Nvcc(Compiler):
    def language_flags(self):
        # The device headers call `constexpr` host functions from device code.
        return ['-std=c++17', '--expt-relaxed-constexpr']

    def target_flags(self, arch):
        return [f'-arch={arch}']

    def report_flags(self):
        return ['-Xptxas=-v']


@dataclass(frozen=True)
class Hipcc(Compiler):
    def target_flags(self, arch):
        return [f'--offload-arch={arch}']

    def report_flags(self):
        return ['-Rpass-analysis=kernel-resource-usage']


@dataclass(frozen=True)
class Icpx(Compiler):
    """oneAPI's `icpx`, for the SPMD and the explicit-SIMD lowering alike.

    A target means an ahead-of-time build: a JIT build never reaches IGC, so
    it reports nothing, and it moves the device compilation into the first
    launch.  `TF_ICPX_DEVICE_OPTIONS` reaches the device compiler, e.g.
    `-internal_options -ze-opt-disable-sendwarwa`.  The binary needs the
    environment `setvars.sh` makes; a path alone reaches the driver, not the
    device compiler behind it.
    """

    def language_flags(self):
        return ['-fsycl', '-std=c++17']

    def target_flags(self, arch):
        extra = os.environ.get('TF_ICPX_DEVICE_OPTIONS', '').strip()
        return ['-fsycl-targets=spir64_gen', '-Xsycl-target-backend',
                f'-device {arch}' + (f' {extra}' if extra else '')]


@dataclass(frozen=True)
class Acpp(Compiler):
    def target_flags(self, arch):
        return [f'--acpp-targets={arch}']


#: Keyed by the generator backend.  `oneapi` and `esimd` are one compiler and
#: two code generators, so they share an entry's settings but not its name.
COMPILERS: Dict[str, Compiler] = {
    'cuda': Nvcc('cuda', 'nvcc', ('TF_NVCC', 'NVCC'), 'tensorforge_aux.cu',
                 suffix='.cu'),
    'hip': Hipcc('hip', 'hipcc', ('TF_HIPCC', 'HIPCC'), 'tensorforge_aux.cpp'),
    'oneapi': Icpx('oneapi', 'icpx', ('TF_ICPX', 'ICPX'),
                   'tensorforge_aux_sycl.cpp'),
    'esimd': Icpx('esimd', 'icpx', ('TF_ICPX', 'ICPX'),
                  'tensorforge_aux_sycl.cpp'),
    'acpp': Acpp('acpp', 'acpp', ('TF_ACPP', 'ACPP'),
                 'tensorforge_aux_sycl.cpp'),
}

#: The compiler that reports on a vendor's hardware.
_VENDOR = {'nvidia': 'cuda', 'amd': 'hip', 'intel': 'oneapi'}


def compiler_for_vendor(vendor: str) -> Optional[Compiler]:
    backend = _VENDOR.get(vendor)
    return COMPILERS[backend] if backend else None


def include_dir() -> str:
    """The device headers of this installation."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')


@dataclass
class Toolchain:
    """Compiler paths a caller names, per vendor.

    A path given here wins; otherwise `Compiler.find` decides, from the
    environment and then from `PATH`.
    """
    nvcc: Optional[str] = None
    hipcc: Optional[str] = None
    icpx: Optional[str] = None
    include: Optional[str] = None

    def compiler(self, vendor: str) -> Optional[str]:
        entry = compiler_for_vendor(vendor)
        if entry is None:
            return None
        given = {'nvidia': self.nvcc, 'amd': self.hipcc,
                 'intel': self.icpx}[vendor]
        return entry.find(given)

    def include_dir(self) -> str:
        return self.include or include_dir()


# -- what they report -------------------------------------------------------- #
#
# Each reader returns the fields it found, keyed in the vocabulary of AMDGPU's
# resource remarks -- `vgprs`, `agprs`, `sgprs`, `scratchsize`, `ldssize`,
# `occupancy`, `vgprsspill` -- so that a caller comparing configurations needs
# no per-vendor case.  A field the compiler did not state is absent, not zero:
# a missing number compared as zero would rank configurations equal instead of
# declining to rank them.  Where a translation unit holds several kernels,
# each field is the largest, since a budget is per kernel and the largest is
# the only reading that cannot understate.

#: `-Rpass-analysis=kernel-resource-usage`, one remark per field.  Not a list
#: of field names: half of them carry their unit in brackets before the colon
#: -- `ScratchSize [bytes/lane]`, `Occupancy [waves/SIMD]`, `LDS Size
#: [bytes/block]` -- and the spills are `SGPRs Spill` and `VGPRs Spill`, not
#: one count.  So this takes whatever `name: integer` the remarks contain and
#: normalizes the name afterwards; `Function Name` and `Dynamic Stack: False`
#: do not match.
_REMARK = re.compile(
    r'remark:\s*(?P<field>[A-Za-z][A-Za-z ]*?)\s*'
    r'(?:\[[^\]]*\])?\s*:\s*(?P<value>-?[0-9]+)\b')

#: `nvcc -Xptxas=-v`: a comma-separated list of `<number> bytes <what>`, with
#: the register count written the other way round (`Used 93 registers`), so
#: it takes two patterns.  `cmem[0]` is constant memory and matches neither.
_PTXAS_REGS = re.compile(r'Used\s+(?P<value>\d+)\s+registers')
_PTXAS_FIELD = re.compile(
    r'(?P<value>\d+)\s+bytes\s+(?P<field>smem|spill stores|spill loads|'
    r'stack frame)')

#: What IGC prints about an ahead-of-time build.  It has no per-kernel
#: resource remark -- the register count goes to a shader dump under
#: `/tmp/IntelIGC`, whose format moves with the driver -- so what the command
#: line carries is when a kernel spills.  `Spill memory used = 33088 bytes for
#: kernel ...` is the spelling of IGC 2026, for ESIMD and SPMD alike; the other
#: two are older ones.
_IGC_SPILL = re.compile(
    r"(?:kernel|Kernel)\s+.*?\bspill(?:s|ed)?\b.*?(?P<value>\d+)\s*bytes"
    r"|spill(?:ed)?\s+(?P<value2>\d+)\s*bytes"
    r"|spill memory used\s*=\s*(?P<value3>\d+)\s*bytes", re.I)
#: IGC's word for a kernel it compiled twice: the first attempt blew the
#: register file and it retries with another strategy.
_IGC_RETRY = re.compile(r'\[RetryManager\]\s+Start recompilation', re.I)


def _keep_max(fields: Dict[str, int], key: str, value: int) -> None:
    fields[key] = max(fields.get(key, value), value)


def remark_fields(log: str) -> Dict[str, int]:
    """Every numeric AMDGPU resource remark, keyed by its squashed lower-case
    name (`VGPRs Spill` -> `vgprsspill`)."""
    fields: Dict[str, int] = {}
    for m in _REMARK.finditer(log):
        key = m.group('field').strip().lower().replace(' ', '')
        _keep_max(fields, key, int(m.group('value')))
    return fields


def ptxas_fields(log: str) -> Dict[str, int]:
    """`nvcc -Xptxas=-v`, in the AMDGPU vocabulary.

    The register count lands under `vgprs`: NVIDIA has one register file where
    CDNA has two, so `vgprs + agprs` reads correctly with `agprs` absent.  The
    stack frame is `scratchsize`, shared memory `ldssize`.  Spill stores and
    spill loads are kept as they are, and `vgprsspill` is the larger of the
    two -- the same traffic seen from both ends -- where either is nonzero.
    """
    fields: Dict[str, int] = {}
    for m in _PTXAS_REGS.finditer(log):
        _keep_max(fields, 'vgprs', int(m.group('value')))
    names = {'smem': 'ldssize', 'stack frame': 'scratchsize',
             'spill stores': 'spillstores', 'spill loads': 'spillloads'}
    for m in _PTXAS_FIELD.finditer(log):
        _keep_max(fields, names[m.group('field')], int(m.group('value')))
    spill = max(fields.get('spillstores', 0), fields.get('spillloads', 0))
    if spill:
        fields['vgprsspill'] = spill
    return fields


def igc_fields(log: str) -> Dict[str, int]:
    """What an Intel build says, which is whether it spilled.

    `vgprsspill` in bytes where IGC states a size, and 1 -- spilled, amount
    unknown -- where it only reports a retry, so that such a build ranks
    behind every one that did not.  Silence is the answer "no spill", not a
    failure to parse: IGC says nothing about a kernel that fits.
    """
    spill = 0
    for m in _IGC_SPILL.finditer(log):
        spill = max(spill, int(m.group('value') or m.group('value2')
                               or m.group('value3')))
    if not spill and _IGC_RETRY.search(log):
        spill = 1
    return {'vgprsspill': spill} if spill else {}


def report_fields(backend: str, log: str) -> Dict[str, int]:
    """What the compiler of `backend` said about a build, in one vocabulary."""
    if backend == 'cuda':
        return ptxas_fields(log)
    if backend == 'hip':
        return remark_fields(log)
    if backend in ('oneapi', 'esimd'):
        return igc_fields(log)
    return {}


@dataclass(frozen=True)
class Resources:
    """What a compiler reports for one kernel, in the terms a ranking uses."""
    registers: Optional[int]
    spill_bytes: int
    #: Blocks the register file admits per SM, where it can be said.
    register_blocks: Optional[int] = None


def resources(backend: str, log: str) -> Optional[Resources]:
    """The report as a ranking reads it, or None where it states nothing.

    On NVIDIA the spilling is the traffic, stores and loads together.  On AMD
    the registers are both files and the spilling is the scratch size; the
    occupancy it reports, waves per SIMD, stands in for the blocks.  An Intel
    build is always a report -- see `igc_fields` -- and never has registers.
    """
    fields = report_fields(backend, log)
    if backend == 'cuda':
        if 'vgprs' not in fields:
            return None
        return Resources(fields['vgprs'], fields.get('spillstores', 0)
                         + fields.get('spillloads', 0))
    if backend == 'hip':
        if 'vgprs' not in fields:
            return None
        return Resources(fields['vgprs'] + fields.get('agprs', 0),
                         fields.get('scratchsize', 0), fields.get('occupancy'))
    if backend in ('oneapi', 'esimd'):
        return Resources(None, fields.get('vgprsspill', 0))
    return None


#: `spill_size:      13888` in the binary's `.ze_info`, which is the only
#: place IGC states it for a SPMD build.
_ZEINFO_SPILL = re.compile(rb'spill_size:\s*(\d+)')
#: The note itself, to tell "it says no spilling" from "it does not say
#: anything because it is not there".
_ZEINFO_NOTE = re.compile(rb'ze_info|payload_arguments|execution_env')
#: What the vector backend writes instead.  A `-vc-codegen` build carries no
#: `spill_size:` line at all; what it spills appears as the per-thread scratch
#: buffer it asks the runtime for, and a build that spills nothing has no such
#: buffer.
_ZEINFO_SCRATCH = re.compile(
    rb'-\s*type:\s*scratch\s*\n\s*usage:\s*\w+\s*\n\s*size:\s*(\d+)')


def zeinfo_spill(path: str) -> Optional[int]:
    """What an ahead-of-time Intel binary says it spilled, or None where
    there is nothing to ask.

    IGC says nothing on the console about a SPMD kernel that spills -- the
    build of `elastic-o6s:neighboringFlux` at sixteen lanes prints not one
    word and carries 13888 bytes of spilling, 55 spill and 72 fill messages in
    its ISA, while the same kernel at 32 lanes has none.  So the log is not
    where to look: the figure is in the `.ze_info` note of the object, and
    that is in the file whatever the compiler chose to print.

    Read as bytes rather than parsed as ELF: the note is text in a section
    whose name has moved between releases, and one regular expression over the
    file is both shorter and harder to break.

    Three answers and not two.  No object: nothing is known.  An object whose
    note is there and names neither a `spill_size` nor a scratch buffer: no
    spilling, which is what the note not mentioning it means.  An object with
    no note at all: nothing is known either, which is not the same as zero.

    Two spellings, because the two backends do not write the same note.  A
    SPMD build states `spill_size:`.  A `-vc-codegen` build -- every
    explicit-SIMD kernel -- states none, ever, and puts what it spills in the
    per-thread scratch buffer it asks the runtime for:

        per_thread_memory_buffers:
          - type:            scratch
            usage:           single_space
            size:            10688

    That is the 32-lane build of `elastic-o6d:localFluxAll`, whose ISA carries
    1376 spill messages; the same kernel at sixteen lanes has no such buffer
    and spills nothing.  Reading only the first spelling would make every
    explicit-SIMD candidate come back spill-free -- the two lane counts
    indistinguishable to the one scorer able to tell them apart, since the
    modelled footprint puts them half a percent apart (79760 B against 79396)
    while the clock puts them at 99.45 ns an element against 33.07.
    """
    try:
        with open(path, 'rb') as f:
            blob = f.read()
    except OSError:
        return None                    # no object: nothing is known
    figures = [int(m.group(1)) for m in _ZEINFO_SPILL.finditer(blob)]
    if figures:
        return max(figures)            # it says so
    scratch = [int(m.group(1)) for m in _ZEINFO_SCRATCH.finditer(blob)]
    if scratch:
        return max(scratch)            # the vector backend says it this way
    if _ZEINFO_NOTE.search(blob):
        return 0                       # the note is there and does not
    return None                        # no note: nothing is known either
