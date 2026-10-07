# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How wide an FMA may be, which is not how wide a load may be.

`vectorize.widths_for` answers a question about an address; this answers one
about an instruction set (`Target.packed_fma_width`), and the two disagree on
every target we build for:

* NVIDIA loads 16 bytes on every architecture since Kepler and has a packed
  FP32 FMA only from sm_100 to sm_11x.  FP64 never.
* AMD has packed FP32 on CDNA2 and later and on gfx125x, and packed FP64 on
  gfx1251, so a `double2` is arithmetic there and pure load traffic on NVIDIA.

Conflating them would mean either emitting `fma.rn.f32x2` on Hopper -- which
ptxas rejects -- or declining a `float4` load on Ampere because its FMAs are
scalar, which throws away the part of the win that works everywhere.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.target import Target

F32, F64 = Datatype.F32, Datatype.F64


def _width(arch, datatype, backend=None):
    backend = backend or ('cuda' if arch.startswith('sm_') else
                          'hip' if arch.startswith('gfx') else 'oneapi')
    return Target(arch, backend).packed_fma_width(datatype)


# --------------------------------------------------------------------------- #
# NVIDIA
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch', ['sm_80', 'sm_86', 'sm_89', 'sm_90'])
def test_nvidia_before_blackwell_has_no_packed_fp32(arch):
    """Two scalar FFMAs. `float2` there is a load optimization, not a math one."""
    assert _width(arch, F32) == 1


@pytest.mark.parametrize('arch', ['sm_100', 'sm_101', 'sm_110'])
def test_nvidia_from_sm_100_packs_fp32(arch):
    assert _width(arch, F32) == 2


def test_sm_120_lowers_the_pair_to_two_ffma():
    """The intrinsic is declared there and is two instructions."""
    assert _width('sm_120', F32) == 1


@pytest.mark.parametrize('arch', ['sm_80', 'sm_90', 'sm_100', 'sm_120'])
def test_nvidia_never_packs_fp64(arch):
    """`double2` on NVIDIA is one load and two scalar FMAs, on every part."""
    assert _width(arch, F64) == 1


# --------------------------------------------------------------------------- #
# AMD, as LLVM records it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch', ['gfx90a', 'gfx940', 'gfx942', 'gfx950',
                                  'gfx1250', 'gfx1251'])
def test_amd_packs_fp32_on_cdna2_and_later_and_on_gfx125x(arch):
    assert _width(arch, F32) == 2


@pytest.mark.parametrize('arch', ['gfx900', 'gfx906', 'gfx908', 'gfx1030',
                                  'gfx1100', 'gfx1150', 'gfx1200'])
def test_amd_before_cdna2_and_rdna_do_not(arch):
    """RDNA 3 pairs two scalar FMAs by dual issue (`vopd`), which is not a
    packed operand and does not halve the arithmetic a lead width of two
    writes down."""
    assert _width(arch, F32) == 1


def test_only_gfx1251_packs_fp64():
    """The one that matters for a code that runs mostly in double precision."""
    assert _width('gfx1251', F64) == 2
    assert _width('gfx1250', F64) == 1
    assert _width('gfx942', F64) == 1


def test_hip_on_nvidia_asks_the_nvidia_part():
    assert _width('sm_100', F32, backend='hip') == 2
    assert _width('sm_86', F32, backend='hip') == 1


# --------------------------------------------------------------------------- #
# Elsewhere
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch,backend', [('pvc', 'oneapi'), ('pvc', 'esimd'),
                                          ('dg1', 'oneapi')])
def test_intel_gets_scalar_fmas(arch, backend):
    assert _width(arch, F32, backend) == 1
    assert _width(arch, F64, backend) == 1
