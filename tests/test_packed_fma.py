# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How wide an FMA may be, which is not how wide a load may be.

`vectorize.widths_for` answers a question about an address; this answers one
about an instruction set, and the two disagree on every target we build for:

* NVIDIA loads 16 bytes on every architecture since Kepler and gained a packed
  FP32 FMA only on Blackwell. FP64 never.
* AMD gained packed FP32 on gfx90a and packed FP64 on gfx1251, so a `double2`
  is arithmetic there and pure load traffic on NVIDIA.

Conflating them would mean either emitting `fma.rn.f32x2` on Hopper -- which
ptxas rejects -- or declining a `float4` load on Ampere because its FMAs are
scalar, which throws away the part of the win that works everywhere.

The DPP exclusion is the reason this returns a number per *operand* rather
than per target: on AMD an instruction carries a DPP modifier or packed
operands, never both, and the AMD path uses DPP to broadcast.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.packed import (
    packed_fma_width, packed_is_exclusive_with_dpp)

F32, F64 = 4, 8


# --------------------------------------------------------------------------- #
# NVIDIA
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch', ['sm_80', 'sm_86', 'sm_89', 'sm_90'])
def test_nvidia_before_blackwell_has_no_packed_fp32(arch):
    """Two scalar FFMAs. `float2` there is a load optimisation, not a math one."""
    assert packed_fma_width('nvidia', arch, F32) == 1


@pytest.mark.parametrize('arch', ['sm_100', 'sm_120'])
def test_nvidia_blackwell_packs_fp32(arch):
    assert packed_fma_width('nvidia', arch, F32) == 2


@pytest.mark.parametrize('arch', ['sm_80', 'sm_90', 'sm_100', 'sm_120'])
def test_nvidia_never_packs_fp64(arch):
    """`double2` on NVIDIA is one load and two scalar FMAs, on every part."""
    assert packed_fma_width('nvidia', arch, F64) == 1


# --------------------------------------------------------------------------- #
# AMD, which is ahead here rather than behind
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942', 'gfx950', 'gfx1100',
                                  'gfx1200', 'gfx1250', 'gfx1251'])
def test_amd_packs_fp32_from_cdna2_and_rdna3(arch):
    assert packed_fma_width('amd', arch, F32) == 2


@pytest.mark.parametrize('arch', ['gfx900', 'gfx906', 'gfx908', 'gfx1030'])
def test_amd_before_cdna2_does_not(arch):
    assert packed_fma_width('amd', arch, F32) == 1


def test_only_gfx1251_packs_fp64():
    """The one that matters for a code that runs mostly in double precision."""
    assert packed_fma_width('amd', 'gfx1251', F64) == 2
    assert packed_fma_width('amd', 'gfx1250', F64) == 1
    assert packed_fma_width('amd', 'gfx942', F64) == 1


# --------------------------------------------------------------------------- #
# The exclusion
# --------------------------------------------------------------------------- #

def test_dpp_and_packed_cannot_be_had_together():
    """Not a preference: the instruction encoding has room for one or the other.

    The AMD path broadcasts an operand with DPP to keep it out of LDS, so this
    is a live conflict on every operator it applies to, not a corner case.
    """
    assert packed_fma_width('amd', 'gfx942', F32, dpp=True) == 1
    assert packed_fma_width('amd', 'gfx942', F32, dpp=False) == 2


def test_the_exclusion_is_an_amd_fact():
    """NVIDIA has no DPP modifier, so the flag says nothing there."""
    assert packed_fma_width('nvidia', 'sm_100', F32, dpp=True) == 2
    assert packed_is_exclusive_with_dpp('amd')
    assert not packed_is_exclusive_with_dpp('nvidia')


# --------------------------------------------------------------------------- #
# Unknown targets get the answer that is correct everywhere
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('vendor,arch', [
    ('intel', 'pvc'), (None, 'sm_100'), ('nvidia', None), ('nvidia', 'sm_95'),
])
def test_an_unrecognised_target_gets_scalar_fmas(vendor, arch):
    """Guessing from the shape of an arch name would emit what the assembler
    rejects, and on a new architecture that is the likely case."""
    assert packed_fma_width(vendor, arch, F32) == 1


def test_the_two_amd_lists_point_in_opposite_directions():
    """Deny for FP32, allow for FP64, and the asymmetry is the point.

    FP32 is a floor AMD has held since gfx90a: with an allow list a new gfx
    would quietly lose packed math and show up only as a performance
    regression nobody attributes.  FP64 exists on one part, and extrapolating
    that from an arch name emits an instruction the assembler rejects.
    """
    assert packed_fma_width('amd', 'gfx1299', F32) == 2
    assert packed_fma_width('amd', 'gfx1299', F64) == 1
    assert packed_fma_width('nvidia', 'sm_130', F32) == 1
