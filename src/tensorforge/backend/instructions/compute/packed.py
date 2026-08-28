# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How many elements one FMA instruction covers, and what that costs.

Two questions kept apart on purpose. `vectorize.widths_for` answers how wide a
*memory* access may be, which is a property of an address. This answers how
wide an *arithmetic* op may be, which is a property of the instruction set --
and the two are not the same number:

* NVIDIA has 16-byte loads since forever and a packed FP32 FMA only on
  Blackwell (`fma.rn.f32x2`, PTX ISA 8.7, sm_100 and sm_120). There is no
  packed FP64 FMA at all. A `float4` there is one load and four FMAs before
  Blackwell, one load and two after.
* AMD is the other way round and is further ahead: packed FP32 since gfx90a
  and on RDNA 3, packed FP64 on gfx1251. So `double2` is a pure load win on
  NVIDIA and a real arithmetic win there.

The DPP entanglement is why this cannot be a single number. Packed math and
DPP are mutually exclusive on AMD, and the AMD path uses DPP to broadcast an
operand across lanes without going through LDS. So the choice per operator is
between two bundles -- DPP broadcast with scalar FMAs, or an LDS/splat
broadcast with packed FMAs -- and which wins depends on whether the kernel is
short of VALU issue or of LDS bandwidth. That decision belongs to the cost
model; this module only reports what the hardware would allow.
"""

from __future__ import annotations

from typing import Optional

#: NVIDIA: `fma.rn.f32x2` needs PTX ISA 8.7 and Blackwell.  `sm_101` is in the
#: same family; `sm_110`/`sm_120` are later.  FP64 is absent everywhere.
_NVIDIA_PACKED_F32 = frozenset({'sm_100', 'sm_101', 'sm_110', 'sm_120'})

#: AMD: `v_pk_fma_f32` from CDNA2 (gfx90a) onward, and on RDNA 3 (gfx11xx) and
#: later.  gfx900/gfx906/gfx908 and RDNA 1/2 (gfx10xx) predate it.
_AMD_NO_PACKED_F32 = frozenset({'gfx900', 'gfx906', 'gfx908',
                                'gfx1010', 'gfx1030'})

#: AMD: packed FP64 is new with gfx1251.
_AMD_PACKED_F64 = frozenset({'gfx1251'})


def packed_fma_width(vendor: Optional[str], arch: Optional[str],
                     elem_bytes: int, dpp: bool = False) -> int:
    """Elements one FMA instruction covers on this target.

    ``dpp`` says the operator's broadcast comes from a DPP modifier.  On AMD
    that is not a hint but an exclusion: an instruction carries either a DPP
    modifier or packed operands, never both, so asking for a packed FMA on a
    DPP operand is asking for an instruction that does not exist.

    The two AMD entries are deliberately asymmetric, and the asymmetry is the
    whole content of them.  Packed FP32 is a floor AMD has held since gfx90a,
    so it is a *deny* list: an unlisted part keeps it, because the likely
    failure with an allow list is a new gfx quietly losing packed math and
    nobody noticing a performance regression.  Packed FP64 exists on exactly
    one part, so it is an *allow* list: extrapolating a brand-new capability
    from the shape of an arch name emits an instruction the assembler
    rejects.  NVIDIA is an allow list throughout for the same reason -- the
    FP32 form is one generation old.

    An unknown vendor gets 1, which is correct everywhere.
    """
    if vendor is None or arch is None:
        return 1
    if dpp and vendor == 'amd':
        return 1
    if vendor == 'nvidia':
        return 2 if (elem_bytes == 4 and arch in _NVIDIA_PACKED_F32) else 1
    if vendor == 'amd':
        if elem_bytes == 4:
            return 1 if arch in _AMD_NO_PACKED_F32 else 2
        if elem_bytes == 8:
            return 2 if arch in _AMD_PACKED_F64 else 1
    return 1


def packed_is_exclusive_with_dpp(vendor: Optional[str]) -> bool:
    """Whether choosing packed math costs the operand broadcast.

    Stated as its own question because the *cost* of the exclusion is not
    symmetric with the exclusion itself: losing DPP means the broadcast has
    to come from LDS or from an explicit splat, and that is a bandwidth cost
    weighed against an issue-rate gain.  A caller that only wants to know
    "may I emit both" reads this; a caller choosing between them needs the
    cost model.
    """
    return vendor == 'amd'
