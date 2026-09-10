# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the runtime actually defines.

Mirrors the `#if` guards in `include/tensorforge_device/hip.h`.  This is
a duplication of a C++ fact, which is a real cost -- so
`tests/test_amd_caps.py` parses those guards and fails if the two drift
apart.  A silent duplicate is the disease; a checked one is a seam.
"""

from tensorforge.common.basic_types import Datatype
from .arch import amdarch


def has_fmacdpp4(ctx):
    """`#if !defined(__gfx900__)` -- everything but gfx900."""
    return amdarch(ctx) != 0x900


def has_fmacdpp8(ctx):
    """The `__GFX10__ || ... || __GFX13__` block: DPP8 is gfx10 and later.

    It used to be `False` for every target, because the runtime had no
    `fmacdpp8` at all and `hfma` had emitted calls to a name that did not
    exist.  The runtime has one now (float only, like `fmacdpp4`), and at
    eight lanes it is what carries the broadcast inside each FMA: without it
    the step narrowed to 4 and a separate move -- `ds_swizzle`, later
    `v_mov_b32_dpp8` -- redistributed the sub-block first.
    """
    return amdarch(ctx) >= 0x1000


def has_fmacdpp16(ctx, datatype):
    """The `__gfx90a__ || __gfx940__ || ... || __GFX10__ ... || __GFX13__` block.

    Float and double specialisations live under the same guard, so the
    datatype does not currently split the answer -- it is a parameter because
    the C++ side declares them separately and could.
    """
    if datatype not in (Datatype.F32, Datatype.F64):
        return False
    arch = amdarch(ctx)
    return arch in (0x90a, 0x940, 0x941, 0x942, 0x950) or arch >= 0x1000
