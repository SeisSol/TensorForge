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
    """Never: the runtime has no `fmacdpp8`, for any target.

    Not an oversight in the guard -- there is no declaration, no
    specialisation, nothing.  `hfma` used to select it for 8 <= threads < 16
    on RDNA and emit a call to a name that does not exist.
    """
    return False


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
