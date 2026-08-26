# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The MFMA tile catalogue.

What a tile *is* -- its intrinsic, the transpose its A operand needs --
is a fact about the target; which tile to use is a policy.  They used to
be interleaved: three dicts keyed by tile width, rebuilt inside the
emitter on every call, plus a fourth condition at the dispatch.  Neither
could be read without the other.
"""

from dataclasses import dataclass
from typing import Optional

from tensorforge.common.basic_types import Datatype
from .arch import cdna1, gfx1251


@dataclass(frozen=True)
class MfmaTile:
    """One square MFMA tile: the intrinsic, and how its A operand is fed."""

    block: int
    builtin: str
    #: Cross-lane transpose applied to the A registers before the tile runs.
    #: `None` means the tile needs none; a name that `hip.h` does not define
    #: means the tile is unusable, which `available_for` reports.
    transpose: Optional[str]
    #: True when the transpose declares its outputs separately from its
    #: inputs -- `tp(w1..wn, v1..vn)` rather than `tp(w1..wn)`.  That is what
    #: lets it be emitted in SSA form: fresh values come out, so they can
    #: carry the layout the exchange produced, and the inputs are left alone
    #: instead of being rewritten underneath whatever else still reads them.
    #: `transpose4x4b32` has it; `transpose16x16b32` is in-place only.
    transpose_has_separate_outputs: bool = False

    def scale(self, threads: int) -> int:
        """The intrinsic's `cbsz`: how many lanes share one tile row."""
        if not self.fits(threads):
            raise ValueError(
                f'{self.builtin} needs threads to be a multiple of '
                f'{self.block}, got {threads}')
        return (threads // self.block).bit_length() - 1

    def fits(self, threads: int) -> bool:
        return threads >= self.block and threads % self.block == 0

    def available_for(self, threads: int, dtype, ctx) -> bool:
        """Can this tile be emitted here at all?

        Three separate questions, kept separate on purpose: does the hardware
        have MFMA, does the tile divide the thread count, and does the runtime
        define the transpose it needs.  The third is the one that bites --- a
        tile whose transpose is missing produces a call to an undeclared
        template, exactly like `fmacdpp4` on gfx900.
        """
        if dtype != Datatype.F32:
            return False
        if not (cdna1(ctx) and not gfx1251(ctx)):
            return False
        if not self.fits(threads):
            return False
        return self.transpose is None or self.transpose in DEFINED_TRANSPOSES


#: Transposes `include/tensorforge_device/hip.h` actually defines.
#: `tests/test_amd_catalog.py` checks this against the header, the same way
#: the `fmacdpp` capabilities are checked --- the list is a copy of a C++ fact
#: and copies drift.
#:
#: `transpose16x2` and `transpose16x4` are in the runtime and unused by any
#: tile here.  They are not square: they permute 2 or 4 registers *within* a
#: 16-lane row rather than transposing a 16x16 tile, so they do not fit the
#: `MfmaTile` shape.  Listing them anyway keeps this set meaning what its name
#: says, which is what makes the check against the header a real check.
DEFINED_TRANSPOSES = frozenset({
    'tensorforge::transpose4x4b32',
    'tensorforge::transpose16x16b32',
    'tensorforge::transpose16x2',
    'tensorforge::transpose16x4',
})


MFMA_TILES = (
    MfmaTile(4, '__builtin_amdgcn_mfma_f32_4x4x1f32',
             'tensorforge::transpose4x4b32',
             transpose_has_separate_outputs=True),
    MfmaTile(16, '__builtin_amdgcn_mfma_f32_16x16x1f32',
             'tensorforge::transpose16x16b32'),
    # No `transpose32x32b32` exists, so `available_for` refuses this tile and
    # the wider path stays unreachable -- which is what the commented-out
    # `write_matmul(32, ...)` call used to express, silently and without
    # saying why.
    MfmaTile(32, '__builtin_amdgcn_mfma_f32_32x32x1f32',
             'tensorforge::transpose32x32b32'),
)


def usable_mfma_tiles(threads, dtype, ctx):
    """Widest first -- the order the tiling loop wants to try them in."""
    return tuple(sorted((t for t in MFMA_TILES
                         if t.available_for(threads, dtype, ctx)),
                        key=lambda t: -t.block))
