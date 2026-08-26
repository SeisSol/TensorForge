# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The MFMA catalogue describes instructions that exist.

`MFMA_TILES` names, per tile width, the intrinsic to call and the cross-lane
transpose its A operand needs.  The transpose names are a copy of a fact that
lives in `hip.h`, so they are checked against it -- the same seam, and the same
reason, as the `fmacdpp` capabilities in `test_amd_caps.py`.

The 32-wide tile is the case worth having a test for.  Its transpose does not
exist, and the way that used to be expressed was a commented-out call site:
nothing said why it was commented out, and uncommenting it would have emitted
a call to an undeclared template. `available_for` refuses the tile instead,
which is a statement a reader can act on.

The `scale` argument gets its own tests because it stopped being data.  It was
three nested dicts of hand-written constants; it is `log2(threads // block)`.
Tabulating a formula does not just duplicate it, it introduces holes -- a legal
combination missing from the table raised `KeyError` from an unrelated entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

HIP_H = (Path(__file__).parent.parent / "tensorforge" / "include" /
         "tensorforge_device" / "hip.h")

ARCHS = ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx942", "gfx950",
         "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx1250", "gfx1251"]

#: The scale table that used to be written out by hand, kept as the reference
#: the formula has to reproduce.
LEGACY_SCALE = {
    4: {64: 4, 32: 3, 16: 2, 8: 1, 4: 0},
    16: {64: 2, 32: 1, 16: 0},
    32: {64: 1, 32: 0},
}


def _ctx(arch, dtype=Datatype.F32):
    return Context(arch=arch, backend="hip", fp_type=dtype)


def _defined_in_header() -> set:
    """Transposes `hip.h` gives a body, not just a declaration."""
    src = HIP_H.read_text()
    out = set()
    for m in re.finditer(r"\bvoid\s*\n?\s*(transpose\w+)\s*\(", src):
        name = m.group(1)
        # a declaration ends in `;`, a definition opens a brace
        rest = src[m.end():]
        depth, i = 1, 0
        while i < len(rest) and depth:
            if rest[i] == "(":
                depth += 1
            elif rest[i] == ")":
                depth -= 1
            i += 1
        after = rest[i:].lstrip()
        if after.startswith("{"):
            out.add(f"tensorforge::{name}")
    return out


# --------------------------------------------------------------------------- #
# The catalogue against the header
# --------------------------------------------------------------------------- #

def test_defined_transposes_matches_the_header():
    assert amd.DEFINED_TRANSPOSES == _defined_in_header(), (
        "the transpose list in amd.py has drifted from hip.h")


def test_every_tile_names_a_transpose_or_says_it_needs_none():
    for tile in amd.MFMA_TILES:
        assert tile.transpose is None or tile.transpose.startswith("tensorforge::")


@pytest.mark.parametrize("arch", ARCHS)
def test_no_usable_tile_needs_a_missing_transpose(arch):
    """The property the 32-wide tile exists to test."""
    for threads in (4, 8, 16, 32, 64):
        for tile in amd.usable_mfma_tiles(threads, Datatype.F32, _ctx(arch)):
            assert (tile.transpose is None
                    or tile.transpose in amd.DEFINED_TRANSPOSES), (
                f"{arch}/threads={threads}: tile {tile.block} would call "
                f"{tile.transpose}, which the runtime does not define")


def test_the_32_wide_tile_is_refused_everywhere():
    """Not a permanent truth -- a statement of the current blocker.

    If someone writes `transpose32x32b32`, this test fails and points at the
    place where the tile becomes selectable.
    """
    tile = next(t for t in amd.MFMA_TILES if t.block == 32)
    assert tile.transpose not in amd.DEFINED_TRANSPOSES
    for arch in ARCHS:
        for threads in (32, 64):
            assert tile not in amd.usable_mfma_tiles(
                threads, Datatype.F32, _ctx(arch))


# --------------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("block,table", sorted(LEGACY_SCALE.items()))
def test_scale_reproduces_the_hand_written_table(block, table):
    tile = next(t for t in amd.MFMA_TILES if t.block == block)
    for threads, expected in table.items():
        assert tile.scale(threads) == expected


@pytest.mark.parametrize("threads", [4, 8, 16, 32, 64])
def test_scale_covers_combinations_the_table_omitted(threads):
    """The formula answers wherever the tile divides the thread count, which
    is the constraint that actually applies."""
    for tile in amd.MFMA_TILES:
        if tile.fits(threads):
            assert tile.scale(threads) >= 0
        else:
            with pytest.raises(ValueError):
                tile.scale(threads)


def test_scale_rejects_a_thread_count_the_tile_does_not_divide():
    tile = next(t for t in amd.MFMA_TILES if t.block == 16)
    with pytest.raises(ValueError):
        tile.scale(8)          # narrower than the tile
    with pytest.raises(ValueError):
        tile.scale(24)         # not a multiple


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arch", ARCHS)
def test_tiles_are_offered_only_where_mfma_exists(arch):
    ctx = _ctx(arch)
    expected = amd.cdna1(ctx) and not amd.gfx1251(ctx)
    got = bool(amd.usable_mfma_tiles(32, Datatype.F32, ctx))
    assert got == expected, f"{arch}: MFMA offered={got}, expected={expected}"


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dtype", [Datatype.F64, Datatype.F32])
def test_only_f32_tiles_are_offered(arch, dtype):
    """Every entry is an `f32` intrinsic; offering one for f64 would emit a
    call whose operand types do not match."""
    tiles = amd.usable_mfma_tiles(32, dtype, _ctx(arch, dtype))
    if dtype != Datatype.F32:
        assert not tiles
    for t in tiles:
        assert "f32" in t.builtin


def test_usable_tiles_come_widest_first():
    ctx = _ctx("gfx90a")
    widths = [t.block for t in amd.usable_mfma_tiles(64, Datatype.F32, ctx)]
    assert widths == sorted(widths, reverse=True)


def test_unused_transposes_are_recorded_not_silently_dropped():
    """`DEFINED_TRANSPOSES` means what its name says.

    Two of its entries belong to no tile. Trimming the set to only what the
    catalogue uses would make the check against `hip.h` pass by construction
    and stop being a check at all.
    """
    used = {t.transpose for t in amd.MFMA_TILES} - {None}
    unused = amd.DEFINED_TRANSPOSES - used
    assert unused, "the set has been trimmed to what the catalogue uses"


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 32, 64])
def test_an_offered_tile_can_always_be_emitted(arch, threads):
    """Offering implies usable, end to end.

    A tile that does not divide the thread count has no `scale`, so `scale()`
    raises -- which means `available_for` and `scale` have to agree about
    which combinations are legal. Checking the transpose alone leaves that
    agreement untested: the 16-wide tile at 4 threads has a transpose and is
    still unemittable.
    """
    ctx = _ctx(arch)
    for tile in amd.usable_mfma_tiles(threads, Datatype.F32, ctx):
        tile.scale(threads)      # must not raise
        assert tile.fits(threads)
