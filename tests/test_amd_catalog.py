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

HIP_H = (Path(__file__).parent.parent / "src" / "tensorforge" / "include" /
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
    """`mai-insts`, where a family range used to stand in for it.

    The two agree on every target here and differ on gfx90b--gfx90f, which
    `cdna1`'s `>= 0x90a` admitted and the hardware does not have. The
    predicate is gone; this is the property it was carrying.
    """
    ctx = _ctx(arch)
    expected = amd.has_feature(ctx, "mai-insts")
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


# --------------------------------------------------------------------------- #
# The catalogue against LLVM
# --------------------------------------------------------------------------- #
#
# `MATRIX_OPS` states shapes, block counts, per-lane fragment widths and
# feature gates.  Those are facts LLVM already writes down, so they are a copy,
# and this is the seam that keeps the copy honest -- the same construction as
# the `hip.h` checks above, one layer out.
#
# The table is vendored rather than fetched: the suite stays offline, and a
# change to what LLVM says shows up in review as a diff to
# `tests/data/amd_matrix_builtins.json` next to the catalogue change it
# justifies.  Regenerate with `tools/amd_matrix_table.py`.

import json

from tensorforge.backend.instructions.compute.primitives.amd import catalog

LLVM_TABLE = Path(__file__).parent / "data" / "amd_matrix_builtins.json"

#: Element formats a split can reach back to F32 or F64 from.  Anything
#: narrower is in `NOT_MODELLED`.
MODELLED_INPUTS = {"f32", "f64", "f16", "bf16", "xf32"}


def _llvm():
    return json.loads(LLVM_TABLE.read_text())


def _llvm_rows():
    """The builtins the catalogue claims to cover: float in, F32/F64 out."""
    return {r["builtin"]: r for r in _llvm()["builtins"]
            if r["out"] in ("f32", "f64") and r["in"] in MODELLED_INPUTS}


def test_the_catalogue_covers_every_float_matrix_builtin():
    """Equality, not containment.

    Containment would let a new generation's instructions arrive unnoticed:
    `wmma_f64_16x16x4_f64` appeared with gfx1251 and is the whole reason the
    FP64 path on that target is direct rather than emulated. `NOT_MODELLED`
    says which families are left out on purpose; nothing else may be.
    """
    assert {op.builtin for op in catalog.MATRIX_OPS} == set(_llvm_rows()), (
        "the catalogue and LLVM disagree about which float matrix "
        "instructions exist; rerun tools/amd_matrix_table.py")


@pytest.mark.parametrize("op", catalog.MATRIX_OPS, ids=lambda op: op.builtin)
def test_every_row_matches_the_llvm_signature(op):
    row = _llvm_rows()[op.builtin]
    assert (op.m, op.n, op.k) == (row["m"], row["n"], row["k"])
    assert op.wave == row["wave"]
    assert (op.gate or op.feature) == row["feature"], (
        "the row names a feature clang does not gate this builtin on")
    assert (op.a.per_lane, op.b.per_lane, op.d.per_lane) == (
        row["a"][0], row["b"][0], row["d"][0])


@pytest.mark.parametrize("op", catalog.MATRIX_OPS, ids=lambda op: op.builtin)
def test_a_stricter_row_is_a_subset_of_what_clang_accepts(op):
    """`feature` may be narrower than clang's gate, never wider.

    Narrower is the interesting direction and the reason `gate` exists: clang
    lets `mfma_f32_16x16x8_xf32` through on any `mai-insts` target, and the
    instruction only exists on gfx942, so a row that trusted the builtin gate
    would offer XF32 on gfx908 and fail in the backend. Wider would be a call
    the front end rejects outright.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import features
    ours = set(features.FEATURE_TARGETS[op.feature])
    theirs = set(features.FEATURE_TARGETS[op.gate or op.feature])
    assert ours <= theirs, f"{op.builtin}: {op.feature} reaches past {op.gate}"


@pytest.mark.parametrize("op", catalog.MATRIX_OPS, ids=lambda op: op.builtin)
def test_blocks_and_replication_account_for_every_slot(op):
    """The invariant that makes `blocks` a checkable claim.

    LLVM states `per_lane * wave`; the catalogue splits it into a block count
    and a replication factor, and only their product is determined. Asserting
    the product turns a wrong split into a failure at whichever of the three
    operands it does not fit -- which is what distinguishes a genuine 4-block
    MFMA from a 1-block WMMA whose operand is held four times over.
    """
    assert op.a.per_lane * op.wave == op.m * op.k * op.blocks * op.replication("a")
    assert op.b.per_lane * op.wave == op.k * op.n * op.blocks * op.replication("b")
    assert op.d.per_lane * op.wave == op.m * op.n * op.blocks * op.replication("d")


def test_only_the_rdna3_fragments_are_replicated():
    """The one place the operands are held more than once.

    Not a curiosity: it is why `wmma-256b-insts` costs 8 VGPRs per operand
    where `wmma-128b-insts` costs 4 for the same 16x16x16 shape. If a later
    generation joins them, this test is the place that says so.
    """
    replicated = {op.builtin for op in catalog.MATRIX_OPS
                  if op.replication("a") > 1}
    assert replicated == {op.builtin for op in catalog.MATRIX_OPS
                          if op.feature == "wmma-256b-insts"}
    for op in catalog.MATRIX_OPS:
        assert op.replication("d") == 1, "no accumulator here is replicated"


@pytest.mark.parametrize("feature,targets",
                         sorted(catalog_features := _llvm()["features"].items()))
def test_feature_targets_cover_what_llvm_lists(feature, targets):
    """Containment, because gfx940 and gfx941 are ours and not LLVM's.

    Both were removed from LLVM main as targets, not for lacking the
    instructions, and `hip.h` still guards on them. So the check is that
    nothing LLVM names is missing here -- the surplus is named in
    `features._REMOVED_FROM_LLVM`.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import features
    ours = features.FEATURE_TARGETS[feature]
    missing = [t for t in targets if int(t[3:], 16) not in ours]
    assert not missing, f"{feature}: LLVM lists {missing}, features.py does not"


@pytest.mark.parametrize("arch", ARCHS)
def test_a_row_is_only_offered_where_its_wave_matches(arch):
    """A matrix instruction is a whole-wave operation.

    `wave` is stated per row because the same shape exists in both widths --
    `wmma_f32_16x16x16_bf16_w32` and `..._w64` differ only in it -- and
    picking the wrong one gives a correctly typed call that reads the wrong
    lanes, which no snapshot would catch.
    """
    ctx = _ctx(arch)
    hw = amd.wave_size(ctx)
    for dtype in (Datatype.F32, Datatype.F64):
        for op in catalog.ops_for(dtype, _ctx(arch, dtype)):
            assert op.wave == hw
            assert op.d.dtype == dtype


def test_gfx1250_and_gfx1251_reach_the_native_paths():
    """The finding that changes what those targets need.

    gfx1250 has a native F32 matrix instruction and gfx1251 adds F64, so
    neither is a split-precision target for the types SeisSol computes in.
    Asserted rather than left to a comment, because the arch predicates in
    `arch.py` still route both to the DPP path.
    """
    f32 = {op.builtin for op in catalog.ops_for(Datatype.F32, _ctx("gfx1250"))}
    assert "wmma_f32_16x16x4_f32" in f32

    f64 = {op.builtin for op in
           catalog.ops_for(Datatype.F64, _ctx("gfx1251", Datatype.F64))}
    assert "wmma_f64_16x16x4_f64" in f64
    assert not catalog.ops_for(Datatype.F64, _ctx("gfx1250", Datatype.F64)), (
        "gfx1251-gemm-insts is gfx1251 only")


def test_fp64_mfma_is_offered_on_cdna2_and_up():
    for arch in ("gfx90a", "gfx942", "gfx950"):
        got = {op.builtin for op in
               catalog.ops_for(Datatype.F64, _ctx(arch, Datatype.F64))}
        assert got == {"mfma_f64_4x4x4f64", "mfma_f64_16x16x4f64"}
    assert not catalog.ops_for(Datatype.F64, _ctx("gfx908", Datatype.F64)), (
        "gfx908 has MFMA but no FP64 MFMA")


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

def test_bf16_needs_three_terms_and_six_products_for_f32():
    """The scheme `mfma_emu_bf16_f32` already writes out, as a formula.

    Three BF16 terms are 24 significand bits, which is F32's, and the products
    with `i + j >= 3` sit at or below its rounding error.
    """
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f32_4x4x4bf16_1k")
    terms = catalog.split_terms(op, Datatype.F32)
    assert terms == 3
    products = catalog.split_products(terms)
    assert len(products) == 6
    assert set(products) == {(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1)}


def test_products_are_ordered_smallest_contribution_first():
    products = catalog.split_products(3)
    weights = [i + j for i, j in products]
    assert weights == sorted(weights, reverse=True), (
        "products are ordered by contribution, largest last")
    assert weights[0] == 2, "the three smallest kept products come first"
    assert products[-1] == (0, 0), "the leading term is accumulated last"


def test_xf32_needs_fewer_terms_than_bf16():
    """Why the gfx942 path is worth having beside the BF16 one.

    Eleven significand bits against eight: two terms reach 22 bits where BF16
    reaches 16, so the reduced variant is much closer to F32 for the same
    three products.
    """
    xf32 = next(o for o in catalog.MATRIX_OPS if "xf32" in o.builtin)
    bf16 = next(o for o in catalog.MATRIX_OPS if o.builtin.endswith("bf16_1k"))
    assert xf32.significand > bf16.significand
    assert len(catalog.split_products(2)) == 3


def test_a_direct_row_needs_a_single_term():
    op = next(o for o in catalog.MATRIX_OPS if o.builtin == "mfma_f64_4x4x4f64")
    assert catalog.split_terms(op, Datatype.F64) == 1
    assert catalog.split_products(1) == ((0, 0),)


# --------------------------------------------------------------------------- #
# cbsz, and which instructions the lane-batched scheme can feed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("op", [o for o in catalog.MATRIX_OPS if o.broadcast],
                         ids=lambda op: op.builtin)
def test_cbsz_never_names_a_block_the_instruction_does_not_have(op):
    """The bound that `threads // block` did not respect.

    `cbsz` selects a broadcast group, so it cannot exceed `log2(blocks)`. That
    was invisible while every entry had `k == 1`, because there `blocks ==
    wave // n` makes `threads // n` and `blocks` the same number at a full
    wave. `mfma_f64_4x4x4f64` has four blocks and sixteen lanes per row, and
    the old expression asked it for `cbsz = 4`.
    """
    for threads in (4, 8, 16, 32, 64):
        if threads % op.n:
            continue
        assert 0 <= op.cbsz(threads) <= max(op.blocks - 1, 0).bit_length()
        assert 2 ** op.cbsz(threads) <= op.blocks


def test_cbsz_reproduces_the_hand_written_table():
    """Same numbers as before, from the instruction instead of the tile.

    `LEGACY_SCALE` is reproduced exactly, which is what makes this a
    refactoring of the K=1 path rather than a change to it.
    """
    for block, table in LEGACY_SCALE.items():
        op = next(o for o in catalog.MATRIX_OPS
                  if o.m == o.n == block and o.k == 1
                  and o.a.dtype is Datatype.F32)
        for threads, expected in table.items():
            assert op.cbsz(threads) == expected, (
                f"{op.builtin} at {threads} threads")


def test_only_the_k1_f32_tiles_are_lane_batched():
    """The precondition, spelled out over the whole catalogue.

    `matmul32` puts the leading dimension in the lanes and one contraction
    value per instruction, so the instruction's `n * blocks` has to be the
    whole wave -- which, at one element per lane, is the same statement as
    `k == 1`. This is the test that changes when the staging that puts k into
    lanes lands.
    """
    got = {op.builtin for op in catalog.MATRIX_OPS if op.lane_batched()}
    assert got == {"mfma_f32_4x4x1f32", "mfma_f32_16x16x1f32",
                   "mfma_f32_32x32x1f32"}


@pytest.mark.parametrize("op", catalog.MATRIX_OPS, ids=lambda op: op.builtin)
def test_lane_batching_is_exactly_k_equals_one(op):
    """The equivalence the docstring derives, checked rather than asserted."""
    scalar_square = (op.a.per_lane == 1 and op.b.per_lane == 1
                     and op.m == op.n)
    if scalar_square:
        assert op.lane_batched() == (op.k == 1)
        assert (op.n * op.blocks == op.wave) == (op.k == 1)


def test_the_fp64_mfmas_are_offered_but_not_lane_batched():
    """Both exist on gfx90a and neither fits today's loop.

    Which contradicts what a matching per-lane width suggests: A and B are one
    element per lane, the same as the K=1 F32 tiles. The difference is the
    assignment, not the width -- `k == 4` means two of the six lane bits carry
    the contraction, and the generator's data operand carries the leading
    dimension there.
    """
    ctx = _ctx("gfx90a", Datatype.F64)
    offered = {op.builtin for op in catalog.ops_for(Datatype.F64, ctx)}
    assert offered == {"mfma_f64_4x4x4f64", "mfma_f64_16x16x4f64"}
    for op in catalog.ops_for(Datatype.F64, ctx):
        assert op.a.per_lane == 1 and op.b.per_lane == 1
        assert not op.lane_batched()
    assert not catalog.lane_batched_ops(Datatype.F64, ctx)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dtype", [Datatype.F32, Datatype.F64])
def test_the_router_and_the_emitter_ask_the_same_question(arch, dtype):
    """`matmul()` gates on `mfma_tile_for`; so does `matmul32`.

    They used to ask differently -- a family predicate at the router, a
    `next()` without a default at the emitter -- and a disagreement surfaced
    as `StopIteration` out of code generation.
    """
    ctx = _ctx(arch, dtype)
    for threads in (4, 16, 32, 64):
        tile = catalog.mfma_tile_for(threads, dtype, ctx)
        if tile is not None:
            assert tile.block == 4
            assert tile.op.lane_batched()
            tile.scale(threads)          # must not raise
        else:
            assert not [t for t in amd.usable_mfma_tiles(threads, dtype, ctx)
                        if t.block == 4]


def test_f64_still_takes_the_dpp_path():
    """Now because no tile fits, not because the condition named F32.

    `fmacdpp16(double&, double, double)` is in the runtime and `select` picks
    it on CDNA 2, so F64 is served -- the point is that the reason is now a
    structural one the catalogue states.
    """
    for arch in ("gfx90a", "gfx942", "gfx950"):
        assert catalog.mfma_tile_for(64, Datatype.F64,
                                     _ctx(arch, Datatype.F64)) is None
