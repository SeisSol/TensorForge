# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Whether a broadcast is worth an instruction of its own.

A DPP modifier replicates a lane at no instruction cost and at the price of
the multiply's right to be one of two.  Both halves of that are hardware
facts, they do not cover the same parts, and a family predicate cannot stand
in for either: `vopd` starts at RDNA 3 while `rdna()` starts four generations
earlier, and `dpp-64bit` splits gfx1250 from gfx1251 where every family
predicate keeps them together.

So the rows are checked here against what they claim, and the decision is
checked against the count it is derived from.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.primitives.amd.features import (
    has_feature)
from tensorforge.backend.instructions.compute.primitives.amd.select import (
    BroadcastForm, MATERIALISE_FROM)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

CDNA = ['gfx90a', 'gfx942', 'gfx950']
RDNA3PLUS = ['gfx1100', 'gfx1200']
GFX125X = ['gfx1250', 'gfx1251']


def _ctx(arch, dtype=Datatype.F32):
    return Context(arch=arch, backend='hip', fp_type=dtype)


# --------------------------------------------------------------------------- #
# what the two doubling mechanisms cover
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch', CDNA + GFX125X)
def test_packed_fp32_covers_cdna_and_gfx125x(arch):
    """`v_pk_fma_f32` exists on both, under two different feature names."""
    assert amd.packed_fma_lanes(Datatype.F32, _ctx(arch)) == 2


@pytest.mark.parametrize('arch', ['gfx900', 'gfx906', 'gfx908',
                                  'gfx1010', 'gfx1030'] + RDNA3PLUS)
def test_packed_fp32_is_absent_from_rdna(arch):
    """RDNA reaches two FP32 FMAs through VOPD and has no VOP3P form of them.

    Worth its own test because the two are easy to conflate: the throughput is
    the same and the instruction is not, and a kernel that packs its operands
    for a `v_pk_fma_f32` that does not exist there gets neither.
    """
    assert amd.packed_fma_lanes(Datatype.F32, _ctx(arch)) == 1


def test_packed_fp64_is_gfx1251_alone():
    assert amd.packed_fma_lanes(Datatype.F64, _ctx('gfx1251')) == 2
    for arch in ['gfx1250', 'gfx942', 'gfx950', 'gfx90a', 'gfx1200']:
        assert amd.packed_fma_lanes(Datatype.F64, _ctx(arch)) == 1


@pytest.mark.parametrize('arch', RDNA3PLUS + GFX125X)
def test_dual_issue_reaches_gfx125x(arch):
    """gfx125x carries VOPD as well, so the chain is not an RDNA-only path."""
    assert amd.dual_issue_fma_lanes(Datatype.F32, _ctx(arch)) == 2


@pytest.mark.parametrize('arch', ['gfx1010', 'gfx1030'] + CDNA)
def test_dual_issue_is_absent_from_rdna2_and_cdna(arch):
    assert amd.dual_issue_fma_lanes(Datatype.F32, _ctx(arch)) == 1


@pytest.mark.parametrize('arch', RDNA3PLUS + GFX125X + CDNA)
def test_nothing_dual_issues_fp64(arch):
    """`v_dual_fmac_f32` has no 64-bit counterpart on any part."""
    assert amd.dual_issue_fma_lanes(Datatype.F64, _ctx(arch)) == 1


# --------------------------------------------------------------------------- #
# the width of one DPP move
# --------------------------------------------------------------------------- #

def test_dpp64_splits_gfx1250_from_gfx1251():
    """One DPP instruction moves a 64-bit unit on gfx1251 and two do on gfx1250.

    The one place the two parts differ here, and no family predicate sees it:
    `rdna()` stops below both and `cdna2()` folds gfx1251 in by hand.  What
    reads the row is the packed arrangement, which pays one move where the
    unit goes in one piece and two where it does not.
    """
    assert has_feature(_ctx('gfx1251'), 'dpp-64bit')
    assert not has_feature(_ctx('gfx1250'), 'dpp-64bit')


@pytest.mark.parametrize('arch', CDNA)
def test_dpp64_is_on_cdna2_and_later(arch):
    assert has_feature(_ctx(arch), 'dpp-64bit')


@pytest.mark.parametrize('arch', ['gfx908', 'gfx1030', 'gfx1100', 'gfx1200'])
def test_a_wide_move_elsewhere_is_two_instructions(arch):
    assert not has_feature(_ctx(arch), 'dpp-64bit')


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch', CDNA + RDNA3PLUS + GFX125X)
def test_a_single_product_keeps_the_modifier(arch):
    """One product cannot repay a move, whatever the target offers."""
    assert amd.broadcast_form(Datatype.F32, 16, 1, _ctx(arch)) \
        is BroadcastForm.FUSED


@pytest.mark.parametrize('arch,expected', [
    ('gfx90a', BroadcastForm.PACKED),
    ('gfx942', BroadcastForm.PACKED),
    ('gfx1251', BroadcastForm.PACKED),
    ('gfx1250', BroadcastForm.PACKED),
    ('gfx1100', BroadcastForm.MOVED),
    ('gfx1200', BroadcastForm.MOVED),
])
def test_reuse_takes_the_move_out_in_the_shape_the_target_wants(arch, expected):
    """Packed where the products have to be paired, scalar where they must not.

    The distinction is the whole reason this returns three answers rather than
    a boolean: both emit one move, and what follows it is a different kernel.
    """
    assert amd.broadcast_form(Datatype.F32, 16, 4, _ctx(arch)) is expected


@pytest.mark.parametrize('arch', ['gfx900', 'gfx906', 'gfx908',
                                  'gfx1010', 'gfx1030'])
def test_without_a_doubling_mechanism_the_modifier_is_free(arch):
    """Nothing to forfeit, so the move buys nothing at any reuse."""
    assert amd.broadcast_form(Datatype.F32, 16, 16, _ctx(arch)) \
        is BroadcastForm.FUSED


def test_fp64_reaches_the_move_only_where_packed_fp64_exists():
    """No VOPD for FP64, so gfx1251 is the one part where it pays."""
    assert amd.broadcast_form(Datatype.F64, 16, 4, _ctx('gfx1251')) \
        is BroadcastForm.PACKED
    for arch in ['gfx90a', 'gfx942', 'gfx950', 'gfx1250']:
        assert amd.broadcast_form(Datatype.F64, 16, 4, _ctx(arch)) \
            is BroadcastForm.FUSED


@pytest.mark.parametrize('step', [1, 4])
def test_a_narrower_broadcast_has_no_move_to_be_taken_out_of_it(step):
    """The runtime materialises a row share and nothing narrower.

    A quad-permute broadcast would need a `movdpp4` that does not exist, so
    reuse cannot reach past the modifier there however high it goes.
    """
    assert amd.broadcast_form(Datatype.F32, step, 16, _ctx('gfx90a')) \
        is BroadcastForm.FUSED


def test_the_threshold_is_where_both_counts_agree():
    """`n` fused issues against `1 + ceil(n/2)`, and `n` instructions against
    `1 + n`.

    The slot count ties at 2 and 3 and the instruction count never ties, so
    below 4 the move costs code for nothing and from 4 it buys a slot per
    broadcast.  Pinning the constant to that derivation is what makes moving
    it a decision rather than a drift.
    """
    def slots_fused(n):
        return n

    def slots_moved(n):
        return 1 + -(-n // 2)

    assert [n for n in range(1, 8) if slots_moved(n) < slots_fused(n)] == \
        [4, 5, 6, 7]
    assert [n for n in range(1, 8) if slots_moved(n) == slots_fused(n)] == [2, 3]
    assert all(1 + n > n for n in range(1, 8))
    assert MATERIALISE_FROM == 4


@pytest.mark.parametrize('arch', RDNA3PLUS + GFX125X)
@pytest.mark.parametrize('reuse', [2, 3])
def test_the_tie_region_keeps_the_modifier(arch, reuse):
    """Where the move buys no slot, it is code size spent on nothing."""
    assert amd.broadcast_form(Datatype.F32, 16, reuse, _ctx(arch)) \
        is BroadcastForm.FUSED
