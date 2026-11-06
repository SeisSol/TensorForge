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
    BroadcastForm, MATERIALIZE_FROM)
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
    """The runtime materializes a row share and nothing narrower.

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
    assert MATERIALIZE_FROM == 4


@pytest.mark.parametrize('arch', RDNA3PLUS + GFX125X)
@pytest.mark.parametrize('reuse', [2, 3])
def test_the_tie_region_keeps_the_modifier(arch, reuse):
    """Where the move buys no slot, it is code size spent on nothing."""
    assert amd.broadcast_form(Datatype.F32, 16, reuse, _ctx(arch)) \
        is BroadcastForm.FUSED


# --------------------------------------------------------------------------- #
# the packed arrangements: one move for a register pair
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch,moves', [
    ('gfx942', 1), ('gfx950', 1), ('gfx1251', 1),
    ('gfx90a', 2), ('gfx1250', 2), ('gfx1150', 2), ('gfx1100', 2)])
def test_a_pair_moves_in_one_instruction_only_with_a_64_bit_move(arch, moves):
    """`dpp-64bit` alone is not enough, and llvm-mc says why: gfx90a has the
    unit and no `v_mov_b64` for it to modify, gfx1250 the move and no unit.
    Both legalize a 64-bit `mov_dpp` into two 32-bit ones."""
    assert amd.dpp_move_instructions(4, _ctx(arch)) == 1
    assert amd.dpp_move_instructions(8, _ctx(arch)) == moves
    assert amd.dpp_move_instructions(16, _ctx(arch)) == 2 * moves


@pytest.mark.parametrize('arch,slots,expected', [
    ('gfx942', 1, False),    # 1 + 1 against 2
    ('gfx942', 2, True),     # 1 + 2 against 4
    ('gfx1251', 2, True),
    ('gfx90a', 2, False),    # 2 + 2 against 4
    ('gfx90a', 3, True),     # 2 + 3 against 6
    ('gfx1250', 2, False),
    ('gfx1250', 3, True),
    ('gfx1150', 8, False),   # no packed FMA to pair into
    ('gfx1100', 8, False),
])
def test_a_column_pair_is_packed_where_the_move_repays(arch, slots, expected):
    """One move and `slots` packed FMAs against `2 * slots` fused ones -- the
    same count in slots and in instructions, so no tie region to argue."""
    assert amd.packed_broadcast(Datatype.F32, 16, 2 * slots, 8,
                                _ctx(arch)) is expected


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1251'])
def test_fp64_and_narrow_broadcasts_stay_fused(arch):
    """No `v_pk_fma_f64` in this LLVM, and no row share to move below 16."""
    assert not amd.packed_broadcast(Datatype.F64, 16, 8, 16, _ctx(arch))
    assert not amd.packed_broadcast(Datatype.F32, 8, 8, 8, _ctx(arch))


def test_a_body_rebuilt_over_budget_takes_no_move():
    """`_fused_if_over_budget` asks again with the moves forbidden."""
    ctx = _ctx('gfx942')
    ctx.force_fused_broadcast = True
    assert not amd.packed_broadcast(Datatype.F32, 16, 4, 8, ctx)


# --------------------------------------------------------------------------- #
# the move on CDNA, where the fused form is assembly (`amd_moved_broadcast`)
# --------------------------------------------------------------------------- #

def _asked(arch, dtype=Datatype.F64, on=True):
    from tensorforge.common.context import Options
    return Context(arch=arch, backend='hip', fp_type=dtype,
                   options=Options(amd_moved_broadcast=on))


@pytest.mark.parametrize('arch', CDNA)
@pytest.mark.parametrize('dtype', [Datatype.F64, Datatype.F32])
def test_cdna_takes_the_move_where_asked(arch, dtype):
    """Nothing pairs a CDNA FMA in double precision, and in single the
    caller holds no pairs: the modifier would stay -- but it is assembly
    there, which the scheduler cannot enter, and the move a builtin it can.
    Asked for, the move is taken from the reuse the other targets take it
    from; not asked for, nothing changes."""
    form = amd.select_broadcast_form(dtype, 16, MATERIALIZE_FROM,
                                     _asked(arch, dtype), can_pack=False)
    assert form is BroadcastForm.MOVED
    form = amd.select_broadcast_form(dtype, 16, MATERIALIZE_FROM,
                                     _asked(arch, dtype, on=False),
                                     can_pack=False)
    assert form is BroadcastForm.FUSED


@pytest.mark.parametrize('arch', CDNA)
def test_cdna_keeps_the_modifier_below_the_reuse(arch):
    """One move per product below four is the tie region again."""
    assert (amd.broadcast_form(Datatype.F64, 16, MATERIALIZE_FROM - 1,
                               _asked(arch))
            is BroadcastForm.FUSED)


@pytest.mark.parametrize('arch', ['gfx908', 'gfx1150', 'gfx1251'])
def test_the_move_on_cdna_is_cdna_only(arch):
    """gfx908 has no row share to move; RDNA and gfx1251 decide by their own
    doubling (VOPD, packed FP64), as before."""
    on = amd.broadcast_form(Datatype.F64, 16, MATERIALIZE_FROM, _asked(arch))
    off = amd.broadcast_form(Datatype.F64, 16, MATERIALIZE_FROM,
                             _asked(arch, on=False))
    assert on is off


def _two_products(dtype=Datatype.F64, m=56, n=9):
    """``T = A0 X; Q = A1 T`` over 56 rows: at sixteen lanes four lead slots a
    lane, so every moved value feeds four products, and each product has an
    operand of its own, as SeisSol's have."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import MultilinearDescr

    def tensor(shape, alias, addressing, is_tmp=False):
        return Tensor(list(shape), addressing, BoundingBox([0, 0], list(shape)),
                      alias=alias, datatype=dtype, is_tmp=is_tmp)

    a0 = tensor([m, m], 'A0', Addressing.NONE)
    a1 = tensor([m, m], 'A1', Addressing.NONE)
    x = tensor([m, n], 'X', Addressing.PTR_BASED)
    t = tensor([m, n], 'T', Addressing.STRIDED, is_tmp=True)
    q = tensor([m, n], 'Q', Addressing.PTR_BASED)
    target, permute = [[0, -1], [-1, 1]], [[0, 1], [0, 1]]
    return [MultilinearDescr(dest=SubTensor(t), ops=[SubTensor(a0), SubTensor(x)],
                             target=target, permute=permute),
            MultilinearDescr(dest=SubTensor(q), ops=[SubTensor(a1), SubTensor(t)],
                             target=target, permute=permute)]


def _built(on):
    import contextlib
    import io
    from tensorforge.common.context import Options
    from tensorforge.generators.generator import Generator
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F64,
                  options=Options(lanes_per_mult=16, amd_moved_broadcast=on,
                                  reassociate_products=False))
    gen = Generator(_two_products(), ctx, attrs={})
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def test_asked_for_on_gfx942_the_products_read_moves():
    """With the option every row share of the two products is a `movdpp16`
    and no fused block is left; without it the blocks stay and no move is
    written."""
    moved, fused = _built(True), _built(False)
    assert 'movdpp16<' in moved and 'v_fmac_f64_dpp' not in moved
    assert 'movdpp16<' not in fused and 'v_fmac_f64_dpp' in fused


def test_the_fused_transpose_is_taken_where_asked():
    """`amd_fused_transpose`: the 4x4 quads of the lane-batched MFMA path go
    through `transpose4x4b32Fused` instead of `transpose4x4b32`."""
    import contextlib
    import io
    import seissol_suite as suite
    from tensorforge.common.context import Options
    from tensorforge.frontend.yateto import DescriptionReader
    from tensorforge.generators.generator import Generator

    def built(on):
        d = DescriptionReader(None, {}).read(suite.description(
            'elastic-linearck', 'elastic-linearck-o4-s', 'gpu_volume'))[0]
        ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                      options=Options(amd_fused_transpose=on))
        gen = Generator(d, ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        return gen.get_kernel()
    on, off = built(True), built(False)
    assert 'transpose4x4b32Fused(' in on and 'transpose4x4b32(' not in on
    assert 'transpose4x4b32Fused(' not in off and 'transpose4x4b32(' in off
