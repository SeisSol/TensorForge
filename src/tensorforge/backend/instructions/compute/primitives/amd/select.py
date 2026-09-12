# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Broadcast width selection, and whether the broadcast is an instruction.

Two questions about the same operand.  How *wide* a broadcast reaches is
`select_fmadpp_step`: strategy first, then narrowed to what the runtime
defines, and the split is the point -- a performance tweak to the first must
not be able to turn into a link error.

What *form* it takes is `broadcast_form`, and it is a question the width does
not answer.  A DPP modifier replicates a lane for free, in the sense that it
costs no instruction of its own -- but it costs the multiply the right to be
one of two: an instruction carries a modifier or packed operands, and a VOPD
pair carries neither.  So a fused broadcast buys its move by halving the FMA
rate, and where the same replicated value feeds several products, buying the
move once outright is cheaper.  Which of those wins is a count over the reuse,
and what the alternative looks like differs per target: packed math wants the
products paired in registers, dual issue wants them independent and scalar.
"""

from enum import Enum

from tensorforge.common.basic_types import Datatype
from .arch import cdna2, gfx1250, gfx1251, rdna, gfx906
from .caps import has_fmacdpp4, has_fmacdpp8, has_fmacdpp16
from .features import has_feature


def wanted_fmadpp_step(datatype, threads, ctx):
    """The widest broadcast the *hardware strategy* wants -- performance only.

    Deliberately says nothing about whether the target defines it; see
    `select_fmadpp_step`.  Keeping the two apart is what stops a performance
    tweak from silently becoming a link error, which is how gfx900 came to
    emit a call to a template that has only a declaration there.
    """
    step = 1
    if threads >= 4 and datatype == Datatype.F32 and gfx906(ctx):
        step = 4
    if threads >= 8 and datatype == Datatype.F32 and (rdna(ctx) or gfx1251(ctx) or gfx1250(ctx)):
        step = 8
    if threads >= 16 and datatype == Datatype.F32 and (rdna(ctx) or gfx1251(ctx) or gfx1250(ctx)):
        step = 16
    if threads >= 16 and (cdna2(ctx) or gfx1251(ctx)) and datatype in (Datatype.F32, Datatype.F64):
        step = 16
    return step


def select_fmadpp_step(datatype, threads, ctx):
    """What we can actually emit: the strategy, narrowed to what links.

    Falls to the next *narrower* width the target defines.  Narrower is always
    correct -- it just costs more broadcasts -- so an unavailable instruction
    degrades performance instead of breaking the build.
    """
    wanted = wanted_fmadpp_step(datatype, threads, ctx)
    available = {
        1: lambda: True,                       # plain FMA, always there
        4: lambda: has_fmacdpp4(ctx),
        8: lambda: has_fmacdpp8(ctx),
        16: lambda: has_fmacdpp16(ctx, datatype),
    }
    return next(s for s in (wanted, 8, 4, 1) if s <= wanted and available[s]())


class BroadcastForm(Enum):
    """How a replicated operand reaches the multiply."""

    #: The broadcast as a modifier on the FMA: `fmacdpp{step}<row>`.  Costs no
    #: instruction and forfeits the doubling, so it is one issue per product.
    FUSED = 'fused'
    #: One move, then ordinary scalar FMAs the target pairs by itself.  What
    #: VOPD wants: the products stay independent and nothing is packed.
    MOVED = 'moved'
    #: One move, then FMAs over packed operands.  What VOP3P wants: the
    #: accumulator and the multiplicand have to sit in register pairs, which
    #: is a constraint on the *caller*, not on the move.
    PACKED = 'packed'


def packed_fma_lanes(datatype, ctx) -> int:
    """Products one packed FMA covers on this target.

    Two rows per type rather than one, because AMDGPU.td names the same
    arithmetic twice: CDNA carries `packed-fp32-ops` and gfx125x gates its
    packed math on the target, leaving the `single-sgpr` refinement as the
    only record that mentions it.  Reading both is what keeps this from
    claiming gfx1250 has no `v_pk_fma_f32` -- and from claiming RDNA 3 has
    one, which is the mistake in the other direction.
    """
    if datatype == Datatype.F32:
        return 2 if (has_feature(ctx, 'packed-fp32-ops')
                     or has_feature(ctx, 'packed-fp32-single-sgpr-ops')) else 1
    if datatype == Datatype.F64:
        return 2 if has_feature(ctx, 'packed-fp64-single-sgpr-ops') else 1
    return 1


def dual_issue_fma_lanes(datatype, ctx) -> int:
    """Products one VOPD pair covers.

    FP32 only: `v_dual_fmac_f32` has no 64-bit counterpart, so an FP64 chain
    reaches the doubling through packed math or not at all.
    """
    if datatype == Datatype.F32 and has_feature(ctx, 'vopd'):
        return 2
    return 1


def dpp_move_instructions(nbytes, ctx) -> int:
    """DPP moves it takes to replicate `nbytes` of one lane across its row.

    64 bits at a time where the target has both a 64-bit DPP unit
    (`dpp-64bit`) and a 64-bit move for it to modify -- `v_mov_b64`, which
    LLVM derives from `gfx940-insts` and `gfx1250-insts`: gfx942, gfx950 and
    gfx1251.  gfx90a has the unit and no move, its DPP64 serving `v_fmac_f64`
    alone; gfx1250 has the move and no unit.  Both take two 32-bit moves for a
    pair, which is what LLVM legalises `mov_dpp` on an `i64` into there.
    """
    wide = (has_feature(ctx, 'dpp-64bit')
            and (has_feature(ctx, 'gfx940-insts')
                 or has_feature(ctx, 'gfx1250-insts')))
    unit = 8 if wide else 4
    return -(-nbytes // unit)


def packed_broadcast(datatype, step, products, moved_bytes, ctx) -> bool:
    """Whether one move of `moved_bytes` and packed FMAs beat fused ones.

    `products` is how many multiplies the moved value feeds, all of them in
    pairs one `v_pk_fma_f32` retires together -- two columns of one lead slot,
    or the two rows a lane holds at lead width two.  Fused, each product is an
    issue of its own; packed, the move costs `dpp_move_instructions` and the
    products half as many.  The same count in slots and in instructions, so
    unlike `broadcast_form` there is no tie region to argue over: a column
    pair over two lead slots is 3 against 4 where a 64-bit move exists and 4
    against 4 where it does not, and only the first is taken.

    FP32 only.  The LLVM of ROCm 7.2 has no `v_pk_fma_f64` on any target, so
    a pair of doubles would be two moves and four scalar FMAs against four
    fused ones.  And only at the row-share width, the one broadcast the
    runtime materialises (`movdpp16`).
    """
    if getattr(ctx, 'force_fused_broadcast', False):
        return False
    if datatype != Datatype.F32 or step < 16:
        return False
    if packed_fma_lanes(datatype, ctx) < 2:
        return False
    moves = dpp_move_instructions(moved_bytes, ctx)
    if moves + -(-products // 2) >= products:
        return False
    # Told to the body, as `select_broadcast_form` does: the moved values and
    # the paired accumulators live in registers until their last product.
    ctx.materialised_broadcast = True
    return True


#: Products sharing one broadcast, from which a materialised move is taken.
#:
#: Two counts, and the move has to win the first without losing the second by
#: more.  In *issue slots* a fused product cannot pair, so `n` of them cost
#: `n`, against the move plus the paired products, `1 + ceil(n/2)`: equal at 2
#: and at 3, and the move ahead from 4.  In *instructions* the move is always
#: one more, `1 + n` against `n`, whatever `n` is.
#:
#: So below 4 the move buys no slot and still costs the instruction, which is
#: the losing side of both counts -- and the instruction is not free.  The
#: kernels this generator is aimed at are already short of instruction cache
#: at order 6, so an arrangement that trades code size for nothing is exactly
#: what that budget cannot afford.  From 4 the trade is real: one instruction
#: for one slot per broadcast, improving as `n` grows.
#:
#: Three things the slot count does not hold argue for going lower -- a
#: DPP-modified FMA carries a throughput tax of its own on several parts, it
#: gives up the chance to co-issue with unrelated VALU work, and it pays the
#: wait states when its source register was just written, which materialising
#: pays once at the move instead of once per product.  None of them is
#: measured here, and in the tie region they would have to be worth a 50%
#: larger inner loop for nothing back.  A measurement is what moves this
#: number.
#:
#: The one taken so far is about the other side of the trade, registers.
#: `local_flux` at 16 lanes on gfx1150 sits exactly at 4: 828 moves, 196 of
#: some 8800 FMAs paired, and 5.6 KB of scratch -- 46 times the runtime, where
#: the fused form was 8 % faster than the default.  So the number stays and
#: the body decides: `_fused_if_over_budget` builds again fused when the
#: materialised form does not fit the target's register budget.
MATERIALISE_FROM = 4


def broadcast_form(datatype, step, reuse, ctx) -> BroadcastForm:
    """Which form the broadcast of `reuse` products should take.

    `reuse` is how many multiplies read the same replicated value -- the lead
    slots one column of the chain accumulates into.  It is what the whole
    decision turns on: at one product a move is pure overhead, and the more
    products share it the less of the move each carries.

    `step` is the broadcast width, and it bounds the answer from the other
    side: the runtime materialises a row share (`movdpp16`) and nothing else,
    so a quad-permute broadcast has no move to be taken out of it and stays
    fused however often it is reused.  Adding the quad-permute move is what
    would lift that, and it belongs with the instruction rather than here.
    """
    if step < 16 or reuse < MATERIALISE_FROM:
        return BroadcastForm.FUSED
    if packed_fma_lanes(datatype, ctx) > 1:
        return BroadcastForm.PACKED
    if dual_issue_fma_lanes(datatype, ctx) > 1:
        return BroadcastForm.MOVED
    # Nothing here retires two of these at once, so the modifier is free after
    # all and a move would be an instruction spent on nothing.
    return BroadcastForm.FUSED


def select_broadcast_form(datatype, step, reuse, ctx,
                          can_pack: bool) -> BroadcastForm:
    """The form, narrowed to what the caller can emit.

    The same split `select_fmadpp_step` makes, for the same reason: what is
    fastest and what this generator can currently write down are two questions,
    and answering them together is how a preference turns into a wrong kernel.

    `can_pack` is whether the caller holds its products in register pairs.
    Where it does not, packed math is unreachable and the move is still worth
    taking on a target that dual-issues -- one move and independent scalar
    FMAs is a complete arrangement, not a degraded one.  Where nothing pairs
    the FMAs either, the modifier costs nothing and the move goes back.
    """
    if getattr(ctx, 'force_fused_broadcast', False):
        # The body this is part of did not fit with a materialised broadcast
        # and is being built again; see `_fused_if_over_budget`.
        return BroadcastForm.FUSED
    form = broadcast_form(datatype, step, reuse, ctx)
    if form is BroadcastForm.PACKED and not can_pack:
        form = (BroadcastForm.MOVED if dual_issue_fma_lanes(datatype, ctx) > 1
                else BroadcastForm.FUSED)
    if form is not BroadcastForm.FUSED:
        # Told to the body, which is the only place the cost can be weighed:
        # the moved values live in registers until their last product.
        ctx.materialised_broadcast = True
    return form
