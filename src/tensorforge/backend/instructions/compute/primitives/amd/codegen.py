# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Code generation for the multilinear kernel.
"""

from tensorforge.common.basic_types import Datatype
from tensorforge.backend.writer import Writer
from tensorforge.backend.pir.core import ScalarType
from .arch import cdna2, gfx1251, rdna
from .catalog import mfma_tile_for
from ... import split
from .emitters import fmadpp4, fmadpp8, fmadpp16, fmascalar
from .exchange_codegen import _writeback, apply_exchange
from .layouts import FRAGMENT_BITS
from .relayout import (MOVDPP16, TRANSPOSE4X4, find_relayout,
                       nest_shared, reach, takes, transposed,
                       fmadpp_operand_layout)
from tensorforge.common.exceptions import GenerationError
from .select import (BroadcastForm, packed_broadcast, select_broadcast_form,
                     select_fmadpp_step)

#: The runtime's BF16 split, in its out-parameter form: each term is a value
#: the generator declared rather than a name bound by a structured binding,
#: which is what lets a layout ride on it and a pass see who consumed it.
SPLIT_BF16 = 'tensorforge::splitFloatx4BF16'

#: Whether the moved broadcast (`BroadcastForm.MOVED`, the VOPD form) keeps
#: its move from being folded back into the FMAs (`movdpp16Kept`).  LLVM's
#: DPP combine does that from gfx10 on wherever it can, and a DPP-modified
#: FMA cannot pair.  Off until the measurement on gfx1150 says which way it
#: goes.
PIN_MOVED = False

#: How the fused chain walks its products.  `'columns'` is the order `hfma`
#: was written for: every `A(i, k)` read first (`_load_a`) and held while the
#: columns go by one after the other.  `'rows'` is the order of the packed
#: arrangements: the contraction outermost, each `A(i, k)` read at its row and
#: dead after the last column, the accumulators pinned at the end of each row.
#: Every accumulator receives the same products in the same order either way,
#: so the two agree to the bit -- what differs is what is live.  `'auto'`
#: picks per chain (`_fused_order`).  Measured on gfx1150, `local_flux`: 219
#: VGPRs down to 130, occupancy 4 to 7, 3.6 % faster; at 16 lanes with the
#: faces merged 23.5 % faster, and no longer spilling.
FUSED_ORDER = 'auto'

#: The size of the `A` image, in registers a lane, from which `'auto'`
#: considers the rows at all.  `local_flux` holds 112 (two slots over 56
#: steps), which on gfx1150 was half of its 219 VGPRs; below this the image
#: does not decide the occupancy, and the rows' extra reads and branches are
#: all that is left of them.
AUTO_ROWS_FROM = 64

#: Whether lead width above one takes the fused chain where the packed one
#: does not pay (`_matmuldpp_wide`), rather than leaving it to the nest.  Off:
#: on gfx1150 (`local_flux`, eight mults, no packed FMA) it ran 7 % behind the
#: nest at the same occupancy -- 153 ns an element against 143, 129 VGPRs
#: against 141 -- in either order.  It was on while the nest computed
#: `slice_offset_a` wrong at lead width two, from a broadcast inside a lane
#: guard; `passes.converge_crosslane` took that out.
FUSED_WIDE = False


def _check_mfma_operand(operand, threads, callee, tile=None):
    """The A operand of an MFMA has to be laid out as the transpose left it.

    `None` means untracked and is allowed through: an absent annotation is
    not evidence of a wrong one, and refusing to emit for want of one would
    make the layouts an obstacle rather than a description.  A present layout
    that disagrees is a wrong kernel.

    Only where the transpose hands back values of its own.  One that writes
    its operands in place (`transpose16x16b32`) leaves them carrying the
    layout they had before, which says nothing about what they hold now.
    """
    if tile is not None and not tile.transpose_has_separate_outputs:
        return
    want = TRANSPOSE4X4.produces(threads=threads)
    got = getattr(operand, 'layout', None)
    if got is not None and got != want:
        raise ValueError(f'{callee} needs its A operand at {want!r}, '
                         f'got {got!r}')


def _refuse_multiwave(threads, ctx):
    """A multiplication wider than the wave is refused on the DPP paths.

    Measured on gfx1150 (local_flux, 64 lanes over 32-wide waves): the kernel
    came out wrong with no error and no spill -- first through the DPP
    broadcast, which ends at the wave, and still wrong with the broadcast
    narrowed to one lane, so it is not the exchange alone.  A wrong kernel is
    worse than none; on a 64-wide wave the same width is one wave and builds.
    """
    hw = ctx.get_vm().get_hw_descr()
    wave = getattr(hw, 'vec_unit_length', None)
    if wave and threads > wave:
        raise GenerationError(
            f'a multiplication of {threads} lanes spans {-(-threads // wave)} '
            f'waves of {wave} on this target, which the AMD SIMT path does not '
            f'compute correctly (the register broadcast ends at the wave)')


def hfma(writer: Writer, Cs, As, Bs, repeat, datatype, threads, ctx):
    """The broadcast chain: one lane of `A` against a row of `B`, per product.

    Two decisions, both asked of `select`.  How wide the broadcast reaches is
    `select_fmadpp_step`.  Whether it is a modifier on each multiply or an
    instruction of its own is `select_broadcast_form`, and `repeat` is what
    that turns on: it is how many products read the same replicated value, so
    it is what a materialised move would be divided by.

    `can_pack=False` states what this emitter holds.  The products of one
    broadcast are separate accumulators here, not a register pair, so packed
    math is out of reach and the move is worth taking only where the target
    pairs scalar FMAs by itself.  The arrangements that do hold register
    pairs -- two columns of a slot, or the rows of a lane at lead width two --
    are `matmuldpp`'s, which asks `packed_broadcast` before it comes here.
    """
    _refuse_multiwave(threads, ctx)

    step = select_fmadpp_step(datatype, threads, ctx)
    form = select_broadcast_form(datatype, step, repeat, ctx, can_pack=False)

    fma = fmascalar
    func = {
        1: fmascalar,
        4: fmadpp4,
        8: fmadpp8,
        16: fmadpp16
    }[step]

    bcstmin = 3 # if cdna2(ctx) else 2
    bcststep = 2 if datatype == Datatype.F32 and (cdna2(ctx) or gfx1251(ctx)) else 1
    bcst = datatype == Datatype.F32 and repeat * bcststep >= bcstmin and (cdna2(ctx) or rdna(ctx))

    # disable for now
    bcst = False
    bcststep = 1

    for b in range(0, len(Cs), bcststep):
        A = [As[bb] for bb in range(b, min(b + bcststep, len(Cs)))]
        B = [Bs[bb] for bb in range(b, min(b + bcststep, len(Cs)))]
        C = [Cs[bb] for bb in range(b, min(b + bcststep, len(Cs)))]

        # assert all(len(B[0]) == len(b) for b in B)

        localstep = len(B)

        ftype = ScalarType(datatype)
        # A packed pair for the `movdpp16` path.  `ScalarType(base, length)`
        # is what the lexic renders as `float2` -- the hard-coded name the
        # commented-out line above was reaching for.
        vtype = ScalarType(datatype, localstep) if localstep > 1 else ftype

        for i in range(0, len(B[0]) // repeat, step):
            if step == threads:
                a = A
            else:
                a = []
                for aa in A:
                    # What the instruction downstream needs, asked of the
                    # table rather than assumed.  `hfma` used to name the
                    # broadcast directly and state its result layout beside
                    # it; the two disagreed for a while, because a name and a
                    # claim written in two places can.  Now the requirement is
                    # stated once, the table answers with the instruction that
                    # meets it, and `fmadpp` checks the same requirement on
                    # arrival.
                    want = fmadpp_operand_layout(step)
                    found = find_relayout(want, threads)
                    if found is None:
                        raise ValueError(
                            f'no instruction reaches {want!r} at {threads} '
                            f'threads')
                    entry, params = found
                    # `lane` picks which sub-block, which is the algorithm's
                    # business: the table cannot know it, and says so.
                    params = dict(params, lane=i // step)
                    # Cross-lane: reads other lanes' registers, so it is only
                    # defined where the wave is converged -- pure (CSE and
                    # inlining stay correct) but not hoistable.
                    a += [writer.call(
                        entry.callee.format(**params),
                        ftype, aa, hint='bc', movable=False,
                        layout=entry.produces(**params))]

            for j in range(min(len(B[0][i*repeat:]) // repeat, step)):
                idx = (j + i) * repeat

                ax = []

                if form is BroadcastForm.MOVED:
                    # The row that `fmacdpp16<j>` would apply per product,
                    # applied once and read `repeat` times.  Same table and
                    # same layout as the fused form asks for, so the two
                    # arrangements compute the same products -- what differs
                    # is that these multiplies carry no modifier and may
                    # therefore be issued in pairs.
                    mv = dict(threads=threads, row=j)
                    callee = (f'tensorforge::movdpp16Kept<{j}>' if PIN_MOVED
                              else MOVDPP16.callee.format(**mv))
                    ax = [writer.call(callee, ftype, aa,
                                      hint='bc', movable=False,
                                      layout=MOVDPP16.produces(**mv))
                          for aa in a]
                    usebcst = True
                elif bcst and all((B[bx][idx + jj] if idx + jj < len(B[bx]) else None) is not None for bx in range(localstep) for jj in range(repeat)):
                    aa = writer.pack(vtype, *a, hint='pk')
                    # Same table, same reason.  This path is switched off
                    # below (`bcst`), so no snapshot exercises it -- which is
                    # exactly when taking the layout from a checked row rather
                    # than writing one out by hand is worth most.
                    mv = dict(threads=threads, row=j)
                    aa2 = writer.call(MOVDPP16.callee.format(**mv), vtype, aa,
                                      hint='bc', movable=False,
                                      layout=MOVDPP16.produces(**mv))
                    ax = ([writer.extract(aa2, bx, ftype) for bx in range(localstep)]
                          if localstep > 1 else [aa2])
                    usebcst = True
                else:
                    ax = a
                    usebcst = False

                for jj in range(repeat):
                    for bx in range(localstep):
                        if idx + jj < len(B[bx]):
                            # NOTE: `b` used to shadow the loop variable of the
                            # enclosing `for b in range(0, len(Cs), bcststep)`,
                            # so the block index was destroyed on the first
                            # iteration that got here.
                            bv = B[bx][idx + jj]
                            c = C[bx][idx + jj]
                            if bv is not None:
                                if usebcst:
                                    fma(writer, c, ax[bx], bv, j)
                                else:
                                    func(writer, c, ax[bx], bv, j)


def _pad(writer, tile, ftype):
    """A zero for a padding lane of a partial block, as `_transpose` can take
    it: a constant where the transpose writes fresh outputs, a zeroed variable
    where it writes its arguments in place (`transpose16x16b32`) -- a
    reference parameter cannot bind a literal."""
    if tile.transpose is None or tile.transpose_has_separate_outputs:
        return writer.const(0.0, ftype)
    return writer.declare(ftype, hint='pad')


def _transpose(writer, tile, ftype, threads, regs):
    """Exchange the register index with the lane index in a quad.

    Emitted in SSA form where the instruction allows it:
    `tp(w1..wn, v1..vn)` declares its outputs separately from its
    inputs, so fresh values come out.  That matters for two
    reasons beyond tidiness.

    The outputs can carry a layout.  The exchange *changes* the
    distribution --- afterwards both the register dimension and
    the lane dimension vary with the lane, which is the one
    genuinely rank-2 layout this generator produces --- and a
    value that already exists cannot say so, because
    `Value.layout` is fixed when the value is created.

    And the inputs are left alone.  Rewriting them underneath
    whatever else still reads them is what forces `call_stmt` to
    pin their producers, which in turn bars those loads from
    being reused.  Returning new values costs four declarations
    and gives that back.

    `transpose16x16b32` has no such form --- all sixteen
    parameters are by reference --- so it keeps the in-place path,
    and its results stay untracked.
    """
    if tile.transpose is None:
        return list(regs)
    if not tile.transpose_has_separate_outputs:
        writer.call_stmt(tile.transpose, *regs, writes=tuple(regs))
        return list(regs)
    out = [writer.declare(ftype, hint='tp',
                          layout=TRANSPOSE4X4.produces(threads=threads))
           for _ in regs]
    writer.call_stmt(tile.transpose, *out, *regs, writes=tuple(out))
    return out


def _accumulator_direct(op) -> bool:
    """Whether element `(m, n)` of block `b` sits in slot `m` of lane
    `b * n + n` -- the way the lane-batched store reads the accumulator, one
    output column a slot and the leading dimension across the lanes.

    The 4-wide tile has it.  The 16-wide one spreads a block's output over
    the whole wave (`catalog.LANE_BATCHED_BLOCKS`), and its columns are
    gathered back by the epilogue the exchange scheme uses
    (`accumulator_gathers`).
    """
    row = FRAGMENT_BITS.get(op.builtin)
    if row is None:
        return False
    blk, m_bits, n_bits = row[6], row[7], row[8]
    return (blk == tuple(op.n << b for b in range(len(blk)))
            and m_bits == tuple(-(1 << b) for b in range(len(m_bits)))
            and n_bits == tuple(1 << b for b in range(len(n_bits))))


def _column(writer, tile, acc, column, ftype):
    """Output column `column` of one MFMA accumulator, at the store's layout:
    the slot itself where the layout is direct, gathered where it is not."""
    if _accumulator_direct(tile.op):
        return writer.extract(acc, column, ftype)
    value = _writeback(writer, tile.op, {(0, 0): acc}, 0, column, ftype)
    if value is None:
        raise GenerationError(
            f'{tile.builtin}: column {column} of the accumulator has no '
            f'gather back to the lanes the store reads')
    return value


def _blgp(mults, q):
    """The `blgp` handing every lane the B operand of the `q`-th of `mults`
    multiplications sharing the wave.

    1 and 2 broadcast the lower and the upper 32 lanes to all 64; 4 to 7 one
    of the four groups of 16.  F32 entries only: on CDNA3 the field is the
    negation modifier of an F64 MFMA.
    """
    if mults == 2:
        return 1 + q
    if mults == 4:
        return 4 + q
    raise ValueError(f'blgp has no pattern for {mults} multiplications')


def _shared_fragment(writer, tile, ftype, threads, regs):
    """The shared matrix at the layout the A fragment wants.

    Asked rather than assumed.  `matmul32` transposed unconditionally because
    that is what its own operands need, which is true and is not the same
    statement as the instruction needing it -- and an operand arriving already
    right would have been transposed anyway.  `None` here is the gap this
    instruction does not close; the caller declines rather than emitting
    something that does not reach the fragment.
    """
    block = tile.block
    route = reach(nest_shared(block, threads), transposed(block, threads),
                  block, [(c, l) for c in range(block) for l in range(threads)],
                  wave=threads)
    if not takes(route):
        # A staged trip, which this emitter does not write: the buffer has to
        # be reserved before any body exists.  Declining sends the operation
        # to the generic nest, which is slower and right, rather than to a
        # reservation that was never made.  Asked through `takes` rather than
        # by testing the shape here, because `strategies` decides whether to
        # offer this arrangement from the same sentence and the two must not
        # drift.
        return None
    if route == 0:
        return list(regs)
    if route == 1:
        return _transpose(writer, tile, ftype, threads, regs)
    # The same exchange assembled out of swaps and merges, which is what a
    # width the runtime has no `transpose*` for gets.  More instructions than
    # the builtin and far fewer than a trip through memory.
    return apply_exchange(writer, regs, route, ftype)


def matmul32(writer: Writer, C, B, A, M, N, K, kx, threads, dtype, sparse,
             ctx, start, stop, width=1, tile=None, lead_wave=None, mults=1,
             lead_quad=None, quad=0):
    with writer.AnonymousScope():

        ftype = ScalarType(dtype)

        def write_matmul(tile, start, end):
            block = tile.block
            scale = tile.scale(threads)
            fn = tile.builtin

            def transpose(regs):
                return _shared_fragment(writer, tile, ftype, threads, regs)

            # The MFMA accumulator layout is deliberately left untracked.
            #
            # It is a hardware register assignment -- which lane of which
            # block holds which element of C -- and it is not derivable from
            # anything in this file. Writing down a plausible one would be
            # worse than writing nothing: `None` means *unknown*, and every
            # check treats an unknown layout as distinct from every other, so
            # a pass stays conservative. A wrong layout is not conservative --
            # it lets a pass merge two values that actually differ.
            acclayout = None

            # TODO: use Bctrl for threads in (16, 32)

            # C <- C + B@A.  `end` bounds the blocks; `N` still bounds the
            # real columns inside one, which is what pads a partial block.
            for j in range(start, end, block):
                with writer.AnonymousScope():
                    tA = {}
                    for k in range(0, K + kx, threads):
                        regs = []
                        for jj in range(min(block, N - j)):
                            regs += [A(writer, None, j + jj, k // threads)]
                        for jj in range(min(block, N - j), block):
                            # The padding lanes of a partial block: real
                            # zeroes, so that the MFMA over the full block
                            # contributes nothing for them.
                            regs += [_pad(writer, tile, ftype)]
                        reached = transpose(regs)
                        if reached is None:
                            return False
                        tA[k // threads] = reached
                    for i in range(0, M):
                        with writer.AnonymousScope():
                            vtype = ScalarType(dtype, block)
                            acc = writer.declare(vtype, hint='acc',
                                                 layout=acclayout)

                            def step(acc, trueK, lead, blgp):
                                """One contraction step: the shared matrix's
                                slot for `trueK`, broadcast from its block by
                                `abid`, against the lead operand `lead`."""
                                km = trueK // threads
                                kkm = (trueK % threads) // block
                                kkkm = trueK % block
                                _check_mfma_operand(tA[km][kkkm], threads, fn,
                                                    tile)
                                return writer.call(
                                    fn, vtype, tA[km][kkkm], lead, acc,
                                    scale, kkm, blgp,
                                    hint='acc', movable=False,
                                    materialize=True, layout=acclayout)

                            if lead_quad is not None:
                                # The operand stored in k-quads
                                # (`amd.prepared_order`): one read is `quad`
                                # steps of the lane's row.  With `blgp` the
                                # `p`-th multiplication reads quad `q + p`, so
                                # one read serves `mults * quad` steps; the
                                # quads a group does not fill read alone.
                                depth = K + kx
                                group = mults * quad
                                for g in range(0, depth, quad):
                                    if g % group == 0 and g + group <= depth:
                                        lead = lead_quad(writer, i, g // quad,
                                                         mults)
                                        for q in range(mults):
                                            for c in range(quad):
                                                acc = step(
                                                    acc, g + q * quad + c,
                                                    writer.extract(lead, c,
                                                                   ftype),
                                                    _blgp(mults, q)
                                                    if mults > 1 else 0)
                                        continue
                                    if mults > 1 and g % group:
                                        continue
                                    for q0 in range(g, min(g + group, depth),
                                                    quad):
                                        lead = lead_quad(writer, i, q0 // quad)
                                        for c in range(min(quad, depth - q0)):
                                            acc = step(
                                                acc, q0 + c,
                                                writer.extract(lead, c, ftype),
                                                0)
                                for jj in range(min(block, N - j)):
                                    C(writer,
                                      _column(writer, tile, acc, jj, ftype),
                                      i, j + jj)
                                continue

                            if lead_wave is not None:
                                # The multiplications of a wave read the same
                                # rows of a batch-constant lead operand, so
                                # each reads a different step of them -- the
                                # `p`-th reads `k + p` -- and every MFMA of
                                # the group takes one of them from its lanes
                                # (`blgp`).  One read of `mults` steps where
                                # each step was read `mults` times.  A tail
                                # shorter than the group reads as before.
                                depth = K + kx
                                for g in range(0, depth, mults):
                                    lead = (lead_wave(writer, i, g, mults)
                                            if g + mults <= depth else None)
                                    if lead is not None and lead is not False:
                                        for q in range(mults):
                                            acc = step(acc, g + q, lead,
                                                       _blgp(mults, q))
                                        continue
                                    for trueK in range(g, min(g + mults, depth)):
                                        lead = B(writer, None, i, trueK)
                                        if lead is None or lead is False:
                                            continue
                                        acc = step(acc, trueK, lead, 0)
                                for jj in range(min(block, N - j)):
                                    C(writer,
                                      _column(writer, tile, acc, jj, ftype),
                                      i, j + jj)
                                continue

                            for k in range(0, K + kx, threads):
                                dk = min(threads, K + kx - k)
                                for kk in range(0, dk, block):
                                    # NOTE: no scope here.  It used to isolate
                                    # the `tmpB_*` names; those now come from
                                    # the shared allocator and are unique
                                    # anyway.  Keeping it would trap the
                                    # accumulator: with the chain in SSA the
                                    # updated value is *declared* at the MFMA,
                                    # not assigned to a variable that outlives
                                    # the braces.
                                    tB = [None] * block
                                    dkk = min(block, dk - kk)
                                    for kkk in range(dkk):
                                        tB[kkk] = B(writer, None, i, k + kk + kkk)
                                    for kkk in range(dkk, block):
                                        tB[kkk] = writer.const(0.0, ftype)
                                    for kkk in range(dkk):
                                        if tB[kkk] is None or tB[kkk] is False:
                                            continue
                                        trueK = k + kk + kkk #+ kx
                                        km = trueK // threads
                                        kkm = ((trueK % threads) // block)
                                        kkkm = trueK % block

                                        assert km == k // threads
                                        assert kkm == kk // block
                                        assert kkkm == kkk
                                        # the index for tmpB is correct
                                        #
                                        # The A operand has to arrive in the
                                        # distribution the transpose left it
                                        # in.  Checked rather than assumed:
                                        # this is the one operand whose layout
                                        # is rank 2, and getting it wrong
                                        # would feed the intrinsic a correctly
                                        # typed value holding the wrong
                                        # elements -- which no snapshot and no
                                        # symbolic comparison would notice,
                                        # since both treat the intrinsic as
                                        # opaque.
                                        _check_mfma_operand(tA[km][kkkm],
                                                            threads, fn, tile)
                                        # MFMA *returns* the updated
                                        # accumulator, so the chain is
                                        # naturally SSA -- each step reads the
                                        # previous result.
                                        acc = writer.call(
                                            fn, vtype,
                                            tA[km][kkkm], tB[kkk], acc,
                                            scale, kkm, 0,
                                            hint='acc', movable=False,
                                            materialize=True,
                                            layout=acclayout)

                            for jj in range(min(block, N - j)):
                                C(writer, _column(writer, tile, acc, jj, ftype),
                                  i, j + jj)
            return True

        def write_wide(tile, start, end):
            """`write_matmul` at lead width `width`, one component at a time.

            The lanes of the lane-batched scheme are independent rows, so
            which row a lane holds is the kernel's business, and at width `w`
            lane `t` holds `w` of them: `w * t + c` in component `c`
            (`componentwise`).  Component `c` of every lane is then a
            lane-batched problem of its own -- one element per lane, as the
            B fragment wants -- with an accumulator of its own, packed back
            into the `w`-vector the store takes.

            The shared matrix arrives the same way along its contraction:
            lane `t` of the block at `k0` holds step `k0 + w * t + c` in
            component `c`.  So each component is transposed on its own, and
            its fragment serves the steps `w` apart that it holds, with `abid`
            picking the quad as before.  The same count of MFMAs as at width
            one -- `M` is a `w`-th of it and each step issues `w`.  Not quite
            the same count of transposes: one per component of a contraction
            block `w` times as long, so a contraction shorter than the block
            transposes `w` times where width one did once (`local_flux`'s 9x9
            products: 32 against 24).
            """
            block = tile.block
            scale = tile.scale(threads)
            fn = tile.builtin
            span = threads * width
            vtype = ScalarType(dtype, block)
            wtype = ScalarType(dtype, width)
            for j in range(start, end, block):
                with writer.AnonymousScope():
                    frag = {}
                    for k0 in range(0, K + kx, span):
                        cols = []
                        for jj in range(min(block, N - j)):
                            v = A(writer, None, j + jj, k0 // span)
                            if not _spread(v, width, threads):
                                return False
                            cols.append(v)
                        for c in range(width):
                            regs = [writer.extract(v, c, ftype) for v in cols]
                            regs += [_pad(writer, tile, ftype)
                                     for _ in range(block - len(cols))]
                            reached = _shared_fragment(writer, tile, ftype,
                                                       threads, regs)
                            if reached is None:
                                return False
                            frag[(k0, c)] = reached
                    for i in range(M):
                        with writer.AnonymousScope():
                            accs = [writer.declare(vtype, hint='acc')
                                    for _ in range(width)]
                            for k0 in range(0, K + kx, span):
                                for t in range(threads):
                                    for c in range(width):
                                        k = k0 + width * t + c
                                        if k >= K + kx:
                                            continue
                                        rows = B(writer, None, i, k)
                                        if rows is None or rows is False:
                                            continue
                                        if _width_of(rows) != width:
                                            return False
                                        a = frag[(k0, c)][t % block]
                                        _check_mfma_operand(a, threads, fn,
                                                            tile)
                                        for h in range(width):
                                            accs[h] = writer.call(
                                                fn, vtype, a,
                                                writer.extract(rows, h, ftype),
                                                accs[h], scale, t // block, 0,
                                                hint='acc', movable=False,
                                                materialize=True)
                            for jj in range(min(block, N - j)):
                                C(writer, writer.pack(wtype, *(
                                    _column(writer, tile, acc, jj, ftype)
                                    for acc in accs)), i, j + jj)
            return True

        # The tiling policy, now separate from what the tiles are.  Only the
        # 4-wide tile is reachable today: the 16-wide one needs a shared-memory
        # staging step that is not written, and the 32-wide one has no
        # transpose in the runtime, which `available_for` already refuses.
        #
        # `matmul()` is the gate; this is the guard for a direct caller, and
        # both ask `mfma_tile_for`.  A `next()` without a default raised
        # `StopIteration` here instead, which unwinds out of generation as an
        # unrelated-looking error.
        # The tile `rank` chose, where the caller passes it; the narrowest
        # otherwise, for a direct caller.
        if tile is None:
            tile = mfma_tile_for(threads, dtype, ctx)
        if tile is None:
            raise ValueError(
                f'no MFMA tile for {dtype} at {threads} threads; '
                f'matmul() should have taken the DPP path')

        return (write_matmul if width == 1 else write_wide)(tile, start, stop)


    # TODO: gfx1200, f'__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12'

def matmulemu(writer: Writer, C, B, A, M, N, K, kx, threads, dtype, sparse,
              ctx, start, stop, tile, terms):
    """`C += B@A` through a narrower matrix instruction, one tile per term
    product.

    The same loop `matmul32` runs, over an instruction whose operands are
    `tile.op.k` contraction values wide rather than one.  Four scalar
    registers become one k-vector, and that is not a rearrangement this
    emitter performs: `layouts.position` puts element `(m, k)` of the wide
    fragment at slot `k` of the lane the narrow fragment puts `(m, 0)` in, so
    the four registers a `kk` step already holds *are* the four slots.  The
    same holds of B.  `tests/test_amd_emulation.py` asserts it rather than
    this comment claiming it, because a wrong stacking here yields a
    correctly typed operand holding the wrong elements -- which no snapshot
    would notice.

    One instruction per term product, all accumulating into the same `acc`.
    That arrangement needs no axis of its own and no epilogue, which is why it
    works under this mapping without anything being freed for it first: the
    products are issued in sequence and the accumulator sums them the way it
    sums the contraction.  `packing.stages` is the layout, and what it costs
    against packing them into spare positions is `packing.saving`.

    Products come from `split.products` in its order, smallest contribution
    first, so the small terms land before the large one rounds them off.
    """
    products = split.products(terms)
    op = tile.op
    width = op.k

    with writer.AnonymousScope():
        ftype = ScalarType(dtype)
        block = tile.block
        scale = tile.scale(threads)
        fn = op.callee
        # The split's outputs, and the instruction's operands.  Not `dtype`:
        # that is what the accumulator keeps, and reaching it is the whole
        # reason there is more than one product.
        termtype = ScalarType(Datatype.I16, width)
        acclayout = None

        def split4(regs, hint):
            """One operand's `width` registers as `terms` narrow k-vectors."""
            out = [writer.declare(termtype, hint=hint) for _ in range(terms)]
            writer.call_stmt(SPLIT_BF16, *out, *regs, writes=tuple(out))
            return out

        for j in range(start, stop, block):
            with writer.AnonymousScope():
                tA = {}
                for k in range(0, K + kx, threads):
                    regs = []
                    for jj in range(min(block, N - j)):
                        regs += [A(writer, None, j + jj, k // threads)]
                    for jj in range(min(block, N - j), block):
                        regs += [writer.const(0.0, ftype)]
                    tA[k // threads] = _transpose(writer, tile, ftype, threads,
                                                  regs)
                for i in range(0, M):
                    with writer.AnonymousScope():
                        vtype = ScalarType(dtype, block)
                        acc = writer.declare(vtype, hint='acc',
                                             layout=acclayout)
                        for k in range(0, K + kx, threads):
                            dk = min(threads, K + kx - k)
                            for kk in range(0, dk, width):
                                dkk = min(width, dk - kk)
                                tB = []
                                for kkk in range(dkk):
                                    tB += [B(writer, None, i, k + kk + kkk)]
                                if any(v is None or v is False for v in tB):
                                    return False
                                for kkk in range(dkk, width):
                                    # A real zero rather than a shorter
                                    # k-vector: the instruction reads all of
                                    # its slots and a tail that does not fill
                                    # them contributes nothing only if they
                                    # hold nothing.
                                    tB += [writer.const(0.0, ftype)]
                                kkm = kk // width

                                aterms = split4(tA[k // threads], 'at')
                                bterms = split4(tB, 'bt')
                                for ti, tj in products:
                                    acc = writer.call(
                                        fn, vtype,
                                        aterms[ti], bterms[tj], acc,
                                        scale, kkm, 0,
                                        hint='acc', movable=False,
                                        materialize=True, layout=acclayout)

                        for jj in range(min(block, N - j)):
                            C(writer, writer.extract(acc, jj, ftype), i,
                              j + jj)
    return True


def _value(v):
    """A read that produced something, as against the `None` or `False` a
    declined one hands back."""
    return v is not None and v is not False


def _width_of(v):
    return getattr(v.type, 'length', None) or 1


def _spread(v, width, threads):
    """Whether a `B` read is what the packed chain broadcasts from.

    `width` elements a lane, one lane per contraction step -- and not a value
    the loader has already replicated, which is what a fixed element read
    through the broadcast path is: an image packed another way than the one
    the chain asks for comes back as `broadcast<16, 1, k>(r[0])`, a correctly
    typed scalar that holds one element for every lane.  The chain would move
    it again and multiply the same step into every row.
    """
    if not _value(v) or _width_of(v) != width:
        return False
    layout = getattr(v, 'layout', None)
    return layout is None or layout == fmadpp_operand_layout(threads)


def _row_source(writer, src, vtype, threads, step, lane):
    """`src` at the distribution `movdpp16` reads.

    One sub-block of `step` lanes, repeated -- asked of the same table and
    for the same layout `hfma` asks it for, so the packed arrangements move
    the same elements the fused one multiplies.
    """
    if step == threads:
        return src
    want = fmadpp_operand_layout(step)
    found = find_relayout(want, threads)
    if found is None:
        raise ValueError(f'no instruction reaches {want!r} at {threads} '
                         f'threads')
    entry, params = found
    params = dict(params, lane=lane)
    return writer.call(entry.callee.format(**params), vtype, src, hint='bc',
                       movable=False, layout=entry.produces(**params))


def _move_row(writer, src, vtype, threads, row):
    """`movdpp16<row>`: lane `row` of each 16-lane row, to all of it."""
    mv = dict(threads=threads, row=row)
    return writer.call(MOVDPP16.callee.format(**mv), vtype, src, hint='bc',
                       movable=False, layout=MOVDPP16.produces(**mv))


def _pin(writer, accumulators):
    """End of a row: its FMAs have to precede the next row's moves.

    `tensorforge::pin` on each accumulator the row wrote.  Arithmetic carries
    no chain through instruction selection, so without it an accumulator chain
    is free to be linearised after every input it reads -- a body's moves all
    first, each live until its FMA.  Measured on `local_flux` (gfx1251, lead
    width two): 740 VGPRs and 252 moves ahead of the first FMA; pinned every
    row, 161 and 10.  Every fourth row was not enough for the column pairs
    (512 VGPRs and 800 B of scratch against 230 and none).
    """
    for acc in accumulators:
        writer.call_stmt('tensorforge::pin', acc, writes=(acc,))


def _load_a(writer, A, M, K, kx):
    """Every `A(i, k)` the chain reads, keyed `(i, k + kx)`.

    `None` asks the loader for the value rather than for a name to fill in:
    the intrinsics below take these as operands, and an operand whose
    definition the IR cannot see is invisible to every pass that reasons about
    ordering or reuse.
    """
    ab = {}
    for k in range(K):
        for i in range(M):
            res = A(writer, None, i, k)
            if _value(res):
                ab[(i, k + kx)] = res
    return ab


def matmuldpp(writer, start, C, A, B, M, N, K, kx, threads, dtype, sparse,
              ctx, stop, width=1, a_resident=False, a_vector=False):
    """`C += A @ B` over columns `[start, stop)`, `B` broadcast across lanes.

    Three arrangements of the same products, all reading `B(k, j)` from lane
    `k` of a row with a row share:

    * the scalar chain (`hfma`): each product carries its broadcast as a DPP
      modifier, or takes a moved one -- `select_broadcast_form`;
    * column pairs, at lead width one: `B(k, j)` and `B(k, j + 1)` moved
      together and multiplied into both columns of a slot by one packed FMA;
    * lead width above one (`_matmuldpp_wide`), where the lanes hold vectors
      of rows and of contraction steps both.

    False where it declines, which the nest takes as "compute it yourself".
    """
    if start >= stop:
        # Nothing left for this path.  Worth an early return rather than
        # letting the loops come out empty: the A operands below are loaded
        # before the first `for j`, so falling through would emit a full set
        # of reads with no consumer.
        return True
    _refuse_multiwave(threads, ctx)
    if width > 1:
        return _matmuldpp_wide(writer, start, stop, C, A, B, M, K, kx,
                               threads, dtype, sparse, ctx, width, a_resident)
    scalar = list(range(start, stop))
    if sparse is None and stop - start >= 2 and M % 2 == 0:
        step = select_fmadpp_step(dtype, threads, ctx)
        if packed_broadcast(dtype, step, 2 * M, 2 * dtype.size(), ctx):
            pairs = list(range(start, stop - 1, 2))
            if _paired_columns(writer, C, A, B, pairs, M, K, kx, threads,
                               dtype, step):
                scalar = scalar[2 * len(pairs):]
    if scalar:
        # What is left goes through the scalar chain: an odd last column, or
        # all of them where a `B` read declined.  Its `A` reads are its own
        # rather than the pairs' -- holding those across the pairs would be
        # the pressure the pairs' order avoids.
        if sparse is None and _fused_order(
                scalar, M, K, dtype, select_fmadpp_step(dtype, threads, ctx),
                threads, a_resident, a_vector) == 'rows':
            _fused_rows(writer, scalar, C, A, B, M, K, kx, threads, dtype,
                        ctx)
        else:
            ab = _load_a(writer, A, M, K, kx)
            _scalar_chain(writer, scalar[0], scalar[-1] + 1, C, B, ab, M, K,
                          kx, threads, dtype, sparse, ctx)
    return True


def _fused_order(cols, M, K, dtype, step, threads, a_resident, a_vector=False):
    """`FUSED_ORDER`, with `'auto'` decided by what each order keeps live.

    The column order keeps two things live that the row order does not.  The
    `A` image, `M * K` values, across every column -- where `A` is read from
    memory: one held in registers anyway (`a_resident`) costs nothing more,
    and a single column reads each value once in either order (`local_flux`
    on gfx942, whose one fused column is what the matrix core leaves over).
    And, where the broadcast reaches one lane at a time (no DPP for the type,
    step one), the value it relays for every step of every column, which
    nothing stops LLVM from issuing ahead of the products they feed.
    `add_true_f64` on gfx1150 has 128 of those in FP64: 256 VGPRs and 176 B
    of scratch, against 83 and none in rows.  A wider broadcast relays one
    register a sub-block only; counting those took one chain of `chain_five`
    into rows, for 4 to 7 % more instructions and not a register less.

    A single column reads each value once in either order -- unless `A` is
    read several steps at a time (`a_vector`, the k-quads of
    `amd.prepared_order`).  Then the column order reads every vector before
    the chain and holds each until its last step is consumed, which is the
    whole `A` image again: `local_flux` on gfx942 with k-quads, 216 VGPRs in
    columns against 164 in rows (b = 35: 200 against 152), for 7 % more
    instructions.  At b = 80 and 120, which spill either way, the two are
    within 3 %.

    The row order keeps every column's accumulators and its `B` register,
    and pays elsewhere: an `A` value read at one row is one LLVM may sink
    into a branch of its own where the read is guarded (`rectangular` on
    gfx1150, 34 more), and one it no longer shares with a neighbouring chain
    (`local_flux` on gfx942, 220 more LDS reads).  So the rows only where the
    column order holds a lot and the row order less than half of it.
    """
    if FUSED_ORDER != 'auto':
        return FUSED_ORDER
    n = len(cols)
    relayed = step == 1 < threads
    held = ((M * K if (n > 1 or a_vector) and not a_resident else 0)
            + (n * K if relayed else 0))
    resident = n * (M + 1) + (n if relayed else 0)
    words = max(1, dtype.size() // 4)
    return ('rows' if held * words >= AUTO_ROWS_FROM and 2 * resident <= held
            else 'columns')


_FUSED = {1: fmascalar, 4: fmadpp4, 8: fmadpp8, 16: fmadpp16}


def _row_products(writer, form, step, src, ftype, threads, row, terms):
    """`acc += src[row] * a` for each `(acc, a)` in `terms`, as `hfma` writes
    them: the row share a modifier on every product, or -- `MOVED` -- one
    move that plain FMAs read."""
    emit = _FUSED[step]
    if form is BroadcastForm.MOVED:
        mv = dict(threads=threads, row=row)
        callee = (f'tensorforge::movdpp16Kept<{row}>' if PIN_MOVED
                  else MOVDPP16.callee.format(**mv))
        src = writer.call(callee, ftype, src, hint='bc', movable=False,
                          layout=MOVDPP16.produces(**mv))
        emit = fmascalar
    for acc, a in terms:
        emit(writer, acc, src, a, row)


def _fused_rows(writer, cols, C, A, B, M, K, kx, threads, dtype, ctx):
    """The fused chain over `cols` with the contraction outermost
    (`FUSED_ORDER`).

    `hfma`'s products in the order `_paired_columns` walks its pairs: for each
    row of a sub-block, every column's broadcast of that row into every slot,
    then the next row.  An `A(i, k)` is read at its row and dead after it
    (`_a_on_demand`); what stays resident is the accumulators and one `B`
    register a column.  In the column order every `A` value is live for the
    whole chain: on `local_flux` (gfx1150, two slots, 56 steps) that is 112
    of its 219 VGPRs, at occupancy 4.
    """
    step = select_fmadpp_step(dtype, threads, ctx)
    form = select_broadcast_form(dtype, step, M, ctx, can_pack=False)
    ftype = ScalarType(dtype)
    sources = {(j, k0): B(writer, None, j, k0 // threads)
               for j in cols for k0 in range(0, K + kx, threads)}
    acc = {(j, s): writer.declare(ftype, hint='acc')
           for j in cols for s in range(M)}
    a_of = _a_on_demand(writer, A, K, kx)
    for k0 in range(0, K + kx, threads):
        dk = min(threads, K + kx - k0)
        for i in range(0, dk, step):
            rows = {j: _row_source(writer, sources[(j, k0)], ftype, threads,
                                   step, i // step) for j in cols}
            for r in range(min(step, dk - i)):
                avs = [(s, a_of(s, k0 + i + r)) for s in range(M)]
                avs = [(s, av) for s, av in avs if av is not None]
                if not avs:
                    continue
                for j in cols:
                    _row_products(writer, form, step, rows[j], ftype, threads,
                                  r, [(acc[(j, s)], av) for s, av in avs])
                _pin(writer, [acc[(j, s)] for j in cols for s, _ in avs])
    for j in cols:
        for s in range(M):
            C(writer, acc[(j, s)], s, j)


def _a_on_demand(writer, A, K, kx):
    """`A(i, k)` read where it is first used, keyed `(i, k + kx)` like
    `_load_a`, and read once.

    The packed arrangements walk the contraction outermost, so each value is
    read, consumed by every column, and dead -- where `_load_a` reads them all
    first and holds every one across every column.  At two lead slots and 56
    contraction steps that is 112 registers held for the whole chain, and a
    packed FMA wants each splat operand in an aligned pair besides: gfx1251
    went from 226 VGPRs to 512 and 1.6 KB of scratch on it.
    """
    cache = {}

    def get(i, key):
        if (i, key) not in cache:
            res = (A(writer, None, i, key - kx)
                   if kx <= key < K + kx else None)
            cache[(i, key)] = res if _value(res) else None
        return cache[(i, key)]
    return get


def _paired_columns(writer, C, A, B, pairs, M, K, kx, threads, dtype, step):
    """Columns `j` and `j + 1`, for each `j` in `pairs`, from one move per row.

    The scalar chain replicates `B(k, j)` into each product of a lead slot
    with a DPP modifier of its own, one issue per product.  Here both
    columns' values sit in one register pair, and a row share moves the pair
    -- one `v_mov_b64_dpp` where the target has it.  The lead slots are
    paired too: `A(2p, k)` and `A(2p + 1, k)` are adjacent in a register
    image, so they are one aligned operand, and `acc += A_pair * b[c]` is one
    `v_pk_fma_f32` with a half of the moved pair splat, which `op_sel` reads
    from any register pair.  The accumulators are `(C(2p, j), C(2p + 1, j))`
    per column.

    Pairing the slots is what keeps the operands where they are.  Splatting
    `A(i, k)` itself across a column pair instead wants each of them in the
    low half of an aligned pair, and an operator held as a register image --
    `chain_three`, 112 values a lane -- came out 100 VGPRs over its fused
    count on gfx1251 and spilled.  Needs an even `M`, which `matmuldpp` asks.

    The contraction is the outer loop and the pairs the inner one, so an
    `A(i, k)` is read, used by every column and dead; what stays resident is
    the accumulators.  See `_a_on_demand`.

    False, before any product is emitted, if a `B` read declines; the caller
    then takes every column through the scalar chain.
    """
    ftype = ScalarType(dtype)
    vtype = ScalarType(dtype, 2)
    zero = writer.const(0.0, ftype)
    sources = {}
    for j in pairs:
        for k0 in range(0, K + kx, threads):
            b0 = B(writer, None, j, k0 // threads)
            b1 = B(writer, None, j + 1, k0 // threads)
            if not (_spread(b0, 1, threads) and _spread(b1, 1, threads)):
                return False
            sources[(j, k0)] = (b0, b1)
    slots = range(M // 2)
    acc = {(j, c, p): writer.declare(vtype, hint='acc')
           for j in pairs for c in range(2) for p in slots}
    a_of = _a_on_demand(writer, A, K, kx)
    for k0 in range(0, K + kx, threads):
        packs = {j: writer.pack(vtype, *sources[(j, k0)]) for j in pairs}
        dk = min(threads, K + kx - k0)
        for i in range(0, dk, step):
            rows = {j: _row_source(writer, packs[j], vtype, threads, step,
                                   i // step) for j in pairs}
            for r in range(min(step, dk - i)):
                k = k0 + i + r
                halves = {p: (a_of(2 * p, k), a_of(2 * p + 1, k))
                          for p in slots}
                halves = {p: h for p, h in halves.items()
                          if h[0] is not None or h[1] is not None}
                if not halves:
                    continue
                apair = {p: writer.pack(vtype, *(zero if h is None else h
                                                 for h in hs))
                         for p, hs in halves.items()}
                for j in pairs:
                    m = _move_row(writer, rows[j], vtype, threads, r)
                    for c in range(2):
                        mc = writer.extract(m, c, ftype)
                        for p, ap in apair.items():
                            writer.accumulate(acc[(j, c, p)], writer.op(
                                'mul', vtype, mc, ap, hint='p'))
                _pin(writer, [acc[(j, c, p)] for j in pairs for c in range(2)
                              for p in apair])
    for j in pairs:
        for c in range(2):
            for p in slots:
                for h in range(2):
                    C(writer, writer.extract(acc[(j, c, p)], h, ftype),
                      2 * p + h, j + c)
    return True


def _matmuldpp_wide(writer, start, stop, C, A, B, M, K, kx, threads, dtype,
                    sparse, ctx, width, a_resident=False):
    """The chain at lead width `width`: each lane holds `width` adjacent rows.

    `A(i, k)` and `C(i, j)` are `width`-vectors then, and so is `B`: its
    contraction axis is spread over the lanes the same way, lane `t` of the
    block at `k0` holding `k = k0 + width * t + c` in component `c`.  A row
    share therefore replicates `width` contraction steps at once, and each of
    them feeds the `width` rows of every lead slot.

    Packed where it pays (`packed_broadcast`): the accumulator is the vector,
    one move per row carries all `width` steps -- one `v_mov_b64_dpp` for a
    float pair -- and `acc[i] += b[c] * A(i, k)` is a `v_pk_fma_f32` with the
    moved component splat.  Otherwise the scalar chain over components:
    `width` scalar accumulators per slot and one broadcast per step, which is
    `hfma` unchanged, fused DPP included.

    Declines where the contraction does not start and end on whole vectors,
    and for a sparse `B`, whose linear image the rows here do not describe.
    """
    if sparse is not None or kx % width or (K + kx) % width:
        return False
    span = threads * width
    ftype = ScalarType(dtype)
    vtype = ScalarType(dtype, width)
    step = select_fmadpp_step(dtype, threads, ctx)
    cols = range(start, stop)
    sources = {}
    for j in cols:
        for k0 in range(0, K + kx, span):
            b = B(writer, None, j, k0 // span)
            if not _spread(b, width, threads):
                # Declined after emitting reads, which the nest discards
                # (`Writer.speculative`) before it computes the product.
                return False
            sources[(j, k0)] = b
    if packed_broadcast(dtype, step, M * width * width, width * dtype.size(),
                        ctx):
        # Contraction outermost, as for the column pairs: each `A(i, k)` is
        # read, used by every column and dead, and what stays resident is the
        # accumulators.
        acc = {(j, s): writer.declare(vtype, hint='acc')
               for j in cols for s in range(M)}
        a_of = _a_on_demand(writer, A, K, kx)
        for k0 in range(0, K + kx, span):
            lanes = -(-min(span, K + kx - k0) // width)
            for i in range(0, lanes, step):
                rows = {j: _row_source(writer, sources[(j, k0)], vtype,
                                       threads, step, i // step)
                        for j in cols}
                for r in range(min(step, lanes - i)):
                    base = k0 + width * (i + r)
                    terms = {c: [(s, a_of(s, base + c)) for s in range(M)]
                             for c in range(width)}
                    terms = {c: [(s, av) for s, av in ts if av is not None]
                             for c, ts in terms.items()}
                    if any(_width_of(av) != width
                           for ts in terms.values() for _, av in ts):
                        # `A` does not hold the rows as the accumulator does.
                        return False
                    if not any(terms.values()):
                        continue
                    moved = {j: _move_row(writer, rows[j], vtype, threads, r)
                             for j in cols}
                    for c, ts in terms.items():
                        for j in cols:
                            mc = writer.extract(moved[j], c, ftype)
                            for s, av in ts:
                                writer.accumulate(acc[(j, s)], writer.op(
                                    'mul', vtype, mc, av, hint='p'))
                    _pin(writer, list(dict.fromkeys(
                        acc[(j, s)] for ts in terms.values() for j in cols
                        for s, _ in ts)))
        for j in cols:
            for s in range(M):
                C(writer, acc[(j, s)], s, j)
        return True
    # Fused: the scalar chain over components, column by column as the chain
    # at width one runs, with `width` scalar accumulators a slot -- a DPP
    # modifier writes its accumulator through a reference, which a vector
    # element cannot bind to.
    if not FUSED_WIDE:
        return False
    if _fused_order(cols, M * width, K, dtype, step, threads,
                    a_resident) == 'rows':
        return _fused_rows_wide(writer, cols, C, A, sources, M, K, kx,
                                threads, dtype, ctx, width)
    ab = _load_a(writer, A, M, K, kx)
    if any(_width_of(av) != width for av in ab.values()):
        return False
    for j in cols:
        acc = [[writer.declare(ftype, hint='acc') for _ in range(width)]
               for _ in range(M)]
        for k0 in range(0, K + kx, span):
            b = sources[(j, k0)]
            lanes = -(-min(span, K + kx - k0) // width)
            for c in range(width):
                # Component `c` of every lane is one contraction step per
                # lane, `width` apart: a scalar source like any other, and
                # each of its steps feeds `M * width` scalar products.
                src = writer.extract(b, c, ftype)
                avs, cvs = [], []
                for t in range(lanes):
                    k = k0 + width * t + c
                    for s in range(M):
                        for row in range(width):
                            if (s, k) in ab:
                                avs.append(writer.extract(ab[(s, k)], row,
                                                          ftype))
                                cvs.append(acc[s][row])
                            else:
                                avs.append(None)
                                cvs.append(None)
                hfma(writer, [cvs], [src], [avs], M * width, dtype, threads,
                     ctx)
        for s in range(M):
            C(writer, writer.pack(vtype, *acc[s]), s, j)
    return True


def _fused_rows_wide(writer, cols, C, A, sources, M, K, kx, threads, dtype,
                     ctx, width):
    """`_matmuldpp_wide`'s fused chain with the contraction outermost.

    Within a block of `threads * width` steps the components go outermost,
    then the rows of each sub-block: the order the column walk gives every
    accumulator, so the sums agree with it to the bit.  `A(i, k)` is a
    `width`-vector of rows, read at its step and dead after it.
    """
    step = select_fmadpp_step(dtype, threads, ctx)
    form = select_broadcast_form(dtype, step, M * width, ctx, can_pack=False)
    ftype = ScalarType(dtype)
    vtype = ScalarType(dtype, width)
    span = threads * width
    acc = {(j, s): [writer.declare(ftype, hint='acc') for _ in range(width)]
           for j in cols for s in range(M)}
    a_of = _a_on_demand(writer, A, K, kx)
    for k0 in range(0, K + kx, span):
        lanes = -(-min(span, K + kx - k0) // width)
        for c in range(width):
            comps = {j: writer.extract(sources[(j, k0)], c, ftype)
                     for j in cols}
            for i in range(0, lanes, step):
                rows = {j: _row_source(writer, comps[j], ftype, threads, step,
                                       i // step) for j in cols}
                for r in range(min(step, lanes - i)):
                    avs = [(s, a_of(s, k0 + width * (i + r) + c))
                           for s in range(M)]
                    avs = [(s, av) for s, av in avs if av is not None]
                    if any(_width_of(av) != width for _, av in avs):
                        # `A` does not hold the rows as the accumulator does.
                        return False
                    if not avs:
                        continue
                    parts = [(s, [writer.extract(av, h, ftype)
                                  for h in range(width)]) for s, av in avs]
                    for j in cols:
                        _row_products(writer, form, step, rows[j], ftype,
                                      threads, r,
                                      [(acc[(j, s)][h], p) for s, ps in parts
                                       for h, p in enumerate(ps)])
                    _pin(writer, [a for j in cols for s, _ in avs
                                  for a in acc[(j, s)]])
    for j in cols:
        for s in range(M):
            C(writer, writer.pack(vtype, *acc[(j, s)]), s, j)
    return True


def _scalar_chain(writer, start, stop, C, B, ab, M, K, kx, threads, dtype,
                  sparse, ctx):
    """The chain `hfma` writes, over columns `[start, stop)`."""
    cx = []
    ax = []
    cb = []
    for j in range(start, stop):
        cbl = []
        for i in range(M):
            # The accumulator is written by `fmacdpp` through a reference, so
            # it is not the result of any statement -- `declare` gives it a
            # definition point without changing the emitted text.
            vC = writer.declare(ScalarType(dtype), hint='acc')
            cb += [vC]
            cbl += [vC]
        for k in range(K + kx):
            for i in range(M):
                if (not sparse or sparse(k, j)) and (i,k) in ab:
                    cx += [cbl[i]]
                    ax += [ab[(i, k)]]
                elif not sparse:
                    cx += [None]
                    ax += [None]

    if sparse is not None:
        stride = threads*M
        for kj in range(0, len(cx), stride):
            vB = B(writer, None, None, kj // M)
            vA = ax[kj: min(kj + stride, len(cx))]
            vC = cx[kj: min(kj + stride, len(cx))]
            hfma(writer, [vC], [vB], [vA], M, dtype, threads, ctx)
    else:
        vA = []
        vB = []
        vC = []
        for j in range(start, stop):
            for k in range(0, K + kx, threads):
                vB += [B(writer, None, j, k // threads)]
                kj = ((K + kx) * (j-start) + k) * M
                stride = min(threads, K + kx - k) * M
                vA += [ax[kj: min(kj + stride, len(cx))]]
                vC += [cx[kj: min(kj + stride, len(cx))]]
        hfma(writer, vC, vB, vA, M, dtype, threads, ctx)

    for j in range(start, stop):
        for i in range(M):
            C(writer, cb[(j-start)*M+i], i, j)
