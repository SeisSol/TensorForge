# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Code generation for the multilinear kernel.
"""

from tensorforge.common.basic_types import Datatype
from tensorforge.backend.writer import Writer
from tensorforge.backend.pir.core import ScalarType
from .arch import cdna2, gfx1251, rdna
from .catalog import usable_mfma_tiles
from .emitters import fmadpp4, fmadpp8, fmadpp16, fmascalar
from .relayout import (MOVDPP16, TRANSPOSE4X4, find_relayout,
                       fmadpp_operand_layout)
from .select import select_fmadpp_step


def _check_mfma_operand(operand, threads, callee):
    """The A operand of an MFMA has to be laid out as the transpose left it.

    `None` means untracked and is allowed through: an absent annotation is
    not evidence of a wrong one, and refusing to emit for want of one would
    make the layouts an obstacle rather than a description.  A present layout
    that disagrees is a wrong kernel.
    """
    want = TRANSPOSE4X4.produces(threads=threads)
    got = getattr(operand, 'layout', None)
    if got is not None and got != want:
        raise ValueError(f'{callee} needs its A operand at {want!r}, '
                         f'got {got!r}')


def hfma(writer: Writer, Cs, As, Bs, repeat, datatype, threads, ctx):
    """
    Strategy:

    * gfx906: shuffle over 4 threads; but use DPP over 16 to broadcast blocks of 4 (already encoded)
    * gfx908+ / CDNA+ and RDNA+: use DPP 16 with broadcasting. Fuse for F32 and little users (to harness pkd math on RDNA 3+).
    * gfx90a+ / CDNA2+: use DPP64 for F64 and broadcasting if 2 or more rows and lots of users.
    """

    step = select_fmadpp_step(datatype, threads, ctx)

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

                if bcst and all((B[bx][idx + jj] if idx + jj < len(B[bx]) else None) is not None for bx in range(localstep) for jj in range(repeat)):
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


def matmul32(writer: Writer, C, B, A, M, N, K, kx, threads, dtype, sparse, ctx):
    with writer.AnonymousScope():

        ftype = ScalarType(dtype)

        def write_matmul(tile, start, cap):
            block = tile.block
            scale = tile.scale(threads)
            fn = tile.builtin

            def transpose(regs):
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

            # C <- C + B@A
            end = ((N // block) * block) if cap else N
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
                            regs += [writer.const(0.0, ftype)]
                        tA[k // threads] = transpose(regs)
                    for i in range(0, M):
                        with writer.AnonymousScope():
                            vtype = ScalarType(dtype, block)
                            acc = writer.declare(vtype, hint='acc',
                                                 layout=acclayout)
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
                                                            threads, fn)
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
                                C(writer, writer.extract(acc, jj, ftype), i, j + jj)

        # The tiling policy, now separate from what the tiles are.  Only the
        # 4-wide tile is reachable today: the 16-wide one needs a shared-memory
        # staging step that is not written, and the 32-wide one has no
        # transpose in the runtime, which `available_for` already refuses.
        tiles = usable_mfma_tiles(threads, dtype, ctx)
        tile = next(t for t in tiles if t.block == 4)

        start = 0
        # A tail of 0 or 1 columns is cheaper through DPP; 2 or 3 are cheaper
        # as one padded MFMA block.  `cap` says which: capped, `write_matmul`
        # stops at the last whole block and leaves the tail; uncapped, it pads
        # the tail block out and does everything.
        cap = N % tile.block < 2
        write_matmul(tile, start, cap)
        # The handoff has to be read off what `write_matmul` *did*, not
        # recomputed.  `(N // block) * block` is the tail only in the capped
        # case; when uncapped it points into a block that was already emitted,
        # and both paths then computed the same columns -- the DPP store
        # landing last and hiding it.
        tail = ((N // tile.block) * tile.block) if cap else N
        matmuldpp(writer, tail, C, B, A, M, N, K, kx, threads, dtype, sparse, ctx)


    # TODO: gfx1200, f'__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12'

def matmuldpp(writer, start, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):
    if start >= N:
        # Nothing left for this path.  Worth an early return rather than
        # letting the loops come out empty: the A operands below are loaded
        # before the first `for j`, so falling through would emit a full set
        # of reads with no consumer.
        return
    # `None` asks the loader for the value rather than for a name to fill in:
    # the intrinsics below take these as operands, and an operand whose
    # definition the IR cannot see is invisible to every pass that reasons
    # about ordering or reuse.
    ab = {}
    for k in range(K):
        for i in range(M):
            res = A(writer, None, i, k)
            if res is not None and res is not False:
                ab[(i, k + kx)] = res
    cx = []
    ax = []
    cb = []
    for j in range(start, N):
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
        for j in range(start, N):
            for k in range(0, K + kx, threads):
                vB += [B(writer, None, j, k // threads)]
                kj = ((K + kx) * (j-start) + k) * M
                stride = min(threads, K + kx - k) * M
                vA += [ax[kj: min(kj + stride, len(cx))]]
                vC += [cx[kj: min(kj + stride, len(cx))]]
        hfma(writer, vC, vB, vA, M, dtype, threads, ctx)

    for j in range(start, N):
        for i in range(M):
            C(writer, cb[(j-start)*M+i], i, j)
