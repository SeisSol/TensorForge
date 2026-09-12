# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""AMD code generation for the multilinear kernel.

`multilinear.py` enters through `matmul()` and nothing else.  The modules
below are layered in dependency order, and the layering carries the lesson of
the bugs that came out of this file:

* `arch`     -- which family a target is
* `caps`     -- what its runtime defines
* `features` -- which LLVM subtarget features it has
* `catalog`  -- what a matrix instruction is
* `layouts`  -- where each element of its operands sits
* `reorder`  -- how to build a fragment from the nest's registers
* `relayout` -- which instruction turns one lane distribution into another
* `select`   -- which instruction width to use
* `emitters` -- how to write one instruction down
* `codegen`  -- the kernel
* `unused`   -- matrix paths kept for repair, with no call site

The split between `arch` and `caps` is the load-bearing one.  Those were the
same thing here, and a family predicate standing in for a capability is what
let gfx900 emit a call to a template that has only a declaration there.

Removed when this became a package: the `dppctrl_*` constant helpers, the raw
`amdgcn_*` intrinsic wrappers, four `shuffle_*` routines, two `reduction`s
written against CUDA's `__shfl_xor_sync`, the `MatrixCore` class with its
`matrixcores`/`archmap` tables and the `matmul` it served, and three empty
stubs -- 350 lines unreachable from `matmul()` through the call graph, not
merely uncovered by tests.  Two of those names, `reduction` and `matmul`, were
defined twice at module level, so Python had been discarding the first
definition since it was written.  `tests/test_amd_reachability.py` keeps the
property.
"""

from dataclasses import replace

from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import InternalError

from ... import bitlayout, broadcast, packing, staging
from ...routes import lead_route as routes_lead_route
from ...strategy import Span, Strategy, whole

from .arch import amdarch, cdna2, gfx1250, gfx1251, rdna
from .caps import has_fmacdpp4, has_fmacdpp8, has_fmacdpp16
from .catalog import (DEFINED_SPLITS, DEFINED_TRANSPOSES, MANTISSA,
                      MATRIX_OPS, MFMA_TILES, emu_tile_for, emu_tiles,
                      NOT_MODELLED, Call, Fragment, MatrixOp,
                      MfmaTile, lane_batched_ops, mfma_tile_for, ops_for,
                      usable_mfma_tiles)
from .features import FEATURE_TARGETS, has_feature, wave_size
from .layouts import (FRAGMENT_BITS, Provenance, covers, established,
                      position, provenance)
from .reorder import (BANK, FED_BY, IDENTITY_DPP, ROW, Gather, Move,
                      Select, Exchange, a_exchange, accumulator_cost,
                      accumulator_gathers, broadcast_feeds_a,
                      fragment_cost, fragment_moves)
from .codegen import hfma, matmul32, matmulemu, matmuldpp
from .exchange_codegen import exchange_op, exchange_ops, matmul_exchange
from .tiling import (EMULATION, EXCHANGE, Fit, Scheme, boundary,
                     candidates, choose, issues, offers, rank,
                     spare_products)
from .emitters import fmadpp, fmadpp4, fmadpp8, fmadpp16, fmascalar
from .relayout import (BROADCAST, MOVDPP16, RELAYOUTS, RUNGS,
                       TRANSPOSE4X4, Relayout, find_relayout, reach,
                       takes)
from .select import (BroadcastForm, MATERIALISE_FROM, broadcast_form,
                     dpp_move_instructions, dual_issue_fma_lanes,
                     packed_broadcast, packed_broadcast_pays,
                     packed_fma_lanes,
                     select_broadcast_form, select_fmadpp_step,
                     wanted_fmadpp_step)
from .unused import (mfma_emu_bf16_f32, mfma_emu_f16_f32, mfma_emu_int8,
                     wmma3atom)

__all__ = [
    'amdarch', 'cdna2', 'gfx1250', 'gfx1251', 'rdna',
    'has_fmacdpp4', 'has_fmacdpp8', 'has_fmacdpp16',
    'FEATURE_TARGETS', 'has_feature', 'wave_size',
    'Call', 'Fragment', 'MatrixOp', 'MATRIX_OPS', 'MANTISSA',
    'DEFINED_SPLITS', 'emu_tile_for', 'matmulemu', 'EMULATION',
    'EXCHANGE', 'Scheme', 'Fit', 'exchange_op', 'exchange_ops',
    'matmul_exchange', 'emu_tiles', 'candidates', 'issues',
    'rank', 'spare_products',
    'NOT_MODELLED', 'ops_for',
    'MfmaTile', 'DEFINED_TRANSPOSES', 'MFMA_TILES', 'usable_mfma_tiles',
    'lane_batched_ops', 'mfma_tile_for',
    'FRAGMENT_BITS', 'Provenance', 'covers', 'established',
    'position', 'provenance',
    'BANK', 'FED_BY', 'IDENTITY_DPP', 'ROW', 'Gather', 'Move', 'Select',
    'broadcast_feeds_a', 'Exchange', 'a_exchange',
    'accumulator_cost', 'accumulator_gathers',
    'fragment_cost', 'fragment_moves',
    'wanted_fmadpp_step', 'select_fmadpp_step',
    'BroadcastForm', 'MATERIALISE_FROM', 'broadcast_form',
    'dpp_move_instructions', 'dual_issue_fma_lanes', 'packed_broadcast',
    'packed_fma_lanes', 'select_broadcast_form',
    'Relayout', 'RELAYOUTS', 'BROADCAST', 'MOVDPP16', 'TRANSPOSE4X4',
    'find_relayout',
    'fmadpp', 'fmadpp4', 'fmadpp8', 'fmadpp16', 'fmascalar',
    'hfma', 'matmul32', 'matmuldpp', 'matmul',
    'mfma_emu_int8', 'mfma_emu_bf16_f32', 'mfma_emu_f16_f32', 'wmma3atom',
]


def strategies(shape, ctx):
    """What this target can emit for this shape.

    The DPP chain always: a broadcast modifier on the multiply needs nothing
    of the shape, and where the widest form does not link, `select.py` falls
    to a narrower one rather than to nothing.

    A matrix core where any of the schemes in `tiling` serves the shape,
    which is a structural question and not a family or a type one.
    `mfma_f64_16x16x4f64` spends two of its lane bits on the contraction, so
    the data operand carries the leading dimension there and the lane-batched
    loop cannot feed it; `MatrixOp.lane_batched` states that as one equation.
    F64 therefore lands on DPP -- where `fmacdpp16(double&, ...)` serves it --
    while the scheme that could feed it is not deployed, rather than because a
    condition names the type.

    A sparse second operand is read by linear index, which no fragment layout
    accepts and which the broadcast chain has no lane to replicate; the DPP
    chain has a branch for it and takes it alone.

    A packed lead operand is taken by the DPP chain and by nothing else here.
    The DPP chain indexes the lanes through the accessors, and at width `w`
    those hand it vectors: `A` and `C` hold `w` rows a lane, and `B` holds `w`
    contraction steps a lane, so one row share replicates `w` steps and each
    feeds `w` rows (`codegen._matmuldpp_wide`).  A sparse `B` is read through
    its linear image, whose entries the rows there do not describe, so it
    stays with the nest.  The readlane chain has been given no such
    conversion.

    The matrix core answers through `takes`: an operand at width one already
    arrives spread one element per lane, and above one it reaches a fragment
    that wants the rows in lane order only through the trip that is priced
    and not yet emitted.  The lane-batched scheme does not want that order
    (`componentwise`), so it takes the packed operand as it is, one component
    at a time.
    """
    if bitlayout.packed(shape.lead_layout):
        offered = set()
        if (not shape.sparse
                and offers(shape.threads, shape.accumulator, ctx)
                and (takes(lead_route(shape)) or componentwise(shape, ctx))):
            offered.add(Strategy.MATRIX)
        if not shape.sparse:
            offered.add(Strategy.DPP)
        return frozenset(offered)
    offered = {Strategy.DPP}
    if not shape.sparse:
        # The same chain the DPP one fuses its broadcast into, available here
        # through `readlane`.  It cannot read a sparse operand, which is the
        # one thing the DPP branch does that this does not.
        offered.add(Strategy.BROADCAST)
        if offers(shape.threads, shape.accumulator, ctx):
            offered.add(Strategy.MATRIX)
    return frozenset(offered)


def scratch(strategy, shape, ctx):
    """Elements the operands need staged before any body exists.

    Nothing while the operands arrive unpacked: every arrangement here keeps
    them in registers, and the relayouts between register layouts are swaps
    and merges.

    A packed lead operand is the exception, and it is why this reads the shape
    rather than the type alone.  `lead_width` puts its low bits inside a
    register and the fragment wants the leading dimension across the lanes;
    unpacking moves them the wrong way and what remains is a permutation
    between lane weights, which no relayout performs.  The trip through memory
    answers it, and one wave of elements is what that trip holds -- the buffer
    carries one operand register at a time, so it does not grow with the
    problem.

    Sized from the same plan the emission will walk, so the two cannot differ:
    a reservation smaller than the plan is an overrun and a larger one is
    memory nobody writes.
    """
    route = lead_route(shape)
    # The trip is the matrix core's: the DPP chain takes a packed operand in
    # its registers and stages nothing, and so does the lane-batched matrix
    # scheme (`componentwise`).
    if (strategy is not Strategy.MATRIX or not isinstance(route, tuple)
            or componentwise(shape, ctx)):
        return 0
    return staging.buffer_elements(route)


def lead_route(shape):
    """`routes.lead_route` with this target's rungs.

    Kept as a name here for the same reason `reach` is: `strategies` and
    `scratch` are this target's, and the rungs are not theirs to pass.
    """
    return routes_lead_route(shape, RUNGS)


def componentwise(shape, ctx) -> bool:
    """Whether the matrix core takes a packed lead operand as it is.

    `lead_route` answers for a fragment that wants the rows in lane order,
    and for a packed operand that is a permutation between lane weights and
    a trip through memory.  The lane-batched scheme does not want that order.
    Its lanes are independent rows -- the instruction's N times its block
    count is the wave, and no product crosses from one lane's row to another's
    -- so which row a lane holds is the kernel's business.  At width `w` lane
    `t` holds rows `w * t + c`, and component `c` of every lane is a
    lane-batched problem of its own: one element per lane, as the B fragment
    takes it (`codegen.matmul32`).

    Only for that scheme: the exchange and the emulated ones spend lane bits
    on the contraction and do want the order.
    """
    if not bitlayout.packed(shape.lead_layout):
        return False
    fit = choose(shape.threads, shape.accumulator, ctx, lead=shape.lead,
                 depth=shape.depth)
    return fit is not None and fit.scheme is Scheme.LANE_BATCHED


#: Whether a batch-constant lead operand is read once for the multiplications
#: sharing a wave and handed between them by the MFMA's `blgp`
#: (`codegen.matmul32`).  A switch, because it changes the loop as well: the
#: multiplications of a wave then take the same trips (`convergence`).
B_DUPLICATION = True

#: The wave of every target with a lane-batched MFMA (CDNA).
_MFMA_WAVE = 64


def wave_mults(threads, a_uniform, width=1, dtype=Datatype.F32) -> int:
    """How many multiplications of a wave read one lead operand together.

    `blgp` has the patterns for two -- the lower half of the wave's B operand
    to the upper, or back -- and for four, one quarter to all.  So a lead
    operand every multiplication reads alike, at lead width one, over 32 or
    16 lanes: one read of a wave's worth of distinct elements serves two or
    four contraction steps where each step read the same elements into every
    multiplication.  One everywhere else, which is the arrangement as it was
    -- and for F64, whose MFMA spends the field on negation (CDNA3).
    """
    if (not B_DUPLICATION or not a_uniform or width != 1 or threads <= 0
            or dtype != Datatype.F32):
        return 1
    mults = _MFMA_WAVE // threads if _MFMA_WAVE % threads == 0 else 1
    return mults if mults in (2, 4) else 1


def convergence(strategy, shape):
    """How far the threads have to run in step for `strategy` over `shape`.

    The lane-batched MFMA asks nothing of its own: each multiplication feeds
    its blocks, and `cbsz` keeps the broadcast of A inside them.  Handing the
    lead operand between the multiplications of a wave does ask: every one of
    them has to have read its part when the instruction issues, so they take
    the same trips through the batch loop.
    """
    width = (1 << bitlayout.unpacked(shape.lead_layout)[1]
             if bitlayout.packed(shape.lead_layout) else 1)
    if (strategy is Strategy.MATRIX
            and wave_mults(shape.threads, shape.a_uniform, width,
                           shape.accumulator) > 1):
        from tensorforge.backend.pir.core import Uniformity
        return Uniformity.MULTGROUP
    return None


def wide_chain_takes(shape, ctx) -> bool:
    """Whether the DPP chain would take a span of this packed shape.

    `_matmuldpp_wide`'s own conditions, asked before any body exists: whole
    vectors of contraction, and the packed form paying where the fused one is
    switched off (`FUSED_WIDE`).  Asked through `packed_broadcast_pays`, which
    leaves the body unmarked.  The one condition the chain also has that a
    shape does not carry is where the contraction starts; an odd start still
    declines there.
    """
    width = 1 << bitlayout.unpacked(shape.lead_layout)[1]
    if shape.sparse or shape.depth % width:
        return False
    from . import codegen
    if codegen.FUSED_WIDE:
        return True
    step = select_fmadpp_step(shape.accumulator, shape.threads, ctx)
    slots = -(-shape.lead // (shape.threads * width))
    return packed_broadcast_pays(shape.accumulator, step, slots * width * width,
                                 width * shape.accumulator.size(), ctx)


def plan(strategy, shape, n, ctx):
    """How the chosen arrangement is laid out over the output.

    The matrix core covers whole tiles; what is left over is a cost question
    with a threshold behind it, and which threshold depends on which scheme
    `tiling` picked.  A tail of two or three columns is cheaper as
    one MFMA block with its spare lanes zeroed than as two or three passes of
    a broadcast chain, and a tail of one is not -- padding a block of four to
    compute one column spends three quarters of it on zeroes.

    So the tail is a second span rather than a handoff inside the emitter,
    and the boundary is stated once instead of computed twice.  Computing it
    twice is what makes the two arrangements overlap: `(n // block) * block`
    is the tail only when the block loop stopped there, and when it padded
    through to the end it points into a block already emitted -- both spans
    then write the same columns, the later store hiding it.
    """
    if strategy is not Strategy.MATRIX:
        return whole(strategy, n)
    fit = choose(shape.threads, shape.accumulator, ctx,
                 columns=n, lead=shape.lead, depth=shape.depth)
    # Asked of the scheme that will run rather than of one of them.  Only the
    # lane-batched one draws a boundary at all, and the threshold behind it is
    # a measurement against its own block width.
    if componentwise(shape, ctx) and not wide_chain_takes(shape, ctx):
        # A packed lead operand whose tail the DPP chain would decline -- a
        # contraction that ends mid-vector, or no 64-bit move with the fused
        # form off -- goes to the matrix core whole, the last block padded.
        # A declined tail takes the matrix span down with it: the whole
        # product went to the nest, `local_flux`'s 9x9 products on gfx942 and
        # all of it on gfx90a (2100 B of scratch).  Where the chain takes the
        # tail the width-one boundary stands, which measured one column
        # cheaper there; at width two that is unmeasured.
        return whole(Strategy.MATRIX, n)
    edge = boundary(fit, n)
    if edge >= n:
        return whole(Strategy.MATRIX, n)
    if edge <= 0:
        # Fewer columns than one block, and too few to repay padding it: there
        # is no matrix span to name, not an empty one.
        return whole(Strategy.DPP, n)
    return (Span(Strategy.MATRIX, 0, edge),
            Span(Strategy.DPP, edge, n))


#: Whether a prepared lead operand is offered in k-quads (`prepared_order`).
K_QUADS = True


def quad_width(dtype) -> int:
    """Contraction steps one 16-byte read of a prepared operand holds."""
    return 16 // dtype.size()


def quad_order(shape, threads, width):
    """`Tensor.storage_order` for an operand read in k-quads.

    Lane `l`'s row in slot `s` holds steps `width * q .. width * q + width -
    1` side by side, and the lanes of one slot one after the other: cell `(r,
    k)` sits at `((k // width * slots + s) * threads + l) * width + k % width`
    for `r = s * threads + l`.  So one lane reads `width` steps of its row as
    one aligned vector, and a wave reads one contiguous run -- where the
    column-major operand gives it one scalar per step.  Rows past the end and
    steps past the end are padding (`-1`).
    """
    rows, cols = (int(x) for x in shape)
    slots = -(-rows // threads)
    order = []
    for q in range(-(-cols // width)):
        for s in range(slots):
            for lane in range(threads):
                r = s * threads + lane
                for c in range(width):
                    k = q * width + c
                    order.append(r + rows * k if r < rows and k < cols else -1)
    return order


def prepared_order(shape, dtype, ctx, columns=0, lead=0, depth=0,
                   threads=32):
    """The order this target would read a two-dimensional A operand in, or
    `None` (`multilinear._offer_order`).

    k-quads for the lane-batched MFMA (`quad_order`): its lanes are rows and
    each step reads one element of the lane's row, so four steps are one
    16-byte read.  F32 only, like the `blgp` it composes with.  Only where the
    operation reads the whole operand, since the order is laid out over the
    tensor and read by slot, with no coordinate left to offset.
    """
    if not K_QUADS or len(shape) != 2 or dtype != Datatype.F32:
        return None
    if tuple(int(x) for x in shape) != (lead, depth):
        return None
    fit = choose(threads, dtype, ctx, columns=columns, lead=lead, depth=depth)
    if fit is None or fit.scheme is not Scheme.LANE_BATCHED:
        return None
    return quad_order(shape, threads, quad_width(dtype))


def _quad_reader(ops, width):
    """`read(writer, i, q, mults)`: the quad of steps `width * q ..` of slot
    `i`, or -- `mults > 1` -- quad `q + p` in the lanes of the `p`-th
    multiplication of the wave, as `blgp` hands them on (`matmul32`)."""
    threads, slots = ops.threads, ops.lead_slots

    def read(writer, i, q, mults=1):
        shift = None
        if mults > 1:
            from tensorforge.backend.pir.core import INDEX
            p = writer.op('rem', INDEX, writer.thread_id('y'), mults, hint='m')
            shift = writer.op('mul', INDEX, p, slots, hint='r')
        return ops.A_slot(writer, (q * slots + i) * threads * width,
                          shift=shift, width=width)
    return read


def _quad_element(ops, width):
    """`ops.A` for an operand stored in k-quads: the step out of its quad.

    Every path that reads `A` by coordinate reads it through this -- the DPP
    chain beside the MFMAs, the broadcast chain -- because the buffer is
    permuted and a coordinate address would read the wrong cell.
    """
    from tensorforge.backend.pir.core import ScalarType
    read = _quad_reader(ops, width)
    ftype = ScalarType(ops.a)

    def A(writer, var, i, k, part=0, parts=1):
        if var is not None or part or parts != 1:
            raise InternalError(
                'an operand stored in k-quads is read as a value of one part')
        return writer.extract(read(writer, i, k // width), k % width, ftype)
    return A


def matmul(writer, ops, ctx, span):
    """Emit one span of the plan."""
    width = quad_width(ops.a) if ops.A_slot is not None else 0
    if width:
        ops = replace(ops, A=_quad_element(ops, width))
    taken = _matmul(writer, ops, ctx, span, width)
    if not taken and width:
        # Declining would hand the operation to the nest, which reads `A` by
        # coordinate out of a buffer stored in k-quads.
        raise InternalError(
            f'{span.strategy.value} declined an operand stored in k-quads; '
            f'no other path reads that order')
    return taken


def _matmul(writer, ops, ctx, span, width):
    C, A, B = ops.C, ops.A, ops.B
    M, N, K, kx = ops.lead_slots, ops.n, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.accumulator, ops.sparse

    if span.strategy is Strategy.BROADCAST:
        return broadcast.matmul(writer, ops, ctx, span)
    if span.strategy is Strategy.MATRIX:
        fit = choose(threads, dtype, ctx, columns=N,
                     lead=ops.lead_slots * threads * ops.lead_width,
                     depth=K + kx)
        if ops.lead_width > 1 and (fit is None
                                   or fit.scheme is not Scheme.LANE_BATCHED):
            # Only the lane-batched scheme takes a packed lead operand; see
            # `componentwise`, which is what offered this span.
            return False
        if fit is None or (ops.a, ops.b) != (fit.reads, fit.reads):
            # The entry was selected by what it accumulates in; what it
            # multiplies is a separate property of the same entry, and an
            # emulated scheme reads a third thing again.  `Fit.reads` is the
            # one the chosen scheme expects, and where the operands do not
            # arrive as it, reaching the instruction is a further split that
            # none of these emitters performs.
            return False
        if fit.scheme is Scheme.EMULATED:
            return matmulemu(writer, C, A, B, M, N, K, kx, threads, dtype,
                             sparse, ctx, span.start, span.stop, fit.tile,
                             fit.terms)
        if fit.scheme is Scheme.EXCHANGE:
            return matmul_exchange(writer, C, A, B, M, N, K, kx, threads,
                                   dtype, sparse, ctx, span.start, span.stop)
        # `convergence` asked for the wave group on the same condition, less
        # the accessor: a lead operand that turns out not to be addressable
        # reads per step, in a loop that merely runs in step for nothing.
        mults = (wave_mults(threads, ops.a_uniform, ops.lead_width, dtype)
                 if ops.lockstep else 1)
        lead_quad = _quad_reader(ops, width) if width else None
        lead_wave = ops.A_wave if mults > 1 and not width else None
        return matmul32(writer, C, A, B, M, N, K, kx, threads, dtype, sparse,
                        ctx, span.start, span.stop, width=ops.lead_width,
                        tile=fit.tile, lead_wave=lead_wave,
                        lead_quad=lead_quad, quad=width,
                        mults=(mults if lead_wave is not None
                               or lead_quad is not None else 1))
    mults = wave_mults(threads, ops.a_uniform, ops.lead_width, dtype)
    if (mults > 1 and ops.lockstep and ops.A_wave is not None
            and not ops.a_shared):
        A = _wave_lead(ops, ctx, mults)
    return matmuldpp(writer, span.start, C, A, B, M, N, K, kx, threads, dtype,
                     sparse, ctx, span.stop, width=ops.lead_width,
                     a_resident=ops.a_resident, a_vector=bool(width))


def _wave_lead(ops, ctx, mults):
    """`ops.A` for a chain beside `blgp` MFMAs, read the way they read it.

    The MFMAs of a wave take step `k + q` from the lanes of its `q`-th
    multiplication (`codegen.matmul32`), which read it through `A_wave`.  A
    chain reading `A(i, k)` itself reads the same elements at other addresses,
    so the two share no load -- `local_flux`'s ninth column on gfx942 at
    b = 80: 480 more global loads than without `blgp`.  Here the chain reads
    through `A_wave` too, the same load as the MFMAs', and takes step `k + q`
    out of the `q`-th multiplication's lanes with `tensorforge::broadcast`:
    `bpermute` on CDNA, one LDS-pipe instruction per step and no memory.  At
    b = 80 that is 7522 instructions and 548 B of scratch against 12585 and
    6712 for the chain's own loads (11251 and 6064 without `blgp`).

    Not from shared memory (`ops.a_shared`), where the chain's own read is an
    LDS instruction as well and two steps come in one `ds_read2`: at b = 56,
    112 reads and 448 moves against 340 reads, and 216 VGPRs against 172.

    Only where the batch loop runs the wave in step (`ops.lockstep`), as the
    MFMAs need it anyway: the lanes read belong to other multiplications.  A
    step whose group runs past the contraction reads itself.
    """
    from tensorforge.backend.pir.core import ScalarType
    lexic = ctx.get_vm().get_lexic()
    ftype = ScalarType(ops.a)

    def A(writer, var, i, k, *rest):
        g = k - k % mults
        if var is not None or rest or g + mults > ops.k:
            return ops.A(writer, var, i, k, *rest)
        lead = ops.A_wave(writer, i, g, mults)
        if lead is None or lead is False:
            return ops.A(writer, var, i, k)
        text = lexic.broadcast('{0}', k % mults, _MFMA_WAVE, ops.threads)
        return writer.rawexpr(text, lead, type_=ftype, hint='bc', pure=True,
                              movable=True, crosslane=True)
    return A
