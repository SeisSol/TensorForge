# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Emitting a matrix path from the plans, for the instructions with `k > 1`.

`codegen.matmul32` covers the K=1 tiles: the instruction broadcasts its own A
operand and its accumulator already sits where the nest wants it, so the whole
path is a transpose, a chain of issues and a store.  Nothing in it generalizes,
because every one of those three is a property of `k == 1`.

This is the other shape.  Three plans do the arranging and this only emits
them:

* `a_exchange` --- one `transpose{ext}x{ext}b32` over the shared matrix, whose
  output register `g` *is* the A fragment for contraction group `g`.  Paid
  once per k-block of `ext * k`, not once per issue.
* `fragment_moves` --- the B fragment, as `swap` sequences merged by
  `dppUpdate`, one region per contraction value.
* `accumulator_gathers` --- the epilogue, the same shape in reverse.

The contraction runs in stride-`ext` groups, which is what makes the first of
those free; `Exchange` says why, and says what that costs elsewhere.

**Not deployed.**  `matmul` dispatches here, but `offers` names the path only
under `tiling.EXCHANGE`, which is off, so nothing generates it by default.
Turning it on changes every F64 kernel on CDNA 2 and later that runs at the
wave's lane count (`mfma_f64_16x16x4f64` takes all 64 lanes); the check
available here is that the emitted call sequence matches the plans --- which
says the emitter is faithful to them and nothing about whether the instruction
then computes the right thing.  That wants a machine.
"""

from functools import lru_cache
from typing import Optional

from tensorforge.backend.pir.core import ScalarType
from .catalog import MATRIX_OPS, Call
from .features import has_feature, wave_size
from .layouts import covers, element_at
from .reorder import (IDENTITY_DPP, a_exchange, accumulator_gathers,
                      broadcast_feeds_a, fragment_moves)


def exchange_op(dtype, threads, ctx) -> Optional[object]:
    """The instruction this path would emit here, or `None`.

    Largest first, and the conditions are the three plans' own: the
    instruction has to be emittable on this target, its A fragment has to come
    from a transpose, and both its B fragment and its accumulator have to have
    a plan.  Asking all three here rather than discovering the third halfway
    through an emission is what lets `matmul` gate on one call.
    """
    found = exchange_ops(dtype, threads, ctx)
    return found[0] if found else None


def exchange_ops(dtype, threads, ctx) -> tuple:
    """Every instruction this path could emit here, largest first.

    The same conditions, kept in one place: a selection that reads them and a
    ranking that reads them again are two chances to disagree about what the
    path can serve.
    """
    out = []
    for op in sorted(MATRIX_OPS, key=lambda o: (-o.m * o.n, o.builtin)):
        if op.call is not Call.MFMA or not op.available_for(dtype, ctx):
            continue
        if threads != op.wave or wave_size(ctx) != op.wave:
            continue
        if not has_feature(ctx, op.feature) or not _planned(op):
            continue
        out.append(op)
    return tuple(out)


@lru_cache(maxsize=None)
def _planned(op) -> bool:
    """Whether the three plans exist for `op`: properties of the instruction
    alone, and searches over its layouts that a storage settlement asking
    `offers` for every operation would otherwise repeat each time (SeisSol's
    `localFluxAll` at order 6 did not finish in 25 minutes)."""
    return (covers(op) and not broadcast_feeds_a(op)
            and a_exchange(op) is not None
            and fragment_moves(op, 'B', 0) is not None
            and accumulator_gathers(op, 0) is not None)


@lru_cache(maxsize=None)
def _b_moves(op, group):
    """`fragment_moves` for the B fragment of contraction group `group`: the
    same search for every fragment of every issue, and of every placement the
    generator tries (4.3 million `locate` calls in two minutes of
    `localFluxAll` at order 6)."""
    return fragment_moves(op, 'B', 0, group)


@lru_cache(maxsize=None)
def _gathers(op, column):
    """`accumulator_gathers`, kept for the same reason as `_b_moves`."""
    return accumulator_gathers(op, column)


def _swapped(writer, value, blocks, ftype):
    """`value` through a `swap` sequence, one call per block."""
    for block in blocks:
        value = writer.call(f'tensorforge::swap<{block}>', ftype, value,
                            hint='sw')
    return value


def _merge(writer, into, value, select, ftype):
    """One region of a register, through `dppUpdate`'s masks.

    The identity control makes it a merge and not also a shuffle, and the two
    masks carry the region.  A region no mask reaches would need a ternary on
    the lane id; `Select` reports that as `cndmask` and this refuses it rather
    than emitting a lane id read the rest of the path does not need --- there
    is no such region in the catalog today, and one appearing is a thing to
    look at rather than to paper over.
    """
    if not select.free:
        # A region finer than a bank -- one lane out of every four, which is
        # what transposing a 4x4 needs -- is outside `dppUpdate`'s masks, and
        # a ternary on the lane id is what serves it.  `transpose4x4b32`
        # writes that by hand four times, so it is not a read this path
        # avoids; it is one it already makes inside a runtime function.
        callee = f'tensorforge::laneMerge<{select.mask}ULL>'
        return writer.call(callee, ftype, value,
                           value if into is None else into, hint='frag')
    callee = (f'tensorforge::dppUpdate<{IDENTITY_DPP}, {select.row_mask}, '
              f'{select.bank_mask}, false>')
    if into is None:
        return writer.call(callee, ftype, value, value, hint='frag')
    return writer.call(callee, ftype, value, into, hint='frag')


def apply_exchange(writer, regs, groups, ftype):
    """The registers at the other layout, or `None`.

    One output at a time: every source that reaches it, swapped into place and
    merged under the region's mask.  The masks of one group partition the
    wave, so the first merge writes rather than combines and `_merge` takes
    `None` for that.

    `None` where a region needs a mask `dppUpdate` cannot express.  The plan
    reports that through `Select`, and refusing here is the same refusal
    `_merge` already makes for the fragments -- the caller falls back rather
    than getting a register with a hole in it.
    """
    out = []
    for group in groups:
        value = None
        for move in group:
            moved = _swapped(writer, regs[move.contraction], move.swaps, ftype)
            value = _merge(writer, value, moved, move.select, ftype)
            if value is None:
                return None
        out.append(value)
    return out


def fragment_index(op, M, block, group, i, g) -> int:
    """Which B fragment of the leading operand `matmul_exchange` issues, in
    the order it issues them: k-block, contraction group, lead slot, lead
    group.  The order a prepared operand is stored in (`exchange_order`)."""
    exchange = a_exchange(op)
    groups = op.wave // (op.n * op.blocks)
    return ((block * exchange.groups + group) * M + i) * groups + g


def exchange_order(op, shape, threads, window, head=0):
    """`Tensor.storage_order` for a leading operand read as B fragments.

    Fragment `fragment_index(...)` holds, in lane `l`, what the plan would
    have swapped there (`fragment_moves`): the row of lane `group * span +
    block * extent + n` of the slot, at the step the lane's contraction index
    stands for -- position `base + stride * kk + group` of the shared
    operand's block, which is the lead's step `head` earlier (`matmul32`
    pairs them the same way).  So one read per fragment, every lane its own
    element, where the plan took swaps and merges.  Steps and rows outside
    the window are padding (`-1`).

    `window` is `((r0, rows), (c0, steps))` as `amd.quad_order` takes it, and
    `head` where the operation's contraction starts inside the shared
    operand's first block of `threads` steps.
    """
    exchange = a_exchange(op)
    extent, span = op.n, op.n * op.blocks
    groups = op.wave // span
    depth = exchange.stride * op.k
    box_rows, box_cols = int(shape[0]), int(shape[1])
    (r0, rows), (c0, steps) = window
    slots = -(-rows // threads)
    lanes = [element_at(op, 'B', 0, lane) for lane in range(op.wave)]
    order = []
    for base in range(0, steps + head, depth):
        for group in range(exchange.groups):
            for i in range(slots):
                for g in range(groups):
                    for block, kk, n in lanes:
                        r = base + exchange.stride * kk + group - head
                        row = i * threads + g * span + block * extent + n
                        inside = (0 <= r < steps and row < rows
                                  and r0 + row < box_rows
                                  and c0 + r < box_cols)
                        order.append((r0 + row) + box_rows * (c0 + r)
                                     if inside else -1)
    return order


def matmul_exchange(writer, C, shared, lead, M, N, K, kx, threads, dtype,
                    sparse, ctx, start, stop, lead_slot=None) -> bool:
    """``C[i,j] += lead[i,k] * shared[j,k]`` through a `k > 1` instruction.

    `shared` and `lead` are the accessors under their own names rather than
    `A` and `B`: the instruction's A fragment is fed by the *shared* matrix
    and its B fragment by the *leading* operand, so carrying the caller's
    letters through here would name each one after the other one.

    The shared operand is read a block of `threads` steps at a time from the
    block holding the contraction's first step, `head = kx % threads` steps
    before it; the lead accessor counts from that first step.  So position
    `p` of a block pairs with the lead's step `p - head`, as in `matmul32`.

    `lead_slot(writer, index)`, where the leading operand is stored in
    `exchange_order`: fragment `index` (`fragment_index`) as one read.

    Returns `False` without emitting where the path does not apply.  A caller
    inside `Writer.speculative` may also see it return `False` after it has
    emitted --- a region that needs a `cndmask` is found per fragment, not up
    front -- and the discard is what makes that safe.
    """
    if sparse is not None:
        return False
    op = exchange_op(dtype, threads, ctx)
    if op is None:
        return False

    exchange = a_exchange(op)
    ftype = ScalarType(dtype)
    acctype = ScalarType(dtype, op.d.per_lane)
    ext, span = op.m, op.n * op.blocks
    groups = op.wave // span
    depth = exchange.stride * op.k
    head = kx % threads

    with writer.AnonymousScope():
        for j in range(start, stop, ext):
            with writer.AnonymousScope():
                # One accumulator per leading-dimension group: the instruction
                # covers `span` of it and the wave holds `threads`.
                acc = {(i, g): writer.declare(acctype, hint='acc')
                       for i in range(M) for g in range(groups)}

                for base in range(0, K + head, depth):
                    columns = []
                    for jj in range(ext):
                        if j + jj < N:
                            columns.append(shared(writer, None, j + jj,
                                                  base // threads))
                        else:
                            # Padding a partial block with real zeroes, so the
                            # issue over the full block contributes nothing.
                            columns.append(writer.const(0.0, ftype))
                    if any(v is None or v is False for v in columns):
                        return False
                    # The transpose writes its operands in place, and a column
                    # can be a constant: the padding, or an operator whose
                    # numbers the kernel spells as literals.  Each gets a
                    # register of its own first.
                    fragments = []
                    for column in columns:
                        own = writer.declare(ftype, hint='xa')
                        writer.assign(own, column)
                        fragments.append(own)
                    writer.call_stmt(f'tensorforge::transpose{ext}x{ext}b32',
                                     *fragments, writes=fragments)

                    for group in range(exchange.groups):
                        # A group none of whose steps is the operation's
                        # multiplies zeros: no issue at all.
                        if not any(0 <= base + p - head < K
                                   for p in exchange.covers(group, op.k)):
                            continue
                        for i in range(M):
                            for g in range(groups):
                                if lead_slot is not None:
                                    built = lead_slot(writer, fragment_index(
                                        op, M, base // depth, group, i, g))
                                else:
                                    built = _b_fragment(writer, op, lead, i,
                                                        g, base, group,
                                                        exchange, ftype,
                                                        head, K)
                                if built is None or built is False:
                                    return False
                                acc[(i, g)] = writer.call(
                                    op.callee, acctype,
                                    fragments[group], built, acc[(i, g)],
                                    0, 0, 0, hint='acc', movable=False,
                                    materialize=True)

                for i in range(M):
                    for column in range(min(ext, N - j)):
                        value = _writeback(writer, op, acc, i, column, ftype)
                        if value is None:
                            return False
                        C(writer, value, i, j + column)
    return True


def _b_fragment(writer, op, lead, i, group, base, kgroup, exchange, ftype,
                head, steps):
    """One B fragment: the plan, emitted.  Position `p` of the block is the
    lead's step `p - head` (`matmul_exchange`); one outside `[0, steps)` is a
    zero."""
    built = None
    for move in _b_moves(op, group):
        r = base + exchange.stride * move.contraction + kgroup - head
        if not 0 <= r < steps:
            value = writer.const(0.0, ftype)
        else:
            value = lead(writer, None, i, r)
            if value is None or value is False:
                return None
        built = _merge(writer, built,
                       _swapped(writer, value, move.swaps, ftype),
                       move.select, ftype)
        if built is None:
            return None
    return built


def _writeback(writer, op, acc, i, column, ftype):
    """One output column, gathered out of the group accumulators."""
    built = None
    for gather in _gathers(op, column):
        value = writer.extract(acc[(i, gather.group)], gather.slot, ftype)
        built = _merge(writer, built,
                       _swapped(writer, value, gather.swaps, ftype),
                       gather.select, ftype)
        if built is None:
            return None
    return built
