# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Emitting a matrix path from the plans, for the instructions with `k > 1`.

`codegen.matmul32` covers the K=1 tiles: the instruction broadcasts its own A
operand and its accumulator already sits where the nest wants it, so the whole
path is a transpose, a chain of issues and a store.  Nothing in it generalises,
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

**Not routed.**  `offers` does not name this path and `matmul` does not
dispatch to it, so nothing generates it yet.  Wiring it up changes every F64
kernel on CDNA 2 and later, and the only check available here is that the
emitted call sequence matches the plans --- which says the emitter is faithful
to them and nothing about whether the instruction then computes the right
thing.  That wants a machine, so the decision to route belongs with someone
who has one.
"""

from typing import Optional

from tensorforge.common.basic_types import Datatype
from tensorforge.backend.pir.core import ScalarType
from .catalog import MATRIX_OPS, Call
from .features import has_feature, wave_size
from .layouts import covers
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
        if not has_feature(ctx, op.feature) or not covers(op):
            continue
        if broadcast_feeds_a(op) or a_exchange(op) is None:
            continue
        if fragment_moves(op, 'B', 0) is None:
            continue
        if accumulator_gathers(op, 0) is None:
            continue
        out.append(op)
    return tuple(out)


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
    is no such region in the catalogue today, and one appearing is a thing to
    look at rather than to paper over.
    """
    if not select.free:
        return None
    callee = (f'tensorforge::dppUpdate<{IDENTITY_DPP}, {select.row_mask}, '
              f'{select.bank_mask}, false>')
    if into is None:
        return writer.call(callee, ftype, value, value, hint='frag')
    return writer.call(callee, ftype, value, into, hint='frag')


def matmul_exchange(writer, C, shared, lead, M, N, K, kx, threads, dtype,
                    sparse, ctx, start, stop) -> bool:
    """``C[i,j] += lead[i,k] * shared[j,k]`` through a `k > 1` instruction.

    `shared` and `lead` are the accessors under their own names rather than
    `A` and `B`: the instruction's A fragment is fed by the *shared* matrix
    and its B fragment by the *leading* operand, so carrying the caller's
    letters through here would name each one after the other one.

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

    with writer.AnonymousScope():
        for j in range(start, stop, ext):
            with writer.AnonymousScope():
                # One accumulator per leading-dimension group: the instruction
                # covers `span` of it and the wave holds `threads`.
                acc = {(i, g): writer.declare(acctype, hint='acc')
                       for i in range(M) for g in range(groups)}

                for base in range(0, K + kx, depth):
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
                    fragments = list(columns)
                    writer.call_stmt(f'tensorforge::transpose{ext}x{ext}b32',
                                     *fragments, writes=fragments)

                    for group in range(exchange.groups):
                        for local in range(op.k):
                            if base + exchange.covers(group, op.k)[local] \
                                    >= K + kx:
                                break
                        for i in range(M):
                            for g in range(groups):
                                built = _b_fragment(writer, op, lead, i, g,
                                                    base, group, exchange,
                                                    ftype, K + kx)
                                if built is None:
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
                depth):
    """One B fragment: the plan, emitted."""
    built = None
    for move in fragment_moves(op, 'B', 0, group):
        k = base + exchange.stride * move.contraction + kgroup
        if k >= depth:
            value = writer.const(0.0, ftype)
        else:
            value = lead(writer, None, i, k)
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
    for gather in accumulator_gathers(op, column):
        value = writer.extract(acc[(i, gather.group)], gather.slot, ftype)
        built = _merge(writer, built,
                       _swapped(writer, value, gather.swaps, ftype),
                       gather.select, ftype)
        if built is None:
            return None
    return built
