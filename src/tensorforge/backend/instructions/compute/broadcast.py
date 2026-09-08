# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The contraction as a chain of broadcasts and multiplies.

`A` is per-lane in the output index `i`; `B` is per-lane in the contraction
index `k`.  Two different meanings of "lane" for the two operands, and every
product needs them reconciled: one of `B`'s lanes replicated across all of
them, then an ordinary multiply.  That is the whole arrangement, and it stages
nothing -- no barrier, no arena, no round trip through shared memory.

What decides whether it is worth taking is what the replication costs, and that
is a property of the lowering rather than of the vendor.  Under an explicit
vector it is `v[k]`, an element read out of this work-item's own registers, and
free.  Under SPMD it is a real cross-lane instruction: `readlane` on CUDA and
HIP, `group_broadcast` under SYCL.  A free broadcast is what lets this beat
staging operands, and a shuffle per product is what stops it.

The AMD DPP chain in `primitives/amd/codegen.py` is this arrangement with the
replication folded into the multiply as an instruction modifier, so it costs no
instruction of its own.  The fusion is bounded by what the modifier can
address: a DPP row is sixteen lanes, so it stands in for the broadcast outright
only while the thread count is at most that.  Above it `relayout.py` supplies
the crossing separately -- which is this chain again, with the broadcast
hoisted out of the product loop and paid once per block instead of once per
lane.

Nothing here is per-target.  Which targets offer the arrangement is
`strategies()` in each vendor module, and what it costs there is `PREFERENCES`.
"""


def matmul(writer, ops, ctx, span):
    """`C[i][j] += B[k][j] * A[i][k]`, entirely in registers."""
    A, B, C = ops.A, ops.B, ops.C
    M, threads = ops.lead_slots, ops.threads
    depth = ops.k + ops.kx

    # `None` asks the loader for the value rather than for a name to fill in:
    # these are operands, and an operand whose definition the IR cannot see is
    # invisible to every pass that reasons about ordering or reuse.
    a = {}
    out_layout = None
    for i in range(M):
        for k in range(depth):
            v = A(writer, None, i, k)
            if v is not None and v is not False:
                a[(i, k)] = v
                # Taken from the operand rather than constructed: A is indexed
                # by the same output index the accumulator is, so whatever
                # distribution its loads came out with is the one to hold.
                if out_layout is None:
                    out_layout = v.layout

    for j in range(span.start, span.stop):
        # The accumulator is spread over the lanes exactly like the output it
        # holds -- one element of the lead dimension per lane.  Declared with
        # that layout rather than left untracked, because untracked is not a
        # conservative default here: an explicitly vectorised declaration
        # cannot be written without it.
        acc = [writer.declare(hint='acc', layout=out_layout) for _ in range(M)]
        for k0 in range(0, depth, threads):
            vb = B(writer, None, j, k0 // threads)
            if vb is None or vb is False:
                continue
            for lane in range(min(threads, depth - k0)):
                bk = writer.lane_broadcast(vb, lane, threads)
                for i in range(M):
                    operand = a.get((i, k0 + lane))
                    if operand is None:
                        continue
                    writer.accumulate(
                        acc[i], writer.op('mul', operand.type, bk, operand,
                                          hint='p'))
        for i in range(M):
            C(writer, acc[i], i, j)
    return True
