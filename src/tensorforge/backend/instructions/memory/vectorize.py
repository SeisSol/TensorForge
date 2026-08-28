# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which widths a linearized transfer may use, and where the hops go.

Two questions that have to be answered together, because the answer to
either alone is wrong.

*How wide* is a property of the base: reinterpreting ``&buf[i]`` as a
``T2``/``T4`` is defined only when that address is aligned to the wider type,
and the base's alignment is the only thing that can promise it.  Nothing
promised anything before -- ``GlbToRegLoader`` carried ``for g in [4, 2, 1]``
commented down to ``[1]``, which is a width decision written as a disabled
list, so re-enabling it would have cast unaligned addresses on whichever
operand happened not to be padded.

*Where the hops go* is a property of the extent, and it was wrong in a way
the width made much worse.  The old loop emitted a hop per ``range`` step and
then set ``start = (total // granularity) * granularity`` from the granularity
it had just finished with, so a partial hop at the end both overran the buffer
and was covered again by the next, narrower width.  At width 1 that overruns
by up to ``threads - 1`` elements; at width 4 by four times as many, and into
a 16-byte access that a padded batch stride no longer covers.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

#: Whether the lead dimension is vectorised at all.  Off by default, and the
#: reason is not doubt about the mechanism: it changes the thread count of
#: every kernel, and the only instrument that can say whether that was a good
#: idea is a register and occupancy measurement on real hardware.  The host
#: oracle checks that the numbers still come out right; it cannot check that
#: they come out faster.
LEAD_VECTORIZE = os.environ.get('TF_LEAD_VEC', '') not in ('', '0')

#: No target loads more than 16 bytes in one instruction: `LDG.128`/`LDS.128`
#: on NVIDIA, `global_load_dwordx4`/`ds_read_b128` on AMD.  So `double4` is
#: not a width, and `float4` is the widest there is.
MAX_ACCESS_BYTES = 16


def widths_for(elem_bytes: int, align_bytes: int,
               cap_bytes: int = MAX_ACCESS_BYTES) -> List[int]:
    """The widths a base of this alignment may be accessed at, widest first.

    ``align_bytes`` is what is *proven* about the base, not what it happens to
    be at runtime.  An unproven alignment answers ``[1]``, the same way
    ``lane_span`` refuses rather than returning 1: "not known to be 16-byte
    aligned" and "known to be 4-byte aligned" are the same permission, and a
    cast that needs 16 must not be able to acquire one by default.
    """
    if elem_bytes < 1:
        raise ValueError(f'element size must be >= 1, got {elem_bytes}')
    return [w for w in (4, 2, 1)
            if w * elem_bytes <= cap_bytes and w * elem_bytes <= max(align_bytes, elem_bytes)]


def plan_hops(total: int, threads: int,
              widths: Sequence[int]) -> Tuple[List[Tuple[int, int]], int]:
    """Cover ``[0, total)`` with whole hops, widest first.

    Returns ``(hops, tail)``: ``hops`` is a list of ``(offset, width)`` where a
    hop moves ``threads * width`` consecutive elements, and ``tail`` is what is
    left over -- fewer than ``threads`` elements once ``widths`` ends in 1.

    Three properties the old arithmetic did not have, and which
    :mod:`tests.test_vector_hops` states as tests rather than as prose:

    * no hop runs past ``total`` -- ``(total - pos) // step`` counts *whole*
      hops, where ``range(pos, total, step)`` counted started ones;
    * no element is covered twice -- ``pos`` advances by the hops actually
      emitted, not by a quantity recomputed from the granularity;
    * every hop offset is a multiple of its own width, which is what makes
      the reinterpret cast at that offset legal given an aligned base.

    The tail is returned rather than emitted as a narrow hop, because covering
    it is a different question: fewer than ``threads`` elements means some
    lanes have nothing to do, and whether they are predicated off or allowed
    to read past the end is a decision about the buffer, not about the width.
    """
    if threads < 1:
        raise ValueError(f'thread count must be >= 1, got {threads}')
    hops: List[Tuple[int, int]] = []
    pos = 0
    for w in widths:
        if w < 1:
            raise ValueError(f'width must be >= 1, got {w}')
        step = threads * w
        n = (total - pos) // step
        hops += [(pos + k * step, w) for k in range(n)]
        pos += n * step
    return hops, total - pos


def register_array_align(volume_bytes: int,
                         cap_bytes: int = MAX_ACCESS_BYTES) -> int:
    """How far a register staging array is declared aligned, in bytes.

    Stated once and read from both ends -- `allocate.py` requests it, and
    `Symbol.linear_align_bytes` reports it -- because the two are the same
    fact, and a width chosen from a guarantee the declaration does not make
    is exactly the bug this whole path is about.

    Declaring it breaks a circularity rather than adding a knob.  The width
    depends on the minimum of the two bases' alignment, and if the register
    end were decided *from* the width there would be nothing to start from;
    with the array always as aligned as one access can use, the width depends
    on the source alone.

    A small array is not over-aligned: an 8-byte one asks for 8, not 16.  The
    padding would be free but the claim would not be true of anything.
    """
    a = 1
    while a * 2 <= min(cap_bytes, max(volume_bytes, 1)):
        a *= 2
    return a


def lead_vector_width(start: int, end: int, threads: int,
                      elem_bytes: int, align_bytes: int,
                      cap: int = 2, pay_registers: bool = False) -> int:
    """How many adjacent elements one lane should hold in the lead dimension.

    This is `LeadIndex`'s width, not `plan_hops`'s.  The two answer different
    questions and share only the alignment part: a staging transfer picks a
    width per hop over a flat run, while this picks one for a distributed
    dimension whose every slot has to agree.

    The interesting condition is not divisibility.  An extent that does not
    divide leaves one lane holding a vector half outside the box, and the
    lane computes that component rather than being excluded -- a few products
    on data that is discarded.  What that costs is not instructions: the
    guarded tail slot occupies the whole warp either way, so the lane was
    already there.  It costs *registers*, and only sometimes:

        floats per lane at width 1:  ceil(extent / threads)
        floats per lane at width w:  ceil(extent / (threads * w)) * w

    For 56 over 32 lanes both are 2, and the width is free.  For 9 over 32
    both schemes need one slot, but the wide one is a `float2`, so it is 2
    floats against 1 -- the lead dimension does not even fill a slot and
    half of every vector is waste.  The default therefore takes a width only
    where the two agree; `pay_registers` lifts that for a caller who has
    measured the occupancy and wants the packed math anyway.

    Two conditions this does *not* check, because they are the caller's:

    * the over-computed component must not be stored.  The destination's own
      guard is at element granularity, so it is not -- unless the store also
      goes wide, which needs a component mask this does not provide.
    * the operand window must cover the rounded-up extent, or the read that
      produces the discarded component leaves the tile.  That is a sizing
      question for whoever allocates the window, and it is a correctness
      problem rather than a wasted lane.

    Returns 1 when nothing wider survives, which is always correct.
    """
    if threads < 1:
        raise ValueError(f'thread count must be >= 1, got {threads}')
    extent = end - start
    if extent <= 0:
        return 1
    scalar_slots = -(-extent // threads)
    for w in widths_for(elem_bytes, align_bytes):
        if w > cap:
            continue
        span = threads * w
        if start % span != 0:
            # The head has the same straddling problem as the tail, and
            # unlike the tail it shifts every later slot.  Left out for now
            # rather than guarded: no operator in the corpus starts a lead
            # dimension at an offset that is not a multiple of the span.
            continue
        if pay_registers or -(-extent // span) * w == scalar_slots:
            return w
    return 1


def _round_up_pow2(n: int, cap: int) -> int:
    """The thread count `MultilinearDescr.get_num_threads` would pick for `n`."""
    t = 1
    while t < n and t < cap:
        t *= 2
    return t


def lead_threads_and_width(extent: int, elem_bytes: int, align_bytes: int,
                           max_threads: int = 32, cap: int = 2):
    """Pick the lane count and the per-lane width together.

    `lead_vector_width` takes the thread count as given, and for most of the
    corpus that is why it answers 1: 403 of 446 lead loops have an extent no
    larger than the thread count, so a lane already holds one element and a
    width of 2 can only mean half the wave runs empty.  The thread count is
    not a constant of the problem, though -- `get_num_threads` derives it from
    the extent -- so the two are one decision.

    At width `w` the lane count needed is `ceil(extent / w)`, rounded up to a
    power of two as before.  A 32-element dimension becomes 16 lanes each
    holding a `float2` instead of 32 lanes each holding a `float`: same
    elements, same total registers for the operator, half the load and address
    instructions, and one packed FMA where the target has one.

    **Total** registers are what is neutral here, not per-lane registers.  A
    lane now carries twice as many, and there are half as many lanes.  Per
    block that cancels; against a per-*thread* register cap it does not, which
    is the constraint that already binds in FP64 at order 6.  So this is safe
    where register pressure is not already the limit and needs a measurement
    where it is.

    The caller has one more thing to get right, and it is the whole occupancy
    story: `mults_per_block` must stay put.  `RegmaxBlockPolicy` sizes it as
    `256 // num_threads`, which binds in every case in the corpus, so halving
    the lane count doubles the mults, doubles the shared memory per block and
    halves the occupancy for any operand above roughly 256 elements per mult.
    Holding the mults instead simply makes the block smaller -- shared memory
    per block unchanged, blocks per SM unchanged or better.  The win is real
    only in the second arrangement.

    Returns ``(threads, width)``; ``width == 1`` reproduces today's choice.
    """
    if extent < 1:
        return 1, 1
    scalar_threads = _round_up_pow2(extent, max_threads)
    for w in widths_for(elem_bytes, align_bytes):
        if w > cap or w == 1:
            continue
        threads = _round_up_pow2(-(-extent // w), max_threads)
        # Total floats for the operator, not per lane: a lane carries `w`
        # times as many and there are `w` times fewer of them, so the two
        # cancel -- unless the thread cap bit before the width could absorb
        # the extent, in which case the lanes hold several vectors each and
        # the rounding has to be checked rather than assumed.
        scalar_total = scalar_threads * -(-extent // scalar_threads)
        wide_total = threads * -(-extent // (threads * w)) * w
        if wide_total > scalar_total:
            continue
        return threads, w
    return scalar_threads, 1


def lead_vectorize_supported(context) -> bool:
    """Whether this backend can spell what the widened compute path emits.

    CUDA and HIP can: `VectorT`/`VectorRelaxedT` are GNU vector types, so
    they carry arithmetic and the naturally-aligned and element-aligned
    spellings convert to each other.

    SYCL cannot, and for two separate reasons.  `sycl::vec` has no
    element-aligned twin, so a relaxed cast has nowhere to go; and it does
    not define `operator*` between two `vec`s the way a GNU vector does, so
    the product does not compile even where the cast would.  The ESIMD
    emitter is further out still -- its whole model puts the lane axis in the
    type, so a per-lane width is a second axis it has no spelling for yet.

    Left as a capability question rather than a `TODO`: the widened path is
    correct on the backends that answer yes, and silently wrong on the ones
    that would need `sycl::vec`'s componentwise API instead.
    """
    if not LEAD_VECTORIZE:
        return False
    lex = context.get_vm().get_lexic()
    return getattr(lex, '_backend', None) in ('cuda', 'hip', 'hipsycl_cuda')


def reduction_vector_width(extent: int, elem_bytes: int, align_bytes: int,
                           k_stride: int, dense: bool, cap: int = 4) -> int:
    """How many reduction steps one load of the broadcast operand may cover.

    A different question from `lead_vector_width`, and the conditions barely
    overlap.  `B` in `C[m,n] += A[m,k] B[k,n]` is not indexed by the lead
    dimension, so it is loaded once and splatted; but `k` *is* `B`'s own
    contiguous axis, so `B[k,n] .. B[k+V-1,n]` are adjacent and one load
    fetches the operands of `V` consecutive reduction steps.

    What this does and does not buy:

    * It removes `V - 1` loads per `(k, n)` pair.  On an operand that lives in
      shared memory or in registers reached by a cross-lane broadcast, that is
      the instruction the inner loop issues most often after the FMA itself.
    * It does **not** remove a single splat.  The `V` components still feed
      `V` separate FMAs against `V` different `A` vectors, and each still
      needs its own `{b, b}`.  Anything that claims otherwise is confusing
      this with vectorising the reduction on *both* operands, which needs `A`
      staged k-contiguous and turns the accumulator into a partial sum.

    Unlike the lead dimension, a remainder here is free: the reduction is a
    sum, so leftover `k` values are simply added scalarly afterwards and no
    lane holds anything half-valid.  Divisibility is therefore not a
    condition, and `cap` may go to 4 where the lead width could not.

    Two conditions that are structural rather than arithmetic:

    * ``k_stride`` must be 1.  A transposed `B` has its reduction axis
      strided, and then the `V` values are not adjacent at all.
    * ``dense`` -- a sparse operand stores only its non-zeros, so a wide load
      reads across whichever entries happen to follow, and the whole point of
      the sparse path is not to touch them.
    """
    if extent < 1 or k_stride != 1 or not dense:
        return 1
    for w in widths_for(elem_bytes, align_bytes):
        if w <= cap and w <= extent:
            return w
    return 1
