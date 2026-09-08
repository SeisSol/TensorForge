# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Cyclic becomes blocked, and that is the whole of what `width` does.

At `width == 1` a lane holds elements `lane, lane + block, lane + 2*block, …`
of the lead dimension. Those are `block` apart, so no alignment of any base
makes them one access -- the per-lane element set is the obstacle, not the
address. `width = 2` changes the map to

    idx = width * (((tid / stride) % block) + nonlead * block) + c

so a lane holds `2*lane, 2*lane+1` and then that pair `2*block` further on.
Adjacent, therefore castable, therefore one `float2`.

What is deliberately *not* changed is `layout()`. Which lane holds which share
is still `LaneAxis(block, stride)`; the width sits on the value's
`ScalarType.length`, where `LaneAxis`'s own docstring says packing belongs.
Two indices that differ only in width address the same distribution at
different granularity, and a pass asking "is this a shuffle?" must keep
getting "no".
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.memory.vectorize import (
    lead_threads_and_width, lead_vector_width)
from tensorforge.backend.symbol import LeadIndex, LeadLoop
from tensorforge.common.exceptions import InternalError


def elements(block, stride, width, slots, threads):
    """The element each (lane, slot) pair holds, as the index map defines it."""
    out = {}
    for slot in range(slots):
        for tid in range(threads):
            lane = (tid // stride) % block
            base = width * (lane + slot * block)
            out[(tid, slot)] = list(range(base, base + width))
    return out


# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #

def test_at_width_one_a_lane_holds_a_strided_set():
    """The state of affairs the width exists to change."""
    e = elements(block=16, stride=1, width=1, slots=3, threads=16)
    assert [e[(3, s)][0] for s in range(3)] == [3, 19, 35]


def test_at_width_two_a_lane_holds_adjacent_pairs():
    e = elements(block=16, stride=1, width=2, slots=3, threads=16)
    assert e[(3, 0)] == [6, 7]
    assert e[(3, 1)] == [38, 39]


def test_the_whole_range_is_still_covered_exactly_once():
    """Blocking is a permutation of the assignment, not a change of extent."""
    for width in (1, 2, 4):
        e = elements(block=16, stride=1, width=width, slots=2, threads=16)
        seen = sorted(i for v in e.values() for i in v)
        assert seen == list(range(16 * 2 * width))


# --------------------------------------------------------------------------- #
# What LeadIndex says about it
# --------------------------------------------------------------------------- #

def test_width_scales_the_whole_index_not_just_the_lane():
    """Scaling the lane term alone would interleave the slots into each other.

    Slot 1 must start `width * block` past slot 0; if only the lane were
    scaled it would start `block` past it and overlap slot 0's second halves.
    """
    assert LeadIndex(3, 16, 1).lead() == 48
    assert LeadIndex(3, 16, 1, width=2).lead() == 96


def test_width_does_not_change_the_distribution():
    """`layout()` answers which lane holds a share, and that is unchanged.

    A pass asking "may these two register images be treated as the same
    distribution?" must keep getting yes -- moving between them is a change of
    granularity, not a shuffle.
    """
    narrow = LeadIndex(0, 16, 1)
    wide = LeadIndex(0, 16, 1, width=2)
    assert narrow.layout() == wide.layout()
    assert narrow.same_layout(wide)


def test_width_is_part_of_index_identity():
    """`same_layout` is the weaker question; `==` is value equality of the
    index, and two indices of different width name different elements."""
    assert LeadIndex(0, 16, 1) != LeadIndex(0, 16, 1, width=2)


def test_width_one_is_the_old_index_unchanged():
    assert repr(LeadIndex(2, 16, 1)) == repr(LeadIndex(2, 16, 1, width=1))
    assert LeadIndex(2, 16, 1).lead() == 32


def test_a_zero_width_is_refused():
    with pytest.raises(InternalError):
        LeadIndex(0, 16, 1, width=0)


# --------------------------------------------------------------------------- #
# The ragged end: over-compute rather than refuse
# --------------------------------------------------------------------------- #

def test_a_ragged_range_is_accepted():
    """The boundary lane holds a vector half outside the box and computes it.

    Excluding it instead would drop an element that *is* inside; there is no
    lane bound that does both, and splitting the components is a different
    mechanism.  What the extra component costs is not instructions -- the
    guarded tail slot occupies the whole warp either way.
    """
    assert LeadLoop('n0', 0, 35, 32, 1, width=2).width == 2


def test_the_straddling_lane_is_excluded_and_its_element_peeled():
    """With 3 elements left and width 2, one whole vector fits and one
    element is left over.

    The bound excludes the straddling lane rather than over-including it, and
    the leftover comes back as a plain element index -- a scalar FMA on the
    machinery that already handles a fixed element of a distributed
    dimension.  Over-computing it instead was safe for the destination, whose
    guard is at element granularity, and safe for the *source* only where the
    operand window happened to be sized past the extent.
    """
    loop = LeadLoop('n0', 0, 35, 32, 1, width=2)
    assert loop._lane_hi(3) == 1
    assert loop._peeled(3) == 1
    assert loop._peeled(4) == 0


def test_a_width_one_loop_peels_nothing():
    loop = LeadLoop('n0', 0, 35, 32, 1)
    for offset in range(6):
        assert loop._peeled(offset) == 0
        assert loop._lane_hi(offset) == offset


def test_the_peel_hands_over_the_elements_the_vectors_missed():
    seen = []
    LeadLoop('n0', 0, 35, 32, 1, width=2)._peel(
        lambda idx: seen.append(idx[0]), 34, 35 % 64)
    assert seen == [34]


def test_at_width_one_the_bounds_are_the_element_offsets():
    """Every existing call site must see the arithmetic it saw before."""
    loop = LeadLoop('n0', 0, 35, 32, 1)
    for offset in range(6):
        assert loop._lane_lo(offset) == offset
        assert loop._lane_hi(offset) == offset


def test_width_one_accepts_everything_as_before():
    for start, end in [(0, 35), (8, 72), (1, 2), (0, 9)]:
        assert LeadLoop('n0', start, end, 16, 1).width == 1


# --------------------------------------------------------------------------- #
# Choosing the width: registers, not divisibility
# --------------------------------------------------------------------------- #

def test_an_unproven_base_gets_width_one():
    assert lead_vector_width(0, 32, 16, elem_bytes=4, align_bytes=0) == 1


def test_a_dividing_extent_gets_two():
    assert lead_vector_width(0, 32, 16, elem_bytes=4, align_bytes=16) == 2


def test_a_ragged_extent_that_costs_no_registers_still_gets_two():
    """35 over 32 lanes: two floats per lane either way, so the width is free.

    This is the case the divisibility rule used to refuse, and refusing it
    was the reason the policy answered 1 for almost the whole corpus.
    """
    assert lead_vector_width(0, 35, 32, elem_bytes=4, align_bytes=16) == 2


def test_a_ragged_extent_that_does_cost_registers_gets_one():
    """9 over 32 lanes: the dimension does not fill one slot, so half of
    every vector is waste and the lane pays a register for it."""
    assert lead_vector_width(0, 9, 32, elem_bytes=4, align_bytes=16) == 1
    assert lead_vector_width(0, 35, 16, elem_bytes=4, align_bytes=16) == 1


def test_paying_the_register_is_available_to_a_caller_who_measured():
    assert lead_vector_width(0, 9, 32, elem_bytes=4, align_bytes=16,
                             pay_registers=True) == 2


def test_an_offset_start_is_left_out():
    """The head straddles like the tail and additionally shifts every later
    slot.  No operator in the corpus starts at such an offset."""
    assert lead_vector_width(8, 72, 16, elem_bytes=4, align_bytes=16) == 1


def test_fp64_reaches_two_from_the_same_base():
    assert lead_vector_width(0, 32, 16, elem_bytes=8, align_bytes=16) == 2


def test_the_default_cap_is_two_and_is_a_judgement():
    assert lead_vector_width(0, 64, 16, elem_bytes=4, align_bytes=16) == 2
    assert lead_vector_width(0, 64, 16, elem_bytes=4, align_bytes=16,
                             cap=4) == 4


def test_an_empty_range_is_width_one():
    assert lead_vector_width(4, 4, 16, elem_bytes=4, align_bytes=16) == 1


# --------------------------------------------------------------------------- #
# Choosing the lane count and the width together
# --------------------------------------------------------------------------- #

def scalar_floats(extent, threads):
    return threads * -(-extent // threads)


def wide_floats(extent, threads, width):
    return threads * -(-extent // (threads * width)) * width


def test_the_thread_count_is_not_a_constant_of_the_problem():
    """Why `lead_vector_width` answers 1 for most of the corpus.

    403 of 446 lead loops have an extent no larger than the thread count, so a
    lane already holds one element and a width of 2 at fixed lane count can
    only mean half the wave runs empty. Halving the lanes instead is the same
    elements in half the instructions.
    """
    assert lead_vector_width(0, 32, 32, elem_bytes=4, align_bytes=16) == 1
    assert lead_threads_and_width(32, elem_bytes=4, align_bytes=16) == (16, 2)


@pytest.mark.parametrize('extent', [9, 12, 16, 20, 32, 35, 56, 64, 120, 512])
def test_the_total_register_count_is_unchanged(extent):
    """The invariant that makes this safe, and it is a *total*, not per lane.

    A lane carries `w` times as many floats and there are `w` times fewer
    lanes. Per block that cancels exactly; against a per-thread register cap
    it does not, which is the constraint that already binds in FP64 at order
    6 -- so this is neutral where register pressure is not already the limit
    and needs a measurement where it is.
    """
    narrow_threads, narrow_width = lead_threads_and_width(extent, 4, 0)
    threads, width = lead_threads_and_width(extent, 4, 16)
    assert narrow_width == 1
    assert (wide_floats(extent, threads, width)
            <= scalar_floats(extent, narrow_threads))


@pytest.mark.parametrize('extent', [9, 12, 16, 20, 32, 35, 56, 64, 120])
def test_the_lanes_still_cover_the_extent(extent):
    threads, width = lead_threads_and_width(extent, 4, 16)
    assert threads * width * -(-extent // (threads * width)) >= extent


def test_an_unproven_base_reproduces_todays_choice():
    """`get_num_threads` rounds the extent up to a power of two, capped at 32."""
    for extent, threads in [(9, 16), (12, 16), (20, 32), (32, 32), (120, 32)]:
        assert lead_threads_and_width(extent, 4, 0) == (threads, 1)


def test_a_short_dimension_halves_the_lanes_rather_than_wasting_them():
    """32 over 32 lanes is the corpus's most common shape by a wide margin."""
    assert lead_threads_and_width(32, 4, 16) == (16, 2)
    assert lead_threads_and_width(16, 4, 16) == (8, 2)


def test_a_long_dimension_keeps_the_lanes_and_takes_the_width():
    """Past the cap the lane count cannot grow, so the width buys slots."""
    assert lead_threads_and_width(120, 4, 16) == (32, 2)
    assert lead_threads_and_width(512, 4, 16) == (32, 2)


def test_fp64_gets_the_same_treatment_from_a_16_byte_base():
    assert lead_threads_and_width(32, 8, 16) == (16, 2)


def test_a_degenerate_extent_is_one_lane():
    assert lead_threads_and_width(0, 4, 16) == (1, 1)


# --------------------------------------------------------------------------- #
# Register blocking: what makes the packed FMA pay for its own splat
# --------------------------------------------------------------------------- #

def test_blocking_reduces_the_lane_count_a_second_time():
    """`R` vectors per lane means `R` times fewer lanes, at constant total.

    The same lever as the width, one level down. What it buys is not loads --
    those are already wide -- but the amortisation of everything that is per
    `b` rather than per element: one load of `b` and one splat of it now feed
    `R` fused multiply-adds instead of one.
    """
    assert lead_threads_and_width(32, 4, 16) == (16, 2)
    assert lead_threads_and_width(32, 4, 16, blocking=2) == (8, 2)
    assert lead_threads_and_width(32, 4, 16, blocking=4) == (4, 2)


@pytest.mark.parametrize('extent', [16, 20, 32, 35, 56, 120])
@pytest.mark.parametrize('blocking', [1, 2, 4])
def test_blocking_keeps_the_total_register_count_neutral(extent, blocking):
    """As the width does, and for the same reason: `R` times as many floats
    per lane against `R` times fewer lanes."""
    narrow_threads, _ = lead_threads_and_width(extent, 4, 0)
    threads, width = lead_threads_and_width(extent, 4, 16, blocking=blocking)
    assert (wide_floats(extent, threads, width)
            <= scalar_floats(extent, narrow_threads) * blocking)


@pytest.mark.parametrize('extent', [16, 20, 32, 35, 56, 120])
@pytest.mark.parametrize('blocking', [1, 2, 4])
def test_the_lanes_still_cover_the_extent_when_blocked(extent, blocking):
    threads, width = lead_threads_and_width(extent, 4, 16, blocking=blocking)
    assert threads * width * -(-extent // (threads * width)) >= extent


def test_blocking_does_nothing_without_a_width():
    """It is a second factor on the same decision, not an independent one.

    An operand that cannot prove its alignment gets no width, and then there
    is no splat to amortise and no reason to give up lanes.
    """
    assert lead_threads_and_width(32, 4, 0, blocking=4) == (32, 1)


def test_a_zero_blocking_is_refused():
    with pytest.raises(ValueError):
        lead_threads_and_width(32, 4, 16, blocking=0)


# --------------------------------------------------------------------------- #
# The cap
# --------------------------------------------------------------------------- #
# It was the constant 2, and the constant hid a question rather than answering
# it: `widths_for` offers 4 for an FP32 base of 16-byte alignment, so `float4`
# was unreachable, and `double2` was reachable only because 2 happened to be
# both the cap and the ceiling for FP64.

from tensorforge.backend.instructions.compute.packed import packed_fma_width
from tensorforge.backend.instructions.memory.vectorize import (  # noqa: E402
    lead_width_cap, lead_threads_and_width)


@pytest.mark.parametrize('elem,align,expected', [
    (4, 0, 1), (4, 8, 2), (4, 16, 4), (4, 32, 4),
    (8, 0, 1), (8, 8, 1), (8, 16, 2), (8, 32, 2),
])
def test_the_cap_is_what_the_address_proves(elem, align, expected):
    assert lead_width_cap(elem, align) == expected


def test_an_unproven_alignment_caps_at_one():
    """Same permission as `widths_for`'s: not known to be aligned and known to
    be element-aligned are one answer, and a cast that needs 16 must not
    acquire the permission by default."""
    assert lead_width_cap(4, 0) == 1
    assert lead_threads_and_width(32, 4, 0, cap=lead_width_cap(4, 0))[1] == 1


def test_float4_is_reachable_and_double4_is_not():
    """The two the cap used to decide together, now decided apart.

    16 bytes is the access ceiling, so FP32 reaches 4 and FP64 stops at 2 --
    not because doubles are special but because two of them are already the
    widest access there is.
    """
    assert lead_threads_and_width(64, 4, 16, cap=lead_width_cap(4, 16))[1] == 4
    assert lead_threads_and_width(64, 8, 16, cap=lead_width_cap(8, 16))[1] == 2


@pytest.mark.parametrize('vendor,arch', [('nvidia', 'sm_86'),
                                         ('nvidia', 'sm_100'),
                                         ('amd', 'gfx90a'), ('amd', 'gfx908')])
def test_the_fma_width_is_not_the_cap(vendor, arch):
    """A vector wider than the packed FMA is several of them, not none.

    The intuition runs the other way, so this states it: every element past
    the first amortises the one load and the one splat further, and the
    per-element instruction count falls with the width whether or not the
    arithmetic packs.  A scalar-FMA target gains *more* from the step to 4
    than a packed one does, which is the opposite of a ceiling.
    """
    packed = packed_fma_width(vendor, arch, 4)
    cap = lead_width_cap(4, 16)
    assert cap >= packed, 'the address ceiling, not the instruction width'

    def per_element(w):
        # one load of b, one splat of it, and w/p fused multiply-adds
        return (1 + 1 + -(-w // packed)) / w

    assert per_element(4) < per_element(2) < per_element(1)


def test_the_address_and_the_validation_agree_and_stay_two_questions():
    """Both 4 now.  Kept apart because they are still different facts.

    `lead_width_cap` is what a 16-byte-aligned FP32 base permits;
    `VALIDATED_LEAD_WIDTH` is what has been shown to compute the right
    numbers.  Collapsing them once they coincide is how the next ceiling --
    32-byte alignment, or a width the register budget cannot carry -- would
    arrive already claimed.
    """
    from tensorforge.backend.instructions.memory.vectorize import (
        VALIDATED_LEAD_WIDTH)
    assert lead_width_cap(4, 16) == 4
    assert lead_width_cap(8, 16) == 2
    assert VALIDATED_LEAD_WIDTH == 4


@pytest.mark.parametrize('extent,threads,width,expected', [
    # what the image used to size, against what a wide read needs
    (12, 4, 4, 4), (16, 4, 4, 4), (20, 8, 4, 4), (32, 8, 4, 4),
    (33, 32, 1, 2), (33, 32, 2, 2), (31, 32, 1, 1),
])
def test_the_slot_count_is_floats_and_not_slots(extent, threads, width,
                                                expected):
    """`w * (ceil(u/(T*w)) - floor(l/(T*w)))`, stated once.

    The old expression was `ceil(u/T)`, which is a lane's slot count and not
    its float count; the two differ at 12/4/4, where it gave 3 and a four-wide
    read needs 4.  It was written out three times -- addressing and both
    allocation sites -- so the width reached none of them, and the failure was
    every destination cell wrong on the extents where the two disagree.
    """
    from tensorforge.backend.symbol import slots_for
    assert slots_for(0, extent, threads, width) == expected


def _run_case(monkeypatch, M, N, width, align=16, dtype=None):
    """`D = A B` at `width`, run over one block on the host interpreter."""
    import re

    from tensorforge.backend.instructions.memory import vectorize
    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.context import Context
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.generators.generator import Generator
    from kernel_eval import evaluate_wave

    dtype = dtype or Datatype.F32
    monkeypatch.setattr(vectorize, 'LEAD_VECTORIZE', True)
    monkeypatch.setattr(vectorize, 'VALIDATED_LEAD_WIDTH', width)

    def t(shape, alias):
        return SubTensor(Tensor(shape, Addressing.STRIDED,
                                BoundingBox([0] * len(shape), list(shape)),
                                alias=alias, datatype=dtype, alignment=align))

    gen = Generator([GemmDescr(False, False, a=t([M, 8], 'A'),
                               b=t([8, N], 'B'), c=t([M, N], 'D'),
                               alpha=1.0, beta=0.0)],
                    Context(arch='sm_86', backend='cuda', fp_type=dtype))
    gen.register()
    gen.generate()
    src = gen.get_kernel()
    lanes = max([int(x) for x
                 in re.findall(r'threadIdx\.x % (\d+)', src)] or [32])
    mem = evaluate_wave(src, lanes, seed=7, globals_only=True,
                        preset={'m0': 0.0})
    return src, [mem.get(('m0', i)) or 0.0 for i in range(M * N)]


def _wrong_lead_indices(monkeypatch, M, N, width, **kw):
    """Which lead elements the widened kernel gets wrong, or a skip.

    The skip is the interpreter's limit and not the generator's: a narrow
    extent puts the operand broadcast in its template form
    (`tensorforge::broadcast<8, 1, 4>`), which `kernel_eval` does not parse.
    Raised as a skip rather than filtered out of the parameter list, so the
    day it is modelled these cases start running instead of staying quietly
    absent.
    """
    from kernel_eval import Abort
    try:
        _, scalar = _run_case(monkeypatch, M, N, 1, **kw)
        _, wide = _run_case(monkeypatch, M, N, width, **kw)
    except Abort as exc:
        pytest.skip(f'host interpreter cannot evaluate this shape: {exc}')
    return sorted({i % M for i, (a, b) in enumerate(zip(scalar, wide))
                   if abs(a - b) > 1e-4})


def _slot_stride(extent, threads, width):
    """What the register image uses, against what a wide read needs."""
    return -(-extent // threads), width * -(-extent // (threads * width))


#: The extents the old slot count got wrong, kept as the parametrisation
#: rather than replaced by a round set: they are where `ceil(u/T)` and
#: `w * ceil(u/(T*w))` disagree, and a regression in `slots_for` shows up here
#: first.
WIDTH4_WAS_BROKEN = [12, 17, 20, 24, 33, 35, 40, 48]
WIDTH4_WAS_FINE = [8, 16, 31, 32, 56, 64]


@pytest.mark.parametrize('extent', WIDTH4_WAS_BROKEN + WIDTH4_WAS_FINE)
@pytest.mark.parametrize('columns', [3, 8])
def test_width_four_computes_the_same_numbers(monkeypatch, extent, columns):
    """Both halves of the old split, now one answer.

    The image sized a lane's share as its slot count where a four-wide read
    needs its float count, so consecutive non-lead indices addressed
    overlapping windows -- column 1 starting one register inside column 0 --
    and every destination cell came out wrong on the extents where the two
    expressions disagree.  Where they happened to agree, width 4 was already
    right, which is what said the defect was the stride and not the width.
    """
    assert _wrong_lead_indices(monkeypatch, extent, columns, 4) == []


@pytest.mark.parametrize('extent', WIDTH4_WAS_BROKEN + WIDTH4_WAS_FINE)
def test_width_two_computes_the_same_numbers(monkeypatch, extent):
    """The width that was already offered, held in place while 4 arrives."""
    assert _wrong_lead_indices(monkeypatch, extent, 3, 2) == []


@pytest.mark.parametrize('extent', [9, 15, 17, 21, 33, 35, 45, 63, 65])
@pytest.mark.parametrize('dtype_align', [(None, 16), (None, 8)],
                         ids=['align16', 'align8'])
def test_an_odd_extent_computes_the_peeled_element(monkeypatch, extent,
                                                   dtype_align):
    """The element no whole vector covers, and the lane that owns it.

    `Symbol.store` guards a fixed lead element to the one lane that holds it,
    and compared the *thread* index against the *element* index to do so --
    the same number only at width 1.  At width 2 a peeled element 32 asked for
    `threadIdx.x == 32` in a 32-lane wave, so the accumulator was never
    written and the store's `readlane` of it read what the guarded main block
    had left there.  One wrong element per column, always the last, on every
    odd extent, in FP32 and FP64 alike.

    The owning lane is `(element // width) % threads`, which is what
    `Symbol.load` already computed to broadcast the same element -- so the
    defect was the two disagreeing, and taking the store's answer from the
    symbol is what stops them.
    """
    _, align = dtype_align
    assert _wrong_lead_indices(monkeypatch, extent, 3, 2, align=align) == []


def test_the_peeled_write_is_now_exact_too():
    """Right value and written once -- two properties, fixed in that order.

    The guard on the *register* accumulator is what made the value right; it
    could not be reused for the global store, because `readlane` is
    `__shfl_sync` over the full warp mask and a single-lane branch is a
    shuffle the other lanes never reach.  Guarding the load along with the
    store removes the shuffle instead, and with the peeled element written by
    one lane the nest partitions its range at every width.

    So `placement` no longer asks whether the write is exact; what remains
    deciding an atomic is whether the target has the instruction at that
    width.  `test_lead_coverage` holds the property, and
    `test_store_exactness` holds the two ends of the capability question.
    """
    from tensorforge.backend import placement
    assert not hasattr(placement, 'atomic_write_is_exact')
