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

from tensorforge.backend.instructions.memory.vectorize import lead_vector_width
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


def test_both_lane_bounds_over_include_the_straddling_lane():
    """Floor below, ceiling above.  Both directions err inwards-inclusive.

    With 3 elements left and width 2, lane 0 covers elements 0-1 and lane 1
    covers 2-3: the bound has to be 2 lanes, which keeps element 2 and picks
    up element 3 from outside.
    """
    loop = LeadLoop('n0', 0, 35, 32, 1, width=2)
    assert loop._lane_hi(3) == 2
    assert loop._lane_lo(3) == 1


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
