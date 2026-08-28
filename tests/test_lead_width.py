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
# The loop precondition
# --------------------------------------------------------------------------- #

def test_a_ragged_range_is_refused_rather_than_guarded():
    """A lane at a ragged end would hold a vector half inside the box.

    Masking the components is a correct answer and a different mechanism;
    refusing here keeps the choice at the one place that can avoid the
    situation entirely, which is where the width is picked.
    """
    with pytest.raises(InternalError):
        LeadLoop('n0', 0, 35, 16, 1, width=2)


def test_an_offset_start_is_refused_too():
    """The head block has the same problem as the tail."""
    with pytest.raises(InternalError):
        LeadLoop('n0', 8, 72, 16, 1, width=2)


def test_a_clean_range_is_accepted():
    assert LeadLoop('n0', 0, 64, 16, 1, width=2).width == 2


def test_width_one_never_refuses():
    """Every existing call site passes no width and must keep working."""
    for start, end in [(0, 35), (8, 72), (1, 2), (0, 9)]:
        assert LeadLoop('n0', start, end, 16, 1).width == 1


# --------------------------------------------------------------------------- #
# Choosing the width
# --------------------------------------------------------------------------- #

def test_an_unproven_base_gets_width_one():
    assert lead_vector_width(0, 32, 16, elem_bytes=4, align_bytes=0) == 1


def test_a_proven_base_and_a_dividing_extent_get_two():
    assert lead_vector_width(0, 32, 16, elem_bytes=4, align_bytes=16) == 2


def test_an_extent_that_does_not_divide_gets_one():
    """35 basis functions over 16 threads: the SeisSol shape that does not."""
    assert lead_vector_width(0, 35, 16, elem_bytes=4, align_bytes=16) == 1


def test_fp64_reaches_two_from_the_same_base():
    """`double2` is 16 bytes, which is the widest access any target has."""
    assert lead_vector_width(0, 32, 16, elem_bytes=8, align_bytes=16) == 2


def test_the_default_cap_is_two_and_is_a_judgement():
    """Wider multiplies the accumulators per lane, on a code that already
    spills at order 6 in FP64.  A caller who has measured may raise it."""
    assert lead_vector_width(0, 64, 16, elem_bytes=4, align_bytes=16) == 2
    assert lead_vector_width(0, 64, 16, elem_bytes=4, align_bytes=16,
                             cap=4) == 4


def test_an_empty_range_is_width_one():
    assert lead_vector_width(4, 4, 16, elem_bytes=4, align_bytes=16) == 1
