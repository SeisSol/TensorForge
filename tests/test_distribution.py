# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Distribution is total, and it is the only statement about lane-varying.

Three properties, each of which was false before and each of which an
explicitly vectorised emitter depends on:

1. ``uniformity`` and ``layout`` cannot disagree.  They did, on 71443 of the
   93837 values in the corpus that carried a layout, and always in the unsafe
   direction --- ``GRID`` on a value spread across the lanes.

2. *Replicated* and *unknown* are different answers.  Both spell ``float x``
   in SPMD, so nothing noticed; in ESIMD one is ``T`` and the other cannot be
   given a type at all.

3. A replicated operand does not destroy the layout of the one it is scaled
   into.  ``alpha * A`` is spread exactly like ``A``, and SeisSol scales
   almost every operator it generates.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.core import (SCALAR_LAYOUT, IRError, LaneAxis,
                                          RegisterLayout, ScalarType,
                                          Uniformity, Value, join_layout)
from tensorforge.backend.symbol import LeadIndex, VarOffset, layout_of
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)
SPREAD = RegisterLayout((LaneAxis(16),))
SPREAD8 = RegisterLayout((LaneAxis(8),))


def val(vid, layout=None, uniformity=Uniformity.GRID):
    return Value(id=vid, type=F32, uniformity=uniformity, layout=layout)


# --------------------------------------------------------------------------
# 1. one fact, one place
# --------------------------------------------------------------------------

def test_a_distributed_value_is_lane_varying_whatever_the_caller_said():
    assert val(1, SPREAD).uniformity is Uniformity.LANE


def test_derivation_only_tightens():
    """An explicit LANE is not an error to be corrected, and MULT is not raised.

    The rule narrows a claim that is too broad; it does not widen one that is
    already narrow.  Otherwise a caller who knows more than the layout does --
    the batch id is ``MULT`` and carries no layout at all -- would be overruled
    by a default.
    """
    assert val(2, SPREAD, Uniformity.LANE).uniformity is Uniformity.LANE
    assert val(3, None, Uniformity.MULT).uniformity is Uniformity.MULT
    assert val(4, SCALAR_LAYOUT, Uniformity.MULT).uniformity is Uniformity.MULT


def test_replication_does_not_make_a_value_lane_varying():
    assert val(5, SCALAR_LAYOUT).uniformity is Uniformity.GRID


# --------------------------------------------------------------------------
# 2. replicated is not unknown
# --------------------------------------------------------------------------

def test_lane_span_distinguishes_replicated_from_untracked():
    assert val(6, SCALAR_LAYOUT).lane_span() == 1
    assert val(7, SPREAD).lane_span() == 16
    # Not 1.  Those are the same number and opposite facts, and an emitter
    # that cannot tell them apart writes a scalar where a vector belongs.
    with pytest.raises(IRError):
        val(8).lane_span()


def test_distributed_predicate():
    assert val(9, SPREAD).distributed
    assert not val(10, SCALAR_LAYOUT).distributed
    assert not val(11).distributed


def test_multi_axis_span_is_the_product():
    lay = RegisterLayout((LaneAxis(4, 1), LaneAxis(4, 4)))
    assert val(12, lay).lane_span() == 16


# --------------------------------------------------------------------------
# 3. a scalar does not veto
# --------------------------------------------------------------------------

def test_scaling_keeps_the_layout_it_is_scaled_into():
    assert join_layout([val(13, SCALAR_LAYOUT), val(14, SPREAD)]) == SPREAD


def test_two_different_distributions_still_give_unknown():
    """Not an error -- a vendor intrinsic may consume two -- but nothing may
    be concluded from it either."""
    assert join_layout([val(15, SPREAD), val(16, SPREAD8)]) is None


def test_untracked_operands_are_ignored_not_fatal():
    assert join_layout([val(17), val(18, SPREAD)]) == SPREAD


def test_no_layout_anywhere_stays_unknown():
    assert join_layout([val(19), val(20)]) is None


def test_all_replicated_agrees():
    assert join_layout([val(21, SCALAR_LAYOUT), val(22, SCALAR_LAYOUT)]) == SCALAR_LAYOUT


# --------------------------------------------------------------------------
# layout_of: the entry point where a distribution enters the IR
# --------------------------------------------------------------------------

def test_thread_independent_index_is_replicated_not_unknown():
    assert layout_of([0, 3]) == SCALAR_LAYOUT
    assert layout_of([]) == SCALAR_LAYOUT


def test_a_lead_index_gives_its_axis():
    assert layout_of([LeadIndex(0, 16, 1)]) == RegisterLayout((LaneAxis(16, 1),))


def test_an_offset_lead_index_is_still_that_axis():
    """A slicing shift does not change *which lane holds what*.

    The shift used to be a `VarOffset` wrapped around the index and is now a
    field on it -- see `LeadIndex._offset`.  Either way the layout is the same
    axis, which is the property this test is about.
    """
    assert (layout_of([LeadIndex(0, 16, 1, offset=32)])
            == RegisterLayout((LaneAxis(16, 1),)))


def test_two_lead_axes_need_the_wave_size_to_be_checkable():
    idx = [LeadIndex(0, 4, 1), LeadIndex(0, 4, 4)]
    assert layout_of(idx, num_threads=None) is None
    assert layout_of(idx, num_threads=16) == RegisterLayout(
        (LaneAxis(4, 1), LaneAxis(4, 4)))


def test_axes_that_do_not_tile_the_wave_stay_unknown():
    """Unknown, not a guess: the axes describe partial replication this
    function has not established, and a wrong layout lets a pass merge two
    values that differ."""
    idx = [LeadIndex(0, 4, 1), LeadIndex(0, 4, 1)]
    assert layout_of(idx, num_threads=16) is None
