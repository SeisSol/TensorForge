# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where a wide transfer is allowed to go, and how far.

A wide transfer can be wrong twice over, and both faults are invisible until a
tensor happens not to be padded:

* a width not asked of the *bases*. A `float4` read of `&buf[i]` is undefined
  unless that address is 16-byte aligned, which only an operand's promise
  establishes;
* hop arithmetic that overruns and overlaps. `range(start, total, granularity)`
  emits a hop per *started* step, and a `start = (total // granularity) *
  granularity` recomputed from the granularity just finished with goes back
  over covered ground. At `total=96`, 16 threads, width 4 that reads to
  element 128 and then covers `[64, 96)` a second time at width 2.

A corpus kernel exercises the plan only at the sizes and widths its case
reaches, and at width 1 neither fault shows: the overrun is one partial hop
and the double-cover cannot arise, because there is no second width. These
are model tests over the planner instead: exhaustive on the sizes that
actually occur, and stating the properties rather than the output.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.memory.vectorize import (
    MAX_ACCESS_BYTES, plan_hops, widths_for)

# The shapes SeisSol generates, plus the awkward ones from the corpus census.
TOTALS = [1, 9, 16, 20, 35, 46, 56, 64, 81, 96, 156, 169, 504]
THREADS = [1, 2, 16, 32, 64]


# --------------------------------------------------------------------------- #
# Which widths a base permits
# --------------------------------------------------------------------------- #

def test_an_unproven_base_permits_nothing_but_scalar():
    """"Not known to be aligned" and "aligned to one element" are one answer.

    The same asymmetry `lane_span` makes by raising instead of returning 1.
    A cast that needs 16 bytes must not be able to acquire the permission by
    defaulting.
    """
    assert widths_for(4, 0) == [1]
    assert widths_for(4, 4) == [1]


def test_alignment_buys_exactly_the_widths_it_covers():
    assert widths_for(4, 8) == [2, 1]
    assert widths_for(4, 16) == [4, 2, 1]
    assert widths_for(4, 32) == [4, 2, 1]      # capped by the access, not the base


def test_no_target_loads_more_than_sixteen_bytes():
    """`double4` is not a width: `LDG.128` and `ds_read_b128` stop at 16."""
    assert widths_for(8, 64) == [2, 1]         # double2 is the widest fp64 access
    assert 4 * 8 > MAX_ACCESS_BYTES


def test_a_wider_element_reaches_fewer_widths_from_the_same_base():
    assert widths_for(8, 16) == [2, 1]
    assert widths_for(4, 16) == [4, 2, 1]


# --------------------------------------------------------------------------- #
# The plan: the three properties the hop arithmetic has to have
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('total', TOTALS)
@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('widths', [[1], [2, 1], [4, 2, 1]])
def test_no_hop_runs_past_the_end(total, threads, widths):
    hops, tail = plan_hops(total, threads, widths)
    for off, w in hops:
        assert off + threads * w <= total


@pytest.mark.parametrize('total', TOTALS)
@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('widths', [[1], [2, 1], [4, 2, 1]])
def test_every_element_is_covered_at_most_once(total, threads, widths):
    hops, tail = plan_hops(total, threads, widths)
    seen = set()
    for off, w in hops:
        span = range(off, off + threads * w)
        assert not (seen & set(span)), 'hops overlap'
        seen |= set(span)
    assert len(seen) == total - tail


@pytest.mark.parametrize('total', TOTALS)
@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('widths', [[1], [2, 1], [4, 2, 1]])
def test_every_hop_offset_is_a_multiple_of_its_width(total, threads, widths):
    """What makes the reinterpret cast at that offset legal, given a base.

    Not automatic: it holds because each width advances by `threads * width`
    from a position that earlier, wider steps also left a multiple of.
    """
    for off, w in plan_hops(total, threads, widths)[0]:
        assert off % w == 0


@pytest.mark.parametrize('total', TOTALS)
@pytest.mark.parametrize('threads', THREADS)
def test_the_tail_is_smaller_than_one_scalar_hop(total, threads):
    _, tail = plan_hops(total, threads, [4, 2, 1])
    assert 0 <= tail < threads


def test_the_regression_the_old_arithmetic_had():
    """96 elements, 16 threads: the case per-step arithmetic both overruns and
    double-covers.

    A hop per started step at width 4 puts hops at 0 and 64 -- the second
    reading to 128, 32 elements past the end -- and a `start` recomputed from
    that width comes back to 64, so width 2 covers `[64, 96)` again.
    """
    hops, tail = plan_hops(96, 16, [4, 2, 1])
    assert hops == [(0, 4), (64, 2)]
    assert tail == 0


def test_a_total_smaller_than_one_hop_is_all_tail():
    """`total=9` over 32 threads: no whole hop exists at any width."""
    hops, tail = plan_hops(9, 32, [4, 2, 1])
    assert hops == []
    assert tail == 9


# --------------------------------------------------------------------------- #
# Degenerate inputs are refused, not silently misplanned
# --------------------------------------------------------------------------- #

def test_zero_threads_is_refused():
    with pytest.raises(ValueError):
        plan_hops(64, 0, [1])


def test_zero_width_is_refused():
    with pytest.raises(ValueError):
        plan_hops(64, 16, [0])
