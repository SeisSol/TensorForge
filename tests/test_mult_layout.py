# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where the lanes of one multiplication sit, across waves (`MultLayout`).

The arrangement these tests hold is the one that makes every wave of a group
look alike: units of `gcd(threads, wave)` lanes, dealt out to the
multiplications of a group in turn.  A width that divides the wave keeps the
layout it always had, and the interesting widths are the ones that do not --
48 lanes on a 32-wide wave is three waves holding two multiplications, 16 lanes
of each per wave.
"""

from __future__ import annotations

import pytest

from tensorforge.common.threads import MultLayout, mults_per_group


@pytest.mark.parametrize('threads,wave', [(8, 32), (16, 32), (32, 32),
                                          (64, 64), (16, 64)])
def test_a_width_that_divides_the_wave_keeps_the_old_layout(threads, wave):
    layout = MultLayout(threads, wave)
    assert layout.contiguous
    assert layout.units_per_mult == 1
    assert layout.mults_per_group == wave // threads


def test_forty_eight_lanes_are_three_waves_holding_two_multiplications():
    layout = MultLayout(48, 32)
    assert (layout.unit, layout.units_per_mult) == (16, 3)
    assert (layout.waves_per_group, layout.mults_per_group) == (3, 2)
    assert not layout.contiguous


def test_the_waves_of_a_group_hold_the_same_shape():
    """Neighbouring units belong to different multiplications, so a wave holds
    a slice of each rather than one whole multiplication and part of another."""
    layout = MultLayout(48, 32)
    waves = [[layout.mult_of_unit(u) for u in range(w * layout.units_per_wave,
                                                    (w + 1) * layout.units_per_wave)]
             for w in range(layout.waves_per_group)]
    assert waves == [[0, 1], [0, 1], [0, 1]]


def test_every_lane_of_a_multiplication_is_covered_once():
    for threads, wave in ((12, 32), (28, 32), (48, 32), (64, 32), (48, 64)):
        layout = MultLayout(threads, wave)
        seen = {}
        for unit in range(layout.units_per_group):
            for lane in range(layout.unit):
                key = (layout.mult_of_unit(unit), layout.lane_of(unit, lane))
                assert key not in seen, f'{key} twice at {threads}/{wave}'
                seen[key] = True
        assert len(seen) == layout.mults_per_group * threads


def test_a_group_is_whole_waves_and_whole_multiplications():
    for threads, wave in ((12, 32), (28, 32), (48, 32), (96, 64)):
        layout = MultLayout(threads, wave)
        assert layout.group_threads % wave == 0
        assert layout.group_threads % threads == 0
        assert layout.mults_per_group == mults_per_group(threads, wave)


def test_a_multiplication_of_whole_waves_needs_no_interleaving():
    layout = MultLayout(64, 32)
    assert layout.whole_waves and not layout.contiguous
    assert layout.mults_per_group == 1
    assert [layout.mult_of_unit(u) for u in range(4)] == [0, 0, 1, 1]


def test_a_layout_needs_positive_widths():
    with pytest.raises(ValueError):
        MultLayout(0, 32)
