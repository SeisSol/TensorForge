# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The runtime's transposes, against what their names claim.

`transpose4x4b32` has a relayout row and a simulator model. The other
three -- `transpose16x16b32`, `transpose16x4`, `transpose16x2` -- are in
`hip.h` with nothing else checking them, because no tile reaches them:
`mfma_tile_for` returns the 4-wide tile and only the 4-wide tile.

With the two rotate controls of an 8x8 stage the wrong way round, lane 4 would
read lane 8 where it wants lane 0 and half of every tile would come out
holding another row's data -- and nothing would fail, because nothing calls
them.

Each is simulated from its own source rather than from the map its name
implies, which is the only way such a defect shows: a simulation written from
the intended map would only restate it.
"""

from __future__ import annotations

import pytest

from harness import wavesim

WAVE = 64


def _tagged(count):
    """`count` registers, each lane carrying where its value came from."""
    return [[(reg, lane) for lane in range(WAVE)] for reg in range(count)]


# --------------------------------------------------------------------------- #
# what each one does
# --------------------------------------------------------------------------- #

def test_transpose16x16b32_exchanges_the_register_with_the_low_lane_bits():
    """Sixteen registers against the low four lane bits, inside each row.

    Lane `l` of output register `r` holds what lane `(l & ~15) + r` of input
    register `l % 16` held. That is the register-to-lane exchange a
    single-block instruction needs for its A operand, and without this test it
    would be an assumption.
    """
    out = wavesim.transpose16x16b32(_tagged(16))
    for reg in range(16):
        for lane in range(WAVE):
            assert out[reg][lane] == (lane % 16, (lane & ~15) + reg), (
                f"register {reg}, lane {lane}")


def test_transpose16x4_exchanges_the_register_with_lane_bits_three_and_two():
    """Four registers against the *middle* two bits of the lane index.

    Not the low ones: `transpose4x4b32` takes those. This one moves a register
    against bits 3 and 2 inside each row of sixteen, which is why the two are
    not interchangeable and why the catalog's transpose table is keyed by
    tile width.
    """
    out = wavesim.transpose16x4(_tagged(4))
    for reg in range(4):
        for lane in range(WAVE):
            assert out[reg][lane] == ((lane % 16) // 4,
                                      (lane & ~15) + 4 * reg + lane % 4), (
                f"register {reg}, lane {lane}")


def test_transpose16x2_exchanges_the_register_with_lane_bit_three():
    out = wavesim.transpose16x2(_tagged(2))
    for reg in range(2):
        for lane in range(WAVE):
            assert out[reg][lane] == ((lane % 16) // 8,
                                      (lane & ~15) + 8 * reg + lane % 8), (
                f"register {reg}, lane {lane}")


# --------------------------------------------------------------------------- #
# the properties a transpose has to have
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,count", [
    ("transpose16x16b32", 16), ("transpose16x4", 4), ("transpose16x2", 2),
    ("transpose4x4b32", 4),
])
def test_each_transpose_is_a_bijection(name, count):
    """Every cell arrives exactly once.

    The property a swapped control breaks and the one worth checking
    separately from the map: a swapped control produces a result where half
    the cells hold a duplicate and half are missing, which shows up here
    without anyone having to know what the right map is.
    """
    out = getattr(wavesim, name)(_tagged(count))
    cells = {(reg, lane) for reg in range(count) for lane in range(WAVE)}
    assert {out[reg][lane] for reg in range(count) for lane in range(WAVE)} \
        == cells


@pytest.mark.parametrize("name,count", [
    ("transpose16x16b32", 16), ("transpose16x4", 4), ("transpose16x2", 2),
    ("transpose4x4b32", 4),
])
def test_each_transpose_is_its_own_inverse(name, count):
    """Applying it twice is the identity.

    True of every exchange of a register index against lane bits, and the
    cheapest end-to-end check there is: it needs no statement of the map at
    all.
    """
    once = getattr(wavesim, name)(_tagged(count))
    twice = getattr(wavesim, name)(once)
    assert twice == _tagged(count)


@pytest.mark.parametrize("name,count,width", [
    ("transpose16x16b32", 16, 16), ("transpose16x4", 4, 16),
    ("transpose16x2", 2, 16), ("transpose4x4b32", 4, 4),
])
def test_no_transpose_crosses_its_own_group(name, count, width):
    """A value never leaves the lane group the transpose works inside.

    `transpose4x4b32` stays within four lanes and the other three within
    sixteen, which is what makes them composable with `swap`: the transpose
    settles the register against the low lane bits and the swaps settle the
    high ones, and neither undoes the other.
    """
    out = getattr(wavesim, name)(_tagged(count))
    for reg in range(count):
        for lane in range(WAVE):
            assert out[reg][lane][1] // width == lane // width, (
                f"{name}: register {reg}, lane {lane} reached another group")
