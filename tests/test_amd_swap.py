# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`swap<Block>`, and the register reordering built out of it.

`swap` is the butterfly step: one bit of the lane index toggled. It is the
primitive a fragment reordering is made of, because a sequence of them is a
permutation of the lane index bits and a matrix fragment differs from the
generator's distribution by exactly that.

It carried two different maps until now. Four branches toggled one bit and two
read the mirror lane, `i ^ (Block - 1)`, which is what the comment in
`reduction` described as well. Nothing distinguished them: the sole caller is a
butterfly reduction over groups that are already uniform, so any lane of the
neighbouring group answers and both maps reduce correctly. An exact
permutation is not so forgiving, which is why the map is now stated and
checked.
"""

from __future__ import annotations

import pytest

from harness import wavesim

from tensorforge.backend.instructions.compute.primitives.amd import (
    catalog, layouts)

BLOCKS = (1, 2, 4, 8, 16, 32, 64)


# --------------------------------------------------------------------------- #
# the map
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("block", BLOCKS)
def test_swap_toggles_exactly_one_lane_bit(block):
    """`swap<Block>` reads lane `i ^ (Block / 2)`, for every Block.

    Simulated from each branch's own source, so this checks the branches
    against the documented map rather than restating one of them.
    """
    lanes = wavesim.swap(list(range(64)), block)
    assert lanes == [i ^ (block // 2) for i in range(64)]


@pytest.mark.parametrize("block", BLOCKS)
def test_swap_is_its_own_inverse(block):
    """Toggling a bit twice is the identity.

    True of a mirror as well, so it does not separate the two maps -- it
    catches the other way this can break, a branch that shifts or rotates
    instead of toggling.
    """
    once = wavesim.swap(list(range(64)), block)
    assert wavesim.swap(once, block) == list(range(64))


def test_the_two_branches_that_disagreed_now_agree():
    """`swap<8>` and `swap<32>` were `i ^ (Block - 1)`.

    Pinned by value rather than by the general property above, because this
    is the change: a mirror and a one-bit toggle coincide at Block 1 and 2 and
    part company at 4. Reading `i ^ 7` where `i ^ 4` was meant lands three
    lanes away and reduces to the same answer.
    """
    assert wavesim.swap(list(range(64)), 8)[0] == 4
    assert wavesim.swap(list(range(64)), 32)[0] == 16

    mirror = [i ^ 7 for i in range(64)]
    assert wavesim.swap(list(range(64)), 8) != mirror


@pytest.mark.parametrize("subblock", (1, 2, 4, 8, 16, 32))
def test_the_reduction_butterfly_still_pairs_neighbours(subblock):
    """The one caller, under the new map.

    `reduction` calls `swap<2 * Subblock>` once each group of `Subblock` lanes
    holds a uniform value, and needs the result to come from the neighbouring
    group. Both maps satisfy that, which is why this cannot be the test that
    pins `swap` -- but it is the test that says the change is safe.
    """
    lanes = wavesim.swap(list(range(64)), subblock * 2)
    for lane, source in enumerate(lanes):
        assert lane // subblock != source // subblock, "left its own group"
        assert lane // (2 * subblock) == source // (2 * subblock), (
            "and stayed inside the pair")


# --------------------------------------------------------------------------- #
# what it is for
# --------------------------------------------------------------------------- #

def _compose(sequence):
    """Lane map of a sequence of swaps applied in order."""
    lanes = list(range(64))
    for block in sequence:
        lanes = wavesim.swap(lanes, block)
    return lanes


def test_swaps_compose_into_a_bit_toggle_mask():
    """A sequence of swaps toggles the union of their bits.

    Which is the property that makes them a building block rather than seven
    unrelated shuffles: the reachable maps are exactly `i ^ mask`, and the
    cost is one instruction per set bit.
    """
    for sequence in ((4, 16), (2, 8, 32), (16, 64), (2, 4, 8, 16, 32, 64)):
        mask = 0
        for block in sequence:
            mask ^= block // 2
        assert _compose(sequence) == [i ^ mask for i in range(64)]


def test_the_fp64_fragment_is_two_swaps_from_the_generator_layout():
    """What this unlocks, priced.

    `mfma_f64_16x16x4f64` wants B at `n = lane & 15`, `k = (lane >> 4) & 3`.
    The generator hands the data operand the other way round: the leading
    dimension across all 64 lanes and the contraction in registers. Moving a
    register into lane bits 4 and 5 is a toggle of those two bits, so the
    exchange is `swap<32>` and `swap<64>` -- two instructions per register,
    no shared memory.

    Under the old map neither would have been a single-bit toggle, and
    `swap<32>` would have scrambled bits 0 through 4 as well.
    """
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f64_16x16x4f64")
    k_terms = layouts.index_terms(op, "B", "first")
    assert k_terms == (layouts.Term("lane", 4, 3, 1),)

    assert _compose((32,)) == [i ^ 16 for i in range(64)]
    assert _compose((64,)) == [i ^ 32 for i in range(64)]
    assert _compose((32, 64)) == [i ^ 48 for i in range(64)]


def test_the_bf16_fragment_needs_no_swap_at_all():
    """And where the answer is that nothing has to move.

    `mfma_f32_4x4x4bf16_1k` takes its contraction from the slot, so a lane's
    four BF16 are `k = 0..3` of one row and the lane index carries only the
    row and the block. Nothing in the operand crosses a lane.
    """
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f32_4x4x4bf16_1k")
    terms = layouts.index_terms(op, "A", "second")
    assert [t.source for t in terms] == ["slot"]
