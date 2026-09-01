# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The register reordering, run through the wave simulator.

`reorder` says which swaps and which masked merges turn the loop nest's
registers into a matrix fragment. That is a claim about three things at once
-- the layout table, `swap`'s lane map, and the plan -- and the way to check
it is to execute it: tag every source slot, run the plan, and read off whether
each lane of the result holds what `layouts` says that lane should hold.

Which makes this the first test in the AMD package that checks the layouts
against something other than their own source. A wrong row in `FRAGMENT_BITS`
survives `test_amd_layouts.py`, because that file checks the table against the
vendored extract of the same table. It does not survive here, because the plan
derived from it would move data to the wrong lanes.
"""

from __future__ import annotations

import pytest

from harness import wavesim

from tensorforge.backend.instructions.compute.primitives.amd import (
    catalog, layouts, reorder)

#: Operands whose source is one scalar per lane -- what the loop nest holds.
#: The 16-bit ones need a split and a pack before any of this applies, which
#: is a different problem and not this module's.
SCALAR = [(op, which)
          for op in catalog.MATRIX_OPS for which in ("A", "B")
          if layouts.covers(op, which)
          and (op.a if which == "A" else op.b).dtype in (
              catalog.Datatype.F32, catalog.Datatype.F64)]


def _run(op, which, slot, group):
    """Execute the plan and return the resulting lane values.

    Source register `k` starts as `data[lead = lane][k]`, tagged as the pair
    `(k, lane)`. Lanes not written by any region keep `None`, which is how a
    plan that leaves a hole shows up as a hole rather than as stale data.
    """
    moves = reorder.fragment_moves(op, which, slot, group)
    assert moves is not None, f"{op.builtin} {which} slot {slot}"

    result = [None] * op.wave
    for move in moves:
        lanes = [(move.contraction, lane) for lane in range(op.wave)]
        for block in move.swaps:
            lanes = wavesim.swap(lanes, block)
        for lane in range(op.wave):
            if move.row_mask >> (lane // reorder.ROW) & 1:
                result[lane] = lanes[lane]
    return result


@pytest.mark.parametrize("op,which", SCALAR,
                         ids=lambda x: x.builtin if hasattr(x, "builtin") else x)
def test_the_plan_puts_every_element_where_the_layout_wants_it(op, which):
    """End to end, every slot, every group, every lane.

    The source register is the contraction and the source lane is the leading
    dimension; the fragment wants a particular element of that in a particular
    lane. Executing the plan and comparing against `element_at` is the only
    check here that could fail if the layout table itself were wrong.
    """
    frag = op.a if which == "A" else op.b
    extent = op.m if which == "A" else op.n
    span = extent * op.blocks

    for group in range(op.wave // span):
        for slot in range(frag.per_lane):
            got = _run(op, which, slot, group)
            for lane in range(op.wave):
                block, first, second = layouts.element_at(op, which, slot, lane)
                contraction, index = ((second, first) if which == "A"
                                      else (first, second))
                source = group * span + block * extent + index
                assert got[lane] == (contraction, source), (
                    f"{op.builtin} {which} slot {slot} group {group} "
                    f"lane {lane}")


@pytest.mark.parametrize("op,which", SCALAR,
                         ids=lambda x: x.builtin if hasattr(x, "builtin") else x)
def test_the_plan_covers_every_lane_exactly_once(op, which):
    """No hole and no lane written twice.

    A merge writes a whole row, so overlapping regions would be a plan whose
    result depends on the order of the moves. Disjointness is what makes the
    order free.
    """
    frag = op.a if which == "A" else op.b
    for slot in range(frag.per_lane):
        moves = reorder.fragment_moves(op, which, slot)
        assert moves is not None
        covered = 0
        for move in moves:
            assert not covered & move.row_mask, f"{op.builtin}: rows overlap"
            covered |= move.row_mask
        assert covered == (1 << (op.wave // reorder.ROW)) - 1, (
            f"{op.builtin}: rows {covered:b} do not cover the wave")


# --------------------------------------------------------------------------- #
# what the plan costs, and where it costs nothing
# --------------------------------------------------------------------------- #

def test_the_fp64_fragment_costs_eight_instructions():
    """The instruction this work has been heading for, priced.

    Four regions, so four merges; the swaps are what each region's XOR needs,
    which over the four is 0, 1, 1 and 2. Eight instructions per k-block for a
    fragment that every output column then reads -- against an MFMA that is 32
    cycles on its own.
    """
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f64_16x16x4f64")
    moves = reorder.fragment_moves(op, "B", 0)
    assert len(moves) == 4
    assert [len(m.swaps) for m in moves] == [0, 1, 1, 2]
    assert [m.row_mask for m in moves] == [0b0001, 0b0010, 0b0100, 0b1000]
    assert reorder.fragment_cost(op, "B") == 8

    # And the swaps are the two that reach lane bits 4 and 5.
    assert set(sum((m.swaps for m in moves), ())) == {32, 64}


def test_a_k1_tile_needs_no_movement_at_all():
    """Where the answer is that nothing moves.

    With `k == 1` the contraction never reaches a lane, so there is one region
    covering the wave with an empty XOR: a single merge, and even that is only
    there because the plan is uniform. This is the path `matmul32` already
    takes, and it is reassuring that the general derivation reduces to it.
    """
    op = next(o for o in catalog.MATRIX_OPS if o.builtin == "mfma_f32_4x4x1f32")
    moves = reorder.fragment_moves(op, "B", 0)
    assert len(moves) == 1
    assert moves[0].swaps == ()
    assert moves[0].row_mask == 0b1111


def test_the_plan_declines_rather_than_approximating():
    """`None` where the structure does not hold, and a caller that sees it
    stays on the generic nest.

    Asked for a slot the fragment does not have, for the accumulator, and for
    a group past the end of the wave -- three ways to be outside the plan, all
    of which would otherwise produce a confidently wrong answer.
    """
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f64_16x16x4f64")
    assert reorder.fragment_moves(op, "B", 1) is None, "one slot only"
    assert reorder.fragment_moves(op, "D", 0) is None, "not an input"
    assert reorder.fragment_moves(op, "B", 0, group=99) is None


def test_every_operand_in_the_catalogue_has_a_layout_to_plan_from():
    """The fourth decline case does not exist, and that is worth saying.

    `fragment_moves` returns `None` for an operand without a layout, and no
    operand is without one: A and B are covered for all forty entries, by
    measurement for thirty-two and by the granule rule for the rest. Only the
    accumulators have gaps, and those are refused a step earlier.

    So a reader looking for the uncovered case will not find one, and this is
    where that is recorded rather than a missing branch in the test above.
    """
    assert all(layouts.covers(op, which)
               for op in catalog.MATRIX_OPS for which in "AB")
    missing = {op.builtin for op in catalog.MATRIX_OPS
               if not layouts.covers(op, "D")}
    assert missing == {"wmma_f32_16x16x32_bf16", "wmma_f32_16x16x32_f16",
                       "wmma_f32_16x16x4_f32", "wmma_f64_16x16x4_f64"}


@pytest.mark.parametrize("op,which", SCALAR,
                         ids=lambda x: x.builtin if hasattr(x, "builtin") else x)
def test_a_region_never_needs_more_than_two_swaps(op, which):
    """The cost bound, and why it holds.

    A region's XOR only touches the lane bits that carry the contraction, and
    those are at most two: `n * blocks` is 16 or 32, so the contraction has
    two lane bits above it at most. Which is what keeps a fragment in single
    digits of instructions rather than in the tens.
    """
    frag = op.a if which == "A" else op.b
    for slot in range(frag.per_lane):
        for move in reorder.fragment_moves(op, which, slot):
            assert len(move.swaps) <= 2, f"{op.builtin} {which}"
            assert all(block in (32, 64) or block >= 32 for block in move.swaps), (
                f"{op.builtin}: a swap below the row width would need a "
                f"finer select than row_mask")
