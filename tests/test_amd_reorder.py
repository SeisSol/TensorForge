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

#: The instruction's B fragment, where the source is one scalar per lane --
#: what the leading operand hands over. The 16-bit ones need a split and a
#: pack before any of this applies, which is a different problem and not this
#: module's; the A fragment is a different problem too, see below.
SCALAR = [(op, "B")
          for op in catalog.MATRIX_OPS
          if layouts.covers(op, "B")
          and op.b.dtype in (catalog.Datatype.F32, catalog.Datatype.F64)]


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
        for lane in move.select.lanes:
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
        covered = set()
        for move in moves:
            assert not covered & move.select.lanes, f"{op.builtin}: overlap"
            covered |= move.select.lanes
        assert covered == set(range(op.wave)), (
            f"{op.builtin}: {len(covered)} of {op.wave} lanes covered")


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
    assert all(m.select.kind == "row" for m in moves)
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
    assert reorder.fragment_moves(op, "A", 0) is None, "not this source"
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


# --------------------------------------------------------------------------- #
# the select, across what the hardware offers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lanes,kind", [
    (range(16), "row"),
    (range(64), "row"),
    (list(range(16, 32)), "row"),
    (list(range(4)) + list(range(16, 20)), "bank"),
    (list(range(8)), "bank"),
    (list(range(5)), "cndmask"),
    ((0, 1), "cndmask"),
    (range(0, 64, 2), "cndmask"),
    (list(range(4)) + list(range(20, 24)), "cndmask"),
])
def test_the_select_picks_the_cheapest_mechanism(lanes, kind):
    """`row_mask`, then `bank_mask`, then a ternary.

    The last case is the one that separates a product from a set: four lanes
    in bank 0 of row 0 and four in bank 1 of row 1. Both rows are enabled and
    both banks are, so the masks would write eight *more* lanes than asked
    for. A mask expresses `rows x banks` and nothing else.
    """
    assert reorder.Select.of(lanes, 64).kind == kind


@pytest.mark.parametrize("lanes", [
    range(16), range(64), list(range(4)) + list(range(16, 20)), list(range(8)),
])
def test_a_mask_writes_exactly_its_region(lanes):
    """What `free` is claiming, checked rather than asserted.

    A select that reports `row` or `bank` says the merge needs no instruction
    of its own -- which is only true if the masks reproduce the region
    exactly. An over-wide mask would corrupt lanes outside it, silently,
    since they hold the other regions' results.
    """
    select = reorder.Select.of(lanes, 64)
    assert select.free
    written = {lane for lane in range(64)
               if select.row_mask >> (lane // reorder.ROW) & 1
               and select.bank_mask >> ((lane % reorder.ROW) // reorder.BANK) & 1}
    assert written == set(lanes)


def test_every_region_in_the_catalogue_is_still_a_row_mask():
    """Recorded, not relied on.

    Every plan the catalogue produces today selects whole rows, because a lane
    bit only carries the contraction once `n * blocks` has used the ones below
    and that is 16 or 32 everywhere. This test is what tells a later reader
    that the bank and `cndmask` paths above are exercised by construction
    rather than by any instruction -- and it is what changes when the
    accumulator writeback or a sub-wave thread count assigns the dimensions
    differently.
    """
    kinds = {move.select.kind
             for op, which in SCALAR
             for slot in range((op.a if which == "A" else op.b).per_lane)
             for move in reorder.fragment_moves(op, which, slot)}
    assert kinds == {"row"}


def test_a_cndmask_region_is_priced_and_a_masked_one_is_not():
    """The cost difference the emitter has to see.

    A mask rides on the merge; a ternary is an instruction, plus reading the
    lane id. `Move.cost` carries that, so a plan can be weighed against
    staging before anything is emitted rather than after.
    """
    masked = reorder.Move(0, (32,), reorder.Select.of(range(16), 64))
    ternary = reorder.Move(0, (32,), reorder.Select.of(range(5), 64))
    assert masked.cost == 2 and ternary.cost == 3
    assert masked.select.free and not ternary.select.free


# --------------------------------------------------------------------------- #
# the accumulator, coming back
# --------------------------------------------------------------------------- #

ACCUMULATORS = [op for op in catalog.MATRIX_OPS if layouts.covers(op, "D")]


def _writeback(op, column):
    """Execute the gather and return the resulting lane values.

    Each accumulator slot starts tagged with the D element it holds, so a
    lane of the result should end up holding the output for its own leading
    dimension element -- which is the property the nest's store assumes and
    never states.
    """
    gathers = reorder.accumulator_gathers(op, column)
    assert gathers is not None, f"{op.builtin} column {column}"

    result = [None] * op.wave
    for gather in gathers:
        lanes = [layouts.element_at(op, "D", gather.slot, lane)
                 for lane in range(op.wave)]
        for block in gather.swaps:
            lanes = wavesim.swap(lanes, block)
        for lane in gather.select.lanes:
            result[lane] = (gather.group, lanes[lane])
    return result


@pytest.mark.parametrize("op", ACCUMULATORS, ids=lambda op: op.builtin)
def test_the_writeback_lands_each_output_in_its_own_lane(op):
    """Every column, every lane.

    The instruction leaves `D[i][j]` spread over slots and lanes; the nest
    stores with the leading dimension across the lanes. This is the claim that
    the gather turns one into the other -- and it fails if the D layout is
    wrong, which no test that reads the layout table can.
    """
    span = op.n * op.blocks
    for column in range(op.m):
        got = _writeback(op, column)
        for lane in range(op.wave):
            group, local = divmod(lane, span)
            block, index = divmod(local, op.n)
            assert got[lane] == (group, (block, column, index)), (
                f"{op.builtin} column {column} lane {lane}")


@pytest.mark.parametrize("op", ACCUMULATORS, ids=lambda op: op.builtin)
def test_the_writeback_covers_every_lane_exactly_once(op):
    for column in range(op.m):
        covered = set()
        for gather in reorder.accumulator_gathers(op, column):
            assert not covered & gather.select.lanes, f"{op.builtin}: overlap"
            covered |= gather.select.lanes
        assert covered == set(range(op.wave))


def test_the_k1_tile_writes_back_without_moving_anything():
    """The path `matmul32` takes today, out of the general derivation.

    `mfma_f32_4x4x1f32` puts the output column in the accumulator's slot and
    the leading dimension in the lanes, which is already what the nest wants:
    one gather per column, no swaps, the whole wave. That is
    `C(writer, extract(acc, jj), i, j + jj)` written out, and the derivation
    reducing to it is the same check that the operand side passed.
    """
    op = next(o for o in catalog.MATRIX_OPS if o.builtin == "mfma_f32_4x4x1f32")
    for column in range(op.m):
        gathers = reorder.accumulator_gathers(op, column)
        assert len(gathers) == 1
        assert gathers[0].swaps == ()
        assert gathers[0].slot == column
        assert gathers[0].select.lanes == frozenset(range(64))
    assert reorder.accumulator_cost(op) == 4


def test_the_epilogue_is_affordable_where_the_fragment_is_not_free():
    """The two halves priced against each other.

    `mfma_f64_16x16x4f64` spends eight instructions on one B fragment and 128
    on the whole writeback -- but the fragment is rebuilt every contraction
    step and the writeback runs once per output tile. Against a contraction of
    length K that is 2*K fragment instructions and a fixed 128, so the
    epilogue stops mattering as soon as K is more than a handful.
    """
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f64_16x16x4f64")
    assert reorder.fragment_cost(op, "B") == 8
    assert reorder.accumulator_cost(op) == 128
    assert reorder.accumulator_cost(op) // op.m == 8


@pytest.mark.parametrize("op", ACCUMULATORS, ids=lambda op: op.builtin)
def test_every_writeback_region_is_a_row_mask_too(op):
    """Recorded like its counterpart on the operand side.

    Both directions come out as whole rows, so `dppUpdate` covers the epilogue
    as well and nothing here needs a lane id. This is the test that changes if
    a sub-wave thread count makes the groups narrower than a row.
    """
    for column in range(op.m):
        for gather in reorder.accumulator_gathers(op, column):
            assert gather.select.kind == "row", f"{op.builtin} column {column}"


# --------------------------------------------------------------------------- #
# which operand the plan is for, and why not the other one
# --------------------------------------------------------------------------- #

def test_the_plan_is_for_the_leading_operand_only():
    """`ops.A` and `ops.B` do not arrive the same way round.

    `multilinear` gives the leading operand `unwindI` for its lead index and a
    plain index for the contraction, so the contraction sits in registers and
    the lead in lanes. It gives the shared matrix the reverse -- `unwindK`
    with `full=False`, a `LeadIndex` on the contraction -- so the contraction
    is in the lanes and the column in registers.

    A `swap` moves a value between lanes. It never moves one between a
    register and a lane, so no sequence of them reaches a fragment from the
    transposed arrangement, and a plan derived as if it could would be
    self-consistent and wrong. The instruction's B fragment is the one the
    leading operand feeds; its A fragment is refused.
    """
    for op in catalog.MATRIX_OPS:
        if not layouts.covers(op, "A"):
            continue
        assert reorder.fragment_moves(op, "A", 0) is None
    assert reorder.FED_BY["A"].startswith("ops.B")
    assert reorder.FED_BY["B"].startswith("ops.A")


def test_the_broadcast_covers_the_a_fragment_wherever_there_are_blocks():
    """The refusal above is not a gap -- but the boundary is not `k == 1`.

    `cbsz` and `abid` push one block's A operand to the others, so an
    instruction with blocks to spare fetches its own A. That needs `blocks >
    1` and nothing else: `mfma_f64_4x4x4f64` has four blocks *and* a
    contraction of four, so it broadcasts its A and still wants the plan for
    its B. Only a single-block instruction is left needing a register-to-lane
    exchange.

    Worth pinning, because `blocks > 1` looks interchangeable with `k == 1`
    and is not. What `k == 1` is equivalent to is `n * blocks == wave`, which
    is `lane_batched` -- a stronger condition, and the one that decides
    whether the whole scheme applies rather than whether the broadcast does.
    """
    scalar = [op for op in catalog.MATRIX_OPS
              if op.a.per_lane == 1 and op.b.per_lane == 1
              and op.a.dtype in (catalog.Datatype.F32, catalog.Datatype.F64)]
    assert scalar
    for op in scalar:
        assert reorder.broadcast_feeds_a(op) == (op.blocks > 1), op.builtin
        if op.k == 1:
            assert reorder.broadcast_feeds_a(op), "every K=1 tile has blocks"

    both = next(o for o in scalar if o.builtin == "mfma_f64_4x4x4f64")
    assert both.k > 1 and reorder.broadcast_feeds_a(both)
    assert not both.lane_batched(), "blocks alone is not the whole scheme"


def test_the_instructions_this_work_targets_all_need_the_exchange():
    """Which is the next piece, and it is not a swap.

    Both FP64 MFMAs, gfx1250's native F32 WMMA and gfx1251's native F64 one
    have a single block, so none of them can broadcast its A operand and all
    four need the register-to-lane exchange. `_TILE_TRANSPOSES` in the
    catalogue already names which transpose belongs to which width; whether
    `hip.h` defines the one a 16-wide tile wants is what
    `DEFINED_TRANSPOSES` answers.
    """
    for name in ("mfma_f64_4x4x4f64", "mfma_f64_16x16x4f64",
                 "wmma_f32_16x16x4_f32", "wmma_f64_16x16x4_f64"):
        op = next(o for o in catalog.MATRIX_OPS if o.builtin == name)
        if op.blocks > 1:
            assert reorder.broadcast_feeds_a(op), name
        else:
            assert not reorder.broadcast_feeds_a(op), name
            assert reorder.fragment_moves(op, "A", 0) is None


# --------------------------------------------------------------------------- #
# the A fragment, and the contraction order that makes it free
# --------------------------------------------------------------------------- #

EXCHANGED = [op for op in catalog.MATRIX_OPS if reorder.a_exchange(op)]

TRANSPOSES = {"tensorforge::transpose16x16b32": wavesim.transpose16x16b32,
              "tensorforge::transpose4x4b32": wavesim.transpose4x4b32}


@pytest.mark.parametrize("op", EXCHANGED, ids=lambda op: op.builtin)
def test_the_transpose_output_is_the_a_fragment(op):
    """Register `g` of the transpose *is* the fragment for k-group `g`.

    The shared matrix arrives as column-in-registers, contraction-in-lanes.
    One `transpose16x16b32` swaps those, and what comes out has the column the
    fragment wants and a contraction running in steps of sixteen down the lane
    rows -- which is the fragment exactly, once the sum is walked in stride-16
    groups instead of contiguous fours.

    Nothing moves after the transpose, and the transpose emits all sixteen
    registers at once, so it is paid once per k-block of 64 rather than once
    per issue.
    """
    exchange = reorder.a_exchange(op)
    source = [[("sh", col, k) for k in range(op.wave)] for col in range(op.m)]
    out = TRANSPOSES[exchange.transpose](source)

    for group in range(exchange.groups):
        for lane in range(op.wave):
            _, column, k_local = layouts.element_at(op, "A", 0, lane)
            assert out[group][lane] == (
                "sh", column, exchange.stride * k_local + group), (
                f"{op.builtin} group {group} lane {lane}")


@pytest.mark.parametrize("op", EXCHANGED, ids=lambda op: op.builtin)
def test_both_operands_name_the_same_contraction(op):
    """The joint run, which is the claim that matters.

    Each fragment being individually well-formed says nothing about the
    product: the instruction sums `A[m][k] * B[k][n]` over its own `k`, so the
    two fragments have to agree about which real contraction value each of its
    `k` stands for. Here the A side gets that from the transpose and the B
    side from taking its source register at a stride, and they have to come
    out the same.

    And over the whole k-block, every contraction value has to appear exactly
    once -- a relabelling that dropped or repeated one would still pass every
    per-fragment check above.
    """
    exchange = reorder.a_exchange(op)
    shared = [[("sh", col, k) for k in range(op.wave)] for col in range(op.m)]
    data = [[("da", lead, k) for lead in range(op.wave)]
            for k in range(op.wave)]
    a_regs = TRANSPOSES[exchange.transpose](shared)

    span = op.n * op.blocks
    covered = set()
    for group in range(exchange.groups):
        for lead_group in range(op.wave // span):
            fragment = [None] * op.wave
            for move in reorder.fragment_moves(op, "B", 0, lead_group):
                lanes = data[exchange.stride * move.contraction + group]
                for block in move.swaps:
                    lanes = wavesim.swap(lanes, block)
                for lane in move.select.lanes:
                    fragment[lane] = lanes[lane]

            for lane in range(op.wave):
                _, column, k_a = layouts.element_at(op, "A", 0, lane)
                _, k_b, index = layouts.element_at(op, "B", 0, lane)
                assert k_a == k_b, "the layouts disagree about this lane's k"
                left, right = a_regs[group][lane], fragment[lane]
                assert left[1] == column
                assert right[1] == lead_group * span + index
                assert left[2] == right[2], (
                    f"{op.builtin}: A holds k={left[2]}, B holds k={right[2]}")
                covered.add(left[2])

    assert covered == set(range(op.wave)), (
        "the k-groups do not cover the contraction exactly once")


def test_the_exchange_declines_where_the_contraction_reaches_the_register():
    """Two slots is a further claim, and it is not made here.

    With more than one element per lane the fragment keeps part of its
    contraction in the register, so the relabelling has to stay injective
    across slots as well as lane rows. That may well hold -- both XF32 entries
    and both native gfx125x WMMAs are in that shape -- but it is not what the
    simulation above checked, so `a_exchange` says nothing about them.
    """
    for name in ("mfma_f32_16x16x8_xf32", "wmma_f64_16x16x4_f64"):
        op = next(o for o in catalog.MATRIX_OPS if o.builtin == name)
        assert op.a.per_lane > 1
        assert reorder.a_exchange(op) is None

    wide = next(o for o in catalog.MATRIX_OPS
                if o.builtin == "mfma_f32_32x32x2f32")
    assert reorder.a_exchange(wide) is None, "no transpose32x32b32 in hip.h"

    tiled = next(o for o in catalog.MATRIX_OPS
                 if o.builtin == "mfma_f32_4x4x1f32")
    assert reorder.a_exchange(tiled) is None, "it broadcasts its own A"


# --------------------------------------------------------------------------- #
# what the reordered sum costs elsewhere
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("op", EXCHANGED, ids=lambda op: op.builtin)
def test_an_issue_never_covers_a_contiguous_run(op):
    """The one thing the reordering does disturb, pinned.

    An issue sums over `{g, g+stride, ...}` and the stride is the transpose
    width, because that is where the lane rows put `k`. So an issue's
    contraction set is never a run -- and a 16x16x4 instruction can skip an
    issue whose four values are all zero, which under a contiguous walk is
    exactly a 4-deep zero block and under this one is nothing.

    Sparse operands do not reach the matrix path today, so this collides with
    a plan rather than with code. Pinned because it is the reason the two
    cannot both come from one walk, and because whichever is chosen later
    should have to change this test to do it.
    """
    exchange = reorder.a_exchange(op)
    assert not exchange.contiguous
    for group in range(exchange.groups):
        covered = exchange.covers(group, op.k)
        assert len(set(covered)) == op.k
        assert covered != tuple(range(covered[0], covered[0] + op.k))


@pytest.mark.parametrize("op", EXCHANGED, ids=lambda op: op.builtin)
def test_the_groups_partition_the_contraction(op):
    """Every value once, across the groups -- the property the sum needs.

    Reordering a sum is only free if it is still the same sum. The joint
    simulation checks this for one k-block; this checks the arithmetic of it
    directly, which is what would catch a stride that shares a factor with
    the group count.
    """
    exchange = reorder.a_exchange(op)
    depth = exchange.stride * op.k
    covered = [k for group in range(exchange.groups)
               for k in exchange.covers(group, op.k)]
    assert sorted(covered) == list(range(depth))


@pytest.mark.parametrize("op", EXCHANGED, ids=lambda op: op.builtin)
def test_the_accumulator_does_not_care_about_the_order(op):
    """Why the writeback plan survives the reordering untouched.

    `D[i][j]` has no `k` in it -- the accumulator is where the contraction has
    already happened -- so `accumulator_gathers` is the same plan whichever
    order the sum ran in. Worth asserting rather than reasoning about, since
    it is the half of the emitter that would be expensive to rediscover.
    """
    for column in range(op.m):
        assert reorder.accumulator_gathers(op, column) is not None
    terms = layouts.index_terms(op, "D", "first") + \
        layouts.index_terms(op, "D", "second")
    assert all(term.source in ("lane", "slot") for term in terms)


def test_the_exchange_needs_the_whole_shared_tile_live():
    """The pressure it trades the movement for.

    The transpose consumes sixteen registers and produces sixteen, and it is
    in place -- `_TILE_TRANSPOSES` records that it has no separate outputs --
    so it is sixteen registers live and not thirty-two. For
    `mfma_f64_16x16x4f64` that is thirty-two VGPRs of shared matrix held
    across a whole k-block, where the DPP chain streams it.

    Not an objection: the contiguous alternative needs the same sixteen
    columns live to gather from. It is a step change against the path that
    exists, and order 6 in double is already where the register budget bites.
    """
    _, separate = catalog._TILE_TRANSPOSES[16]
    assert not separate, "in place, so the inputs are the outputs"
