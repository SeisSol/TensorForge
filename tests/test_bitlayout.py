# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The bit vocabulary against the two languages it has to subsume.

The whole value of one vocabulary is that both sides can be read in it, so
these check exactly that and nothing else: every fragment element the AMD
tables place lands where `bitlayout` says, and every `LaneAxis` that
decomposes into bits at all decomposes into the ones that reproduce its own
`holders`.

The first is a check on the translation and not on the tables -- both read the
same rows, so a wrong row stays wrong.  That is what `Provenance.MEASURED` and
the LLVM cross-check are for; this says the new reading is the old one.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute import bitlayout
from tensorforge.backend.instructions.compute.bitlayout import (
    Bit, BitLayout, Place, Position)
from tensorforge.backend.instructions.compute.primitives.amd import (
    MATRIX_OPS, layouts)
from tensorforge.backend.pir.core import LaneAxis, RegisterLayout

FRAGMENTS = ('A', 'B', 'D')


def _extents(op, which):
    return {'A': (op.m, op.k), 'B': (op.k, op.n), 'D': (op.m, op.n)}[which]


def _rows(op, which):
    row = layouts._row(op)
    if row is None:
        return None
    block, one, two = row[{'A': 0, 'B': 3, 'D': 6}[which]:][:3]
    return None if block is None else (block, one, two)


# -- the fragment side ----------------------------------------------------- #

@pytest.mark.parametrize('name', sorted(op.builtin for op in MATRIX_OPS))
def test_every_fragment_element_lands_where_the_table_puts_it(name):
    """Element for element, for every entry, both operands and the
    accumulator.  A translation that is right for the common rows and wrong
    for one interleaved one would pass a spot check."""
    op = next(o for o in MATRIX_OPS if o.builtin == name)
    checked = 0
    for which in FRAGMENTS:
        rows = _rows(op, which)
        if rows is None:
            continue
        layout = bitlayout.from_weights(*rows)
        first, second = _extents(op, which)
        for block in range(op.blocks):
            for a in range(first):
                for b in range(second):
                    want = layouts.position(op, which, a, b, block=block)
                    got = layout.locate(block, a, b)
                    assert (got.slot, got.lane) == want, (which, block, a, b)
                    assert got.element == 0, 'no table bit is a vector bit'
                    checked += 1
    assert checked or layouts._row(op) is None


def test_the_tables_reach_the_interleaved_rows():
    """The check is only worth anything if it covers a row where an index's
    bits are split between lane and slot, which is what `LaneAxis` cannot
    say."""
    split = []
    for op in MATRIX_OPS:
        for which in FRAGMENTS:
            rows = _rows(op, which)
            if rows is None:
                continue
            for bits in rows[1:]:
                places = {Place.LANE if w > 0 else Place.SLOT for w in bits}
                if len(places) > 1:
                    split.append((op.builtin, which))
    assert split, 'no interleaved index in the catalogue'


def test_a_fragment_bit_is_never_a_vector_bit():
    """A matrix fragment is register-resident; the third destination only ever
    arrives from the value side."""
    for op in MATRIX_OPS:
        for which in FRAGMENTS:
            rows = _rows(op, which)
            if rows is None:
                continue
            for bits in bitlayout.from_weights(*rows).axes:
                assert all(b.place is not Place.VECTOR for b in bits)


# -- the value side -------------------------------------------------------- #

@pytest.mark.parametrize('block', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('stride', [1, 2, 4])
def test_a_lane_axis_decomposes_into_the_bits_it_describes(block, stride):
    """`LaneAxis` is one cut point -- the low bits to the lanes, the rest to
    the slots -- and that is a special case of what a fragment row says.  The
    check is against its own `holders`, so the two readings of one axis have
    to agree about which thread holds an element."""
    threads = 64
    if block * stride > threads:
        pytest.skip('the axis does not fit the wave')
    extent = block * 4
    axis = LaneAxis(block=block, stride=stride)
    bits = bitlayout.from_lane_axis(block, stride, extent)
    assert bits is not None
    layout = BitLayout((bits,))
    for element in range(extent):
        where = layout.locate(element)
        holders = RegisterLayout((axis,)).holders((element,), threads)
        assert where.lane in holders, (element, where, holders)
        assert where.slot == element // block


def test_an_axis_that_does_not_wrap_at_a_power_of_two_has_no_bits():
    """Not a gap to fill: an axis wrapping at nine does not decompose at all,
    and moving between two such distributions is a shuffle by lane index
    rather than a permutation of bits."""
    assert bitlayout.from_lane_axis(9, 1, 36) is None
    assert bitlayout.from_lane_axis(4, 3, 16) is None


def test_the_undistributed_axis_is_all_slot_bits():
    """`block == 1` is the degenerate case: every lane holds the whole
    extent, so nothing reaches the lanes."""
    bits = bitlayout.from_lane_axis(1, 1, 8)
    assert bits is not None
    assert all(b.place is Place.SLOT for b in bits)


# -- the vocabulary itself ------------------------------------------------- #

def test_a_vector_bit_is_neither_a_slot_nor_a_lane():
    """What a packed operand needs and what makes `lead_width > 1` a starting
    layout rather than an exclusion: a `float4` is one register holding four
    elements, so a move between two of them is neither a shuffle nor a
    register rename."""
    packed = BitLayout(((Bit(Place.VECTOR, 1), Bit(Place.VECTOR, 2),
                         Bit(Place.LANE, 1)),))
    assert packed.locate(0) == Position()
    assert packed.locate(1) == Position(element=1)
    assert packed.locate(3) == Position(element=3)
    assert packed.locate(4) == Position(lane=1)
    assert packed.locate(5) == Position(lane=1, element=1)


def test_an_index_with_no_bits_contributes_nothing():
    """Which is how the single-block entries state that they do not spread
    the block index."""
    assert bitlayout.from_weights((), (1, 2), ()).locate(3, 3, 7) == Position(
        lane=3)


def test_the_index_count_has_to_match():
    with pytest.raises(ValueError):
        bitlayout.from_weights((1,), (2,)).locate(1)


# -- the first `have == want` ---------------------------------------------- #

def _lane_batched():
    return [op for op in MATRIX_OPS if op.lane_batched()]


@pytest.mark.parametrize('name', sorted(op.builtin for op in _lane_batched()))
def test_the_transpose_output_is_the_a_fragment(name):
    """What `matmul32` rests on, and what nothing could state before.

    `_check_mfma_operand` compares the operand it was handed against what the
    transpose produces -- it never asks whether that is what the instruction
    wants, because the two were written in languages that do not compare.  In
    one vocabulary they do, and they agree: the transpose puts one dimension
    on the low lane bits and the other above it, which is where the fragment
    puts `m` and the block.

    The correspondence is stated here rather than assumed, because there is
    none that is right for both operands: A's first index is the output
    dimension and B's is the contraction.
    """
    from tensorforge.backend.instructions.compute.primitives.amd.relayout \
        import TRANSPOSE4X4
    op = next(o for o in MATRIX_OPS if o.builtin == name)
    threads = op.wave
    fragment = layouts.fragment_layout(op, 'A')
    produced = bitlayout.from_register_layout(
        TRANSPOSE4X4.produces(threads=threads), (4, max(threads // 4, 1)))
    if fragment is None or produced is None or op.m != 4:
        pytest.skip('the 4-wide transpose does not feed this entry')
    for block in range(op.blocks):
        for m in range(op.m):
            assert fragment.locate(block, m, 0) == produced.locate(m, block)


@pytest.mark.parametrize('name', sorted(op.builtin for op in _lane_batched()))
def test_the_leading_operand_is_the_b_fragment(name):
    """The nest hands the leading operand over spread flat across the wave,
    one element per lane.  The fragment states the same lanes as an `n` inside
    a block, which looks like a different distribution and is the same bit
    map -- which is the work the vocabulary is for."""
    op = next(o for o in MATRIX_OPS if o.builtin == name)
    fragment = layouts.fragment_layout(op, 'B')
    flat = bitlayout.from_register_layout(
        RegisterLayout((LaneAxis(op.wave, 1),)), (op.wave,))
    assert fragment is not None and flat is not None
    for block in range(op.blocks):
        for n in range(op.n):
            assert fragment.locate(block, 0, n) == flat.locate(
                block * op.n + n)


def test_a_disagreement_would_be_seen():
    """The checks above are only worth something if the comparison can fail.
    A block bit moved one place over is a wrong kernel and no snapshot would
    notice, since both treat the intrinsic as opaque."""
    op = next(o for o in MATRIX_OPS if o.builtin == 'mfma_f32_4x4x1f32')
    rows = _rows(op, 'A')
    shifted = bitlayout.from_weights(
        tuple(w * 2 for w in rows[0]), rows[1], rows[2])
    good = layouts.fragment_layout(op, 'A')
    assert any(shifted.locate(b, m, 0) != good.locate(b, m, 0)
               for b in range(op.blocks) for m in range(op.m))


# -- the first piece of a solver ------------------------------------------- #

def _nest_register(op):
    """The nest's own register for one output column: lane `l` holds lead
    element `l`, and nothing varies with the column."""
    return BitLayout((
        tuple(Bit(Place.LANE, op.n << b)
              for b in range((op.blocks - 1).bit_length())),
        (),
        tuple(Bit(Place.LANE, 1 << b)
              for b in range((op.n - 1).bit_length())),
    ))


def _accumulator_cases():
    from tensorforge.backend.instructions.compute.primitives.amd import reorder
    for op in MATRIX_OPS:
        span = op.n * op.blocks
        if op.wave % span or layouts.fragment_layout(op, 'D') is None:
            continue
        if any(reorder.accumulator_gathers(op, c) is None
               for c in range(op.m)):
            continue
        yield op.builtin


@pytest.mark.parametrize('name', sorted(set(_accumulator_cases())))
def test_the_solver_finds_the_swaps_the_plan_states(name):
    """`accumulator_gathers` computes the epilogue by hand: for each region it
    checks that one XOR carries every lane and turns it into a `swap`
    sequence.  Written against two layouts that is one question -- where does
    each element sit, where does it have to sit -- and the answer has to be
    the same, or the vocabulary describes something other than what is
    emitted.

    Every region of every column of every entry whose accumulator has a plan.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import reorder
    op = next(o for o in MATRIX_OPS if o.builtin == name)
    span = op.n * op.blocks
    fragment = layouts.fragment_layout(op, 'D')
    have = _nest_register(op)
    checked = 0
    for column in range(op.m):
        stated = {(g.group, g.slot): g.swaps
                  for g in reorder.accumulator_gathers(op, column)}
        indices = [(b, column, i)
                   for b in range(op.blocks) for i in range(op.n)]
        for group in range(op.wave // span):
            found = bitlayout.moves(have, fragment, indices,
                                    base=Position(lane=group * span))
            assert found is not None, (name, column, group)
            for move in found:
                assert stated[(group, move.target)] == reorder._swaps_for(
                    move.xor), (name, column, group, move)
                checked += 1
    assert checked


def test_a_region_needing_two_toggles_is_refused():
    """Not a limitation to route around: where two elements sharing a pair of
    slots need different toggles, no sequence of reads serves the region, and
    a plan that pretended otherwise would write some of them from the wrong
    lane."""
    have = BitLayout(((Bit(Place.LANE, 1), Bit(Place.LANE, 2)),))
    want = BitLayout(((Bit(Place.LANE, 2), Bit(Place.LANE, 1)),))
    assert bitlayout.moves(have, want, [(i,) for i in range(4)]) is None


def test_an_identical_distribution_moves_by_nothing():
    layout = BitLayout(((Bit(Place.LANE, 1), Bit(Place.SLOT, 1)),))
    found = bitlayout.moves(layout, layout, [(i,) for i in range(4)])
    assert found is not None
    assert all(move.xor == 0 for move in found)


def test_a_vector_element_is_out_of_this_family_s_reach():
    """A lane toggle does not reach inside a register, so a packed operand
    needs its element bits accounted for before this is the right question."""
    packed = BitLayout(((Bit(Place.VECTOR, 1),),))
    plain = BitLayout(((Bit(Place.LANE, 1),),))
    assert bitlayout.moves(packed, plain, [(0,), (1,)]) is None


# -- the other half: bits that cross between lane and slot ----------------- #

def _before_transpose(ext, threads):
    """The shared matrix as the nest holds it: one register per column, the
    contraction across the lanes."""
    return BitLayout((
        tuple(Bit(Place.SLOT, 1 << b) for b in range((ext - 1).bit_length())),
        tuple(Bit(Place.LANE, 1 << b) for b in range((threads - 1).bit_length())),
    ))


def _after_transpose(ext, threads):
    """The same elements afterwards: the column on the low lane bits, and the
    contraction bits it displaced now in the registers."""
    low = (ext - 1).bit_length()
    return BitLayout((
        tuple(Bit(Place.LANE, 1 << b) for b in range(low)),
        tuple(Bit(Place.SLOT, 1 << b) if b < low else Bit(Place.LANE, 1 << b)
              for b in range((threads - 1).bit_length())),
    ))


@pytest.mark.parametrize('threads', [32, 64])
@pytest.mark.parametrize('ext', [4, 16])
def test_the_gap_across_a_transpose_is_the_exchange_it_performs(ext, threads):
    """The second half of a solver, and the half the swap family cannot
    serve: `swap` moves where a copy is read from, a transpose moves a bit
    between a register and a lane.  What the gap needs and what the
    instruction does are the same statement here, which is what lets one be
    answered by the other."""
    from tensorforge.backend.instructions.compute.primitives.amd import relayout
    gap = bitlayout.displacement(_before_transpose(ext, threads),
                                 _after_transpose(ext, threads))
    assert bitlayout.is_exchange(gap) == relayout.transpose_exchange(ext)


def test_no_gap_is_the_answer_an_emitter_wants_most():
    """`()` and not `None`: nothing to do is a result, and it is the one an
    operand that already arrives right should get."""
    layout = _after_transpose(4, 64)
    assert bitlayout.is_exchange(
        bitlayout.displacement(layout, layout)) == ()


def test_a_permutation_inside_the_lanes_is_no_transpose():
    """Nothing in `RELAYOUTS` moves a bit from one lane weight to another, so
    the honest answer is that this family does not reach it."""
    have = BitLayout(((Bit(Place.LANE, 1), Bit(Place.LANE, 2)),))
    want = BitLayout(((Bit(Place.LANE, 2), Bit(Place.LANE, 1)),))
    assert bitlayout.is_exchange(bitlayout.displacement(have, want)) is None


def test_an_unpaired_move_is_no_exchange():
    """A bit leaving the slots with nothing coming back is not what a
    transpose does -- it would leave two elements in one place."""
    have = BitLayout(((Bit(Place.SLOT, 1),), (Bit(Place.LANE, 2),)))
    want = BitLayout(((Bit(Place.LANE, 1),), (Bit(Place.LANE, 2),)))
    assert bitlayout.is_exchange(bitlayout.displacement(have, want)) is None


def test_index_spaces_that_do_not_line_up_have_no_displacement():
    """`produces` re-factors the index space -- it names one register's
    distribution -- so it is not what a displacement compares."""
    assert bitlayout.displacement(BitLayout(((Bit(Place.LANE, 1),),)),
                                  BitLayout(())) is None
    assert bitlayout.displacement(
        BitLayout(((Bit(Place.LANE, 1), Bit(Place.LANE, 2)),)),
        BitLayout(((Bit(Place.LANE, 1),),))) is None


# -- what the emitter now asks --------------------------------------------- #

@pytest.mark.parametrize('threads', [32, 64])
@pytest.mark.parametrize('ext', [4, 16])
def test_the_nest_needs_one_transpose_to_reach_the_fragment(ext, threads):
    """What `matmul32` does, now as an answer rather than an assumption."""
    from tensorforge.backend.instructions.compute.primitives.amd import relayout
    assert relayout.transposes_between(
        relayout.nest_shared(ext, threads),
        relayout.transposed(ext, threads), ext) == 1


def test_an_operand_that_already_arrives_right_needs_none():
    """The answer worth having: transposing unconditionally is correct for
    the arrangement the nest hands over and wrong for any other, and nothing
    could tell the two apart."""
    from tensorforge.backend.instructions.compute.primitives.amd import relayout
    ready = relayout.transposed(4, 64)
    assert relayout.transposes_between(ready, ready, 4) == 0


def test_a_packed_operand_is_not_reached_by_a_transpose():
    """Which is the branch that makes the question worth asking.  A `float4`
    holds its low lead bits inside a register, and no exchange of lane and
    slot bits reaches an element index -- so the emitter declines instead of
    emitting a transpose that does not land."""
    from tensorforge.backend.instructions.compute.primitives.amd import relayout
    packed = BitLayout((
        (Bit(Place.VECTOR, 1), Bit(Place.VECTOR, 2)),
        tuple(Bit(Place.LANE, 1 << b) for b in range(6)),
    ))
    assert relayout.transposes_between(
        packed, relayout.transposed(4, 64), 4) is None
