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
