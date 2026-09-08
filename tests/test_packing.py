# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Filling an instruction axis, counted rather than argued.

Two decisions turn out to be one: whether to pad a partial block and run its
spare lanes against zeroes, and whether to put a second term product of a
split-precision emulation where the contraction does not reach.  Both ask what
an axis has left over.  These check the counting, and in particular that
packing never claims a saving the arithmetic does not allow -- a matrix
instruction computes the same number of products whatever is put in it, so a
full tile cannot be made cheaper by rearrangement.
"""

from __future__ import annotations

import itertools

import pytest

from tensorforge.backend.instructions.compute import packing, split
from tensorforge.common.basic_types import Datatype


# -- what an axis has left over -------------------------------------------- #

@pytest.mark.parametrize('demand,capacity,expected', [
    (0, 4, 0), (1, 4, 3), (3, 4, 1), (4, 4, 0), (5, 4, 3), (8, 4, 0),
])
def test_waste_is_what_the_last_tile_does_not_use(demand, capacity, expected):
    assert packing.waste(demand, capacity) == expected


def test_a_full_axis_is_never_wholly_wasted():
    """`capacity` would mean an instruction issued for nothing, which is an
    instruction not issued instead."""
    for capacity in range(1, 33):
        for demand in range(0, 200):
            assert 0 <= packing.waste(demand, capacity) < capacity


def test_tiles_and_waste_agree():
    for capacity in range(1, 17):
        for demand in range(0, 100):
            covered = packing.tiles(demand, capacity) * capacity
            assert covered - demand == (packing.waste(demand, capacity)
                                        if demand else 0)


def test_an_axis_holds_at_least_one_position():
    with pytest.raises(ValueError):
        packing.tiles(4, 0)


# -- the two layouts ------------------------------------------------------- #

PRODUCTS = split.products(2)


def _work(slots):
    return sorted((slot.product, slot.step) for slot in slots)


@pytest.mark.parametrize('steps', [1, 2, 3, 4, 7, 8, 9])
@pytest.mark.parametrize('capacity', [1, 2, 4, 8])
def test_both_layouts_compute_the_same_work_once_each(steps, capacity):
    """The property everything else rests on.  Packing rearranges where a
    pair is issued and must not drop one or issue it twice -- the second is
    the quieter failure, since the result is merely wrong rather than absent.
    """
    a = packing.stages(PRODUCTS, steps, capacity)
    b = packing.packed(PRODUCTS, steps, capacity)
    expected = sorted((p, s) for p in PRODUCTS for s in range(steps))
    assert _work(a) == expected
    assert _work(b) == expected


@pytest.mark.parametrize('steps', [1, 2, 3, 4, 5, 7, 8, 12])
@pytest.mark.parametrize('capacity', [1, 2, 3, 4, 8])
def test_no_instruction_overfills_its_axis(steps, capacity):
    for layout in (packing.stages(PRODUCTS, steps, capacity),
                   packing.packed(PRODUCTS, steps, capacity)):
        held = {}
        for slot in layout:
            assert 0 <= slot.position < capacity
            held.setdefault(slot.instruction, set()).add(slot.position)
        for instruction, positions in held.items():
            assert len(positions) <= capacity


def test_a_product_keeps_its_operands_in_one_step():
    """Both halves of a term product come from the same contraction step.
    Pairing across steps computes a different sum rather than a rounder one,
    and nothing downstream would notice."""
    for slot in packing.packed(split.products(3), steps=5, capacity=4):
        assert 0 <= slot.step < 5


# -- what packing is worth ------------------------------------------------- #

@pytest.mark.parametrize('steps', range(1, 17))
@pytest.mark.parametrize('capacity', [1, 2, 3, 4, 8, 16])
def test_packing_never_costs_instructions(steps, capacity):
    assert packing.saving(PRODUCTS, steps, capacity) >= 0


@pytest.mark.parametrize('capacity', [1, 2, 4, 8])
def test_a_divided_contraction_leaves_nothing_to_share(capacity):
    """Where the capacity divides the depth, every instruction of the
    unpacked layout is already full and there is no spare space to move a
    product into."""
    for multiple in (1, 2, 3):
        assert packing.saving(PRODUCTS, capacity * multiple, capacity) == 0


def test_packing_recovers_a_partial_instruction_per_product():
    """Two products of a three-deep contraction into four positions: unpacked
    issues one half-empty instruction each, packed issues one and a half."""
    assert packing.instructions(
        packing.stages(PRODUCTS[:2], 3, capacity=4)) == 2
    assert packing.instructions(
        packing.packed(PRODUCTS[:2], 3, capacity=4)) == 2
    assert packing.saving(PRODUCTS[:2], 3, capacity=4) == 0

    # Two steps instead of three, and the four pairs fit one instruction
    # where the unpacked layout still issues a half-empty one per product.
    assert packing.saving(PRODUCTS[:2], 2, capacity=4) == 1


def test_a_full_axis_cannot_be_made_cheaper():
    """The conservation statement, as a property rather than a claim: the
    products a layout issues are `instructions * capacity` at most, and the
    work is fixed, so no arrangement drops below the ceiling."""
    for capacity in (2, 4, 8):
        for steps in range(1, 13):
            work = len(PRODUCTS) * steps
            floor = packing.tiles(work, capacity)
            for layout in (packing.stages(PRODUCTS, steps, capacity),
                           packing.packed(PRODUCTS, steps, capacity)):
                assert packing.instructions(layout) >= floor
            assert packing.instructions(
                packing.packed(PRODUCTS, steps, capacity)) == floor


def test_spare_counts_what_a_further_product_could_use():
    layout = packing.packed(PRODUCTS, steps=3, capacity=4)
    assert packing.spare(layout, 4) == (
        packing.instructions(layout) * 4 - len(layout))


# -- the padding decision is the same question ----------------------------- #

@pytest.mark.parametrize('block', [4, 8, 16, 32])
def test_the_matrix_span_boundary_reads_the_same_waste(block):
    """`amd.plan` asks this module where its tail goes, and the answer is the
    one it stated in remainders: a tail is left to the chain when the block
    would be empty or would spend all but one position on zeroes."""
    for n in range(0, 200):
        empty = packing.waste(n, block)
        assert (empty in (0, block - 1)) == (n % block < 2)


def test_padding_is_packing_with_one_product():
    """Which is why they are counted in one place.  A padded block is the
    layout for a single product whose last instruction is partly spare."""
    direct = split.products(1)
    assert direct == ((0, 0),)
    layout = packing.stages(direct, steps=6, capacity=4)
    assert packing.instructions(layout) == packing.tiles(6, 4) == 2
    assert packing.spare(layout, 4) == packing.waste(6, 4) == 2


def test_an_exact_arithmetic_packs_to_the_plain_contraction():
    """One term, one product: both layouts are the unemulated loop, and the
    module says nothing about it."""
    direct = split.products(
        split.terms(split.MANTISSA[Datatype.F64], Datatype.F64))
    assert len(direct) == 1
    for steps, capacity in itertools.product((1, 4, 7), (1, 4, 8)):
        assert packing.saving(direct, steps, capacity) == 0
