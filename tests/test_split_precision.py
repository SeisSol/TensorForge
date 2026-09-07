# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Split-precision arithmetic, asked of one place instead of two.

The term count and the product order are an error analysis, and an error
analysis written out per target is one that nothing compares.  The AMD
catalogue derived both from the significand widths and no emitter read it; the
DPAS path wrote its three products out as a tuple and named the product count
`TF32_TERMS`, which is not the term count.  These pin what the two agree on
and what the move changed.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute import split
from tensorforge.backend.instructions.compute.primitives import intel
from tensorforge.common.basic_types import Datatype


# -- the counts ------------------------------------------------------------ #

def test_three_bf16_terms_cover_an_f32_significand():
    """Exactly, which is what makes three the unreduced answer."""
    assert split.terms(split.MANTISSA[Datatype.BF16], Datatype.F32) == 3
    assert split.covered(split.MANTISSA[Datatype.BF16], 3) >= 24


def test_a_reduced_split_is_a_number_rather_than_a_habit():
    """Two BF16 terms carry 16 bits, two TF32 terms 22, and FP32 wants 24.
    Both are usable and neither is the same claim as covering the target, so
    the shortfall is statable rather than implied by an arity somewhere."""
    assert split.covered(split.MANTISSA[Datatype.BF16], 2) == 16
    assert split.covered(split.MANTISSA[Datatype.TF32], 2) == 22
    assert split.MANTISSA[Datatype.F32] == 24


def test_an_exact_arithmetic_needs_one_term():
    """F64 through an F64 instruction is not a split, and the formula says so
    rather than needing a caller to check first."""
    assert split.terms(split.MANTISSA[Datatype.F64], Datatype.F64) == 1
    assert split.products(1) == ((0, 0),)


# -- the products ---------------------------------------------------------- #

@pytest.mark.parametrize('count,expected', [(1, 1), (2, 3), (3, 6)])
def test_keeping_everything_above_the_rounding_is_triangular(count, expected):
    assert len(split.products(count)) == expected


def test_products_are_ordered_smallest_contribution_first():
    """So the small ones accumulate before the large one rounds them off."""
    for count in (2, 3, 4):
        weights = [i + j for i, j in split.products(count)]
        assert weights == sorted(weights, reverse=True), count


def test_dropping_below_keep_removes_the_smallest():
    """`keep` is a rounding threshold, so what it removes is the tail."""
    full = set(split.products(3))
    reduced = set(split.products(3, keep=2))
    assert reduced < full
    assert all(i + j < 2 for i, j in reduced)


# -- what the DPAS path now reads ------------------------------------------ #

def test_the_dpas_products_are_the_same_set_as_before():
    """The move reorders and nothing else.  Written out, the path accumulated
    `hi*hi`, `hi*lo` and `lo*hi`; the formula returns those three, largest
    last instead of first."""
    assert set(split.products(intel.TF32_SPLIT_TERMS)) == {(0, 0), (0, 1),
                                                           (1, 0)}


def test_the_product_count_is_derived_from_the_term_count():
    """Two names for two different numbers, and the second follows from the
    first: a path taking a different number of terms gets the right number of
    products without a second edit."""
    assert intel.TF32_SPLIT_TERMS == 2
    assert intel.TF32_TERMS == len(split.products(intel.TF32_SPLIT_TERMS))


def test_the_dpas_split_is_short_of_the_target_by_a_stated_amount():
    """What `splitFloatTF32` returning a pair costs, in bits, as a number the
    file can be asked for rather than one a reader has to work out."""
    tf32 = split.MANTISSA[Datatype.TF32]
    assert split.terms(tf32, Datatype.F32) == 3
    assert intel.TF32_SPLIT_TERMS == 2
    assert split.covered(tf32, intel.TF32_SPLIT_TERMS) == 22
