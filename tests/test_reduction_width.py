# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How wide the broadcast operand may be loaded along the reduction axis.

`B` in `C[m,n] += A[m,k] B[k,n]` is not indexed by the lead dimension, so the
lead width does nothing for it -- it is loaded once per `(k, n)` and splatted.
But `k` is `B`'s own contiguous axis, so `B[k,n] .. B[k+V-1,n]` are adjacent
and one load fetches the operands of `V` consecutive reduction steps.

The conditions are almost disjoint from `lead_vector_width`'s, and the reason
is worth stating: a remainder on a *reduction* is free. Leftover `k` values
are added scalarly afterwards and no lane ends up holding a vector that is
half outside anything, so divisibility -- the condition that decides the lead
width -- does not appear here at all, and the cap can go to 4 where the lead
cap is 2.

What this does not do is remove a splat. Each of the `V` components still
feeds its own FMA against its own `A` vector and still needs its own `{b, b}`.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.memory.vectorize import (
    reduction_vector_width)


def w(extent=16, elem=4, align=16, stride=1, dense=True, cap=4):
    return reduction_vector_width(extent, elem, align, stride, dense, cap)


def test_an_aligned_dense_contiguous_operand_reaches_the_cap():
    assert w() == 4


def test_an_unproven_base_gets_one():
    assert w(align=0) == 1
    assert w(align=4) == 1


def test_a_transposed_operand_gets_one():
    """A strided reduction axis means the `V` values are not adjacent at all,
    which no alignment fixes."""
    assert w(stride=16) == 1


def test_a_sparse_operand_gets_one():
    """Only the non-zeros are stored, so a wide load reads across whichever
    entries happen to follow -- and not touching them is the point."""
    assert w(dense=False) == 1


def test_divisibility_is_not_a_condition():
    """Unlike the lead dimension.  A leftover `k` is added scalarly and
    nothing is left half-valid, so 13 reduction steps still take width 4."""
    assert w(extent=13) == 4
    assert w(extent=9) == 4


def test_a_reduction_shorter_than_the_width_narrows_to_it():
    assert w(extent=2) == 2
    assert w(extent=1) == 1


def test_fp64_stops_at_two():
    """16 bytes is the widest access any target has."""
    assert w(elem=8) == 2


def test_the_cap_is_honoured():
    assert w(cap=2) == 2
    assert w(cap=1) == 1


def test_a_degenerate_extent_is_one():
    assert w(extent=0) == 1
