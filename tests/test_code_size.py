# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The size estimate, and what it is allowed to be wrong about.

Pinned loosely on purpose.  The constant is a fit, so a test that asserted a
figure would only assert that nobody refitted it; what has to hold is the
shape -- more arithmetic is more code, more lanes is less code per lane, and a
body that is one of several in a list is smaller than the list.
"""

import pytest

from tensorforge.analysis.cost import (LINES_FIXED, LINES_PER_LANE_FLOP,
                                       estimated_lines)
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

DTYPE = Datatype.F32


def op(alias_in, alias_out, k=9):
    def T(alias, shape):
        return SubTensor(Tensor(list(shape), Addressing.STRIDED,
                                BoundingBox([0] * len(shape), list(shape)),
                                alias=alias, datatype=DTYPE))
    return GemmDescr(trans_a=False, trans_b=False, a=T('A', [k, k]),
                     b=T(alias_in, [k, 4]), c=T(alias_out, [k, 4]))


def test_more_arithmetic_is_more_code():
    small = estimated_lines([op('i', 'o', k=9)], 32)
    large = estimated_lines([op('i', 'o', k=56)], 32)
    assert large > small


def test_more_lanes_is_less_code():
    descrs = [op('i', 'o', k=56)]
    assert estimated_lines(descrs, 64) < estimated_lines(descrs, 32)


def test_a_body_is_smaller_than_the_list_it_repeats_in():
    body = [op('i0', 'o0')]
    whole = [op(f'i{k}', f'o{k}') for k in range(4)]
    assert estimated_lines(body, 32) < estimated_lines(whole, 32)


def test_a_list_with_no_arithmetic_is_the_fixed_part():
    assert estimated_lines([], 32) == LINES_FIXED


def test_no_lanes_is_refused():
    with pytest.raises(ValueError):
        estimated_lines([op('i', 'o')], 0)


def test_the_constant_is_positive_and_stated_per_lane_flop():
    """A sign check, so a refit that inverts the model is caught."""
    assert LINES_PER_LANE_FLOP > 0
    assert LINES_FIXED >= 0
