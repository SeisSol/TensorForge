# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An operand whose shift along one axis arrives as data.

`offset_from` is how one of several matrices is selected from the frontend
side: the family is a tensor with one axis more, and the shift along that axis
is a value rather than a number.

Most of what is tested here is the *refusal*.  The field is read, the selected
axis is named, and the shift is declined -- and the reason to pin that is the
failure mode it stands in front of.  A selector silently dropped would produce
an operand pointing at the first member of the family, which is a kernel that
compiles, runs, verifies and is right for one configuration in sixteen.  So the
absent case, the all-zero case and the selecting case each have to come out
differently, and the difference has to be stated somewhere that fails when it
stops holding.
"""

import pytest

from tensorforge.backend.symbol import LeadIndex, VarOffset
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.exceptions import InternalError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.frontend.yateto import DescriptionReader


def _reader(shape=(4, 56, 56)):
    """A frontend with one tensor in its cache, named as a reference will."""
    reader = DescriptionReader.__new__(DescriptionReader)
    reader._prefix = ''
    reader._cache = {
        'family': Tensor(list(shape), Addressing.NONE,
                         BoundingBox([0] * len(shape), list(shape)),
                         alias='family', datatype=Datatype.F32)}
    return reader


def _ref(**extra):
    d = {'name': 'family'}
    d.update(extra)
    return d


# --- the occurrence without a selector --------------------------------------


def test_a_reference_without_the_field_is_unchanged():
    """A description from a yateto that predates the field states none."""
    view = _reader().tensor_ref(_ref())
    assert list(view.offset) == [0, 0, 0]
    assert not view.sliced


def test_an_all_zero_selector_selects_nothing():
    """Present and empty is the same as absent, and has to stay so.

    Every reference in a description carries the field once the frontend emits
    it, so treating a zero selector as a selection would route every operand
    through a path that refuses.
    """
    view = _reader().tensor_ref(_ref(offset_from=[0, 0, 0]))
    assert list(view.offset) == [0, 0, 0]


def test_a_constant_offset_still_arrives():
    view = _reader().tensor_ref(_ref(offset=[2, 0, 0], offset_from=[0, 0, 0]))
    assert list(view.offset) == [2, 0, 0]


def test_the_box_of_the_occurrence_is_kept():
    view = _reader().tensor_ref(_ref(bbox=[[0, 0, 0], [1, 56, 56]]))
    assert list(view.bbox.upper()) == [1, 56, 56]


# --- the occurrence with one -------------------------------------------------


def test_a_selector_is_refused_and_not_dropped():
    """The failure mode it stands in front of is silence, not an error."""
    with pytest.raises(NotImplementedError):
        _reader().tensor_ref(_ref(offset_from=[1, 0, 0]))


def test_the_refusal_names_the_axis_that_selects():
    with pytest.raises(NotImplementedError) as raised:
        _reader().tensor_ref(_ref(offset_from=[0, 0, 1]))
    assert '[2]' in str(raised.value)


def test_selectors_on_several_axes_are_all_named():
    with pytest.raises(NotImplementedError) as raised:
        _reader().tensor_ref(_ref(offset_from=[1, 0, 1]))
    assert '[0, 2]' in str(raised.value)


# --- what will carry it ------------------------------------------------------


def test_a_variable_offset_writes_the_sum():
    class Index:
        def is_thread_dependent(self):
            return False

        def write_nonlead(self):
            return 'v'

    assert VarOffset(Index(), 3).write_nonlead() == '(v + 3)'


def test_a_variable_offset_inherits_thread_dependence():
    class Index:
        def __init__(self, dependent):
            self.dependent = dependent

        def is_thread_dependent(self):
            return self.dependent

    assert VarOffset(Index(True), 0).is_thread_dependent()
    assert not VarOffset(Index(False), 0).is_thread_dependent()


def test_a_lead_index_may_not_be_wrapped():
    """It carries its own offset, in slots where this counts elements."""
    with pytest.raises(InternalError):
        VarOffset(LeadIndex.__new__(LeadIndex), 1)
