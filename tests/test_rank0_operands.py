# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A tensor described without axes, in an operation that has some.

yateto sends `rhoInv[]` -- one value per element, in memory -- as a tensor of
rank 0, and `Tensor` carries it with one axis of extent one (`Tensor.rank0`).
Which axis that is, is the operation's to say: `MultilinearDescr` gives it the
destination's axis 0 where the destination has no axes either, and an axis of
its own, contracted, where it does.  The frontend used to decide it and chose
the first for both, so `t[i,j] = rhoInv * S[i,j]` wrote row 0 of `t` and left
the rest to whatever the buffer held.
"""

from __future__ import annotations

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr


def _t(shape, alias, addressing=Addressing.STRIDED, tmp=False):
    return Tensor(list(shape), addressing, alias=alias, is_tmp=tmp,
                  datatype=Datatype.F32)


def test_an_axisless_tensor_is_carried_with_one_axis():
    t = _t([], 'rhoInv', tmp=True)
    assert t.rank0 and t.shape == (1,)
    assert (list(t.bbox.lower()), list(t.bbox.upper())) == ([0], [1])


def test_its_data_is_carried_the_same_way():
    t = Tensor([], Addressing.NONE, alias='c', datatype=Datatype.F32,
               data={(): 0.5})
    assert t.data.shape == (1,) and float(t.data[0]) == 0.5


def test_a_scalar_is_not_carried():
    s = Tensor([], Addressing.SCALAR, datatype=Datatype.F32)
    assert not s.rank0 and s.shape == ()


def test_into_a_destination_with_axes_it_is_a_contracted_axis():
    s = _t([], 'rhoInv', tmp=True)
    m = _t([6, 11], 'S', Addressing.NONE)
    d = _t([6, 11], 'T', tmp=True)
    descr = MultilinearDescr(SubTensor(d), [SubTensor(s), SubTensor(m)],
                             [[], [0, 1]], [[], [0, 1]])
    assert descr.target == [[-1], [0, 1]]
    assert descr.permute == [[0], [0, 1]]


def test_the_contracted_axis_is_one_no_other_operand_uses():
    a = _t([4, 3], 'A')
    b = _t([3, 5], 'B')
    s = _t([], 's', tmp=True)
    d = _t([4, 5], 'D', tmp=True)
    descr = MultilinearDescr(SubTensor(d),
                             [SubTensor(a), SubTensor(b), SubTensor(s)],
                             [[0, -1], [-1, 1], []], [[0, 1], [0, 1], []])
    assert descr.target[2] == [-2]


def test_into_an_axisless_destination_it_shares_axis_zero():
    x = _t([], 'x', tmp=True)
    v = _t([13], 'v')
    p = _t([13], 'p', Addressing.NONE)
    r = _t([], 'r', tmp=True)
    descr = MultilinearDescr(SubTensor(r),
                             [SubTensor(x), SubTensor(v), SubTensor(p)],
                             [[], [-1], [-1]], [[], [0], [0]])
    assert descr.target == [[0], [-1], [-1]]


def test_accumulating_onto_an_axisless_destination_names_its_axis():
    """The empty mask is yateto's for a destination without axes."""
    v = _t([13], 'v')
    p = _t([13], 'p', Addressing.NONE)
    r = _t([], 'r', tmp=True)
    descr = MultilinearDescr(SubTensor(r), [SubTensor(v), SubTensor(p)],
                             [[-1], [-1]], [[0], [0]], add=[])
    assert descr.add and descr.add_dims == [0]


def test_the_caller_s_lists_are_left_alone():
    s = _t([], 's', tmp=True)
    m = _t([6, 11], 'S', Addressing.NONE)
    d = _t([6, 11], 'T', tmp=True)
    target, permute = [[], [0, 1]], [[], [0, 1]]
    MultilinearDescr(SubTensor(d), [SubTensor(s), SubTensor(m)],
                     target, permute)
    assert target == [[], [0, 1]] and permute == [[], [0, 1]]
