# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a kernel asks yateto to store for it."""

import numpy as np
import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.frontend.yateto import YatetoFrontend


def _tensor(order=None, parts=1, planar=False, addressing=Addressing.NONE):
    data = np.arange(6.0).reshape(3, 2, order='F')
    t = Tensor(shape=[3, 2], addressing=addressing, datatype=Datatype.F64,
               data=data, alias='A')
    if order is not None:
        t.storage_order = order
    if parts != 1:
        t.storage_parts = parts
    t.storage_planar = planar
    return t


class FakeEmitter:
    def __init__(self, tensors):
        self._tensors = tensors

    def tensors(self):
        return self._tensors


def _offerings(tensors):
    frontend = YatetoFrontend.__new__(YatetoFrontend)
    frontend._emitter = FakeEmitter(tensors)
    return frontend.layout_offerings()


class TestLayoutOfferings:
    def test_nothing_is_offered_without_a_decided_order(self):
        assert _offerings({'A': _tensor()}) == {}

    def test_a_decided_order_is_offered_as_numbers(self):
        order = (5, 4, 3, 2, 1, 0)

        offerings = _offerings({'A': _tensor(order=order)})

        assert offerings['A']['data'] == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        assert offerings['A']['parts'] == 1

    def test_a_slot_with_no_cell_reads_zero(self):
        # an order over a tensor whose image is tiled past its end
        t = _tensor()
        t.storage_order = (0, 1, 2, 3, 4, 5)
        assert _offerings({'A': t})['A']['data'] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_an_operand_that_is_not_batch_constant_is_left_alone(self):
        t = _tensor(order=(5, 4, 3, 2, 1, 0), addressing=Addressing.STRIDED)

        assert _offerings({'A': t}) == {}

    def test_split_elements_are_not_offered(self):
        t = _tensor(order=(5, 4, 3, 2, 1, 0), parts=2)

        assert _offerings({'A': t}) == {}

    def test_without_a_kernel_nothing_is_offered(self):
        frontend = YatetoFrontend.__new__(YatetoFrontend)
        frontend._emitter = None

        assert frontend.layout_offerings() == {}
