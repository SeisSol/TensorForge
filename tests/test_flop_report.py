# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a kernel tells yateto about the arithmetic it contains."""

from tensorforge.analysis.cost import list_cost
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.frontend.yateto import YatetoFrontend
from tensorforge.generators.descriptions import MultilinearDescr


def _tensor(name, shape):
    return Tensor(list(shape), Addressing.STRIDED, BoundingBox([0] * len(shape), list(shape)),
                  alias=name, datatype=Datatype.F64)


def _gemm(add):
    a, b, c = (SubTensor(_tensor(name, [8, 8])) for name in ('A', 'B', 'C'))
    return MultilinearDescr(c, [a, b], [[0, -1], [-1, 1]], [[0, 1], [0, 1]], add=add)


class FakeEmitter:
    def __init__(self, descrs):
        self._descrs = descrs

    def descriptors(self):
        return list(self._descrs)


def _report(descrs):
    frontend = YatetoFrontend.__new__(YatetoFrontend)
    frontend._emitter = FakeEmitter(descrs)
    return frontend.flop_report()


class TestFlopReport:
    def test_a_kernel_reports_the_arithmetic_of_its_operations(self):
        # 8x8x8, assigned: 512 products and 448 additions -- the first term of
        # each point is written, not added (see test_cost_model.py)
        assert _report([_gemm(add=False)]) == {'plain': 512 + 448}

    def test_it_is_the_cost_model_s_figure_summed_over_the_kernel(self):
        descrs = [_gemm(add=False), _gemm(add=True)]
        assert _report(descrs) == {'plain': list_cost(descrs).flops}

    def test_nothing_is_reported_before_the_kernel_exists(self):
        frontend = YatetoFrontend.__new__(YatetoFrontend)
        frontend._emitter = None
        assert frontend.flop_report() == {}

    def test_a_kernel_without_arithmetic_reports_nothing(self):
        assert _report([]) == {}
