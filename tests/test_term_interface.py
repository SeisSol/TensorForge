# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel that arrives as terms rather than as a description.

One route into this side sends a kernel one operation at a time, as the
objects yateto's own codegen works with: a term states its indices, its
memory layout and its equivalent sparsity pattern, and everything a
description would state outright has to be derived from those.  This is the
interface SeisSol generates against.

What is asserted here is that the derivation happens -- the box comes off the
eqspp, a view becomes an offset and marks the operand a slice, a named factor
becomes a scalar operand -- and that a kernel built this way reaches the
emitter, with nothing announcing it and no description ever existing.

The stand-ins below are named after the yateto classes they stand for, since
two of them are told apart by the name of their type.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Addressing
from tensorforge.frontend.yateto import TermReader, YatetoFrontend
from tensorforge.generators.descriptions import MultilinearDescr


class _Arch:
    name = "sm_86"
    backend = "cuda"
    typename = "float"
    alignment = 64

    def alignedLower(self, index):
        return index

    def alignedUpper(self, index):
        return index


ARCH = _Arch()


class DenseMemoryLayout:
    """A layout with nothing to say about sparsity."""

    def __init__(self, shape, aligned=False):
        self._shape = tuple(shape)
        self._aligned = aligned

    def shape(self):
        return self._shape

    def bbox(self):
        return [range(0, extent) for extent in self._shape]

    def alignedStride(self):
        return self._aligned

    def storage(self):
        return self


class MemoryLayoutView:
    """A slice of a layout: its own index space, plus the shift onto the base."""

    def __init__(self, base, index, start, end):
        self.base = base
        self.index = index
        self.start = start
        self.end = end

    def shape(self):
        return tuple(extent if i != self.index else self.end - self.start
                     for i, extent in enumerate(self.base.shape()))

    def relidx(self, index):
        return tuple(value if i != self.index else value + self.start
                     for i, value in enumerate(index))

    def alignedStride(self):
        return self.base.alignedStride()

    def storage(self):
        return self.base.storage()


class _Eqspp:
    """The non-zero bounds of a term, inclusive at the upper end as yateto has them."""

    def __init__(self, lower, upper):
        self._bounds = [(lo, hi - 1) for lo, hi in zip(lower, upper)]

    def nnzbounds(self):
        return list(self._bounds)


class _Term:
    def __init__(self, name, indices, memoryLayout, eqspp=None,
                 is_temporary=False, is_compute_constant=False,
                 values=None, addressing=None):
        self.name = name
        self.indices = indices
        self.memoryLayout = memoryLayout
        self.eqspp = eqspp or _Eqspp([0] * len(memoryLayout.shape()),
                                     memoryLayout.shape())
        self.is_temporary = is_temporary
        self.is_compute_constant = is_compute_constant
        self.values = values
        self.addressing = addressing
        self.datatype = None


class _Scalar:
    def __init__(self, name):
        self._name = name
        self.datatype = None

    def name(self):
        return self._name


def _dense(name, shape, **kwargs):
    return _Term(name, "ij"[:len(shape)], DenseMemoryLayout(shape), **kwargs)


def _matmul(extra_ops=(), extra_target=(), extra_permute=()):
    """``C['ij'] = A['ik'] B['kj']``, indexed the way yateto numbers it.

    Non-negative numbers are axes of the result; contracted axes are negative
    and numbered as they are met.  Returned as arguments rather than fed
    somewhere, because the reader and the frontend spell the call differently.
    """
    dest = _Term("C", "ij", DenseMemoryLayout([16, 16]))
    ops = [_Term("A", "ik", DenseMemoryLayout([16, 16])),
           _Term("B", "kj", DenseMemoryLayout([16, 16]))]
    return (dest,
            ops + list(extra_ops),
            [[0, -1], [-1, 1]] + list(extra_target),
            [[0, 1], [0, 1]] + list(extra_permute),
            False)


def test_a_term_becomes_a_multilinear_over_the_box_its_eqspp_marks():
    reader = TermReader(ARCH)
    dest = _Term("C", "ij", DenseMemoryLayout([16, 16]),
                 eqspp=_Eqspp([0, 0], [16, 16]))
    ops = [_Term("A", "ik", DenseMemoryLayout([16, 16]),
                 eqspp=_Eqspp([0, 0], [8, 16])),
           _Term("B", "kj", DenseMemoryLayout([16, 16]))]
    reader.add_operation(dest, ops, [[0, -1], [-1, 1]], [[0, 1], [0, 1]], False)

    descrs, cache = reader.result()
    assert len(descrs) == 1
    descr = descrs[0]
    assert isinstance(descr, MultilinearDescr)
    assert set(cache) == {"A", "B", "C"}
    # the operand runs over what its pattern says is non-zero, not over what
    # the layout stores
    assert list(descr.ops[0].bbox.upper()) == [8, 16]
    assert list(descr.dest.bbox.upper()) == [16, 16]


def test_an_operation_of_terms_runs_unguarded():
    """There is no guard on this route, and no guard means the operation runs.

    Worth stating, because `None` means the opposite one field earlier: a
    description whose condition is `None` names an operation yateto has
    already decided is dead.
    """
    reader = TermReader(ARCH)
    reader.add_operation(*_matmul())
    descrs, _ = reader.result()
    assert descrs[0].condition is None
    assert not descrs[0].guarded()


def test_a_view_becomes_an_offset_and_marks_the_operand_a_slice():
    reader = TermReader(ARCH)
    dest = _Term("C", "ij", DenseMemoryLayout([16, 16]))
    sliced = _Term("A", "ik",
                   MemoryLayoutView(DenseMemoryLayout([16, 16]), 0, 4, 12))
    ops = [sliced, _Term("B", "kj", DenseMemoryLayout([16, 16]))]
    reader.add_operation(dest, ops, [[0, -1], [-1, 1]], [[0, 1], [0, 1]], False)

    descr, = reader.result()[0]
    assert descr.ops[0].offset == [4, 0]
    assert descr.ops[0].sliced
    assert not descr.ops[1].sliced
    # the box stays in the space the operand names; only the address is shifted
    assert list(descr.ops[0].bbox.lower()) == [0, 0]
    assert list(descr.ops[0].bbox.upper()) == [8, 16]


def test_a_slice_starting_at_zero_is_still_a_slice():
    """Its offset is zero, so nothing about the box gives it away."""
    reader = TermReader(ARCH)
    dest = _Term("C", "ij", DenseMemoryLayout([16, 16]))
    ops = [_Term("A", "ik",
                 MemoryLayoutView(DenseMemoryLayout([16, 16]), 0, 0, 8)),
           _Term("B", "kj", DenseMemoryLayout([16, 16]))]
    reader.add_operation(dest, ops, [[0, -1], [-1, 1]], [[0, 1], [0, 1]], False)

    descr, = reader.result()[0]
    assert descr.ops[0].offset == [0, 0]
    assert descr.ops[0].sliced


def test_a_named_factor_arrives_as_a_scalar_operand_over_no_axis():
    reader = TermReader(ARCH)
    reader.add_operation(*_matmul(extra_ops=[_Scalar("alpha")],
                                  extra_target=[[]], extra_permute=[[]]))

    descr, = reader.result()[0]
    assert len(descr.ops) == 3
    assert descr.ops[2].tensor.addressing is Addressing.SCALAR
    assert descr.target[2] == []


def test_a_literal_factor_carries_its_value():
    reader = TermReader(ARCH)
    reader.add_operation(*_matmul(extra_ops=[2.0], extra_target=[[]],
                                  extra_permute=[[]]))

    descr, = reader.result()[0]
    assert descr.ops[2].tensor.addressing is Addressing.SCALAR
    assert descr.ops[2].tensor.get_values() == 2.0


def test_a_temporary_destination_is_not_a_kernel_parameter():
    reader = TermReader(ARCH)
    dest = _Term("tmp", "ij", DenseMemoryLayout([16, 16]), is_temporary=True)
    ops = [_Term("A", "ik", DenseMemoryLayout([16, 16])),
           _Term("B", "kj", DenseMemoryLayout([16, 16]))]
    reader.add_operation(dest, ops, [[0, -1], [-1, 1]], [[0, 1], [0, 1]], False)

    _, cache = reader.result()
    assert cache["tmp"].is_tmp
    assert cache["tmp"].addressing is Addressing.STRIDED


class _Cpp:
    def __init__(self):
        self.lines = []

    def __call__(self, line):
        self.lines.append(line)


class _RoutineCache:
    def __init__(self):
        self.routines = {}

    def addRoutine(self, name, writer):
        self.routines[name] = writer


def test_a_kernel_of_terms_reaches_the_emitter():
    """No call opens such a kernel, so `generate` is what completes it."""
    frontend = YatetoFrontend(ARCH, attrs={})
    frontend.add_linear_operation(*_matmul())

    cpp, cache = _Cpp(), _RoutineCache()
    frontend.generate(cpp, cache)

    assert len(cache.routines) == 1
    assert cpp.lines and any("(" in line for line in cpp.lines)


def test_operations_keep_arriving_until_generate_asks():
    frontend = YatetoFrontend(ARCH, attrs={})
    frontend.add_linear_operation(*_matmul())
    frontend.add_linear_operation(*_matmul())

    descrs, _ = frontend._terms.result()
    assert len(descrs) == 2


def test_a_frontend_that_was_handed_nothing_refuses_to_generate():
    frontend = YatetoFrontend(ARCH, attrs={})
    with pytest.raises(NotImplementedError, match="nothing to build"):
        frontend.generate(_Cpp(), _RoutineCache())


def test_a_kernel_of_terms_is_captured_without_a_description():
    """The capture hook covers both routes, and states which one it saw."""
    recorded = []
    with YatetoFrontend.capture(recorded.append):
        frontend = YatetoFrontend(ARCH, attrs={})
        frontend.add_linear_operation(*_matmul())
        # the second operation arrives after the sink has already been handed
        # the kernel, and still has to show up in what it was handed
        frontend.add_linear_operation(*_matmul())
        frontend.generate(_Cpp(), _RoutineCache())

    item, = recorded
    assert item.description is None
    assert len(item.descrs) == 2
    assert item.name is not None
