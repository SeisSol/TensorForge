# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`verify` bounds every access it can resolve against its buffer.

A register array is a fixed number of registers. An index one past either end
is a neighboring register or a spill slot, so the access reads a value rather
than nothing, and every check that asks whether something was computed is
satisfied by it. The host oracle is no help either: it keeps registers in a
dict, serves index -1, and answers.

What makes the check worth having at this layer rather than only over a dump
is that the index is an expression tree over loop counters with stated bounds,
so the range is derived rather than pattern-matched.
"""

from __future__ import annotations

import pytest

from tensorforge.backend import pir
from tensorforge.backend.pir import IRBuilder, MemSpace
from tensorforge.common.basic_types import Datatype

SIZE = 6


def _body(index):
    builder = IRBuilder(fptype=Datatype.F32)
    array = builder.alloc(Datatype.F32, (SIZE,), MemSpace.REGISTER, hint='r0')
    builder.load(array, index)
    return builder.finish()


def _bounds_diagnostics(body):
    return [d for d in pir.verify(body, strict=False) if 'addressed at' in d]


@pytest.mark.parametrize("index", [0, SIZE - 1])
def test_an_index_inside_the_array_is_accepted(index):
    assert _bounds_diagnostics(_body(index)) == []


@pytest.mark.parametrize("index", [-1, SIZE, SIZE + 3])
def test_an_index_outside_the_array_is_reported(index):
    diagnostics = _bounds_diagnostics(_body(index))
    assert len(diagnostics) == 1, diagnostics
    assert f'r0 holds {SIZE} elements' in diagnostics[0]


HALF = SIZE // 2


class _Symbol:
    """What an access through a symbol shows the check: a name."""
    stype = None

    def __init__(self, name):
        self.name = name


def _namesakes(access):
    """Two register arrays of one name, as every lane's images of its own
    parts of two buffers are (`pir.carry.keep_own`), and an access `access`
    makes, given the builder and the larger and the smaller array."""
    builder = IRBuilder(fptype=Datatype.F32)
    larger = builder.alloc(Datatype.F32, (SIZE,), MemSpace.REGISTER, hint='own')
    smaller = builder.alloc(Datatype.F32, (HALF,), MemSpace.REGISTER, hint='own')
    access(builder, larger, smaller)
    return builder.finish()


@pytest.mark.parametrize("access,reported", [
    (lambda b, larger, smaller: b.load(larger, SIZE - 1), None),
    (lambda b, larger, smaller: b.load(smaller, HALF - 1), None),
    (lambda b, larger, smaller: b.load(smaller, HALF), HALF),
    (lambda b, larger, smaller: b.store(smaller, 0.0, HALF), HALF),
], ids=["larger", "smaller", "smaller past its end", "store past its end"])
def test_an_array_is_bounded_by_its_own_size_and_not_a_namesakes(
        access, reported):
    """The larger array's last element is past the end of the smaller one,
    and the smaller's first past its end lies inside the larger one: either
    array bounded by the other's size is reported wrongly, or not at all."""
    diagnostics = _bounds_diagnostics(_namesakes(access))
    if reported is None:
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1, diagnostics
        assert f'holds {reported} elements' in diagnostics[0]


@pytest.mark.parametrize("index,reported", [(SIZE - 1, False), (SIZE, True)])
def test_a_name_is_bounded_by_the_largest_array_of_it(index, reported):
    """Through a symbol, the name is all that says which array is meant, and
    an index is wrong whichever of two arrays of one name is meant only past
    the larger."""
    diagnostics = _bounds_diagnostics(_namesakes(
        lambda b, larger, smaller: b.load(_Symbol('own'), index,
                                          space=MemSpace.REGISTER)))
    assert bool(diagnostics) == reported, diagnostics
