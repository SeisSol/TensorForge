# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`verify` bounds every access it can resolve against its buffer.

A register array is a fixed number of registers. An index one past either end
is a neighbouring register or a spill slot, so the access reads a value rather
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
