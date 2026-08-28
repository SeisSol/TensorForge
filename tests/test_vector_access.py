# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A wide access is an access, not a string.

`load_linear` and `store_linear` have taken a `vec` argument all along, and it
has always left the structured path: `pir_buffer` was consulted only for
`vec == 1`, so a vectorised read became a `load_expr` around a hand-formatted
reinterpret cast. The buffer was then not an operand of anything, which costs
every pass its view of the one access that moves the most bytes -- CSE cannot
match two of them, LICM cannot hoist one, the scratch checker cannot see the
window it touches, and liveness cannot see the read at all.

The width itself needed nothing new. `ScalarType.length` is where a value
spanning several consecutive elements already lived, and `LaneAxis` says so
in as many words: packing is a vector type over the slot dimension, not a
lane axis. The ESIMD emitter already reads it -- `span * (length or 1)` is
its `simd<>` width. All that was missing was an emitter that spells a
subscript for a value wider than the buffer's element.

These tests do not run through the corpus, and cannot: `GlbToRegLoader`
still iterates `for g in [1]` with the wider widths commented out, so nothing
generated today takes this path. That is the reason to pin it here rather
than in a snapshot -- a snapshot of a path nothing reaches proves nothing,
and this is the path the vectorisation work turns on next.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BufferType, MemSpace, Op,
                                          ScalarType)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)
F32X2 = ScalarType(Datatype.F32, 2)
F32X4 = ScalarType(Datatype.F32, 4)


def builder():
    return IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 512))


def emitted(body):
    lines = []
    Emitter(lines.append).run(body)
    return '\n'.join(lines)


def buf(b, n=64, space=MemSpace.SHARED):
    return b.alloc(Datatype.F32, (n,), space=space)


# --------------------------------------------------------------------------- #
# The spelling
# --------------------------------------------------------------------------- #

def test_a_scalar_load_is_a_plain_subscript():
    """The unchanged case, stated so the vector one is visibly a departure."""
    b = builder()
    v = b.load(buf(b), 'i', type_=F32)
    b.store(buf(b), v, 'j')
    text = emitted(b.finish())
    assert '[i]' in text
    assert ')&' not in text


@pytest.mark.parametrize('t,length', [(F32X2, 2), (F32X4, 4)])
def test_a_vector_load_reinterprets_the_buffer(t, length):
    b = builder()
    b.load(buf(b), 'i', type_=t, hint='lin')
    text = emitted(b.finish())
    # Through a pointer of the wider type: the buffer is typed by its
    # element, so a subscript of it cannot name `length` of them.
    assert ')&' in text and '[i]' in text
    assert str(length) in text


def test_a_vector_store_reinterprets_the_destination_too():
    """The width comes off the *stored value*, not off the buffer.

    A store has no target value to read a type from, so taking it from the
    buffer would give the element type and silently narrow every wide write
    to its first component -- the failure mode with no diagnostic.
    """
    b = builder()
    v = b.load(buf(b), 'i', type_=F32X4, hint='lin')
    b.store(buf(b), v, 'j')
    text = emitted(b.finish())
    store_line = [l for l in text.split('\n') if '[j]' in l]
    assert store_line and ')&' in store_line[0]


# --------------------------------------------------------------------------- #
# What it buys: the access is visible again
# --------------------------------------------------------------------------- #

def test_a_vector_load_is_an_op_with_the_buffer_as_an_operand():
    """The point of the change, and the part a text diff cannot show.

    Same emitted characters as the old cast string; a completely different
    IR. `load_expr` produced a value with no buffer operand and no recorded
    access, so nothing downstream could tell which memory it read.
    """
    b = builder()
    a = buf(b)
    b.load(a, 'i', type_=F32X4, hint='lin')
    body = b.finish()
    loads = [s for s in body if s.op == Op.LOAD]
    assert len(loads) == 1
    assert a in loads[0].args
    assert loads[0].accesses


def test_the_width_rides_on_the_type_not_on_a_second_field():
    """No new state: a wide value is a `ScalarType` with a length."""
    b = builder()
    v = b.load(buf(b), 'i', type_=F32X4, hint='lin')
    assert isinstance(v.type, ScalarType)
    assert v.type.is_vector and v.type.length == 4
    assert v.type.base is Datatype.F32
