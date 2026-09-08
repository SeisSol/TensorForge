# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Masking a value to zero is a select, not a branch.

The expansion of a compressed operand into a dense image needs to write zero
where the pattern has nothing.  Which lane that applies to is known only at
run time, so the choice is per lane -- and how it is spelled decides whether
it costs a predicate or a divergent branch.

`if_else` is what the sparse compute path reaches for today, and it emits an
`if`: a 16x16 band operand comes out with 144 more of them than the dense
build of the same shape, all inside the inner loop.  `select` emits `a ? b :
c`, which the compiler predicates, and it has been in the emitter the whole
time with nothing producing one.

So this pins the primitive before anything is built on it: that a select
survives to a ternary, that it does not become a branch on the way, and that
it takes its operands the way an if/else would have yielded them.
"""

from __future__ import annotations

from tensorforge.backend.pir import (BOOL, IRBuilder, MemSpace, ScalarType,
                                     emit)
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)


def _masked_load(builder, array, index, guard):
    """The shape the expansion wants: a value where occupied, zero elsewhere."""
    loaded = builder.load(array, index, hint='stored')
    return builder.op('select', F32, guard, loaded, builder.const(0.0),
                      hint='masked')


def _emit(builder) -> str:
    writer = Writer()
    emit(builder.finish(), writer)
    return writer.get_src()


def test_select_becomes_a_ternary():
    builder = IRBuilder(fptype=Datatype.F32)
    image = builder.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='image')
    lane = builder.thread_id('x')
    occupied = builder.op('lt', BOOL, lane, 9, hint='occupied')
    builder.store(image, _masked_load(builder, image, lane, occupied), 0)

    src = _emit(builder)
    assert '?' in src and ':' in src, src
    # The point of the exercise: no control flow, so nothing to diverge on.
    assert 'if' not in src, src


def test_the_zero_arm_is_a_literal_and_not_a_load():
    """Nothing is read where the pattern has nothing.

    A masked *load* would still address the compressed buffer on every lane,
    including the ones whose cell is not stored -- past the end of it for the
    last word of the image.  The select takes an already-loaded value and a
    constant, so the arm that is not taken reads nothing.
    """
    builder = IRBuilder(fptype=Datatype.F32)
    image = builder.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='image')
    lane = builder.thread_id('x')
    occupied = builder.op('lt', BOOL, lane, 4, hint='occupied')
    builder.store(image, _masked_load(builder, image, lane, occupied), 0)

    src = _emit(builder)
    ternary = [line for line in src.splitlines() if '?' in line]
    assert len(ternary) == 1, src
    assert '0.0' in ternary[0], ternary[0]


def test_if_else_still_branches():
    """The contrast, so the difference is recorded rather than remembered."""
    builder = IRBuilder(fptype=Datatype.F32)
    image = builder.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='image')
    lane = builder.thread_id('x')
    occupied = builder.op('lt', BOOL, lane, 9, hint='occupied')

    handle = builder.if_else(occupied, (F32,))
    with handle.then():
        handle.yield_(builder.load(image, lane, hint='stored'))
    with handle.otherwise():
        handle.yield_(builder.const(0.0))
    builder.store(image, handle.result, 0)

    assert 'if' in _emit(builder)
