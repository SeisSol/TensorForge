# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a wait may move past, and what pins it.

A pass that reorders is only as good as the cases where it declines, so most of
this file is the declining.  Each test names the reason, because "the wait
stayed put" is the same observation whether the predicate was right or merely
timid.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import verify, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Effect, MemSpace, Op
from tensorforge.backend.pir.schedule import can_reorder, overlap, sink_waits
from tensorforge.common.basic_types import Datatype


def _builder():
    return IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 64))


def _ops(body):
    return [s.op for s, _ in walk(body)]


def test_a_wait_sinks_past_independent_work():
    b = _builder()
    dst = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s')
    src = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    other = b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, hint='r')
    lane = b.thread_id('x')
    tok = b.copy_async(dst, src, dst_index=(lane,), src_index=(lane,))
    b.wait(tok)
    for k in range(3):
        b.load(other, k, hint='ind')
    body = b.finish()
    verify(body)

    assert overlap(body) == {tok.id: 0}
    sunk = sink_waits(body)
    verify(sunk)
    assert overlap(sunk) == {tok.id: 3}, (
        "three register reads touch a different buffer in a different space, "
        "so every one of them is overlap the transfer should have had")
    assert _ops(sunk)[-1] is Op.WAIT


def test_a_wait_does_not_sink_past_a_read_of_what_it_released():
    """The binding case, and the reason to wait at all."""
    b = _builder()
    dst = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s')
    src = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    other = b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, hint='r')
    lane = b.thread_id('x')
    tok = b.copy_async(dst, src, dst_index=(lane,), src_index=(lane,))
    b.wait(tok)
    b.load(dst, lane, hint='use')          # reads what the copy wrote
    b.load(other, 0, hint='ind')
    body = b.finish()
    verify(body)

    sunk = sink_waits(body)
    verify(sunk)
    assert overlap(sunk) == {tok.id: 0}, (
        "the copy and the wait carry the same accesses, so a read of the "
        "destination conflicts with the wait and pins it")


def test_a_wait_does_not_sink_past_a_barrier():
    b = _builder()
    dst = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s')
    src = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    other = b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, hint='r')
    lane = b.thread_id('x')
    tok = b.copy_async(dst, src, dst_index=(lane,), src_index=(lane,))
    b.wait(tok)
    b.barrier()
    b.load(other, 0, hint='ind')
    body = b.finish()
    verify(body)

    sunk = sink_waits(body)
    verify(sunk)
    assert overlap(sunk) == {tok.id: 0}, (
        "a barrier orders every thread, not just this one's memory, so no "
        "access analysis can license crossing it")


def test_a_wait_does_not_sink_into_a_loop():
    b = _builder()
    dst = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s')
    src = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    other = b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, hint='r')
    lane = b.thread_id('x')
    tok = b.copy_async(dst, src, dst_index=(lane,), src_index=(lane,))
    b.wait(tok)
    with b.For('int32_t i = 0; i < 4; ++i'):
        b.load(other, 0, hint='inloop')
    body = b.finish()
    verify(body)

    sunk = sink_waits(body)
    verify(sunk)
    assert overlap(sunk) == {tok.id: 0}, (
        "a loop body may touch what its header does not say; treating it as a "
        "wall is the cheap answer and widening it is a later pass")


def test_the_predicate_refuses_two_writes_to_one_buffer():
    b = _builder()
    reg = b.alloc(Datatype.F32, (8,), MemSpace.REGISTER, hint='r')
    lane = b.thread_id('x')
    b.store(reg, lane, 0)
    b.store(reg, lane, 1)
    stmts = [s for s, _ in walk(b.finish()) if s.op is Op.STORE]
    assert len(stmts) == 2
    assert not can_reorder(stmts[0], stmts[1]), (
        "two writes to the same buffer conflict even though neither reads the "
        "other's result")
