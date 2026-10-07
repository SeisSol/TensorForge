# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where the threads of a multiplication meet, decided on the final body.

`pir.barriers` reads the order the accesses ended up in and puts a barrier in
front of the first statement that needs one: a read of shared memory written
since the last barrier, a write over bytes read since then -- the buffer's own
or another one the allocation put on the same bytes -- a store on either side
of a clearing store, and a value one lane stores for all of them.  These pin
each rule on the smallest body that shows it, the place it goes, and what it
does not do: put a barrier where nothing is outstanding.
"""

from __future__ import annotations

from tensorforge.backend.pir.barriers import BLOCK, MULT, Arena, place_barriers
from tensorforge.backend.pir.build import IRBuilder, barrier_stmt
from tensorforge.backend.pir.core import (BOOL, INDEX, MemSpace, Op,
                                          Participants, Uniformity)
from tensorforge.common.basic_types import Datatype

ARENA = Arena({'shrMem': (MULT, 0), 'block': (BLOCK, 0)})
WAVE = 32


def _make(handoff, block):
    if block:
        return barrier_stmt(Participants.BLOCK, WAVE, handoff=handoff)
    return barrier_stmt(Participants.MULT, WAVE, 16, handoff)


def _place(body, handoff=frozenset()):
    return place_barriers(tuple(body), arena=ARENA, make_barrier=_make,
                          arrival=Uniformity.MULT, handoff=handoff)


def _builder():
    return IRBuilder(fptype=Datatype.F32, arena='shrMem')


def _ops(body, depth=0):
    """The statements as `(depth, op)`, nested regions flattened."""
    out = []
    for s in body:
        out.append((depth, s.op))
        for r in s.regions:
            out.extend(_ops(r.body, depth + 1))
    return out


def _barriers(body):
    return [k for k, (_, op) in enumerate(_ops(body)) if op == Op.BARRIER]


def _between(body, first, second):
    """Whether a barrier sits between the first statement `first` picks and
    the first one after it `second` picks, in the flattened order."""
    flat = []

    def walk(stmts):
        for s in stmts:
            flat.append(s)
            for r in s.regions:
                walk(r.body)
    walk(body)
    i = next(k for k, s in enumerate(flat) if first(s))
    j = next(k for k, s in enumerate(flat) if k > i and second(s))
    return any(s.op == Op.BARRIER for s in flat[i + 1:j])


def _is(op):
    return lambda s: s.op == op


def _flatten(body):
    out = []
    for s in body:
        out.append(s)
        for r in s.regions:
            out.extend(_flatten(r.body))
    return out


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #

def test_a_read_of_what_another_lane_wrote_waits_for_it():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.store(buf, b.const(1.0), lane)
    b.load(buf, lane, hint='d')
    out = _place(b.finish())
    assert _between(out, _is(Op.STORE), _is(Op.LOAD))


def test_two_reads_need_nothing():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.load(buf, lane, hint='d')
    b.load(buf, lane, hint='d')
    assert _barriers(_place(b.finish())) == []


def test_a_write_over_bytes_read_since_the_last_barrier_waits():
    """Another buffer, on the bytes the allocation gave both."""
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='a', offset=0)
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c', offset=16)
    b.load(a, lane, hint='d')
    b.store(c, b.const(1.0), lane)
    assert _between(_place(b.finish()), _is(Op.LOAD), _is(Op.STORE))


def test_a_write_over_other_bytes_needs_nothing():
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='a', offset=0)
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c', offset=32)
    b.load(a, lane, hint='d')
    b.store(c, b.const(1.0), lane)
    assert _barriers(_place(b.finish())) == []


def test_two_stores_need_nothing_by_themselves():
    """The cells two stores of one buffer share are the cells something
    reads in between, and that read is the reason."""
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.store(buf, b.const(1.0), lane)
    b.store(buf, b.const(2.0), lane)
    assert _barriers(_place(b.finish())) == []


def test_a_store_after_a_clearing_store_waits():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.mark('clears', buf)
    b.store(buf, b.const(0.0), lane)
    b.mark('cleared', buf)
    b.store(buf, b.const(2.0), lane)
    out = _place(b.finish())
    assert len(_barriers(out)) == 1
    assert _between(out, lambda s: s.op == Op.MARK
                    and s.attr('mark') == 'cleared', _is(Op.STORE))


def test_a_clearing_store_after_a_store_waits():
    """The zeros are what the buffer holds where nothing was computed, and a
    store from another lane landing after them would put its value back."""
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.store(buf, b.const(1.0), lane)
    b.mark('clears', buf)
    b.store(buf, b.const(0.0), lane)
    b.mark('cleared', buf)
    out = _place(b.finish())
    assert len(_barriers(out)) == 1
    assert _between(out, _is(Op.STORE), lambda s: s.op == Op.MARK
                    and s.attr('mark') == 'clears')


def test_a_value_one_lane_stores_for_all_of_them_is_handed_off():
    b = _builder()
    lane = b.thread_id('x')
    owner = object()
    buf = b.alloc(Datatype.F32, (1,), MemSpace.SHARED, hint='one', offset=0,
                  identity=owner)
    with b.if_(b.op('eq', BOOL, lane, 0, hint='g')):
        b.store(buf, b.const(1.0), 0)
    b.load(buf, 0, hint='d')
    out = _place(b.finish(), handoff=frozenset({owner}))
    found = [s for s in _flatten(out) if s.op == Op.BARRIER]
    assert len(found) == 1 and found[0].attr('handoff')


def test_the_block_s_memory_is_met_by_a_block_barrier():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='op', offset=0,
                  arena='block')
    b.store(buf, b.const(1.0), lane)
    b.load(buf, lane, hint='d')
    found = [s for s in _flatten(_place(b.finish())) if s.op == Op.BARRIER]
    assert len(found) == 1
    assert found[0].attr('participants') == Participants.BLOCK


def test_a_statement_that_says_nothing_is_taken_to_touch_everything():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.store(buf, b.const(1.0), lane)
    b('opaque();')
    assert _between(_place(b.finish()), _is(Op.STORE), _is(Op.RAWSTMT))


def test_the_wait_is_the_write_of_an_asynchronous_copy():
    """The copy lands at the wait, visible to its own lane: the barrier goes
    behind the wait, and one between the issue and the wait fences nothing."""
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    src = b.alloc(Datatype.F32, (32,), MemSpace.GLOBAL, hint='g')
    tok = b.copy_async(buf, src, dst_index=(lane,), src_index=(lane,))
    b.wait(tok)
    b.load(buf, lane, hint='d')
    out = _place(b.finish())
    assert _between(out, _is(Op.WAIT), _is(Op.LOAD))
    assert not _between(out, _is(Op.COPY_ASYNC), _is(Op.WAIT))


# --------------------------------------------------------------------------- #
# Where it goes
# --------------------------------------------------------------------------- #

def test_a_barrier_already_there_counts():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.store(buf, b.const(1.0), lane)
    b.barrier(Participants.MULT, threads=16)
    b.load(buf, lane, hint='d')
    assert len(_barriers(_place(b.finish()))) == 1


def test_a_barrier_goes_in_front_of_a_block_some_lanes_skip():
    """A guard on the lane is entered by some lanes only, so a barrier inside
    it would wait for lanes that never arrive: it goes in front of the
    guard."""
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    b.store(buf, b.const(1.0), lane)
    with b.if_(b.op('lt', BOOL, lane, 8, hint='g')):
        b.load(buf, lane, hint='d')
    ops = _ops(_place(b.finish()))
    barrier = next(k for k, (_, op) in enumerate(ops) if op == Op.BARRIER)
    guard = next(k for k, (_, op) in enumerate(ops) if op == Op.IF)
    assert barrier < guard and ops[barrier][0] == 0


def test_what_a_loop_leaves_for_its_next_iteration_is_met_inside_it():
    """Read at the top of the body, written at its bottom: the write of one
    iteration meets the read of the next, around the back edge, and only a
    barrier inside the body can stand between them."""
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    with b.for_(0, 4, 1):
        b.load(buf, lane, hint='d')
        b.store(buf, b.const(1.0), lane)
    ops = _ops(_place(b.finish()))
    assert any(op == Op.BARRIER and depth == 1 for depth, op in ops), ops


def test_nothing_outstanding_at_the_end_of_an_iteration_needs_nothing_inside():
    b = _builder()
    lane = b.thread_id('x')
    buf = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='buf', offset=0)
    with b.for_(0, 4, 1):
        b.load(buf, lane, hint='d')
    assert _barriers(_place(b.finish())) == []


def test_an_index_computation_is_no_access():
    b = _builder()
    lane = b.thread_id('x')
    b.op('add', INDEX, lane, 1, hint='a')
    assert _barriers(_place(b.finish())) == []
