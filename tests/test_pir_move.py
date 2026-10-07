# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A transfer issued ahead of where its builder wrote it.

`pir.move.move_loads` on bodies small enough to see what moved: how far a
transfer goes, what it takes along, what it leaves where it was -- and mostly
what stops it, since a transfer moved past a write of what it reads computes
from a value that is gone, and nothing about the kernel looks wrong.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import verify
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, SIZE, BufferType, Effect,
                                          MemSpace, Op, Participants,
                                          Uniformity)
from tensorforge.backend.pir.move import move_loads
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32


class Body:
    """A body written the way the builders write one: each transfer right in
    front of what first reads it."""

    def __init__(self):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        self.k = b.extern_value('element', SIZE, uniform=Uniformity.MULT,
                                hint='k')
        self.out = self.binding('m9', writable=True)
        self.marks = {}

    def binding(self, base, writable=False):
        b = self.b
        return b.decl_expr(
            f'const float *const __restrict__ glb_{base}',
            f'&{base}[{{0}} * 64 + 0]',
            BufferType(F32, (64,), MemSpace.GLOBAL, readonly=not writable),
            base, args=(self.k,), kind=Effect.READ, space=MemSpace.GLOBAL,
            hint=f'glb_{base}', extern=f'glb_{base}')

    def register(self, src, name, *, index=None):
        """A transfer into a register array of its own, with its declaration
        in front of it; `index` is arithmetic its addresses read."""
        b = self.b
        dest = b.alloc(F32, (8,), MemSpace.REGISTER, hint=name, extern=name)
        self.marks[name] = b.Comment(f'{name} = load{{g>r}}(...)')
        with b.for_(0, 8, 1, unroll=True) as i:
            at = i.induction if index is None else b.op(
                'add', INDEX, i.induction, index, hint='at')
            b.store(dest, b.load(src, at, hint='g'), i.induction)
        return dest

    def shared(self, src, name):
        """A transfer into a shared window, and the wait behind it."""
        b = self.b
        dest = b.alloc(F32, (64,), MemSpace.SHARED, hint=name, extern=name)
        self.marks[name] = b.Comment(f'{name} = load{{g>s}}(...)')
        b.mark('defines', dest)
        lane = b.thread_id('x')
        token = b.copy_async(dest, src, dst_index=(lane,), src_index=(lane,))
        b.wait(token)
        return dest

    def use(self, buf, slot=0):
        b = self.b
        b.store(self.out, b.load(buf, 0, hint='u'), slot)

    def finish(self):
        self.body = self.b.finish()
        verify(self.body)
        return self.body

    def moved(self, distance=1):
        out = move_loads(self.body, distance=distance)
        verify(out)
        return out


def _at(body, stmt):
    return next(i for i, s in enumerate(body) if s is stmt)


def _first_write(body, buf):
    """Where the transfer filling `buf` starts: its leading comment."""
    return next(i for i, s in enumerate(body)
                if s.op is Op.FOR and any(
                    a.writes and a.base is buf
                    for x in s.regions[0].body for a in x.accesses))


def _reads(body, buf):
    return [i for i, s in enumerate(body) if s.op is Op.LOAD
            and any(a.base is buf and not a.writes for a in s.accesses)]


# --------------------------------------------------------------------------- #
# how far
# --------------------------------------------------------------------------- #

def test_a_transfer_moves_up_to_the_one_before_it():
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    r0 = t.register(m0, 'r0')
    t.use(r0)
    r1 = t.register(m1, 'r1')
    t.use(r1)
    body = t.finish()
    out = t.moved()
    assert _first_write(out, r1) < min(_reads(out, r0)), (
        'r1 is issued ahead of the read of r0')
    assert _first_write(out, r1) > _first_write(out, r0), (
        'and not ahead of r0, the transfer before it')
    assert len(out) == len(body)


@pytest.mark.parametrize('distance,ahead_of', [(1, 1), (2, 0), (3, 0)])
def test_the_distance_is_how_many_transfers_it_passes(distance, ahead_of):
    """At 1 the last transfer stops below the one before it: ahead of that
    one's read, behind the read of the one before.  Each step of distance
    takes it past one transfer more -- here, at 2, to the bindings."""
    t = Body()
    ms = [t.binding(f'm{j}') for j in range(3)]
    regs = []
    for j, m in enumerate(ms):
        regs.append(t.register(m, f'r{j}'))
        t.use(regs[-1], j)
    t.finish()
    out = t.moved(distance)
    last = _first_write(out, regs[2])
    assert last < min(_reads(out, regs[ahead_of]))
    if ahead_of:
        assert last > min(_reads(out, regs[ahead_of - 1]))
    # The transfers keep the order they were written in.
    assert sorted(regs, key=lambda r: _first_write(out, r)) == regs


def test_a_transfer_stops_below_where_its_predecessor_was_written():
    """Moved from the bottom up: the transfer above still stands where its
    builder wrote it when the one below arrives, and it leaves only after."""
    t = Body()
    m0, m1, m2 = (t.binding(f'm{j}') for j in range(3))
    r0 = t.register(m0, 'r0')
    t.use(r0, 0)
    r1 = t.register(m1, 'r1')
    t.use(r1, 1)
    r2 = t.register(m2, 'r2')
    t.use(r2, 2)
    t.finish()
    out = t.moved()
    assert (_first_write(out, r0) < _first_write(out, r1)
            < min(_reads(out, r0)) < _first_write(out, r2)
            < min(_reads(out, r1))), 'each one transfer ahead, not more'


def test_the_wait_stays_where_the_transfer_was():
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    s0 = t.shared(m0, 's0')
    t.use(s0)
    s1 = t.shared(m1, 's1')
    t.use(s1)
    body = t.finish()
    out = t.moved()
    copies = [i for i, s in enumerate(out) if s.op is Op.COPY_ASYNC]
    waits = [i for i, s in enumerate(out) if s.op is Op.WAIT]
    assert copies[1] < waits[0], 'the second copy is issued before the first wait'
    # The waits keep their order and their places relative to the reads.
    assert [out[i] for i in waits] == [s for s in body if s.op is Op.WAIT]
    assert waits[1] < min(_reads(out, s1)) and waits[1] > min(_reads(out, s0))


# --------------------------------------------------------------------------- #
# what stops it
# --------------------------------------------------------------------------- #

def test_a_store_to_what_it_reads_stops_it():
    """The element's operand written before the transfer reads it: moved
    above the store, the transfer reads the value from before it."""
    t = Body()
    m0 = t.binding('m0')
    m1 = t.binding('m1', writable=True)
    r0 = t.register(m0, 'r0')
    t.use(r0)
    store = t.b.store(m1, 1.0, 0)
    r1 = t.register(m1, 'r1')
    t.use(r1)
    t.finish()
    out = t.moved()
    assert _first_write(out, r1) > _at(out, store)


def test_a_read_of_its_buffer_stops_it():
    """What the buffer held before is read ahead of the transfer; a transfer
    moved above that read overwrites it first."""
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    s0 = t.shared(m0, 's0')
    t.use(s0)
    s1 = t.b.alloc(F32, (64,), MemSpace.SHARED, hint='s1', extern='s1')
    early = t.b.load(s1, 3, hint='early')
    t.b.store(t.out, early, 7)
    t.b.mark('defines', s1)
    lane = t.b.thread_id('x')
    token = t.b.copy_async(s1, m1, dst_index=(lane,), src_index=(lane,))
    t.b.wait(token)
    t.use(s1)
    t.finish()
    out = t.moved()
    copy = next(i for i, s in enumerate(out) if s.op is Op.COPY_ASYNC
                and s.args[0] is s1)
    read = next(i for i, s in enumerate(out) if s.op is Op.LOAD
                and s.args[0] is s1)
    assert copy > read


@pytest.mark.parametrize('kind', ['shared', 'register'])
def test_a_barrier_stops_a_shared_transfer_and_not_a_register_one(kind):
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    first = t.register(m0, 'r0')
    t.use(first)
    barrier = t.b.barrier(Participants.MULT, threads=32)
    if kind == 'shared':
        buf = t.shared(m1, 's1')
    else:
        buf = t.register(m1, 'r1')
    t.use(buf)
    t.finish()
    out = t.moved()
    at = (next(i for i, s in enumerate(out) if s.op is Op.COPY_ASYNC)
          if kind == 'shared' else _first_write(out, buf))
    if kind == 'shared':
        assert at > _at(out, barrier)
    else:
        assert at < _at(out, barrier)


def test_a_pointer_binding_stops_it():
    """Bound behind the transfer above, another operand's pointer: the
    transfer below does not go ahead of it."""
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    r0 = t.register(m0, 'r0')
    t.use(r0)
    m2 = t.binding('m2')
    r1 = t.register(m1, 'r1')
    t.use(r1)
    t.use(t.register(m2, 'r2'))
    t.finish()
    out = t.moved()
    bound = next(s for s in out if m2 in s.target)
    assert _first_write(out, r1) > _at(out, bound)


# --------------------------------------------------------------------------- #
# what it takes along
# --------------------------------------------------------------------------- #

def test_it_takes_its_declaration_and_its_arithmetic_along():
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    r0 = t.register(m0, 'r0')
    t.use(r0)
    base = t.b.extern_value('base', INDEX, uniform=Uniformity.MULT,
                            hint='base')
    offset = t.b.op('add', INDEX, base, 8, hint='offset')
    r1 = t.register(m1, 'r1', index=offset)
    t.use(r1)
    t.finish()
    out = t.moved()
    alloc = next(i for i, s in enumerate(out) if s.op is Op.ALLOC
                 and s.target[0] is r1)
    arithmetic = next(i for i, s in enumerate(out) if offset in s.target)
    write = _first_write(out, r1)
    assert alloc < write and arithmetic < write
    assert write < min(_reads(out, r0)), 'and it moved all the same'


def test_a_transfer_does_not_leave_its_list():
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    r0 = t.register(m0, 'r0')
    t.use(r0)
    flag = t.b.extern_value('flag', None, hint='flag')
    with t.b.if_(flag) as _:
        r1 = t.register(m1, 'r1')
        t.use(r1)
    t.finish()
    out = t.moved()
    guard = next(s for s in out if s.op is Op.IF)
    assert any(s.op is Op.FOR for s in guard.regions[0].body), (
        'the transfer is still inside the guard')


def test_where_nothing_may_move_the_body_is_the_same():
    t = Body()
    r0 = t.register(t.binding('m0'), 'r0')
    t.use(r0)
    body = t.finish()
    assert t.moved() is body


def test_a_distance_below_one_is_refused():
    t = Body()
    t.finish()
    with pytest.raises(ValueError):
        move_loads(t.body, distance=0)
