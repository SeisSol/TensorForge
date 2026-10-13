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
                                          Uniformity, walk_stmts)
from tensorforge.backend.pir.move import move_loads
from tensorforge.backend.pir.transfers import one_pipeline
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
        b.Comment(f'wait({name} = load{{g>s}}(...))')
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


# --------------------------------------------------------------------------- #
# out of a counted loop
# --------------------------------------------------------------------------- #

def _loop(body):
    return next(s for s in body if s.op is Op.FOR and s.attr('rolled'))


def _inside(loop, buf):
    """Whether anything in `loop` still writes `buf`."""
    return any(a.writes and a.base is buf
               for x in walk_stmts(loop.regions[0].body) for a in x.accesses)


def test_a_transfer_the_same_every_trip_leaves_the_loop():
    """A merged run's body copies an operand every iteration shares as well
    as the one its counter selects; the first is one copy, ahead of the
    loop, and the reads in the loop stay where they are."""
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        r0 = t.register(m0, 'r0')
        t.use(r0, trip.induction)
        r1 = t.register(m1, 'r1', index=trip.induction)
        t.use(r1, trip.induction)
    t.finish()
    out = t.moved()
    loop = _loop(out)
    assert _first_write(out, r0) < _at(out, loop)
    assert not _inside(loop, r0)
    assert _inside(loop, r1), 'the copy the counter selects stays'
    assert _reads(loop.regions[0].body, r0), 'and so do the reads'


def test_its_declaration_and_its_arithmetic_leave_with_it():
    t = Body()
    m0 = t.binding('m0')
    base = t.b.extern_value('base', INDEX, uniform=Uniformity.MULT,
                            hint='base')
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        offset = t.b.op('add', INDEX, base, 8, hint='offset')
        r0 = t.register(m0, 'r0', index=offset)
        t.use(r0, trip.induction)
    t.finish()
    out = t.moved()
    loop = _loop(out)
    alloc = next(i for i, s in enumerate(out) if s.op is Op.ALLOC
                 and s.target[0] is r0)
    arithmetic = next(i for i, s in enumerate(out) if offset in s.target)
    assert alloc < _first_write(out, r0) < _at(out, loop)
    assert arithmetic < _first_write(out, r0)


def test_a_copy_leaves_with_its_wait():
    """Nothing the hardware carries out on its own is in flight across the
    loop: the wait goes right behind the copy -- and the comment saying what
    it waits for with it."""
    t = Body()
    m0 = t.binding('m0')
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        s0 = t.shared(m0, 's0')
        t.use(s0, trip.induction)
    t.finish()
    out = t.moved()
    loop = _loop(out)
    copy = next(i for i, s in enumerate(out) if s.op is Op.COPY_ASYNC)
    wait = next(i for i, s in enumerate(out) if s.op is Op.WAIT)
    assert copy < wait < _at(out, loop)
    assert out[wait - 1].text.startswith('// wait(s0')
    assert not any(x.op in (Op.COPY_ASYNC, Op.WAIT) or x.text
                   for x in walk_stmts(loop.regions[0].body))


def test_a_transfer_through_what_the_trip_binds_stays():
    """The member the counter selects is bound in the loop: a copy through
    that binding is a different copy every trip."""
    t = Body()
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        member = t.binding('m5')
        r0 = t.register(member, 'r0')
        t.use(r0, trip.induction)
    t.finish()
    assert _inside(_loop(t.moved()), r0)


def test_a_transfer_from_what_the_loop_writes_stays():
    t = Body()
    m1 = t.binding('m1', writable=True)
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        r0 = t.register(m1, 'r0')
        t.use(r0, trip.induction)
        t.b.store(m1, 1.0, trip.induction)
    t.finish()
    assert _inside(_loop(t.moved()), r0)


def test_a_buffer_written_by_something_else_in_the_loop_keeps_its_transfer():
    t = Body()
    m0 = t.binding('m0')
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        r0 = t.register(m0, 'r0')
        t.use(r0, trip.induction)
        t.b.store(r0, 0.0, 1)
    t.finish()
    assert _inside(_loop(t.moved()), r0)


@pytest.mark.parametrize('bounds', ['empty', 'unknown'])
def test_a_loop_that_may_not_run_keeps_its_transfers(bounds):
    """A copy ahead of a loop of no trips would be one that never ran."""
    t = Body()
    m0 = t.binding('m0')
    count = (4 if bounds == 'empty' else
             t.b.extern_value('count', INDEX, uniform=Uniformity.MULT,
                              hint='count'))
    with t.b.for_(4 if bounds == 'empty' else 0, count, 1,
                  rolled=True) as trip:
        r0 = t.register(m0, 'r0')
        t.use(r0, trip.induction)
    t.finish()
    assert _inside(_loop(t.moved()), r0)


def test_a_loop_no_run_was_rolled_into_keeps_its_transfers():
    """Only a merged run's loop is written with the operands every trip
    shares in it, and only that one is measured without them."""
    t = Body()
    m0 = t.binding('m0')
    with t.b.for_(0, 4, 1) as trip:
        r0 = t.register(m0, 'r0')
        t.use(r0, trip.induction)
    t.finish()
    loop = next(s for s in t.moved() if s.op is Op.FOR
                and s.loop_bounds == (0, 4, 1))
    assert _inside(loop, r0)


@pytest.mark.parametrize('kind', ['shared', 'register'])
def test_a_barrier_in_the_loop_keeps_a_shared_copy_and_not_a_register_one(
        kind):
    t = Body()
    m0 = t.binding('m0')
    with t.b.for_(0, 4, 1, rolled=True) as trip:
        buf = (t.shared(m0, 's0') if kind == 'shared'
               else t.register(m0, 'r0'))
        t.use(buf, trip.induction)
        t.b.barrier(Participants.MULT, threads=32)
    t.finish()
    assert _inside(_loop(t.moved()), buf) is (kind == 'shared')


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


@pytest.mark.parametrize('rolled', [True, False])
def test_a_merged_run_keeps_the_reads_it_shares_and_licm_lifts_others(
        rolled):
    """A read the trips share leaves an ordinary loop (`passes.licm`) and
    stays in a merged run's: the run is rolled to keep its code one trip
    long, and a read taken ahead of it is a register held across every trip
    -- every element of a staged operand, at once.  What leaves the run is
    the staging, a transfer at a time."""
    from tensorforge.backend.pir.passes import licm
    t = Body()
    m0 = t.binding('m0')
    with t.b.for_(0, 4, 1, rolled=rolled) as trip:
        t.b.store(t.out, t.b.load(m0, 3, hint='shared'), trip.induction)
    t.finish()
    out = licm(t.body)
    loop = next(s for s in out if s.op is Op.FOR)
    inside = any(s.op is Op.LOAD for s in walk_stmts(loop.regions[0].body))
    assert inside is rolled


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #

def _issue(body, buf):
    """Where the transfer filling `buf` starts: its first write."""
    return next(i for i, s in enumerate(body)
                if any(a.writes and a.base is buf
                       for x in walk_stmts((s,)) for a in x.accesses))


def _wait(body, buf):
    """Where the copy into `buf` is waited for."""
    return next(i for i, s in enumerate(body) if s.op is Op.WAIT
                and any(a.base is buf for a in s.accesses))


def _mixed():
    """A copy into shared memory, a load into registers, and a second copy:
    each written right in front of what reads it."""
    t = Body()
    m0, m1, m2 = t.binding('m0'), t.binding('m1'), t.binding('m2')
    s0 = t.shared(m0, 's0')
    t.use(s0)
    r1 = t.register(m1, 'r1')
    t.use(r1, 1)
    s2 = t.shared(m2, 's2')
    t.use(s2, 2)
    t.finish()
    return t, s0, r1, s2


def test_with_one_pipeline_a_copy_waits_for_a_load_into_registers():
    """One pipeline: the load into r1 is the transfer before the copy into
    s2, which stops below it -- behind the wait for s0, so that one copy is
    in flight at a time."""
    t, s0, r1, s2 = _mixed()
    out = move_loads(t.body, distance=1, pipeline=one_pipeline)
    verify(out)
    assert _issue(out, r1) < _issue(out, s2)
    assert _issue(out, s2) > _wait(out, s0)


def test_a_copy_passes_a_load_into_registers_in_a_pipeline_of_its_own():
    """Two pipelines, one per memory path (the default): the load into r1 is not one of the
    copy's, so the copy into s2 goes past it -- and ahead of the wait for s0,
    the copy before it in its own pipeline: two copies in flight."""
    t, s0, r1, s2 = _mixed()
    out = move_loads(t.body, distance=1)
    verify(out)
    assert _issue(out, s0) < _issue(out, s2) < _wait(out, s0)


def test_a_load_into_registers_passes_a_copy_in_a_pipeline_of_its_own():
    """And the other way round: the load into r2 passes the copy into s1 and
    stops below r0, the load before it in its own pipeline -- ahead of the
    first read of r0."""
    t = Body()
    m0, m1, m2 = t.binding('m0'), t.binding('m1'), t.binding('m2')
    r0 = t.register(m0, 'r0')
    t.use(r0)
    s1 = t.shared(m1, 's1')
    t.use(s1, 1)
    r2 = t.register(m2, 'r2')
    t.use(r2, 2)
    t.finish()
    one = move_loads(t.body, distance=1, pipeline=one_pipeline)
    two = move_loads(t.body, distance=1)
    verify(one)
    verify(two)
    assert _issue(one, r2) > min(_reads(one, r0))
    assert _issue(two, r0) < _issue(two, r2) < min(_reads(two, r0))


# --------------------------------------------------------------------------- #
# the cover a transfer needs
# --------------------------------------------------------------------------- #

def _far():
    """A load into registers written in front of its read, thirty statements
    behind the transfer before it: `move` issues it past all of them."""
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    r1 = t.register(m1, 'r1')
    t.use(r1, 1)
    for n in range(30):
        t.b.store(t.out, float(n), 16 + n)
    r0 = t.register(m0, 'r0')
    t.use(r0)
    t.finish()
    return t, r0, r1


def _ahead(body, buf):
    """The cover of the transfer into `buf`: what runs between its last
    statement and the first read of what it filled."""
    from tensorforge.backend.pir.transfers import executed
    last = max(i for i, s in enumerate(body)
               if any(a.writes and a.base is buf
                      for x in walk_stmts((s,)) for a in x.accesses))
    read = min(_reads(body, buf))
    return executed(body[last + 1:read])


def test_a_transfer_with_more_cover_than_it_needs_goes_back_down():
    from tensorforge.backend.pir.move import cap_cover
    t, r0, r1 = _far()
    moved = move_loads(t.body, distance=2)
    assert _ahead(moved, r0) >= 30, 'move issues r0 past the thirty'
    capped = cap_cover(moved, 10)
    verify(capped)
    assert 10 <= _ahead(capped, r0) <= 11
    assert len(capped) == len(moved)


def test_a_cap_of_zero_leaves_the_body_as_it_is():
    from tensorforge.backend.pir.move import cap_cover
    t, r0, r1 = _far()
    moved = move_loads(t.body, distance=2)
    assert cap_cover(moved, 0) is moved


def test_a_capped_transfer_stays_above_one_of_its_own_pipeline():
    """Going down, a transfer keeps the order of its pipeline: it stops
    above the next transfer of its own kind."""
    from tensorforge.backend.pir.move import cap_cover
    t = Body()
    m0, m1 = t.binding('m0'), t.binding('m1')
    r0 = t.register(m0, 'r0')
    r1 = t.register(m1, 'r1')
    for n in range(30):
        t.b.store(t.out, float(n), 16 + n)
    t.use(r1, 1)
    t.use(r0)
    t.finish()
    capped = cap_cover(t.body, 5)
    verify(capped)
    assert _first_write(capped, r0) < _first_write(capped, r1)


def test_a_capped_copy_goes_down_to_its_cover_ahead_of_its_wait():
    """A copy the hardware carries out on its own is wanted at its wait,
    not at the read behind it: the cover it keeps runs up to the wait."""
    from tensorforge.backend.pir.move import cap_cover
    from tensorforge.backend.pir.transfers import executed
    t = Body()
    b = t.b
    m0 = t.binding('m0')
    s0 = b.alloc(F32, (64,), MemSpace.SHARED, hint='s0', extern='s0')
    b.Comment('s0 = load{g>s}(...)')
    b.mark('defines', s0)
    lane = b.thread_id('x')
    token = b.copy_async(s0, m0, dst_index=(lane,), src_index=(lane,))
    for n in range(30):
        b.store(t.out, float(n), 16 + n)
    b.Comment('wait(s0 = load{g>s}(...))')
    b.wait(token)
    for n in range(10):
        b.store(t.out, float(n), 50 + n)
    t.use(s0)
    t.finish()
    capped = cap_cover(t.body, 10)
    verify(capped)
    copy = next(i for i, s in enumerate(capped) if s.op is Op.COPY_ASYNC)
    wait = next(i for i, s in enumerate(capped) if s.op is Op.WAIT)
    assert 10 <= executed(capped[copy + 1:wait]) <= 11


def test_a_capped_transfer_stays_above_a_write_of_what_it_reads():
    """Going down, a transfer passes only what it may cross: a store into
    the memory it loads from holds it right above."""
    from tensorforge.backend.pir.move import cap_cover
    t = Body()
    m0 = t.binding('m0', writable=True)
    r0 = t.register(m0, 'r0')
    for n in range(10):
        t.b.store(t.out, float(n), 16 + n)
    t.b.store(m0, 1.0, 0)
    for n in range(20):
        t.b.store(t.out, float(n), 26 + n)
    t.use(r0)
    t.finish()
    capped = cap_cover(t.body, 5)
    verify(capped)
    # A global access is recorded against the operand it goes through.
    write = next(i for i, s in enumerate(capped) if s.op is Op.STORE
                 and any(a.base == 'm0' and a.writes for a in s.accesses))
    assert _first_write(capped, r0) == write - 1
