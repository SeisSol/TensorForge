# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle's lockstep with roles that wait on each other
(`kernel_eval.Lockstep`, `kernel_eval.COROUTINE_ARMS`): the two arms of an
`if` run as coroutines -- an arm whose next statement is a wait on a ring's
mbarrier for a phase that has not completed hands over to the other arm,
and is resumed once the other has gone as far as it can.  Before, an arm
ran whole before the other, so a role could only wait for what the arm
ahead of it in that order had done: a producer running ahead of its
consumers by more trips than the ring has slots, or consumers whose arm
comes first in a trip, hung the oracle where the device runs.  Where
neither arm can go on, the run is still the hang it would be, and the wait
refuses as it did.

What is checked here, on the ring of `test_mbarrier_ring` (a producer warp,
two consumer warps, a slot a trip of `WIDTH` floats): roles that are loops
of their own over the trips -- the producer filling the ring as far as the
consumers have freed it -- and consumers whose arm comes first in each trip
hang the old lockstep and run under the new one, every float delivered and
no access unordered; a ring never released, and roles that wait for each
other first, hang under both.
"""

from __future__ import annotations

import pytest

import test_mbarrier_ring as tmr
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.build import IRBuilder, RingSlot, Role
from tensorforge.backend.pir.core import (BOOL, INDEX, MemSpace,
                                          Participants, Uniformity)
from tensorforge.common.basic_types import Datatype
from tensorforge.reference import kernel_eval

F32 = Datatype.F32
WIDTH = tmr.WIDTH


def _ring_body(trips=5, slots=2, looped=True, consumers_first=False,
               release=True, consumers_wait_first=False):
    """The producer and its consumers over a ring of `slots`.  `looped`:
    each role a loop of its own over the trips, the roles entered once --
    the producer fills a slot as soon as its consumers have freed it, so
    from its trip `slots` on it waits for their release of a trip it is
    ahead of.  Otherwise the trip loop around the roles, as in
    `test_mbarrier_ring._body`, `consumers_first` putting the consumers'
    arm ahead of the producer's.  `release` off: the consumers never free
    a slot.  `consumers_wait_first`: the consumers wait for the empty side
    before anything -- for each other, as the producer waits on it too."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    ring = b.mbarrier_ring('pc', slots, full=1, empty=2,
                           leader='threadIdx.x == 0 && threadIdx.y == 0')
    slot_buf = b.alloc(F32, (slots * WIDTH,), MemSpace.SHARED, hint='slot',
                       extern='slot', place_align=4)
    src = b.alloc(F32, (trips * WIDTH,), MemSpace.GLOBAL, hint='g',
                  extern='m1')
    out = b.alloc(F32, (trips * WIDTH,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    x, y = b.thread_id('x'), b.thread_id('y')
    lane = b.op('add', INDEX, x, b.op('mul', INDEX, y, 32), hint='c')

    def produce(t):
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        issue = b.op('eq', BOOL, x, 0, hint='issue')
        b.ring_wait(ring, 'empty', slot,
                    b.op('sub', INDEX, 1, parity, hint='free'),
                    writes=[slot_buf])
        tok = b.copy_bulk(slot_buf, src, barrier=RingSlot(ring, slot),
                          dst_index=[at],
                          src_index=[b.op('mul', INDEX, t, WIDTH)],
                          elems=WIDTH, predicate=issue)
        b.ring_arrive(ring, 'full', slot, issue, tx=WIDTH * 4, tokens=[tok])

    def consume(t):
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        if consumers_wait_first:
            b.ring_wait(ring, 'empty', slot, parity, writes=[slot_buf])
        b.ring_wait(ring, 'full', slot, parity, writes=[slot_buf])
        value = b.load(slot_buf, b.op('add', INDEX, at, lane), hint='v')
        b.store(out, value, b.op('add', INDEX, b.op('mul', INDEX, t, WIDTH),
                                 lane))
        b.barrier(Participants.WAVE)
        if release:
            b.ring_arrive(ring, 'empty', slot,
                          b.op('eq', BOOL, x, 0, hint='first'))

    def trips_of(step):
        with b.for_(0, trips, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
            step(loop.induction)

    producer_role = Role('producer', 32, tmr.IDENT)
    consumer_role = Role('consumers', WIDTH, tmr.IDENT)
    if looped:
        h = b.roles(b.op('ge', BOOL, y, 2, hint='prod'), producer_role,
                    consumer_role)
        with h.then():
            trips_of(produce)
        with h.otherwise():
            trips_of(consume)
    else:
        with b.for_(0, trips, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
            t = loop.induction
            if consumers_first:
                h = b.roles(b.op('lt', BOOL, y, 2, hint='cons'),
                            consumer_role, producer_role)
                with h.then():
                    consume(t)
                with h.otherwise():
                    produce(t)
            else:
                h = b.roles(b.op('ge', BOOL, y, 2, hint='prod'),
                            producer_role, consumer_role)
                with h.then():
                    produce(t)
                with h.otherwise():
                    consume(t)
    built = b.finish()
    placed, _ = allocate(tmr._placed(built), arenas={'shrMem': MULT})
    return tmr._emit(placed)


def _old_lockstep(monkeypatch):
    monkeypatch.setattr(kernel_eval, 'COROUTINE_ARMS', False)


@pytest.mark.parametrize('slots,trips', [(1, 3), (2, 5), (3, 7)])
def test_roles_that_are_loops_of_their_own_run_as_on_the_device(slots,
                                                                 trips):
    """The producer fills the ring as far as its consumers have freed it,
    the consumers drain it as far as it is filled -- each role a loop over
    all the trips, waiting on the other's mbarriers inside it: every float
    arrives, and no access is unordered."""
    lines = _ring_body(trips=trips, slots=slots)
    bad, races = tmr._oracle(lines, trips=trips)
    assert not bad and not races, (bad[:4], races[:3])


def test_the_old_lockstep_hung_on_roles_that_are_loops(monkeypatch):
    """An arm whole before the other: the producer's loop waits, at its
    trip `slots`, for a release its consumers' arm -- behind it -- has not
    run yet.  The device runs it; the old oracle refused it as a hang."""
    lines = _ring_body(trips=5, slots=2)
    _old_lockstep(monkeypatch)
    with pytest.raises(kernel_eval.Abort, match='never return'):
        tmr._oracle(lines, trips=5)


def test_consumers_ahead_of_their_producer_in_a_trip(monkeypatch):
    """The trip loop around the roles, the consumers' arm first: they wait
    for the full side the producer's arm, behind theirs, completes.  The
    old lockstep hung there; the arms as coroutines deliver every float."""
    lines = _ring_body(trips=5, slots=2, looped=False, consumers_first=True)
    bad, races = tmr._oracle(lines, trips=5)
    assert not bad and not races, (bad[:4], races[:3])
    _old_lockstep(monkeypatch)
    with pytest.raises(kernel_eval.Abort, match='never return'):
        tmr._oracle(lines, trips=5)


def test_the_producer_ahead_of_its_consumers_in_a_trip_is_unchanged(
        monkeypatch):
    """`test_mbarrier_ring`'s order -- the producer's arm first in each
    trip -- runs as before under both: no arm waits, and the arms run whole
    one after the other."""
    lines = _ring_body(trips=5, slots=2, looped=False)
    for coroutines in (True, False):
        monkeypatch.setattr(kernel_eval, 'COROUTINE_ARMS', coroutines)
        bad, races = tmr._oracle(lines, trips=5)
        assert not bad and not races, (coroutines, bad[:4], races[:3])


@pytest.mark.parametrize('coroutines', [True, False])
def test_a_ring_never_released_still_hangs(coroutines, monkeypatch):
    """The consumers never free a slot: the producer waits at its trip
    `slots`, the consumers then for a round it never fills -- every arm
    waits, and the wait refuses as the hang it is, under both."""
    monkeypatch.setattr(kernel_eval, 'COROUTINE_ARMS', coroutines)
    with pytest.raises(kernel_eval.Abort, match='never return'):
        tmr._oracle(_ring_body(trips=5, slots=2, release=False), trips=5)


@pytest.mark.parametrize('coroutines', [True, False])
def test_roles_waiting_for_each_other_first_still_hang(coroutines,
                                                       monkeypatch):
    """The consumers wait for the empty side's phase of the round before
    they take the full side's, while the producer waits for the other
    parity of it: neither phase ever completes."""
    monkeypatch.setattr(kernel_eval, 'COROUTINE_ARMS', coroutines)
    with pytest.raises(kernel_eval.Abort, match='never return'):
        tmr._oracle(_ring_body(trips=3, slots=1, consumers_wait_first=True),
                    trips=3)
