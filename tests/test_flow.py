# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which store every load of a body reads, lane by lane (`pir.flow`), asked
of bodies built by hand.

The relation is the ground the passes that forward through memory stand on:
a load said to read one store for certain, where it may read another, is a
value replaced by the wrong one.  So the cases here are the ways a read can
be uncertain -- a lane the store skipped, a guard the read is not under, a
loop that runs again, a write nothing can place -- and the ways it is not.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, BufferType,
                                          Effect, MemSpace, Op, ScalarType,
                                          Uniformity, walk_stmts)
from tensorforge.backend.pir.flow import INITIAL, Flow, TooLarge
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32
LANES = 32
ROWS = 24
COLS = 4
REAL = ScalarType(F32)


class Body:
    """A builder with an element's two buffers bound: `q`, `ROWS x COLS`
    of them, and `p`, one a lane."""

    def __init__(self):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        self.q = b.decl_expr(
            'float *const __restrict__ glb_m0', '&m0[{0} * 96]',
            BufferType(F32, (ROWS * COLS,), MemSpace.GLOBAL), 'm0', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
            extern='glb_m0')
        self.p = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
            BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        self.lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
        self.valid = b.op('lt', BOOL, self.lane, ROWS, hint='g')

    def element(self, j, lane=None):
        b = self.b
        return b.op('add', INDEX, self.lane if lane is None else lane,
                    b.op('mul', INDEX, j, ROWS), hint='a')

    def registers(self, name, n=COLS):
        return self.b.alloc(F32, (n,), MemSpace.REGISTER, hint=name,
                            extern=name, init='{}')

    def flow(self):
        body = self.b.finish()
        verify(body)
        self.body = body
        return Flow(body, LANES, arena='shrMem')


def _stmts(flow, op, space):
    return [s for s in walk_stmts(flow.body) if s.op == op and s.accesses
            and s.accesses[0].space is space]


def _reaches(flow, load):
    return [r for n in flow.executions(load) for r in flow.reach[n]]


def test_a_register_read_reads_the_store_of_its_trip():
    """`r[j] = f(j)` in one unrolled loop, `g(r[j])` in another: each trip
    of the second reads the store of the same trip of the first."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    with b.for_(0, COLS, 1, unroll=True) as hop:
        b.store(r, b.op('mul', REAL, 2.0, 3.0), hop.induction)
    with b.for_(0, COLS, 1, unroll=True) as hop:
        b.store(t.p, b.load(r, hop.induction), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.REGISTER)
    store, = _stmts(f, Op.STORE, MemSpace.REGISTER)
    for n in f.executions(load):
        (reach,) = f.reach[n]
        assert reach.certain and reach.mask == f.full
        w = f.insts[reach.writer]
        assert w.stmt is store and w.hops[-1][1] == f.insts[n].hops[-1][1]


def test_lanes_a_store_skips_read_what_was_there():
    """A store under `lane < ROWS` leaves the lanes past it with what the
    allocation put there, and only the lanes below read the store."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    with b.if_(t.valid):
        b.store(r, 1.0, 0)
    b.store(t.p, b.load(r, 0), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.REGISTER)
    stored = (1 << ROWS) - 1
    by = {r.mask: r for r in _reaches(f, load)}
    assert set(by) == {stored, f.full & ~stored}
    assert f.insts[by[stored].writer].stmt.op == Op.STORE
    assert f.insts[by[f.full & ~stored].writer].stmt.op == Op.ALLOC


def test_a_store_under_a_guard_spelled_in_text_is_certain_only_under_it():
    """`if (active)` is the same wherever the body tests it: a load under it
    reads the store under it for certain, a load outside may read either."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    b.store(r, 2.0, 0)
    with b.Block('if (active)'):
        b.store(r, 1.0, 0)
        b.store(t.p, b.load(r, 0), t.lane)
    b.store(t.p, b.load(r, 0), t.lane)
    f = t.flow()
    inside, outside = _stmts(f, Op.LOAD, MemSpace.REGISTER)
    (a,) = _reaches(f, inside)
    assert a.certain
    (o,) = _reaches(f, outside)
    assert not o.certain and o.maybe


def test_memory_tells_the_lane_that_stored_from_another():
    """A lane that reads back its own element finds its own store; one that
    reads its neighbor's finds another lane's."""
    t = Body()
    b = t.b
    b.store(t.q, 1.0, t.element(0))
    b.store(t.p, b.load(t.q, t.element(0)), t.lane)
    nxt = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), LANES)
    b.store(t.p, b.load(t.q, t.element(0, nxt)), t.lane)
    f = t.flow()
    own, other = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    assert all(r.own and r.certain for r in _reaches(f, own))
    lanes_past = [r for r in _reaches(f, other)
                  if r.writer != INITIAL]
    assert lanes_past and not any(r.own for r in lanes_past)


def test_a_broadcast_tells_the_lane_that_stored_from_the_others():
    """Every lane reads `q[0]`, which lane 5 stored: lane 5 reads its own
    store, the others what lane 5 stored."""
    t = Body()
    b = t.b
    with b.if_(b.op('eq', BOOL, t.lane, 5)):
        b.store(t.q, 1.0, 0)
    b.store(t.p, b.load(t.q, 0), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    reaches = _reaches(f, load)
    assert sum(bin(r.mask).count('1') for r in reaches) == LANES
    own = [r for r in reaches if r.own]
    assert len(own) == 1 and own[0].mask == 1 << 5
    assert all(r.certain and r.writer != INITIAL for r in reaches)
    assert all(r.by == 5 for r in reaches if not r.own)


def test_an_access_counts_once_against_the_budget():
    """A read every lane makes of one element, and a store of an element a
    lane, count once each, whatever their lanes reach: the two run on a
    budget of 2, not on 1."""
    t = Body()
    b = t.b
    b.store(t.p, b.load(t.q, 0), t.lane)
    body = b.finish()
    verify(body)
    assert Flow(body, LANES, budget=2).insts
    with pytest.raises(TooLarge):
        Flow(body, LANES, budget=1)


def test_a_broadcast_takes_what_each_lane_loaded_before_lane_by_lane():
    """Each lane loaded its own element of `q` into `r` before; of the
    lanes that then all read `q[0]`, lane 0 loaded it, and reads it from
    `r`, and no other lane does."""
    t = Body()
    b = t.b
    r = t.registers('r', 1)
    b.store(r, b.load(t.q, t.lane), 0)
    b.store(t.p, b.load(t.q, 0), t.lane)
    body = b.finish()
    verify(body)
    f = Flow(body, LANES, arena='shrMem', values=True, before=False)
    load = [s for s in walk_stmts(body) if s.op == Op.LOAD and s.accesses
            and s.accesses[0].space is MemSpace.GLOBAL][1]
    found = [(m, o) for n in f.executions(load) for m, o in f.forward.get(n, ())]
    assert [(m, o[0]) for m, o in found] == [(1, 'reg')]


def test_an_index_that_picks_by_lane_is_asked_lane_by_lane():
    """Lanes below 16 store their own element of `q`, the others the one 32
    past theirs.  Each then reads the element 32 past its own: a lane above
    reads its own store, a lane below an element nobody stored."""
    t = Body()
    b = t.b
    low = b.op('lt', BOOL, t.lane, 16, hint='low')
    past = b.op('add', INDEX, t.lane, 32, hint='past')
    b.store(t.q, 1.0, b.op('select', INDEX, low, t.lane, past, hint='at'))
    b.store(t.p, b.load(t.q, past), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    reaches = _reaches(f, load)
    lanes = sorted(l for r in reaches for l in range(LANES) if r.mask >> l & 1)
    assert lanes == list(range(LANES))
    for r in reaches:
        if r.mask >> 16:
            assert r.mask & 0xFFFF == 0
            assert r.own and r.certain and r.writer != INITIAL
        else:
            assert r.writer == INITIAL


# -- memory, the lanes of an access as one (`_Cells`) -------------------------

def test_the_lanes_that_read_their_own_stores_alike_read_in_one_reach():
    """Every valid lane stores its element of a column of `q` and reads it
    back: one reach for all of them, each lane's own store, and the element
    each lane read."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, t.element(2))
        b.store(t.p, b.load(t.q, t.element(2)), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    (r,) = _reaches(f, load)
    assert r.mask == (1 << ROWS) - 1 and r.own and r.certain
    assert r.cell_at(3)[1] == (3 + 2 * ROWS,)


def test_a_read_of_another_lanes_store_names_the_lane_that_stored():
    """Lane `l` reads the element lane `l + 1` stored: each lane apart, with
    the lane that stored it."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, t.lane)
    with b.if_(b.op('lt', BOOL, t.lane, ROWS - 1)):
        b.store(t.p, b.load(t.q, b.op('add', INDEX, t.lane, 1)), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    reaches = _reaches(f, load)
    assert sorted(r.mask for r in reaches) == [1 << l for l in range(ROWS - 1)]
    assert all(not r.own and r.certain and r.by == r.mask.bit_length()
               for r in reaches)


def test_a_store_over_some_elements_of_a_read_tells_its_lanes_apart():
    """The valid lanes store their elements, those below 8 once more: read
    back, those find the second store, the others the first."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, t.lane)
    with b.if_(b.op('lt', BOOL, t.lane, 8)):
        b.store(t.q, 2.0, t.lane)
    with b.if_(t.valid):
        b.store(t.p, b.load(t.q, t.lane), t.lane)
    f = t.flow()
    second = _stmts(f, Op.STORE, MemSpace.GLOBAL)[1]
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    by = {f.insts[r.writer].stmt is second: r for r in _reaches(f, load)}
    assert by[True].mask == 0xFF
    assert by[False].mask == (1 << ROWS) - 1 & ~0xFF
    assert all(r.own and r.certain for r in by.values())


def test_elements_in_two_blocks_are_read_in_one_reach():
    """`q[l + 50]`: the lanes' elements lie in two blocks of 64, and the
    lanes read their own stores in one reach all the same."""
    t = Body()
    b = t.b
    at = b.op('add', INDEX, t.lane, 50)
    with b.if_(t.valid):
        b.store(t.q, 1.0, at)
        b.store(t.p, b.load(t.q, at), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    (r,) = _reaches(f, load)
    assert r.mask == (1 << ROWS) - 1 and r.own and r.certain


def test_an_element_keeps_its_last_writes_of_uncertain_standing():
    """Nine stores of `q[l]`, each under a guard of its own: a read under
    none may read the last eight, and the first has gone out of sight."""
    t = Body()
    b = t.b
    for i in range(9):
        with b.Block(f'if (flag{i})'):
            b.store(t.q, float(i), t.lane)
    b.store(t.p, b.load(t.q, t.lane), t.lane)
    f = t.flow()
    stores = _stmts(f, Op.STORE, MemSpace.GLOBAL)[:9]
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    (r,) = _reaches(f, load)
    assert r.writer == INITIAL and r.mask == f.full
    assert {id(f.insts[w].stmt) for w in r.maybe} == {id(s) for s in stores[1:]}


def test_a_store_of_one_element_by_every_lane_is_the_last_lanes():
    """Every valid lane stores `q[0]`: memory holds the last lane's, which
    that lane reads as its own and every other one as lane 23's; and each
    lane's store replaced the one of the lane before it."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, 0)
    b.store(t.p, b.load(t.q, 0), t.lane)
    body = b.finish()
    verify(body)
    f = Flow(body, LANES, arena='shrMem')
    store = _stmts(f, Op.STORE, MemSpace.GLOBAL)[0]
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    reaches = _reaches(f, load)
    own = [r for r in reaches if r.own]
    assert len(own) == 1 and own[0].mask == 1 << (ROWS - 1)
    assert all(r.by == ROWS - 1 for r in reaches if not r.own)
    n, = f.executions(store)
    cell = f.cells_of(n)[0][1]
    assert f.before[n][(0, cell)] == (INITIAL, True)
    assert all(f.before[n][(lane, cell)] == (n, True)
               for lane in range(1, ROWS))


def test_a_loop_that_runs_again_tells_the_lanes_it_writes_for_apart():
    """A batch loop reads `q[l]`, which a store ahead of it wrote, and its
    lanes below 8 store it anew behind the read: those may read that store
    of the run before and take nothing; the others take the number of the
    first."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, t.lane)
    n = b.extern_value('numElements', SIZE, uniform=Uniformity.GRID, hint='n')
    with b.for_(0, n, 1):
        with b.if_(t.valid):
            b.store(t.p, b.load(t.q, t.lane), t.lane)
        with b.if_(b.op('lt', BOOL, t.lane, 8)):
            b.store(t.q, 2.0, t.lane)
    body = b.finish()
    verify(body)
    f = Flow(body, LANES, arena='shrMem', values=True, before=False)
    load, = [s for s in walk_stmts(body) if s.op == Op.LOAD]
    reaches = _reaches(f, load)
    low = [r for r in reaches if r.mask & 0xFF]
    high = [r for r in reaches if r.mask & ~0xFF]
    assert [r.mask for r in low] == [0xFF] and not low[0].certain
    assert [r.mask for r in high] == [(1 << ROWS) - 1 & ~0xFF]
    assert high[0].certain
    found = [(m, o) for x in f.executions(load) for m, o in f.forward.get(x, ())]
    lanes = 0
    for m, o in found:
        assert o == ('const', 1.0)
        lanes |= m
    assert lanes == (1 << ROWS) - 1 & ~0xFF


def test_a_loop_that_runs_again_hands_on_only_the_elements_it_writes():
    """A batch loop reads `q[l]`, which a store ahead of it wrote, and stores
    `q[l + 32]` behind the read, in the same block of memory: nothing the
    loop writes is what the read reads, and it stays certain."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, t.lane)
    n = b.extern_value('numElements', SIZE, uniform=Uniformity.GRID, hint='n')
    with b.for_(0, n, 1):
        with b.if_(t.valid):
            b.store(t.p, b.load(t.q, t.lane), t.lane)
            b.store(t.q, 2.0, b.op('add', INDEX, t.lane, 32))
    body = b.finish()
    verify(body)
    f = Flow(body, LANES, arena='shrMem', before=False)
    load, = [s for s in walk_stmts(body) if s.op == Op.LOAD]
    (r,) = _reaches(f, load)
    assert r.certain and r.own and r.mask == (1 << ROWS) - 1


def test_a_store_through_the_symbol_is_read_through_the_binding():
    """A store names the buffer's symbol, a load the value bound for it:
    one buffer.  Asked of the generator's own shape, where the store's
    first operand is the symbol and the load's the binding."""
    t = Body()
    b = t.b
    b.store(t.q, 1.0, t.element(1))
    b.store(t.p, b.load(t.q, t.element(1)), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    assert all(r.certain and r.writer != INITIAL for r in _reaches(f, load))


def test_a_loop_that_runs_again_reads_what_its_last_run_left():
    """A register the batch loop's body reads before it writes it holds, on
    every run but the first, what the run before left there."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    b.store(r, 1.0, 0)
    n = b.extern_value('numElements', SIZE, uniform=Uniformity.GRID, hint='n')
    with b.for_(0, n, 1):
        b.store(t.p, b.load(r, 0), t.lane)
        b.store(r, 2.0, 0)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.REGISTER)
    (reach,) = _reaches(f, load)
    assert not reach.certain
    late = [s for s in _stmts(f, Op.STORE, MemSpace.REGISTER)
            if s.args[1] == 2.0][0]
    assert any(f.insts[w].stmt is late for w in reach.maybe)


def test_what_text_reads_escapes_and_what_it_writes_is_possible():
    """An access only its text names reads every element of its buffer, and
    may have written any of them."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    b.store(r, 1.0, 0)
    first = b.load(r, 0)
    b.access_stmt('touch(r0);', r, Effect.READ | Effect.WRITE)
    b.store(t.p, b.load(r, 0), t.lane)
    b.store(t.p, first, t.lane)
    f = t.flow()
    store = _stmts(f, Op.STORE, MemSpace.REGISTER)[0]
    (w,) = f.executions(store)
    assert w in f.escaped
    second = _stmts(f, Op.LOAD, MemSpace.REGISTER)[1]
    (reach,) = _reaches(f, second)
    assert not reach.certain


def test_a_store_knows_what_it_replaced():
    """A store's record of what the element held right before it."""
    t = Body()
    b = t.b
    b.store(t.q, 1.0, t.element(0))
    b.store(t.q, 2.0, t.element(0))
    f = t.flow()
    first, second = _stmts(f, Op.STORE, MemSpace.GLOBAL)
    (n1,) = f.executions(first)
    (n2,) = f.executions(second)
    assert all(v == (INITIAL, True) for v in f.before[n1].values())
    assert all(v == (n1, True) for v in f.before[n2].values())
    assert n2 in f.final and n1 not in f.final


def test_a_store_behind_one_under_a_guard_is_unsure_what_it_replaced():
    """A store under a guard spelled in text, and one outside it: what the
    second replaces is the first's where the guard holds and what was there
    before where it does not."""
    t = Body()
    b = t.b
    with b.Block('if (active)'):
        b.store(t.q, 1.0, t.element(0))
    b.store(t.q, 2.0, t.element(0))
    f = t.flow()
    _, second = _stmts(f, Op.STORE, MemSpace.GLOBAL)
    (n2,) = f.executions(second)
    assert f.before[n2]
    assert all(v == (INITIAL, False) for v in f.before[n2].values())


def test_a_register_store_behind_one_under_a_guard_is_unsure_what_it_replaced():
    """The same in a register: what the second store replaces is the
    first's where the guard holds, and the zeros of the allocation where it
    does not."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    with b.Block('if (active)'):
        b.store(r, 1.0, 0)
    b.store(r, 2.0, 0)
    f = t.flow()
    alloc, = [s for s in walk_stmts(f.body) if s.op == Op.ALLOC
              and s.target[0] is r]
    (zeros,) = f.executions(alloc)
    _, second = _stmts(f, Op.STORE, MemSpace.REGISTER)
    (n2,) = f.executions(second)
    assert f.before[n2]
    assert all(v == (zeros, False) for v in f.before[n2].values())


def test_a_buffer_defined_anew_holds_nothing_of_before():
    """After a `mark defines`, a read of the buffer finds no store of
    before it."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    b.store(s, 1.0, t.lane)
    b.mark('defines', s)
    b.store(t.p, b.load(s, t.lane), t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.SHARED)
    assert all(r.writer == INITIAL for r in _reaches(f, load))


def test_a_binding_a_loop_computes_is_another_buffer_every_trip():
    """A binding computed in a loop -- a table's member, the next element --
    points somewhere else every trip: a store of one trip is not what the
    next trip reads through it."""
    t = Body()
    b = t.b
    with b.for_(0, 2, 1, rolled=True) as trip:
        m = b.decl_expr(
            'float *const t', '&m2[{0} * 32]',
            BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm2',
            args=(trip.induction,), kind=Effect.READ, space=MemSpace.GLOBAL,
            hint='t', extern='t')
        b.store(t.p, b.load(m, t.lane), t.lane)
        b.store(m, 1.0, t.lane)
    f = t.flow()
    load, = _stmts(f, Op.LOAD, MemSpace.GLOBAL)
    assert all(r.writer == INITIAL for r in _reaches(f, load))


def test_a_body_that_places_more_than_its_budget_is_too_large():
    """What `Flow` keeps grows with the accesses a body places, trip by
    trip, and they count against its budget: `ROWS` lanes storing an
    element each in each of `COLS` trips place 4, which a budget of 4 runs
    and one of 3 does not."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            b.store(t.q, 1.0, t.element(hop.induction))
    body = b.finish()
    verify(body)
    assert Flow(body, LANES, budget=COLS).insts
    with pytest.raises(TooLarge):
        Flow(body, LANES, budget=COLS - 1)
