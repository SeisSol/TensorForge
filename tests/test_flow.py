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

from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, BufferType,
                                          Effect, MemSpace, Op, ScalarType,
                                          Uniformity, walk_stmts)
from tensorforge.backend.pir.flow import INITIAL, Flow
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
