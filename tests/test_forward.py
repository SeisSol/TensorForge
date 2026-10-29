# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A load that reads what its own lane stored, taken from where the store
took the value, one that reads what its lanes loaded before, from where that
load put it, and a store nothing reads, dropped (`pir.forward`), asked of
bodies built by hand.

The generated bodies are asked in `test_rolling.py` and the corpus in
`tools/host/corpus_check.py`; here are the ways a load reads something else
than the value it would be given -- another lane's store, a register written
again since, a guard the load does not stand under, a register no compiler
can name -- and the ways a store is read after all.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.core import (BOOL, INDEX, SCALAR_LAYOUT, SIZE,
                                          Access, BufferType, Effect, LaneAxis,
                                          MemSpace, Op, RegisterLayout,
                                          ScalarType, Uniformity, walk_stmts)
from tensorforge.backend.pir.flow import Flow
from tensorforge.backend.pir.forward import drop_dead_stores, forward_memory
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32
LANES = 32
ROWS = 24
COLS = 4
REAL = ScalarType(F32)


class Body:
    """`q`, an element's `ROWS x COLS` in global memory, `p`, one a lane,
    and `s`, a buffer of the multiplication's own shared memory."""

    def __init__(self):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = self.k = b.extern_value('element', SIZE,
                                    uniform=Uniformity.MULT, hint='k')
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

    def fill(self, r, value=2.0):
        """`r[j] = value * p[l]` over the columns: a number no pass knows."""
        b = self.b
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            b.store(r, b.op('mul', REAL, value, b.load(self.p, self.lane)), j)

    def store_q(self, r, guard=None, lane=None):
        """`q[e(j)] = r[j]` over the columns, under `guard` where given."""
        b = self.b
        with b.if_(self.valid):
            with b.for_(0, COLS, 1, unroll=True) as hop:
                j = hop.induction
                value = b.load(r, j)
                with (guard() if guard else contextlib.nullcontext()):
                    b.store(self.q, value, self.element(j, lane))

    def load_q(self, into, lane=None, rolled=False, guard=None):
        """`into[j] = q[e(j)]` over the columns, and `p[l] = into[0]`, the
        store under `guard` where given."""
        b = self.b
        loop = (b.for_(0, COLS, 1, rolled=True) if rolled
                else b.for_(0, COLS, 1, unroll=True))
        with b.if_(self.valid):
            with loop as hop:
                j = hop.induction
                b.store(into, b.load(self.q, self.element(j, lane)), j)
        value = b.load(into, 0)
        with (guard() if guard else contextlib.nullcontext()):
            b.store(self.p, value, self.lane)

    def run(self, rounds=3):
        body = self.b.finish()
        verify(body)
        self.body = body
        self.report = []
        out = forward_memory(body, self.b.scratch, threads=LANES,
                             arena='shrMem', rounds=rounds,
                             report=self.report)
        verify(out)
        return out


def _loads(body, space, base=None):
    return [s for s in walk_stmts(body) if s.op == Op.LOAD and s.accesses
            and s.accesses[0].space is space
            and (base is None or s.args[0] is base)]


def _stores(body, space, base=None):
    return [s for s in walk_stmts(body) if s.op == Op.STORE and s.accesses
            and s.accesses[0].space is space
            and (base is None or s.accesses[0].base is base
                 or s.args[0] is base)]


def test_a_result_loaded_back_is_read_from_its_registers():
    """Stored from a register image and loaded straight back: the load
    reads the image, and the store stays -- the element is what the body
    leaves behind."""
    t = Body()
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    t.load_q(t.registers('r1'))
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)
    assert _stores(out, MemSpace.GLOBAL, t.q)


def test_a_copy_is_read_where_the_copy_read():
    """`r1 = r0` and `q = r1`: the store reads `r0`, and the copy, which
    nothing reads any more, goes."""
    t = Body()
    b = t.b
    r0, r1 = t.registers('r0'), t.registers('r1')
    t.fill(r0)
    with b.for_(0, COLS, 1, unroll=True) as hop:
        b.store(r1, b.load(r0, hop.induction), hop.induction)
    t.store_q(r1)
    out = t.run()
    assert not _stores(out, MemSpace.REGISTER, r1)
    assert not _loads(out, MemSpace.REGISTER, r1)


def test_a_store_the_next_one_replaces_unread_goes():
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 1.0, t.element(0))
        b.store(t.q, 2.0, t.element(0))
    out = t.run()
    stores = _stores(out, MemSpace.GLOBAL, t.q)
    assert len(stores) == 1 and stores[0].args[1] == 2.0


def test_a_register_written_again_keeps_the_load():
    """The image is written again between the store and the load, for
    something else to read: what the load reads is in memory alone."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    t.fill(r, 3.0)
    b.store(t.p, b.load(r, 0), t.lane)
    t.load_q(t.registers('r1'))
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_another_lanes_element_keeps_the_load():
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    nxt = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), ROWS, hint='n')
    t.load_q(t.registers('r1'), lane=nxt)
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_a_store_under_a_guard_spelled_in_text_keeps_the_load():
    """Where `active` does not hold, the load reads what memory held."""
    t = Body()
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r, guard=lambda: t.b.Block('if (active)'))
    t.load_q(t.registers('r1'))
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_a_store_under_an_elements_mask_is_read_outside_it():
    """Where an element's mask does not hold, its lanes compute what nobody
    keeps: the load outside the mask reads the store under it."""
    t = Body()
    r = t.registers('r0')
    t.fill(r)
    mask = lambda: t.b.If('active', guard='element')  # noqa: E731
    t.store_q(r, guard=mask)
    t.load_q(t.registers('r1'), guard=mask)
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)


def test_a_global_store_outside_every_mask_takes_none_to_hold():
    """A body that writes global memory where no mask guards it does not
    keep the contract: a lane past the batch would leave that behind."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    mask = lambda: b.If('active', guard='element')  # noqa: E731
    t.store_q(r, guard=mask)
    b.store(t.p, 1.0, t.lane)
    t.load_q(t.registers('r1'), guard=mask)
    body = b.finish()
    assert not Flow(body, LANES, 'shrMem', masks=True).masks
    out = forward_memory(body, b.scratch, threads=LANES, arena='shrMem')
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_a_branch_on_what_memory_holds_takes_no_mask_to_hold():
    """A lane past the batch reads another element's numbers; a branch on
    them could take it past a barrier the others wait at."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    mask = lambda: b.If('active', guard='element')  # noqa: E731
    t.store_q(r, guard=mask)
    x = b.load(t.p, t.lane)
    with b.if_(b.op('lt', BOOL, x, 0.0, hint='neg')):
        t.load_q(t.registers('r1'), guard=mask)
    body = b.finish()
    assert not Flow(body, LANES, 'shrMem', masks=True).masks


def test_a_store_padded_with_zeros_is_read_from_the_register_it_took():
    """`w[l + 32 j] = valid ? r0[j] : 0` by every lane -- the rows a lane
    holds and the zeros past them, as an operation stores its result into a
    buffer laid out for every lane -- loaded back under `valid`: the lanes
    that load it took the select's first arm, and read `r0`."""
    t = Body()
    b = t.b
    w = b.decl_expr(
        'float *const __restrict__ glb_m2', '&m2[{0} * 128]',
        BufferType(F32, (LANES * COLS,), MemSpace.GLOBAL), 'm2', args=(t.k,),
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m2',
        extern='glb_m2')
    r0, r1 = t.registers('r0'), t.registers('r1')
    t.fill(r0)
    with b.for_(0, COLS, 1, unroll=True) as hop:
        j = hop.induction
        at = b.op('add', INDEX, t.lane, b.op('mul', INDEX, j, LANES))
        b.store(w, b.op('select', REAL, t.valid, b.load(r0, j), 0.0,
                        hint='pad'), at)
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            at = b.op('add', INDEX, t.lane, b.op('mul', INDEX, j, LANES))
            b.store(r1, b.load(w, at), j)
    b.store(t.p, b.load(r1, 0), t.lane)
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, w)
    assert _stores(out, MemSpace.GLOBAL, w), 'what the body ends with stays'


def test_a_number_stored_is_read_as_the_number():
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 0.0, t.element(0))
        b.store(t.p, b.load(t.q, t.element(0)), t.lane)
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)


def test_a_register_a_rolled_loop_picks_keeps_the_load():
    """The image's register would be picked by the hop of a loop the
    compiler does not unroll: an array in memory, where the load reads one
    the compiler names."""
    t = Body()
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    t.load_q(t.registers('r1'), rolled=True)
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_a_store_into_shared_memory_others_read_stays():
    """A buffer of shared memory outside the multiplication's arena is read
    by lanes no account here follows."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s1',
                arena='other')
    b.store(s, 1.0, t.lane)
    out = t.run()
    assert _stores(out, MemSpace.SHARED, s)


def test_a_store_into_the_arena_nothing_reads_goes():
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    b.store(s, 1.0, t.lane)
    out = drop_dead_stores(t.b.finish(), threads=LANES, arena='shrMem')
    assert not _stores(out, MemSpace.SHARED, s)


def test_a_value_stored_is_read_as_the_value():
    t = Body()
    b = t.b
    v = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v')
    with b.if_(t.valid):
        b.store(t.q, v, t.element(0))
        b.store(t.p, b.load(t.q, t.element(0)), t.lane)
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)
    assert any(s.op == Op.STORE and s.args[1] is v and s.args[0] is t.p
               for s in walk_stmts(out))


def test_a_value_a_loop_computes_is_not_read_behind_it():
    """The value of every trip is the trip's own: behind the loop, no value
    stands for all of them."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            v = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v')
            b.store(t.q, v, t.element(j))
    t.load_q(t.registers('r1'))
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_a_value_a_branch_computes_is_not_read_behind_it():
    """Every lane takes the branch, and stores there what it computes
    there: behind the branch, the value stands nowhere to be read."""
    t = Body()
    b = t.b
    with b.if_(b.op('lt', BOOL, t.lane, LANES, hint='every')):
        v = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v')
        b.store(t.q, v, t.element(0))
    b.store(t.p, b.load(t.q, t.element(0)), t.lane)
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


@pytest.mark.parametrize('where', ['first', 'every', 'inner', 'carried'])
def test_a_value_a_trip_stored_is_read_in_that_trip_only(where):
    """`v`, which each trip of a loop computes anew, stored into `s` and
    read back by every trip.  Stored in the first trip only, the second
    reads what the first stored, and its own `v` is another: the load stays
    -- the same for `v` the loop hands from trip to trip.  Stored in every
    trip, every trip reads its own: the value.  Stored in the first trip of
    a loop inside, the value of the loop around: the same in every trip of
    the loop inside."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')

    def trip(hop, v):
        first = b.op('eq', BOOL, hop, 0, hint='first')
        with (b.if_(first) if where != 'every' else contextlib.nullcontext()):
            b.store(s, v, t.lane)
        b.store(t.p, b.load(s, t.lane, hint='back'), t.lane)

    if where == 'carried':
        with b.for_(0, 2, 1, inits=(b.load(t.q, t.element(0)),),
                    types=(REAL,)) as f:
            trip(f.induction, f.iter_args[0])
            f.yield_(b.op('mul', REAL, f.iter_args[0], 2.0, hint='twice'))
    else:
        with b.for_(0, 2, 1, unroll=True) as outer:
            v = b.load(t.q, t.element(outer.induction), hint='v')
            if where == 'inner':
                with b.for_(0, 2, 1, unroll=True) as hop:
                    trip(hop.induction, v)
            else:
                trip(outer.induction, v)
    out = t.run()
    assert bool(_loads(out, MemSpace.SHARED, s)) == (where in ('first',
                                                               'carried'))


def test_a_load_taken_from_a_load_the_same_round_takes_reads_what_that_became():
    """`s = x`, `r0 = s` and `p = r0`: the load of `s` reads `x`, and the
    load of `r0` the value the load of `s` gave -- which the same round takes
    away.  The second reads `x` as well."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    r = t.registers('r0', 1)
    x = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='x')
    b.store(s, x, t.lane)
    b.store(r, b.load(s, t.lane), 0)
    b.store(t.p, b.load(r, 0), t.lane)
    out = t.run(rounds=1)
    stored = [y.args[1] for y in walk_stmts(out)
              if y.op == Op.STORE and y.args[0] is t.p]
    assert stored == [x]


def test_a_predicated_load_keeps_what_its_predicate_gives():
    """Read from the registers, the rows past the load's predicate still
    get what the load gives them."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    few = b.op('lt', BOOL, t.lane, ROWS - 4, hint='few')
    into = t.registers('r1')
    with b.for_(0, COLS, 1, unroll=True) as hop:
        j = hop.induction
        b.store(into, b.load(t.q, t.element(j), predicate=few, other=0.0), j)
    b.store(t.p, b.load(into, 0), t.lane)
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)
    taken = [s for s in _loads(out, MemSpace.REGISTER, r)
             if s.predicate is not None]
    assert taken and all(s.predicate is few and s.attr('other') == 0.0
                         for s in taken)


def test_a_named_load_is_taken_from_a_register_alone():
    """Text names the load's value: a number in its place would leave the
    name undeclared."""
    t = Body()
    b = t.b
    with b.if_(t.valid):
        b.store(t.q, 0.0, t.element(0))
        b.store(t.p, b.load(t.q, t.element(0), extern='named'), t.lane)
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_a_register_a_load_reads_from_now_on_stays():
    """`r0[0] = 3`, `q = r0[0]`, and `q` loaded back: the store's read of
    `r0` becomes the number, and the load reads `r0` -- which nothing else
    reads any more, and which has to stay all the same."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    b.store(r, 3.0, 0)
    with b.if_(t.valid):
        b.store(t.q, b.load(r, 0), t.element(0))
        b.store(t.p, b.load(t.q, t.element(0)), t.lane)
    out = t.run(rounds=1)
    assert not _loads(out, MemSpace.GLOBAL, t.q)
    assert _stores(out, MemSpace.REGISTER, r)


def test_rows_the_store_left_out_keep_the_load():
    """The store writes the first twenty rows, the load reads twenty-four:
    the last four are in memory alone."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    few = b.op('lt', BOOL, t.lane, ROWS - 4, hint='few')
    with b.if_(few):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            b.store(t.q, b.load(r, j), t.element(j))
    t.load_q(t.registers('r1'))
    out = t.run()
    assert _loads(out, MemSpace.GLOBAL, t.q)


def test_rows_stored_from_registers_and_rows_of_zeros_read_as_both():
    """The first twenty rows stored from registers, the last four as zeros
    -- a promise of what the product does not reach: the load is the
    register where the guard of the first holds, and zero where not."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    few = b.op('lt', BOOL, t.lane, ROWS - 4, hint='few')
    rest = b.op('and', BOOL, b.op('ge', BOOL, t.lane, ROWS - 4, hint='r'),
                t.valid, hint='rest')
    with b.if_(few):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            b.store(t.q, b.load(r, j), t.element(j))
    with b.if_(rest):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            b.store(t.q, 0.0, t.element(hop.induction))
    t.load_q(t.registers('r1'))
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)
    picks = [s for s in walk_stmts(out) if s.op == 'select'
             and s.args[0] is few]
    assert picks


def test_lanes_a_copy_leaves_out_read_the_zeros_of_the_declaration():
    """`r1 = r0` for the first twenty rows and `q = r1` for every row: the
    last four read the zeros `r1` was declared with.  The store reads `r0`
    where the copy's guard holds and a zero where not, and the copy goes."""
    t = Body()
    b = t.b
    r0, r1 = t.registers('r0'), t.registers('r1')
    t.fill(r0)
    few = b.op('lt', BOOL, t.lane, ROWS - 4, hint='few')
    with b.if_(few):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            b.store(r1, b.load(r0, hop.induction), hop.induction)
    t.store_q(r1)
    out = t.run()
    assert not _loads(out, MemSpace.REGISTER, r1)
    assert not _stores(out, MemSpace.REGISTER, r1)
    assert any(s.op == 'select' and s.args[0] is few
               for s in walk_stmts(out))


def test_registers_declared_without_zeros_hold_nothing_known():
    """The same copy into registers the declaration leaves unset: the last
    four rows store what nobody wrote, and the copy stays."""
    t = Body()
    b = t.b
    r0 = t.registers('r0')
    r1 = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r1', extern='r1',
                 init='')
    t.fill(r0)
    few = b.op('lt', BOOL, t.lane, ROWS - 4, hint='few')
    with b.if_(few):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            b.store(r1, b.load(r0, hop.induction), hop.induction)
    t.store_q(r1)
    out = t.run()
    assert _loads(out, MemSpace.REGISTER, r1)


def test_what_a_statement_says_others_reach_keeps_the_mask():
    """A merged run says what its reads and writes through the pointer its
    counter selects reach (`IRBuilder.touches`), outside every mask: the
    statement writes nothing itself, and the store under the mask is still
    what the load outside it reads."""
    t = Body()
    b = t.b
    member = b.decl_expr(
        'float *const __restrict__ glb_m2', '&m2[0]',
        BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm2',
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m2',
        extern='glb_m2')
    b.touches([Access(Effect.READ | Effect.WRITE, MemSpace.GLOBAL, member)])
    r = t.registers('r0')
    t.fill(r)
    mask = lambda: b.If('active', guard='element')  # noqa: E731
    t.store_q(r, guard=mask)
    t.load_q(t.registers('r1'), guard=mask)
    out = t.run()
    assert not _loads(out, MemSpace.GLOBAL, t.q)


def test_text_that_writes_global_memory_outside_every_mask_takes_none():
    """Text whose effect is known writes for itself: outside every mask, a
    lane past the batch would leave that write behind."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    t.fill(r)
    mask = lambda: b.If('active', guard='element')  # noqa: E731
    t.store_q(r, guard=mask)
    b.access_stmt('glb_m1[0] = 1.0f;', t.p, Effect.WRITE)
    t.load_q(t.registers('r1'), guard=mask)
    body = b.finish()
    assert not Flow(body, LANES, 'shrMem', masks=True).masks


def test_a_buffer_of_the_arena_nothing_reaches_any_more_goes():
    """A temporary staged in the multiplication's shared memory that each
    lane reads back itself: the load reads the registers, the store goes,
    and so does the buffer -- the `mark` that said the store defines it,
    and the declaration, which would otherwise name a window of an arena
    the allocator lays nothing out in."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    r = t.registers('r0')
    t.fill(r)
    b.mark('defines', s)
    b.store(s, b.load(r, 0), t.lane)
    with b.if_(t.valid):
        b.store(t.q, b.load(s, t.lane), t.element(0))
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, s)
    assert not any(x.op in (Op.ALLOC, Op.MARK) and (s in x.target
                                                    or s in x.args)
                   for x in walk_stmts(out))


def test_a_buffer_of_the_arena_read_across_the_lanes_stays():
    """The same temporary read by the next lane, beside a load the pass
    does take: the temporary's load stays, and the buffer with it."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    r = t.registers('r0')
    t.fill(r)
    b.mark('defines', s)
    b.store(s, b.load(r, 0), t.lane)
    nxt = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), LANES, hint='n')
    with b.if_(t.valid):
        b.store(t.q, b.load(s, nxt), t.element(0))
        b.store(t.q, 0.0, t.element(1))
        b.store(t.p, b.load(t.q, t.element(1)), t.lane)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 0
    assert _loads(out, MemSpace.SHARED, s)
    assert any(x.op == Op.ALLOC and s in x.target for x in walk_stmts(out))
    assert any(x.op == Op.MARK and s in x.args for x in walk_stmts(out))


def _across(t, value, place):
    """`s[w] = value` on every lane, then `p = s[r]`: the next lane's
    element (`w` the lane, `r` the next), or element zero, which every lane
    writes and the last one wrote last."""
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    if place == 'next':
        written = t.lane
        read = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), LANES,
                    hint='n')
    else:
        written = read = 0
    b.store(s, value, written)
    b.store(t.p, b.load(s, read), t.lane)
    return s


def _summed(t, term):
    """`term` added up over a loop of three trips, from zero."""
    b = t.b
    with b.for_(0, 3, 1, inits=(b.const(0.0, REAL),), types=(REAL,)) as f:
        f.yield_(b.op('add', REAL, f.iter_args[0], term, hint='acc'))
    return f.results[0]


@pytest.mark.parametrize('place', ['next', 'zero'])
@pytest.mark.parametrize('kind', ['number', 'value', 'loop'])
def test_what_every_lane_stores_alike_is_read_from_another_lanes_store(
        kind, place):
    """Every lane stores the same into shared memory -- a number, a value no
    lane's index went into, or such a value summed up over a loop -- and
    reads an element another lane wrote: what that lane stored is what this
    one did, and the load reads it from where the lane's own store took
    it."""
    t = Body()
    b = t.b
    if kind == 'number':
        value = 1.5
    else:
        alpha = b.extern_value('alpha', REAL, uniform=Uniformity.MULT,
                               hint='alpha')
        value = b.op('mul', REAL, alpha, 2.0, hint='u')
        if kind == 'loop':
            value = _summed(t, value)
    s = _across(t, value, place)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, s)
    stored = [x.args[1] for x in walk_stmts(out)
              if x.op == Op.STORE and x.args[0] is t.p]
    assert len(stored) == 1
    if kind == 'number':
        assert any(x.op == Op.CONST and stored[0] in x.target
                   and x.attr('value') == value for x in walk_stmts(out))
    else:
        assert stored[0] is value


@pytest.mark.parametrize('place', ['next', 'zero'])
@pytest.mark.parametrize('kind', ['lane', 'register', 'loop'])
def test_what_the_lanes_store_apart_keeps_the_load_of_another_lanes_store(
        kind, place):
    """The same with a value that differs between the lanes: the element
    holds what the other lane stored, which is not what this one did.  One
    the lane's index went into; one read out of the lane's registers, a read
    of the same register in every lane, which its uniformity calls the same
    everywhere, since nothing says how the registers are spread over the
    lanes; or a sum over a loop of what the lane loaded, which the loop
    carries from a zero the same everywhere."""
    t = Body()
    b = t.b
    if kind == 'register':
        r = t.registers('r0')
        t.fill(r)
        x = b.load(r, 1)
    else:
        x = b.load(t.p, t.lane)
    value = b.op('mul', REAL, x, 2.0, hint='v')
    if kind == 'loop':
        value = _summed(t, x)
    s = _across(t, value, place)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, s)


@pytest.mark.parametrize('fold', [True, False], ids=['all_reduce', 'text'])
def test_a_fold_every_lane_holds_alike_is_read_from_its_register(fold):
    """A fold over every lane of the multiplication hands each of them the
    fold of all of them (`IRBuilder.all_reduce`): stored by every lane into
    its element and read from the next lane's, it is read from the lane's
    own register.  The same text as a raw expression says nothing of the
    lanes, and the load stays."""
    t = Body()
    b = t.b
    text = ('tensorforge::reduction<tensorforge::ReductionOperation<float, '
            'tensorforge::Operation::Add>, 32, 1, float>({0})')
    x = b.load(t.p, t.lane)
    value = (b.all_reduce(text, x, type_=REAL) if fold
             else b.rawexpr(text, x, type_=REAL, pure=True, movable=False))
    s = _across(t, value, 'next')
    out = t.run()
    assert bool(_loads(out, MemSpace.SHARED, s)) is not fold


def test_a_number_read_as_a_truth_value_is_whether_it_is_zero():
    """A comparison computed as a number and stored into a buffer of truth
    values, which the store makes of it: read back, it is `v != 0`."""
    t = Body()
    b = t.b
    s = b.alloc(Datatype.BOOL, (LANES,), MemSpace.SHARED, hint='s',
                extern='s0', arena='shrMem')
    v = b.op('lt', REAL, b.load(t.p, t.lane), 1.0, hint='e')
    b.store(s, v, t.lane)
    x = b.load(s, t.lane)
    b.store(t.p, b.op('select', REAL, x, 1.0, 0.0, hint='pick'), t.lane)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, s)
    picked = next(x for x in walk_stmts(out) if x.op == 'select')
    truth = next(x for x in walk_stmts(out) if picked.args[0] in x.target)
    assert truth.op == 'ne' and truth.args[0] is v


def test_a_register_of_numbers_read_as_truth_values_is_whether_they_are_zero():
    """Comparisons computed into a register image of numbers, stored into a
    buffer of truth values and read back by the lane that stored each:
    the load reads the register, as whether its number is not zero."""
    t = Body()
    b = t.b
    r = t.registers('r0')
    s = b.alloc(Datatype.BOOL, (LANES * COLS,), MemSpace.SHARED, hint='s',
                extern='s0', arena='shrMem')
    x = b.load(t.p, t.lane)
    with b.for_(0, COLS, 1, unroll=True) as hop:
        b.store(r, b.op('lt', REAL, x, 1.0, hint='e'), hop.induction)
    with b.for_(0, COLS, 1, unroll=True) as hop:
        j = hop.induction
        at = b.op('add', INDEX, t.lane, b.op('mul', INDEX, j, LANES),
                  hint='a')
        b.store(s, b.load(r, j), at)
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            at = b.op('add', INDEX, t.lane, b.op('mul', INDEX, j, LANES),
                      hint='a')
            pick = b.op('select', REAL, b.load(s, at), 1.0, 0.0, hint='pick')
            b.store(t.q, pick, t.element(j))
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, s)
    defs = {v.id: x for x in walk_stmts(out) for v in x.target}
    picked = next(x for x in walk_stmts(out) if x.op == 'select')
    truth = defs[picked.args[0].id]
    assert truth.op == 'ne'
    held = defs[truth.args[0].id]
    assert held.op == Op.LOAD and held.args[0] is r


def _exchange_body(arch='sm_86'):
    """A body built against a target, which spells the exchange, and `s`, a
    buffer of the arena with an element a lane: `(t, b, s)`."""
    from tensorforge.common.context import Context
    t = Body()
    b = t.b
    b.context = Context(arch=arch, backend='cuda' if arch.startswith('sm')
                        else 'hip', fp_type=F32)
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    return t, b, s


def _one_lane(store='register', rewrite=False, around=None, arch='sm_86',
              named=False, predicated=None):
    """Lane 0 stores into element 0 of `s`, and the lanes load element 0
    behind it.  Lane 0 stores what its register holds, which the last trip
    of a loop wrote and no value stands for behind the loop -- or, with
    `store='value'`, a value computed ahead.  With `rewrite` the register is
    written again in between by another such loop, and read; `around` gives
    a block the load stands in; with `named` text names the load's value,
    and with `predicated` the lanes of a row load, or every lane but lane 0,
    and the others take zero.  `(t, s, r, value)`."""
    t, b, s = _exchange_body(arch)
    r = t.registers('r0', 1)
    value = None
    if store == 'value':
        value = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v')
    else:
        with b.for_(0, 2, 1, unroll=True) as hop:
            loaded = b.load(t.q, t.element(hop.induction))
            b.store(r, b.op('mul', REAL, loaded, 2.0, hint='v'), 0)
    with b.if_(b.op('eq', BOOL, t.lane, 0, hint='w')):
        b.store(s, b.load(r, 0) if value is None else value, t.lane)
    if rewrite:
        with b.for_(0, 2, 1, unroll=True) as hop:
            loaded = b.load(t.q, t.element(hop.induction))
            b.store(r, b.op('add', REAL, loaded, 1.0, hint='u'), 0)
        b.store(t.q, b.load(r, 0), t.element(0))
    predicate = {'row': lambda: t.valid,
                 'others': lambda: b.op('ne', BOOL, t.lane, 0, hint='others'),
                 None: lambda: None}[predicated]()
    with (around(t) if around else contextlib.nullcontext()):
        b.store(t.p, b.load(s, 0, extern='named' if named else None,
                            predicate=predicate,
                            other=0.0 if predicated else None), t.lane)
    return t, s, r, value


def _exchanged(t, exchange=LANES):
    body = t.b.finish()
    verify(body)
    out = forward_memory(body, t.b.scratch, threads=LANES, arena='shrMem',
                         exchange=exchange)
    verify(out)
    return out


@pytest.mark.parametrize('arch,exchange,spelled', [
    ('sm_86', LANES, 'tensorforge::readlane({0}, 0)'),
    ('gfx942', 2 * LANES, 'tensorforge::broadcast<32, 1, 0>({0})')],
    ids=['wave', 'half'])
@pytest.mark.parametrize('store', ['register', 'value'])
def test_what_one_lane_stored_is_read_from_it_across_the_lanes(
        store, arch, exchange, spelled):
    """Lane 0 stores, every lane reads it: what lane 0 stored, from the
    register or the value it took it from, handed to every lane -- no store,
    no load, and nothing of the arena.  The same on every lane of the
    multiplication, which an exchange of the whole wave is, or of the
    multiplication's own lanes within a wider one."""
    t, s, r, value = _one_lane(store, arch=arch)
    out = _exchanged(t, exchange)
    assert not _loads(out, MemSpace.SHARED, s)
    assert not _stores(out, MemSpace.SHARED, s)
    read, = [x for x in walk_stmts(out)
             if x.op == Op.RAWEXPR and x.attr('crosslane')]
    assert read.text == spelled and read.attr('alike')
    if value is not None:
        assert read.args[0] is value
    else:
        held, = [x for x in walk_stmts(out) if read.args[0] in x.target]
        assert held.op == Op.LOAD and held.args[0] is r


@pytest.mark.parametrize('arch,backend,across', [
    ('gfx942', 'hip', True), ('gfx1250', 'hip', True),
    ('sm_86', 'cuda', False), ('pvc', 'oneapi', False)])
def test_what_one_lane_stored_is_read_across_where_the_vendor_says(
        arch, backend, across):
    """On AMD a read across the lanes is a scalar read or a DPP move; on
    NVIDIA the operands it stages through shared memory would become a
    shuffle per element, and nothing has been measured on Intel."""
    from tensorforge.common.context import Context
    options = Context(arch=arch, backend=backend,
                      fp_type=F32).get_user_options()
    assert options.forward_across is across


def test_without_an_exchange_one_lanes_store_is_read_from_memory():
    t, s, _, _ = _one_lane()
    assert _loads(_exchanged(t, None), MemSpace.SHARED, s)


def test_lanes_past_one_exchange_read_one_lanes_store_from_memory():
    """The multiplication's lanes are more than one exchange reaches."""
    t, s, _, _ = _one_lane()
    assert _loads(_exchanged(t, LANES // 2), MemSpace.SHARED, s)


def test_lanes_no_block_of_an_exchange_holds_read_one_lanes_store_from_memory():
    """24 lanes, which a broadcast's block of a power of two does not
    hold."""
    t, s, _, _ = _one_lane()
    body = t.b.finish()
    out = forward_memory(body, t.b.scratch, threads=24, arena='shrMem',
                         exchange=LANES)
    verify(out)
    assert _loads(out, MemSpace.SHARED, s)


@pytest.mark.parametrize('predicated', [None, 'others'])
def test_a_register_written_since_keeps_the_load_of_one_lanes_store(
        predicated):
    """Lane 0's register holds another number by the time the lanes load:
    what lane 0 stored is in memory alone -- for the other lanes as well,
    where lane 0 does not read it itself."""
    t, s, _, _ = _one_lane(rewrite=True, predicated=predicated)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


def test_a_predicated_load_of_one_lanes_store_is_read_across():
    """The lanes of a row load, the others take zero: what lane 0 stored,
    read across by every lane, and zero where the predicate says so."""
    t, s, r, _ = _one_lane(predicated='row')
    out = _exchanged(t)
    assert not _loads(out, MemSpace.SHARED, s)
    read, = [x for x in walk_stmts(out)
             if x.op == Op.RAWEXPR and x.attr('crosslane')]
    picked, = [x for x in walk_stmts(out) if x.op == 'select'
               and read.target[0] in x.args]
    assert picked.args[0] is t.valid and picked.args[2] == 0.0


def test_a_named_load_of_one_lanes_store_keeps_its_load():
    """Text names the load's value, and a broadcast is another value."""
    t, s, _, _ = _one_lane(named=True)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


@pytest.mark.parametrize('block', ['row', 'apart', 'text', 'loop', 'alike',
                                   'alike loop'])
def test_a_load_in_a_block_the_lanes_may_take_apart_keeps_its_load(block):
    """The lanes load inside a block: of the lanes of a row, which lane 0 may
    stand outside of; on what each lane loaded, under a head spelled in
    text, or in a loop of as many trips as the lane's index, which any lane
    may -- and an exchange reads the registers of the lanes that take part.
    On a value the same on every lane, or in a loop of a count the same on
    every lane, every lane runs it as often as the others: read across."""
    def around(t):
        b = t.b
        if block == 'row':
            return b.if_(t.valid)
        if block == 'apart':
            return b.if_(b.op('gt', BOOL, b.load(t.p, t.lane), 0.0,
                              hint='pos'))
        if block == 'text':
            return b.Block('if (active)')
        if block == 'loop':
            return b.for_(0, t.lane, 1, hint='n')
        if block == 'alike loop':
            count = b.extern_value('count', SIZE, uniform=Uniformity.MULT,
                                   hint='count')
            return b.for_(0, count, 1, hint='n')
        alpha = b.extern_value('alpha', REAL, uniform=Uniformity.MULT,
                               hint='alpha')
        return b.if_(b.op('gt', BOOL, alpha, 0.0, hint='pos'))
    t, s, _, _ = _one_lane(around=around)
    kept = bool(_loads(_exchanged(t), MemSpace.SHARED, s))
    assert kept == (block not in ('alike', 'alike loop'))


def test_a_value_one_lane_stored_in_a_branch_is_not_read_behind_it():
    """Every lane takes the branch and computes the value there, and lane 0
    stores it: behind the branch, no lane holds it to be read across."""
    t, b, s = _exchange_body()
    with b.if_(b.op('lt', BOOL, t.lane, LANES, hint='every')):
        v = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v')
        with b.if_(b.op('eq', BOOL, t.lane, 0, hint='w')):
            b.store(s, v, t.lane)
    b.store(t.p, b.load(s, 0), t.lane)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


def test_one_lanes_store_of_what_its_register_held_perhaps_is_read():
    """Every lane stores what its register holds, which every lane but
    lane 0 may have written again, and every lane loads lane 0's element:
    the value lane 0 read from its register, read across -- whatever the
    register held there, the value is that on every lane."""
    t, b, s = _exchange_body()
    r = t.registers('r0', 1)
    with b.for_(0, 2, 1, unroll=True) as hop:
        loaded = b.load(t.q, t.element(hop.induction))
        b.store(r, b.op('mul', REAL, loaded, 2.0, hint='v'), 0)
    with b.if_(b.op('ne', BOOL, t.lane, 0, hint='others')):
        with b.if_(b.op('gt', BOOL, b.load(t.p, t.lane), 0.0, hint='pos')):
            b.store(r, 5.0, 0)
    b.store(s, b.load(r, 0), t.lane)
    b.store(t.p, b.load(s, 0), t.lane)
    out = _exchanged(t)
    assert not _loads(out, MemSpace.SHARED, s)


def test_what_one_lane_stored_in_another_trip_keeps_its_load():
    """Lane 0 stores what the first trip of a loop computes, and every lane
    loads it in every trip: in the second, lane 0's value is another than
    it stored."""
    t, b, s = _exchange_body()
    with b.for_(0, 2, 1, unroll=True) as hop:
        v = b.load(t.q, t.element(hop.induction), hint='v')
        with b.if_(b.op('eq', BOOL, hop.induction, 0, hint='first')):
            with b.if_(b.op('eq', BOOL, t.lane, 0, hint='w')):
                b.store(s, v, t.lane)
        b.store(t.p, b.load(s, 0), t.lane)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


@pytest.mark.parametrize('stores', ['registers', 'values', 'both'])
def test_one_lanes_stores_of_two_things_keep_their_load(stores):
    """Lane 0 stores one thing in the first trip of a loop and another in
    the second -- two registers, two values, or a register and a value --
    and every lane loads what it stored in each trip: no one register or
    value holds what they read."""
    t, b, s = _exchange_body()
    kinds = (('register', 'value') if stores == 'both'
             else (stores[:-1],) * 2)
    things = []
    for k, kind in enumerate(kinds):
        if kind == 'register':
            r = t.registers(f'r{k}', 1)
            with b.for_(0, 2, 1, unroll=True) as hop:
                loaded = b.load(t.q, t.element(hop.induction))
                b.store(r, b.op('mul', REAL, loaded, float(k + 2), hint='v'),
                        0)
            things.append(lambda r=r: b.load(r, 0))
        else:
            v = b.op('mul', REAL, b.load(t.p, t.lane), float(k + 2),
                     hint='v')
            things.append(lambda v=v: v)
    with b.for_(0, 2, 1, unroll=True) as hop:
        for k, thing in enumerate(things):
            with b.if_(b.op('eq', BOOL, hop.induction, k, hint='trip')):
                with b.if_(b.op('eq', BOOL, t.lane, 0, hint='w')):
                    b.store(s, thing(), t.lane)
        b.store(t.p, b.op('add', REAL, b.load(t.p, t.lane), b.load(s, 0)),
                t.lane)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


@pytest.mark.parametrize('held', ['alike', 'alike apart', 'apart', 'trip'])
def test_one_lanes_store_of_a_register_every_lane_holds_alike(held):
    """Lane 0 stores what its register holds, and every lane loads it.  A
    value the same on every lane: every lane holds it as well, and reads its
    own -- nothing is exchanged, and so the lanes need not load it together:
    the same in a branch they take apart.  A value of each lane's own: lane
    0's register, read across.  In a loop, a value the same on every lane
    that the first trip stores and both trips load: in the second trip the
    value is another, and the register, which holds the first trip's still,
    is read across."""
    t, b, s = _exchange_body()
    r = t.registers('r0', 1)

    def store_and_load(first=None):
        with (b.if_(first) if first is not None
              else contextlib.nullcontext()):
            with b.if_(b.op('eq', BOOL, t.lane, 0, hint='w')):
                b.store(s, b.load(r, 0), t.lane)
        apart = b.op('gt', BOOL, b.load(t.p, t.lane), 0.0, hint='pos')
        with (b.if_(apart) if held == 'alike apart'
              else contextlib.nullcontext()):
            b.store(t.p, b.op('add', REAL, b.load(t.p, t.lane),
                              b.load(s, 0)), t.lane)

    value = None
    if held == 'trip':
        with b.for_(0, 2, 1, unroll=True) as hop:
            first = b.op('eq', BOOL, hop.induction, 0, hint='first')
            with b.if_(first):
                b.store(r, b.op('mul', REAL, b.load(t.q, hop.induction), 2.0,
                                hint='u'), 0)
            store_and_load(first)
    else:
        if held == 'apart':
            value = b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='u')
        else:
            alpha = b.extern_value('alpha', REAL, uniform=Uniformity.MULT,
                                   hint='alpha')
            value = b.op('mul', REAL, alpha, 2.0, hint='u')
        b.store(r, value, 0)
        store_and_load()
    out = _exchanged(t)
    assert not _loads(out, MemSpace.SHARED, s)
    across = [x for x in walk_stmts(out)
              if x.op == Op.RAWEXPR and x.attr('crosslane')]
    assert bool(across) == (held in ('apart', 'trip'))
    if held.startswith('alike'):
        added, = [x for x in walk_stmts(out) if x.op == 'add']
        assert added.args[1] is value


def test_a_lane_reading_its_own_store_beside_one_lanes_keeps_its_load():
    """Every lane stores into its own element; lane 1 loads its own, every
    other lane lane 0's: two lanes hold what the lanes read."""
    t, b, s = _exchange_body()
    b.store(s, b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v'),
            t.lane)
    one = b.op('eq', BOOL, t.lane, 1, hint='one')
    b.store(t.p, b.load(s, b.op('select', INDEX, one, 1, 0, hint='which')),
            t.lane)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


def test_each_lane_reading_another_lanes_store_keeps_its_load():
    """Every lane stores into its own element and loads the next lane's: no
    one lane holds what they read."""
    t, b, s = _exchange_body()
    b.store(s, b.op('mul', REAL, b.load(t.p, t.lane), 2.0, hint='v'),
            t.lane)
    nxt = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), LANES, hint='n')
    b.store(t.p, b.load(s, nxt), t.lane)
    assert _loads(_exchanged(t), MemSpace.SHARED, s)


def _reload(t, between=None, lane=None, guard=None):
    """`r0[j] = q[e(j)]` and `p = r0[0]`, then `between`, then `r1[j] =
    q[e(j)]` and `p = r1[1]`: the same elements loaded twice by the same
    lanes -- the first time by the lanes `lane` gives where given, under
    `guard` where given."""
    b = t.b
    r0, r1 = t.registers('r0'), t.registers('r1')
    with b.if_(t.valid):
        with (guard() if guard else contextlib.nullcontext()):
            with b.for_(0, COLS, 1, unroll=True) as hop:
                j = hop.induction
                b.store(r0, b.load(t.q, t.element(j, lane)), j)
    b.store(t.p, b.load(r0, 0), t.lane)
    if between is not None:
        between(r0)
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            b.store(r1, b.load(t.q, t.element(j)), j)
    b.store(t.p, b.load(r1, 1), t.lane)
    return r0, r1


def test_a_load_of_what_the_lanes_loaded_before_reads_that_register():
    """Nothing writes `q` between the two loads, and `r0` holds what the
    first read: the second reads `r0`, and `r1`, which nothing reads any
    more, goes."""
    t = Body()
    _, r1 = _reload(t)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 1
    assert not _stores(out, MemSpace.REGISTER, r1)


def test_what_the_lane_stored_comes_before_what_it_loaded():
    """`q = r0`, `r1 = q`, then `q` loaded again: both loads read what the
    lane stored, from `r0`, in one round, and the second not from `r1`,
    where the first put it -- the two would be one load taken from two
    registers."""
    t = Body()
    r0 = t.registers('r0')
    t.fill(r0)
    t.store_q(r0)
    _reload(t)
    out = t.run(rounds=1)
    assert not _loads(out, MemSpace.GLOBAL, t.q)


def test_a_third_load_of_an_element_reads_the_register_of_the_first():
    """`q` loaded into `r0`, `r1` and `r2`: the third load reads `r0`, as
    the second does, and not `r1`, which the same round takes from `r0` in
    turn.  Two rounds, and no copy is left: the reads of `r1` and `r2` read
    `r0`."""
    t = Body()
    b = t.b
    _, r1 = _reload(t)
    r2 = t.registers('r2')
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            b.store(r2, b.load(t.q, t.element(j)), j)
    b.store(t.p, b.load(r2, 2), t.lane)
    out = t.run(rounds=2)
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 1
    assert not _stores(out, MemSpace.REGISTER, r1)
    assert not _stores(out, MemSpace.REGISTER, r2)


def test_a_load_behind_a_store_of_the_element_keeps_its_load():
    """Between the loads, the lanes store into element 1, and behind them
    read elements 0 and 1: the second load finds that store in element 1,
    and in element 0 what the first loaded -- a number and a register, which
    no condition of the lanes tells apart."""
    t = Body()
    b = t.b

    def store(r0):
        with b.if_(t.valid):
            b.store(t.q, 1.0, t.element(1))
    _, r1 = _reload(t, store)
    b.store(t.p, b.op('add', REAL, b.load(r1, 0), b.load(r1, 1)), t.lane)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 2


def test_a_load_behind_another_lanes_store_of_the_element_keeps_its_load():
    """Each lane stores into the next lane's element between the loads: the
    second load finds that store, which the first did not.  A value that
    differs between the lanes, so that the lane's own store does not answer
    for it, as it would for a number."""
    t = Body()
    b = t.b
    nxt = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), LANES, hint='n')

    def store(r0):
        with b.if_(t.valid):
            b.store(t.q, b.op('mul', REAL, b.load(r0, 2), 3.0, hint='v'),
                    t.element(1, nxt))
    _reload(t, store)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 2


def test_a_load_behind_a_write_of_the_register_keeps_its_load():
    """`r0` written again between the loads, and read behind them: what the
    second load reads is in memory alone."""
    t = Body()
    b = t.b
    r0, r1 = _reload(t, lambda r0: t.fill(r0, 3.0))
    b.store(t.p, b.op('add', REAL, b.load(r1, 2), b.load(r0, 2)), t.lane)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 2


def test_an_element_another_lane_loaded_keeps_its_load():
    """Each lane loads the next lane's elements first: what the second load
    wants sits in the registers of another lane."""
    t = Body()
    nxt = t.b.op('rem', INDEX, t.b.op('add', INDEX, t.lane, 1), LANES,
                 hint='n')
    _reload(t, lane=nxt)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 2


def test_an_element_loaded_under_a_guard_spelled_in_text_keeps_its_load():
    """Where `active` does not hold, `r0` holds nothing the first load
    read."""
    t = Body()
    _reload(t, guard=lambda: t.b.Block('if (active)'))
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 2


def test_a_body_past_the_budget_is_left_as_it_is(monkeypatch):
    """The body of the first test, too large for the flow to keep track of:
    the load stays where it is, and the report says why."""
    from tensorforge.backend.pir import flow
    monkeypatch.setattr(flow, 'BUDGET', 20)
    t = Body()
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    t.load_q(t.registers('r1'))
    out = t.run()
    assert out is t.body
    assert _loads(out, MemSpace.GLOBAL, t.q)
    assert any('more than 20 accesses' in line for line in t.report), t.report
    assert drop_dead_stores(t.body, threads=LANES, arena='shrMem') is t.body


# -- what a producer promises of its image (`IRBuilder.promise_zero`) --------

def _handed_over(promise, predicated=False):
    """`r` computed on every lane, its rows below 10 stored into `s` and
    loaded back into `t` by those lanes, the rest of `t` its zeros, and `t`
    read on every lane into `p` -- the way a product hands its result to the
    next through shared memory.  `promise` says `r` holds zeros past row 10;
    `predicated` loads back with a predicate instead of in a branch."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    r = t.registers('r0', 1)
    b.store(r, b.op('mul', REAL, 2.0, b.load(t.p, t.lane)), 0)
    rows = b.op('lt', BOOL, t.lane, 10, hint='g')
    if promise:
        past = b.op('ge', BOOL, t.lane, 10, hint='o')
        b.promise_zero(r, 0, predicate=past)
    with b.if_(rows):
        b.store(s, b.load(r, 0), t.lane)
    back = t.registers('r1', 1)
    if predicated:
        b.store(back, b.load(s, t.lane, predicate=rows), 0)
    else:
        with b.if_(rows):
            b.store(back, b.load(s, t.lane), 0)
    b.store(t.p, b.load(back, 0), t.lane)
    return t, r


def _selects(body):
    return [x for x in walk_stmts(body) if x.op == 'select']


def test_lanes_a_promise_holds_zero_read_the_register_as_the_rest():
    """Lanes below 10 read what `r` holds, the rest the zeros `t` was
    declared with -- which `r` holds there as well, as its producer
    promises: the read is `r`'s, on every lane, and no select tells the
    lanes apart."""
    t, r = _handed_over(promise=True)
    out = t.run()
    assert not _selects(out)
    assert not _loads(out, MemSpace.SHARED)


def test_without_the_promise_a_select_tells_the_lanes_apart():
    t, r = _handed_over(promise=False)
    out = t.run()
    assert _selects(out)


def test_a_predicated_load_of_a_promised_image_reads_it_on_every_lane():
    """The load back stands under a predicate whose lanes left out are
    given a zero, and `r` holds one there: no predicate, no select."""
    t, r = _handed_over(promise=True, predicated=True)
    out = t.run()
    assert not _selects(out)
    assert not [x for x in walk_stmts(out) if x.op == Op.LOAD
                and x.predicate is not None]
    t, r = _handed_over(promise=False, predicated=True)
    out = t.run()
    assert _selects(out) or [x for x in walk_stmts(out) if x.op == Op.LOAD
                             and x.predicate is not None]


def test_the_marks_around_a_clearing_store_write_nothing():
    """`mark clears` and `mark cleared` tell the barrier placement about the
    order of the writes (`pir.barriers`) and change no element: what the
    clearing store wrote between them is read from its registers."""
    t = Body()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    r = t.registers('r0', 1)
    b.store(r, b.op('mul', REAL, 2.0, b.load(t.p, t.lane)), 0)
    b.mark('clears', s)
    b.store(s, b.load(r, 0), t.lane)
    b.mark('cleared', s)
    b.store(t.p, b.load(s, t.lane), t.lane)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED)


def test_a_chain_of_copies_is_read_where_the_first_read():
    """`q` loaded back into `r1`, copied into `r2`, that into `r3`, and `r3`
    read: a round takes one link of the chain, and the rounds run until
    nothing changes -- the read is `r0`'s, and no copy is left."""
    t = Body()
    b = t.b
    r0, r1, r2, r3 = (t.registers(f'r{i}') for i in range(4))
    t.fill(r0)
    t.store_q(r0)
    with b.if_(t.valid):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            j = hop.induction
            b.store(r1, b.load(t.q, t.element(j)), j)
    for src, dst in ((r1, r2), (r2, r3)):
        with b.for_(0, COLS, 1, unroll=True) as hop:
            b.store(dst, b.load(src, hop.induction), hop.induction)
    b.store(t.p, b.load(r3, 0), t.lane)
    body = b.finish()
    verify(body)
    out = forward_memory(body, b.scratch, threads=LANES, arena='shrMem')
    verify(out)
    assert not _loads(out, MemSpace.GLOBAL, t.q)
    for r in (r1, r2, r3):
        assert not _loads(out, MemSpace.REGISTER, r), r
    assert _loads(out, MemSpace.REGISTER, r0)


# --------------------------------------------------------------------------- #
# Under the explicit vector
# --------------------------------------------------------------------------- #

#: The lanes of a multiplication under the explicit vector, and how a value
#: spread over them is laid out: one vector, lane `l` its `l`-th element --
#: and a vector of the first half of them, a tail's.
WIDE = 16
SPREAD = RegisterLayout((LaneAxis(WIDE, 1),))
HALF = RegisterLayout((LaneAxis(WIDE // 2, 1),))


class Vectors:
    """A body as the explicit vector writes it (`Flow.simd`): one work-item,
    its values vectors of `WIDE` lanes and its indices without a lane.  `p`,
    two vectors of global memory, `s`, two of the multiplication's shared
    memory, and `r` and `t`, register images of two vectors each."""

    def __init__(self):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        self.p = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
            BufferType(F32, (2 * WIDE,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        self.s = b.alloc(F32, (2 * WIDE,), MemSpace.SHARED, hint='s',
                         extern='s0', arena='shrMem')
        self.r = b.alloc(F32, (2 * WIDE,), MemSpace.REGISTER, hint='r',
                         extern='r', init='{}')
        self.t = b.alloc(F32, (2 * WIDE,), MemSpace.REGISTER, hint='t',
                         extern='t', init='{}')

    def vector(self, at=0):
        """A vector no pass knows: twice the one of `p` from `at` on."""
        b = self.b
        return b.op('mul', REAL, b.load(self.p, at, layout=SPREAD), 2.0,
                    hint='v')

    def lanes(self):
        """The lanes' own index, `0, 1, ...` (`IRBuilder.lane_index`)."""
        return self.b.rawexpr(f'simd<int32_t, {WIDE}>(0, 1)', type_=INDEX,
                              hint='lead', pure=True, movable=True,
                              layout=SPREAD, attrs=(('lanes', WIDE),))

    def through_s(self):
        """`r` holds a vector, and `s` what `r` held."""
        b = self.b
        b.store(self.r, self.vector(), 0)
        b.store(self.s, b.load(self.r, 0, layout=SPREAD), 0)

    def run(self, exchange=None):
        body = self.b.finish()
        verify(body)
        self.report = []
        out = forward_memory(body, self.b.scratch, threads=WIDE,
                             arena='shrMem', simd=True, exchange=exchange,
                             report=self.report)
        verify(out)
        return out


def test_a_vector_loaded_back_as_it_was_stored_is_read_from_its_register():
    """What `r` held, stored to `s` and loaded back as the same vector, is
    read from `r`: lane `l` reads the element lane `l` stored.  The store
    nothing reads any more goes."""
    t = Vectors()
    b = t.b
    t.through_s()
    b.store(t.t, b.load(t.s, 0, layout=SPREAD), 0)
    b.store(t.p, b.load(t.t, 0, layout=SPREAD), WIDE)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, t.s)
    assert not _stores(out, MemSpace.SHARED, t.s)


def test_a_vector_loaded_from_another_element_on_keeps_its_load():
    """From element 1 on, the vector is lane `l + 1`'s element in lane `l`,
    and the next vector's first: what no lane stored itself."""
    t = Vectors()
    b = t.b
    t.through_s()
    b.store(t.s, t.vector(WIDE), WIDE)
    b.store(t.p, b.load(t.s, 1, layout=SPREAD), WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_an_element_of_a_vector_every_lane_reads_keeps_its_load():
    """One element every lane reads is the element one lane stored."""
    t = Vectors()
    b = t.b
    t.through_s()
    b.store(t.p, b.load(t.s, 3, layout=SCALAR_LAYOUT), WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_a_vector_of_a_zeroed_image_is_not_read_as_a_number():
    """Under SPMD a load of a zeroed image is the number; a vector is not
    one."""
    t = Vectors()
    b = t.b
    b.store(t.p, b.load(t.r, 0, layout=SPREAD), 0)
    out = t.run()
    assert _loads(out, MemSpace.REGISTER, t.r)


def _stored_to_p(t, out):
    """The statement that defines what the last store into `p` stores."""
    defs = {x.target[0].id: x for x in walk_stmts(out) if x.target}
    last = [x for x in walk_stmts(out)
            if x.op == Op.STORE and x.args[0] is t.p][-1]
    return defs.get(getattr(last.args[1], 'id', None))


def test_a_register_vector_between_two_is_not_what_was_stored_from_it():
    """A vector of `r` from half a vector on holds the second half of one
    and the first of the next, each in other lanes than they are there: `r`
    is placed nowhere after it, and what `s` holds is not read from `r`,
    which holds something else by then.  What `r` held stands in a loop,
    where nothing behind it reads it either."""
    t = Vectors()
    b = t.b
    with b.for_(0, 1, 1, unroll=True):
        t.through_s()
    b.store(t.r, t.vector(WIDE), WIDE // 2)
    b.store(t.p, b.load(t.s, 0, layout=SPREAD), WIDE)
    out = t.run()
    d = _stored_to_p(t, out)
    assert d is not None and d.op == Op.LOAD and d.args[0] is t.s


def test_a_tail_reads_the_register_its_lanes_were_stored_from():
    """A tail stored as the lanes that hold rows and zeros past them -- a
    select over the lanes' own index -- and loaded back as the lanes that
    hold rows: read from `r`, which those lanes were stored from."""
    t = Vectors()
    b = t.b
    b.store(t.r, t.vector(), 0)
    rows = b.op('lt', BOOL, t.lanes(), WIDE // 2, hint='g')
    pad = b.op('select', REAL, rows, b.load(t.r, 0, layout=SPREAD), 0.0,
               hint='pad')
    b.store(t.s, pad, 0)
    b.store(t.p, b.load(t.s, 0, layout=HALF), WIDE)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, t.s)
    assert any(s.op == Op.LOAD and s.args[0] is t.r
               and s.target[0].layout == HALF for s in walk_stmts(out))


def test_a_vector_loaded_under_a_mask_keeps_its_load():
    """A predicate of a vector is a mask over its lanes."""
    t = Vectors()
    b = t.b
    t.through_s()
    rows = b.op('lt', BOOL, t.lanes(), WIDE // 2, hint='g')
    b.store(t.p, b.load(t.s, 0, layout=SPREAD, predicate=rows, other=0.0),
            WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_a_value_spread_over_more_lanes_than_the_load_keeps_its_load():
    """A vector of every lane is not the vector of the first half of
    them."""
    t = Vectors()
    b = t.b
    b.store(t.s, t.vector(), 0)
    b.store(t.p, b.load(t.s, 0, layout=HALF), WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_a_vector_of_more_lanes_than_the_multiplications_keeps_its_load():
    """A vector of twice the lanes writes the vector behind the first as
    well, from lanes the multiplication does not have: what `s` held there
    before is no more."""
    t = Vectors()
    b = t.b
    b.store(t.s, t.vector(), WIDE)
    wide = b.load(t.p, 0, layout=RegisterLayout((LaneAxis(2 * WIDE, 1),)))
    b.store(t.s, wide, 0)
    b.store(t.p, b.load(t.s, WIDE, layout=SPREAD), 0)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_a_vector_of_two_elements_a_lane_keeps_a_load_of_what_it_wrote():
    """Two elements a lane write two vectors of one."""
    t = Vectors()
    b = t.b
    b.store(t.s, t.vector(), WIDE)
    two = b.load(t.p, 0, type_=ScalarType(F32, 2), layout=SPREAD, align=8)
    b.store(t.s, two, 0, align=8)
    b.store(t.p, b.load(t.s, WIDE, layout=SPREAD), 0)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_an_element_every_lane_writes_into_a_register_vector_is_one_lanes():
    """`r[0]` written as an element of `t` every lane holds is the first
    lane's part of the vector `r` is read as, the others hold what `r` was:
    the vector is not the vector of `t` that element is part of."""
    t = Vectors()
    b = t.b
    b.store(t.t, t.vector(WIDE), WIDE)
    b.store(t.r, t.vector(), 0)
    b.store(t.r, b.load(t.t, WIDE, layout=SCALAR_LAYOUT), 0)
    b.store(t.p, b.load(t.r, 0, layout=SPREAD), WIDE)
    out = t.run()
    d = _stored_to_p(t, out)
    assert d is not None and d.op == Op.LOAD and d.args[0] is t.r


def test_another_lanes_store_is_not_read_across_under_the_explicit_vector():
    """Every lane reading the element one lane stored is no exchange of
    registers here: the work-item holds every lane, and the vector the
    store wrote is no register a lane holds."""
    t = Vectors()
    b = t.b
    t.through_s()
    b.store(t.p, b.load(t.s, 3, layout=SCALAR_LAYOUT), WIDE)
    out = t.run(exchange=WIDE)
    assert _loads(out, MemSpace.SHARED, t.s)


def test_a_vector_read_from_its_second_lane_on_keeps_its_load():
    """A window's head: the first lane reads nothing, and holds zero."""
    t = Vectors()
    b = t.b
    t.through_s()
    b.store(t.p, b.load(t.s, 0, layout=SPREAD, head=1), WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


#: The lanes of a full-lane tail, fewer than the vector's.
TAIL = 10


def test_a_tail_loaded_back_is_read_from_its_register_with_zeros_past_it():
    """What `r` held, stored to `s` and loaded back as a full-lane tail of
    `TAIL` lanes: those read what their own lanes stored, from `r`, and the
    read of `r` that takes the load's place gives the lanes past them zeros,
    as the load did.  The store nothing reads any more goes."""
    t = Vectors()
    b = t.b
    t.through_s()
    b.store(t.t, b.load(t.s, 0, layout=SPREAD, valid=TAIL), 0)
    b.store(t.p, b.load(t.t, 0, layout=SPREAD), WIDE)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, t.s)
    assert not _stores(out, MemSpace.SHARED, t.s)
    tails = [s for s in walk_stmts(out) if s.op == Op.LOAD
             and s.accesses[0].base is t.r and s.attr('valid') == TAIL]
    assert tails, 'the read of `r` gives zeros past the tail'


def test_a_tail_reads_nothing_past_its_lanes():
    """`s` holds a vector, then a tail of what `r` held over it, and a tail
    loads it back: the load reads the lanes the second store wrote, and
    what the first left past them nothing reads -- both stores go."""
    t = Vectors()
    b = t.b
    b.store(t.r, t.vector(), 0)
    b.store(t.s, t.vector(WIDE), 0)
    b.store(t.s, b.load(t.r, 0, layout=SPREAD), 0, valid=TAIL)
    b.store(t.p, b.load(t.s, 0, layout=SPREAD, valid=TAIL), WIDE)
    out = t.run()
    assert not _loads(out, MemSpace.SHARED, t.s)
    assert not _stores(out, MemSpace.SHARED, t.s)


def test_a_load_behind_a_tail_store_reads_what_was_there_past_it():
    """`s` holds a vector, then a tail of what `r` held over it, and the
    whole vector is loaded back: its lanes up to the tail's read the tail,
    those past them the vector before -- two stores, no one vector to take
    instead.  The load stays, and both stores with it."""
    t = Vectors()
    b = t.b
    b.store(t.r, t.vector(), 0)
    b.store(t.s, t.vector(WIDE), 0)
    b.store(t.s, b.load(t.r, 0, layout=SPREAD), 0, valid=TAIL)
    b.store(t.p, b.load(t.s, 0, layout=SPREAD), WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)
    assert len(_stores(out, MemSpace.SHARED, t.s)) == 2


def test_a_tail_loaded_back_from_a_value_keeps_its_load():
    """A vector no register holds, stored to `s` and loaded back as a tail:
    the value holds what it holds past the tail's lanes, where the load
    gives zeros, so the load stays."""
    t = Vectors()
    b = t.b
    b.store(t.s, t.vector(), 0)
    b.store(t.p, b.load(t.s, 0, layout=SPREAD, valid=TAIL), WIDE)
    out = t.run()
    assert _loads(out, MemSpace.SHARED, t.s)


def test_a_tail_store_into_a_register_array_writes_every_lane():
    """A store into a register array writes every lane of its value, a
    full-lane tail's as well: a load behind it reads the store in every
    lane, and none what the allocation left."""
    t = Vectors()
    b = t.b
    b.store(t.r, t.vector(), 0, valid=TAIL)
    b.store(t.p, b.load(t.r, 0, layout=SPREAD), WIDE)
    body = b.finish()
    verify(body)
    f = Flow(body, WIDE, arena='shrMem', simd=True)
    load = next(x for x in walk_stmts(body) if x.op == Op.LOAD
                and x.accesses[0].base is t.r)
    reach = [r for n in f.executions(load) for r in f.reach[n]]
    assert [r.mask for r in reach] == [(1 << WIDE) - 1]
    assert f.insts[reach[0].writer].stmt.op == Op.STORE


def test_the_lanes_own_index_says_so_under_the_explicit_vector():
    """`IRBuilder.lane_index` writes a vector's `0, 1, ...` as text, and
    says in an attribute what the text is, for the flow to read."""
    from tensorforge.common.context import Context
    b = IRBuilder(fptype=F32, context=Context(arch='pvc', backend='esimd',
                                              fp_type=F32))
    b.lane_index(WIDE)
    body = b.finish()
    assert [s.attr('lanes') for s in walk_stmts(body)
            if s.op == Op.RAWEXPR] == [WIDE]


def test_the_forward_under_the_explicit_vector_asks_for_vectors(monkeypatch):
    """Built for ESIMD, the forward runs, and asks of vectors."""
    import importlib.util
    import pathlib
    from tensorforge.backend.pir import pipeline
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    asked = []
    real = pipeline.forward_memory

    def spy(body, scratch, **kwargs):
        asked.append(kwargs.get('simd'))
        return real(body, scratch, **kwargs)
    monkeypatch.setattr(pipeline, 'forward_memory', spy)
    path = pathlib.Path(__file__).parent / 'cases' / 'mixed' / 'ml_then_ew.py'
    spec = importlib.util.spec_from_file_location('ml_then_ew', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    gen = Generator(case.descr_list(), Context(arch='pvc', backend='esimd',
                                               fp_type=case.DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    assert asked and all(asked)
