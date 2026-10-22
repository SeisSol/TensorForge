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

from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, Access,
                                          BufferType, Effect, MemSpace, Op,
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
    t = Body()
    b = t.b

    def store(r0):
        with b.if_(t.valid):
            b.store(t.q, 1.0, t.element(1))
    _reload(t, store)
    out = t.run()
    assert len(_loads(out, MemSpace.GLOBAL, t.q)) == 2


def test_a_load_behind_another_lanes_store_of_the_element_keeps_its_load():
    """Each lane stores into the next lane's element between the loads: the
    second load finds that store, which the first did not."""
    t = Body()
    b = t.b
    nxt = b.op('rem', INDEX, b.op('add', INDEX, t.lane, 1), LANES, hint='n')

    def store(r0):
        with b.if_(t.valid):
            b.store(t.q, 1.0, t.element(1, nxt))
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
    monkeypatch.setattr(flow, 'BUDGET', 50)
    t = Body()
    r = t.registers('r0')
    t.fill(r)
    t.store_q(r)
    t.load_q(t.registers('r1'))
    out = t.run()
    assert out is t.body
    assert _loads(out, MemSpace.GLOBAL, t.q)
    assert any('more than 50 elements' in line for line in t.report), t.report
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
