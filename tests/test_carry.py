# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the trips of a merged run hand each other through memory, handed
over in registers (`pir.carry`), asked of bodies built by hand.

A loop whose trips are closed loads an element's accumulator at the head of
every trip and stores it at the tail.  The pass takes the load ahead of the
loop and the store behind it where every lane reads back what it stored and
nothing else in the loop touches the buffer -- and leaves the loop alone
otherwise, which costs a round trip through memory and never a value.  The
merged runs of the generator are asked in `test_rolling.py`; here are the
refusals, one at a time.
"""

from __future__ import annotations

from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.carry import carry_trips
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, BufferType,
                                          Effect, MemSpace, Op, ScalarType,
                                          Uniformity, walk_stmts)
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32
LANES = 32
ROWS = 24
COLS = 5


class Run:
    """`Q += f(i)` over a closed trip: the head loads the element's `Q` into
    an image, the trip adds onto it, the tail stores the result.

    `table` hands `Q`'s binding to a table of pointers; `elsewhere` reads `Q`
    in the trip besides the load; `other_rows` stores the result one row
    down from where the load read it; `writes_head` writes the image the load
    filled; `atomic` stores the tail with an atomic update and loads nothing.
    `stored_ahead` stores that many rows of `Q` from an image of the code in
    front of the loop, which is how a closed trip meets a result that code
    left in registers (`Generator._emit_closed_loop`).  `late_read` reads the
    loaded image once more behind the tail's computation, into another
    output; `twice` adds one slot of the tail into a second element;
    `tail_cols` is how many columns the tail stores, `head_cols` how many
    the head loads.  `late_slot` computes the slot the head writes behind the
    load of the element, as an unrolled copy does.
    """

    def __init__(self, *, table=False, elsewhere=False, other_rows=False,
                 writes_head=False, atomic=False, stored_ahead=None,
                 late_read=False, twice=False, tail_cols=COLS, trips=4,
                 late_slot=False, head_cols=COLS):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        q = b.decl_expr(
            'float *const __restrict__ glb_m0', '&m0[{0} * 120]',
            BufferType(F32, (ROWS * COLS,), MemSpace.GLOBAL), 'm0', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
            extern='glb_m0')
        self.q = q
        if table:
            b.decl_expr('float *const *const table', '{{{0}}}',
                        BufferType(F32, (1,), MemSpace.GLOBAL), 'table',
                        args=(q,), kind=Effect.READ, space=MemSpace.GLOBAL,
                        hint='table', extern='table')
        p = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
            BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
        inside = b.op('lt', BOOL, lane, ROWS, hint='g')
        real = ScalarType(F32)

        def element(j, down=0):
            return b.op('add', INDEX, b.op('add', INDEX, lane, down),
                        b.op('mul', INDEX, j, ROWS), hint='a')

        if stored_ahead is not None:
            before = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r0',
                             extern='r0', init='{}')
            with b.for_(0, COLS, 1, unroll=True) as hop:
                b.store(before, b.op('mul', real, 2.0, 3.0), hop.induction)
            with b.if_(b.op('lt', BOOL, lane, stored_ahead, hint='h')):
                with b.for_(0, COLS, 1, unroll=True) as hop:
                    j = hop.induction
                    b.store(q, b.load(before, j), element(j))
            self.before = before
        with b.for_(0, trips, 1, rolled=True, hint='i'):
            head = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r1',
                           extern='r1', init='{}')
            if not atomic:
                with b.transfer():
                    with b.for_(0, head_cols, 1, unroll=True) as hop:
                        j = hop.induction
                        value = b.load(q, element(j), predicate=inside)
                        slot = (b.op('add', INDEX, j, 0, hint='s')
                                if late_slot else j)
                        b.store(head, value, slot)
            if elsewhere:
                b.store(head, b.load(q, element(0), predicate=inside), 0)
            tail = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r2',
                           extern='r2', init='{}')
            with b.if_(inside):
                with b.for_(0, COLS, 1, unroll=True) as hop:
                    j = hop.induction
                    b.store(tail, b.op('add', real, b.load(head, j), 1.0), j)
            if writes_head:
                b.store(head, 0.0, 0)
            if late_read:
                b.store(p, b.load(head, 0), lane)
            with b.if_(inside):
                with b.for_(0, tail_cols, 1, unroll=True) as hop:
                    j = hop.induction
                    at = element(j, 1 if other_rows else 0)
                    b.store(q, b.load(tail, j), at, atomic=atomic)
                if twice:
                    b.store(q, b.load(tail, 0), element(1), atomic=atomic)
        self.body = b.finish()
        verify(self.body)

    def carry(self):
        self.report = []
        out = carry_trips(self.body, self.b.scratch, threads=LANES,
                          report=self.report)
        verify(out)
        return out


def _split(body):
    """`(ahead, loop, behind)` around the rolled loop of `body`."""
    at = next(i for i, s in enumerate(body) if s.op is Op.FOR
              and s.attr('rolled'))
    return list(body[:at]), body[at], list(body[at + 1:])


def _touches(stmts, buffer):
    """The loads and stores among `stmts` that reach `buffer` -- not the
    declaration of a binding, which names what it points into."""
    return [a for s in walk_stmts(tuple(stmts)) for a in s.accesses
            if s.op in (Op.LOAD, Op.STORE) and (
                a.base is buffer or getattr(a.base, 'id', None)
                == getattr(buffer, 'id', None))]


def _q_root(run):
    """`Q` as its accesses name it."""
    return next(a.base for s in walk_stmts(run.body) for a in s.accesses
                if s.op in (Op.LOAD, Op.STORE) and a.space is MemSpace.GLOBAL
                and getattr(a.base, 'name', a.base) == 'm0')


def test_the_load_goes_ahead_and_the_store_behind():
    run = Run()
    out = run.carry()
    ahead, loop, behind = _split(out)
    q = _q_root(run)
    assert not _touches(loop.regions[0].body, q)
    assert any(not a.writes for a in _touches(ahead, q))
    assert any(a.writes for a in _touches(behind, q))
    assert any('in registers over 4 trips' in line for line in run.report)


def test_the_trips_add_into_the_image_they_were_loaded_into():
    """One image across the trips: what the tail of a trip computes is what
    the head of the next reads, slot by slot, so it is computed in place
    rather than copied over at the end of each trip."""
    run = Run()
    out = run.carry()
    ahead, loop, behind = _split(out)
    images = {s.target[0].id for s in ahead if s.op is Op.ALLOC}
    inside = list(walk_stmts(loop.regions[0].body))
    written = {a.base.id for s in inside for a in s.accesses
               if a.space is MemSpace.REGISTER and a.writes}
    assert written == images
    assert not any(s.op is Op.ALLOC for s in inside)


def test_a_binding_handed_on_keeps_the_buffer_in_memory():
    """A table of pointers holding `Q`'s binding reaches `Q` from wherever it
    is read, which no access naming `Q` says."""
    run = Run(table=True)
    assert run.carry() == run.body
    assert any('handed on' in line for line in run.report)


def test_a_buffer_the_trip_reads_elsewhere_stays_in_memory():
    run = Run(elsewhere=True)
    assert run.carry() == run.body


def test_a_lane_that_stores_another_element_keeps_it_in_memory():
    """The store one row down from where the load read: what the next trip
    loads is what another lane stored, which no register of this lane
    holds."""
    run = Run(other_rows=True)
    assert run.carry() == run.body


def test_a_column_stored_and_never_loaded_keeps_the_buffer_in_memory():
    """The tail stores a column the head does not load: its slot of the head's
    image starts every trip from nothing, and carried, it would start from
    what the last trip left."""
    run = Run(head_cols=COLS - 1)
    assert run.carry() == run.body
    assert any('does not read into that slot' in line for line in run.report)


def test_an_image_the_trip_writes_is_not_the_one_it_loaded():
    run = Run(writes_head=True)
    assert run.carry() == run.body


def test_an_atomic_update_is_summed_and_applied_once():
    run = Run(atomic=True)
    out = run.carry()
    ahead, loop, behind = _split(out)
    q = _q_root(run)
    assert not _touches(loop.regions[0].body, q)
    assert any(a.kind & Effect.ATOMIC for a in _touches(behind, q))
    declared, = [s for s in ahead if s.op is Op.ALLOC]
    assert declared.attr('init') == '{}'


def test_what_the_code_ahead_stored_is_read_from_its_registers():
    """The store ahead of the loop and the load the carry puts there meet:
    the load reads the image the store read, and since the store behind the
    loop writes every element again and nothing in between reads one, the
    store ahead of the loop goes."""
    run = Run(stored_ahead=ROWS)
    out = run.carry()
    ahead, loop, behind = _split(out)
    q = _q_root(run)
    assert not _touches(ahead, q), 'still stored, or loaded, ahead'
    assert any(not a.writes for a in _touches(ahead, run.before))
    assert any('not stored there' in line for line in run.report)


def test_a_slot_computed_behind_the_load_is_read_where_it_is_written():
    """The load of the element comes first and the slot it fills after it:
    read from the store's image, the slot is read where the head writes it,
    and not where the load stood, ahead of its index."""
    run = Run(stored_ahead=ROWS, late_slot=True)
    out = run.carry()
    ahead, _, _ = _split(out)
    assert not _touches(ahead, _q_root(run))
    assert any(not a.writes for a in _touches(ahead, run.before))


def test_what_the_code_ahead_stored_only_in_part_is_loaded():
    """Where the store ahead of the loop leaves rows out, the load reads them
    from memory, so it stays a load -- and the store stays as well."""
    run = Run(stored_ahead=ROWS // 2)
    out = run.carry()
    ahead, loop, behind = _split(out)
    q = _q_root(run)
    assert any(not a.writes for a in _touches(ahead, q))
    assert any(a.writes for a in _touches(ahead, q))
    assert not _touches(loop.regions[0].body, q)


def test_the_store_ahead_stays_where_the_one_behind_writes_less():
    """The store behind the loop writes one column fewer than the store
    ahead of it: that column is the store ahead's to write, and it stays."""
    run = Run(stored_ahead=ROWS, tail_cols=COLS - 1)
    out = run.carry()
    ahead, _, _ = _split(out)
    assert any(a.writes for a in _touches(ahead, _q_root(run)))


def test_an_image_read_once_more_late_is_copied_and_not_shared():
    """The trip reads the loaded image behind the computation of the new
    value: computed in place, that read would see the new value.  So the new
    value is computed in an image of its own and copied over at the end of
    the trip."""
    run = Run(late_read=True)
    out = run.carry()
    _, loop, _ = _split(out)
    assert any('in registers' in line for line in run.report)
    assert any(s.op is Op.ALLOC for s in walk_stmts(loop.regions[0].body))


def test_an_atomic_update_that_adds_a_slot_twice_stays_as_it_is():
    """Summed in registers, a slot added into two elements would go into
    each of them twice."""
    run = Run(atomic=True, twice=True)
    assert run.carry() == run.body
