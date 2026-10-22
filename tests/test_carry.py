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
refusals, one at a time -- and the same of a temporary in shared memory that
every lane keeps to itself (`Own`).
"""

from __future__ import annotations

import contextlib

from tensorforge.backend.pir import IRBuilder, dce, verify
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
    output, and `early_read` ahead of it; `twice` adds one slot of the tail
    into a second element;
    `tail_cols` is how many columns the tail stores, `head_cols` how many
    the head loads.  `late_slot` computes the slot the head writes behind the
    load of the element, as an unrolled copy does.

    `rows` is how many rows the head loads and the tail stores; `padded`
    stores ahead of the loop every row of `stored_ahead`, those past `rows`
    as zeros, in one statement -- a `select` per element, as a store does
    that writes its column whole.  `read_behind` loads `Q` once more behind
    the loop, into an image the code there reads.  `spelled` makes the
    atomic update a raw statement with its address spelled into its text,
    and `guarded` puts every store of `Q` under a guard spelled in text,
    `guard`, as a batch element's guard is on AMD, and `guarded_load` the
    load at the head.  `tail_read` reads the tail's image
    for another output as well, and `overwrite` writes its first slot once
    more before the tail stores it.  `split` computes the columns past the
    second under a guard of their own, which holds for the same lanes, as
    the rows past a multiplication's first lanes get theirs; `wide` computes
    the tail for every row, the rows the head does not load included.
    """

    def __init__(self, *, table=False, elsewhere=False, other_rows=False,
                 writes_head=False, atomic=False, stored_ahead=None,
                 late_read=False, early_read=False, twice=False,
                 tail_cols=COLS, trips=4, late_slot=False, head_cols=COLS,
                 rows=ROWS, padded=False, read_behind=False, spelled=False,
                 guarded=False, guarded_load=False, guard='if (active)',
                 tail_read=False, overwrite=False, split=False, wide=False):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')

        def guarding(on):
            return b.Block(guard) if on else contextlib.nullcontext()
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
        valid = (inside if rows == ROWS
                 else b.op('lt', BOOL, lane, rows, hint='v'))
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
                    value = b.load(before, j)
                    if padded:
                        value = b.op('select', real, valid, value, 0.0,
                                     hint='pad')
                    with guarding(guarded):
                        b.store(q, value, element(j))
            self.before = before
        with b.for_(0, trips, 1, rolled=True, hint='i'):
            head = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r1',
                           extern='r1', init='{}')
            if not atomic:
                with b.transfer():
                    with b.for_(0, head_cols, 1, unroll=True) as hop:
                        j = hop.induction
                        with guarding(guarded_load):
                            value = b.load(q, element(j), predicate=valid)
                            slot = (b.op('add', INDEX, j, 0, hint='s')
                                    if late_slot else j)
                            b.store(head, value, slot)
            if elsewhere:
                b.store(head, b.load(q, element(0), predicate=inside), 0)
            if early_read:
                b.store(p, b.load(head, 0), lane)
            tail = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r2',
                           extern='r2', init='{}')
            cols = 2 if split else COLS
            with b.if_(inside if wide else valid):
                with b.for_(0, cols, 1, unroll=True) as hop:
                    j = hop.induction
                    b.store(tail, b.op('add', real, b.load(head, j), 1.0), j)
            if split:
                same = b.op('le', BOOL, lane, rows - 1, hint='s')
                with b.if_(same):
                    with b.for_(cols, COLS, 1, unroll=True) as hop:
                        j = hop.induction
                        b.store(tail, b.op('add', real, b.load(head, j), 1.0),
                                j)
            if writes_head:
                b.store(head, 0.0, 0)
            if late_read:
                b.store(p, b.load(head, 0), lane)
            if tail_read:
                b.store(p, b.load(tail, 0), lane)
            if overwrite:
                with b.if_(valid):
                    b.store(tail, b.op('mul', real, 5.0, 1.0), 0)
            with b.if_(valid):
                with b.for_(0, tail_cols, 1, unroll=True) as hop:
                    j = hop.induction
                    at = element(j, 1 if other_rows else 0)
                    if spelled:
                        named = b.op('add', INDEX, at, 0, hint='a',
                                     escapes=True)
                        b.access_stmt(f'atomicAdd(&glb_m0[{named}], {{0}});',
                                      q, Effect.ATOMIC,
                                      args=(b.load(tail, j), named), fmt=True)
                    else:
                        value = b.load(tail, j)
                        with guarding(guarded):
                            b.store(q, value, at, atomic=atomic)
                if twice:
                    b.store(q, b.load(tail, 0), element(1), atomic=atomic)
        if read_behind:
            after = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r3',
                            extern='r3', init='{}')
            with b.transfer():
                with b.for_(0, COLS, 1, unroll=True) as hop:
                    j = hop.induction
                    b.store(after, b.load(q, element(j), predicate=valid), j)
            b.store(p, b.load(after, 0), lane)
        self.body = b.finish()
        verify(self.body)

    def carry(self):
        self.report = []
        out = carry_trips(self.body, self.b.scratch, threads=LANES,
                          report=self.report)
        verify(out)
        return out


class Own:
    """A temporary in shared memory, `S`, that every trip adds onto, each
    lane its own rows: `S[l, c] += 1` for the lanes `l` below `ROWS`, column
    0 in one statement and the columns past it in another, whose register is
    one on from its hop -- the way the products of the free-surface kernel
    leave their temporaries.  The code ahead of the loop stores `S` from
    registers of its own.

    `across` reads `S` behind the loop, lane `l` the row of lane `l + 1`;
    `other` makes a lane add the next lane's row onto its own; `parity` makes
    an odd lane add its second column onto its first; `readonly` has the
    trips read the columns past the first and store nothing of them, and
    `plain` reaches them in loops that are not unrolled; `partial` leaves the
    last column out of the store ahead of the loop, and
    `extra` has the trips leave it alone; `mark` defines `S` anew in the
    trip; `outer` puts it all in a loop of its own, whose body `defines`
    begins with a `mark defines` of `S`.
    """

    def __init__(self, *, across=False, other=False, parity=False,
                 readonly=False, plain=False, partial=False, extra=False,
                 mark=False, outer=False, defines=False, trips=3):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        p = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
            BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
        inside = b.op('lt', BOOL, lane, ROWS, hint='g')
        s = self.s = b.alloc(F32, (ROWS * COLS,), MemSpace.SHARED, hint='s',
                             extern='s0')
        real = ScalarType(F32)
        cols = COLS - 1 if extra else COLS

        def element(j, shift=0):
            row = lane if not shift else b.op(
                'rem', INDEX, b.op('add', INDEX, lane, shift), ROWS)
            return b.op('add', INDEX, row, b.op('mul', INDEX, j, ROWS),
                        hint='a')

        def adds(first, count, shift=0, odd=False, store=True,
                 unroll=True):
            temp = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='t',
                           extern=f't{first}', init='{}')
            with b.if_(inside):
                with b.for_(0, count, 1, unroll=unroll) as hop:
                    col = (hop.induction if not first
                           else b.op('add', INDEX, hop.induction, first))
                    read = (b.op('add', INDEX, col,
                                 b.op('rem', INDEX, lane, 2, hint='odd'))
                            if odd else col)
                    b.store(temp, b.op('add', real,
                                       b.load(s, element(read, shift)), 1.0),
                            col)
            if not store:
                b.store(p, b.load(temp, first), lane)
                return
            with b.if_(inside):
                with b.for_(0, count, 1, unroll=unroll) as hop:
                    col = (hop.induction if not first
                           else b.op('add', INDEX, hop.induction, first))
                    b.store(s, b.load(temp, col), element(col))

        def run():
            if defines:
                b.mark('defines', s)
            before = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r0',
                             extern='r0', init='{}')
            with b.for_(0, COLS, 1, unroll=True) as hop:
                b.store(before, b.op('mul', real, 2.0, 3.0), hop.induction)
            with b.if_(inside):
                with b.for_(0, COLS - 1 if partial else COLS, 1,
                            unroll=True) as hop:
                    b.store(s, b.load(before, hop.induction),
                            element(hop.induction))
            with b.for_(0, trips, 1, rolled=True, hint='i'):
                if mark:
                    b.mark('defines', s)
                adds(0, 1, 1 if other else 0, odd=parity)
                adds(1, cols - 1, store=not readonly, unroll=not plain)
            if across:
                with b.if_(inside):
                    b.store(p, b.load(s, element(0, 1)), lane)

        if outer:
            with b.for_(0, 2, 1, hint='o'):
                run()
        else:
            run()
        self.body = b.finish()
        verify(self.body)

    def carry(self):
        self.report = []
        out = carry_trips(self.body, self.b.scratch, threads=LANES,
                          arena='shrMem', report=self.report)
        verify(out)
        return out


def _split(body):
    """`(ahead, loop, behind)` around the rolled loop of `body`."""
    at = next(i for i, s in enumerate(body) if s.op is Op.FOR
              and s.attr('rolled'))
    return list(body[:at]), body[at], list(body[at + 1:])


def _shared(stmts, buffer):
    """The loads and stores among `stmts` of `buffer` in shared memory."""
    return [a for s in walk_stmts(tuple(stmts)) for a in s.accesses
            if s.op in (Op.LOAD, Op.STORE) and a.base is buffer]


def _touches(stmts, buffer):
    """The loads and stores among `stmts` that reach `buffer` -- not the
    declaration of a binding, which names what it points into."""
    key = getattr(buffer, 'id', None)
    return [a for s in walk_stmts(tuple(stmts)) for a in s.accesses
            if s.op in (Op.LOAD, Op.STORE) and (
                a.base is buffer
                or key is not None and getattr(a.base, 'id', None) == key)]


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


def test_the_trips_add_in_place_under_guards_that_hold_for_the_same_lanes():
    """Two guards, one condition: the lanes that write a slot of the tail's
    image are the lanes that read the slot of the head's, so the trip
    computes in the head's image whichever guard says so."""
    run = Run(split=True)
    out = run.carry()
    _, loop, _ = _split(out)
    assert not any(s.op is Op.ALLOC for s in walk_stmts(loop.regions[0].body))


def test_rows_computed_and_not_stored_keep_their_own_image():
    """The tail is computed for rows the store leaves out: computed in the
    head's image, those would reach the next trip, which reads them from
    memory, as it was before."""
    run = Run(wide=True, rows=ROWS - 4)
    out = run.carry()
    _, loop, _ = _split(out)
    assert any('in registers' in line for line in run.report)
    assert any(s.op is Op.ALLOC for s in walk_stmts(loop.regions[0].body))


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


def test_a_store_ahead_padded_with_numbers_goes_behind_the_loop():
    """The code ahead stores the rows of `Q` from its registers and zeros
    past them, each element in one statement (`select`).  The load reads
    only the rows with values, so it is taken from those registers; and the
    store, which still has the zeros to write, goes behind the loop reading
    the carried image -- in place of the store there, which writes nothing
    it does not."""
    run = Run(stored_ahead=ROWS, padded=True, rows=ROWS - 4)
    out = run.carry()
    ahead, _, behind = _split(out)
    q = _q_root(run)
    assert not _touches(ahead, q)
    stores = [s for s in walk_stmts(tuple(behind)) if s.op is Op.STORE
              and any(a.base is q for a in s.accesses)]
    defs = {t.id: s for s in walk_stmts(tuple(behind)) for t in s.target}
    assert stores and all(defs[s.args[1].id].op == 'select' for s in stores)
    assert any('stored behind it instead' in line for line in run.report)


def test_an_update_onto_what_the_code_ahead_stored_is_one_store():
    """The code ahead stores `Q` from its registers, and the trips add onto
    it atomically: the sum starts from those registers instead of zeros,
    and the store ahead goes behind the loop in place of the update -- a
    store and an update of every element become one store."""
    run = Run(atomic=True, stored_ahead=ROWS)
    out = run.carry()
    ahead, _, behind = _split(out)
    q = _q_root(run)
    assert not _touches(ahead, q)
    written = _touches(behind, q)
    assert written and not any(a.kind & Effect.ATOMIC for a in written)
    assert any(not a.writes for a in _touches(ahead, run.before))
    assert any('summed onto' in line for line in run.report)


def test_an_update_onto_part_of_what_the_code_ahead_stored_stays():
    """The store ahead writes rows the update does not add to: taken behind
    the loop, those would be stored from a sum that never held them."""
    run = Run(atomic=True, stored_ahead=ROWS, rows=ROWS - 4)
    out = run.carry()
    ahead, _, behind = _split(out)
    q = _q_root(run)
    assert any(a.writes for a in _touches(ahead, q))
    assert any(a.kind & Effect.ATOMIC for a in _touches(behind, q))


def test_the_code_behind_the_loop_reads_the_accumulator_from_registers():
    """Behind the loop the store writes `Q` from the carried image, and the
    code there loads it straight back: the load reads the image."""
    run = Run(read_behind=True)
    out = run.carry()
    _, _, behind = _split(out)
    q = _q_root(run)
    assert not [a for a in _touches(behind, q) if not a.writes]
    assert any('taken from registers behind the loop' in line
               for line in run.report)


def test_what_the_store_behind_leaves_out_is_loaded_behind_the_loop():
    """The store behind the loop writes one column fewer than the code
    behind it loads: that column is in memory alone, and the load stays."""
    run = Run(read_behind=True, tail_cols=COLS - 1)
    _, _, behind = _split(run.carry())
    assert [a for a in _touches(behind, _q_root(run)) if not a.writes]


def test_an_update_spelled_in_text_leaves_no_address_in_the_trip():
    """An atomic update names its address in its text, which keeps the
    address from being dropped; summed into registers, the update is gone
    from the trip, and so is what computed its address.  (The trip reads its
    tail besides, so that the sum is added where the update stood.)"""
    run = Run(atomic=True, spelled=True, tail_read=True)
    _, loop, _ = _split(dce(run.carry()))
    assert not [s for s in walk_stmts(loop.regions[0].body)
                if s.attr('escapes') and s.op is not Op.ALLOC]


def _tail_image(run, stmts):
    """The accesses among `stmts` of the image the tail of `run` computes."""
    tail, = [s.target[0] for s in walk_stmts(run.body) if s.op is Op.ALLOC
             and s.attr('extern') == 'r2']
    return [a for s in walk_stmts(tuple(stmts)) for a in s.accesses
            if a.base is tail]


def test_the_trips_add_into_the_sum_itself():
    """The tail's image is written once a slot and then only added onto: the
    trips add each term into the sum instead, as the list written out adds
    into its accumulator, and the image and the copy into the sum go."""
    run = Run(atomic=True)
    _, loop, _ = _split(run.carry())
    assert not _tail_image(run, loop.regions[0].body)


def test_an_image_the_trip_reads_besides_is_added_at_its_end():
    """The trip reads the tail's image for another output as well: summed
    in place, that read would see the sum.  So the tail keeps its image, and
    the end of the trip adds it into the sum."""
    run = Run(atomic=True, tail_read=True)
    _, loop, _ = _split(run.carry())
    assert _tail_image(run, loop.regions[0].body)
    assert any('in registers' in line for line in run.report)


def test_a_slot_written_twice_is_added_at_the_end_of_the_trip():
    """The trip writes a slot of the tail's image afresh after it computed
    it: what the update adds is the second value alone, which a sum that
    took the first would add as well."""
    run = Run(atomic=True, overwrite=True)
    _, loop, _ = _split(run.carry())
    assert _tail_image(run, loop.regions[0].body)


def test_a_guard_spelled_in_text_stays_with_the_update_behind_the_loop():
    """An atomic update under a guard spelled in text: summed in registers,
    the guard holds nothing a trip has to keep from memory, and goes from the
    trip -- and stays around the update behind the loop, which is what it
    guarded.  (The trip reads its tail besides, so that the sum is added at
    its end, where the guard stood.)"""
    run = Run(atomic=True, guarded=True, tail_read=True)
    out = run.carry()
    _, loop, behind = _split(out)
    q = _q_root(run)
    assert not _touches(loop.regions[0].body, q)
    assert not [s for s in walk_stmts(loop.regions[0].body)
                if s.op is Op.RAWBLOCK]
    guards = [s for s in walk_stmts(tuple(behind)) if s.op is Op.RAWBLOCK]
    assert guards and all(
        any(a.kind & Effect.ATOMIC for a in _touches(g.regions[0].body, q))
        for g in guards)


def test_an_update_onto_a_store_under_the_same_guard_is_one_store():
    run = Run(atomic=True, stored_ahead=ROWS, guarded=True)
    out = run.carry()
    ahead, _, behind = _split(out)
    q = _q_root(run)
    assert not _touches(ahead, q)
    assert not any(a.kind & Effect.ATOMIC for a in _touches(behind, q))
    assert any('summed onto' in line for line in run.report)


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


# -- guards spelled in text ---------------------------------------------------

def _guards(stmts, q):
    """The guards spelled in text among `stmts` that hold a store of `q`."""
    return [s for s in walk_stmts(tuple(stmts)) if s.op is Op.RAWBLOCK
            and any(a.base is q and a.writes
                    for a in _touches(s.regions[0].body, q))]


def test_a_store_under_a_guard_spelled_in_text_is_carried():
    """The store under a batch element's guard, the load not: where the
    guard holds, the trips are what the per-lane account says, and where it
    does not, nothing stores and every trip reads what the first one loaded,
    as each loaded it from memory.  The guard stays around the store behind
    the loop, which is what it guarded."""
    run = Run(guarded=True)
    out = run.carry()
    ahead, loop, behind = _split(out)
    q = _q_root(run)
    assert not _touches(loop.regions[0].body, q)
    assert any(not a.writes for a in _touches(ahead, q))
    assert _guards(behind, q)
    assert any('in registers over 4 trips' in line for line in run.report)


def test_under_a_guard_the_trips_add_in_place_where_only_the_store_reads():
    """Nothing but the tail's slots reads the head's image, and nothing but
    the store the tail's: where the guard does not hold, the sum the trips
    compute in the head's image reaches nothing, so they compute in it."""
    run = Run(guarded=True)
    _, loop, _ = _split(run.carry())
    assert not any(s.op is Op.ALLOC for s in walk_stmts(loop.regions[0].body))


def test_under_a_guard_an_image_read_for_another_output_is_copied():
    """The trip reads the loaded image for another output ahead of its
    computation: where the guard does not hold, a sum computed in place
    would reach that output, where the copy -- which the guard skips --
    leaves what the first trip loaded.  Without the guard, it is computed in
    place."""
    run = Run(guarded=True, early_read=True)
    _, loop, _ = _split(run.carry())
    assert any('in registers' in line for line in run.report)
    assert _tail_image(run, loop.regions[0].body)
    run = Run(early_read=True)
    _, loop, _ = _split(run.carry())
    assert not _tail_image(run, loop.regions[0].body)


def test_a_load_under_a_guard_the_store_is_not_under_stays():
    """Where the guard does not hold, the trips would store onto an image
    the load never filled, trip after trip, where each stored onto what it
    found in the image."""
    run = Run(guarded_load=True)
    assert run.carry() == run.body
    assert any('a guard spelled in text the store does not' in line
               for line in run.report)


def test_a_guard_that_names_what_the_trip_computes_stays():
    """A guard that reads the tail's image holds for one trip and not for
    the next: it does not mean the same behind the loop."""
    run = Run(guarded=True, guard='if (r2[0] > 0.0f)')
    assert run.carry() == run.body
    assert any('names what the trip computes' in line for line in run.report)


def test_a_store_ahead_under_the_same_guard_is_read_from_its_registers():
    """The store ahead of the loop and the one behind it under one guard:
    where it holds, as without it; where it does not, neither stores, and
    what the trips compute from the registers ahead reaches nothing."""
    run = Run(stored_ahead=ROWS, guarded=True)
    out = run.carry()
    ahead, _, behind = _split(out)
    q = _q_root(run)
    assert not _touches(ahead, q)
    assert any(not a.writes for a in _touches(ahead, run.before))
    assert _guards(behind, q)
    assert any('not stored there' in line for line in run.report)


def test_a_store_ahead_under_a_guard_stays_where_the_trips_read_more():
    """The trip reads the tail's image for another output: where the guard
    does not hold, that output would see the registers ahead where it saw
    memory."""
    run = Run(stored_ahead=ROWS, guarded=True, tail_read=True)
    out = run.carry()
    ahead, _, _ = _split(out)
    q = _q_root(run)
    assert any(a.writes for a in _touches(ahead, q))
    assert any(not a.writes for a in _touches(ahead, q))


def test_a_load_behind_the_loop_outside_the_guard_reads_memory():
    """The load behind the loop runs where the store behind it does not:
    there it reads what memory holds, which the image need not."""
    run = Run(guarded=True, read_behind=True)
    _, _, behind = _split(run.carry())
    assert [a for a in _touches(behind, _q_root(run)) if not a.writes]
    assert any('a guard spelled in text the load does not' in line
               for line in run.report)


def test_a_temporary_each_lane_keeps_to_itself_is_held_in_registers():
    """Every lane adds onto its own rows of `S`, which the code ahead of the
    loop stored from registers and nothing behind the loop reads: the trips
    add into registers of their own, which take what the code ahead stored
    from its registers -- and `S` is stored neither ahead of the loop nor
    behind it any more."""
    run = Own()
    out = run.carry()
    ahead, loop, behind = _split(out)
    assert not _shared(loop.regions[0].body, run.s)
    assert not _shared(ahead, run.s) and not _shared(behind, run.s)
    image, = [s.target[0] for s in ahead if s.op is Op.ALLOC
              and s.attr('extern') == 'carry_s0']
    assert image.type.shape == (COLS,)
    assert any('not stored behind the loop' in line for line in run.report)
    assert any('not stored there' in line for line in run.report)


def test_a_register_one_on_from_its_hop_is_computed():
    """The columns past the first are reached at the hop plus one: in the
    trip, their register is that sum."""
    run = Own()
    ahead, loop, _ = _split(run.carry())
    image, = [s.target[0] for s in ahead if s.op is Op.ALLOC
              and s.attr('extern') == 'carry_s0']
    every = walk_stmts(loop.regions[0].body)
    defs = {t.id: s for s in every for t in s.target}
    slots = [s.args[-1] for s in every if s.op in (Op.LOAD, Op.STORE)
             and any(a.base is image for a in s.accesses)]
    assert 0 in slots
    assert any(getattr(x, 'id', None) in defs and defs[x.id].op == 'add'
               and 1 in defs[x.id].args for x in slots)


def test_what_the_code_behind_reads_is_stored_once_behind_the_loop():
    """Lane `l` reads the row of lane `l + 1` behind the loop: what the trips
    wrote goes there once, behind the loop, and the store ahead of it -- of
    the same elements -- goes."""
    run = Own(across=True)
    ahead, loop, behind = _split(run.carry())
    assert not _shared(loop.regions[0].body, run.s)
    assert not [a for a in _shared(ahead, run.s) if a.writes]
    image, = [s.target[0] for s in ahead if s.op is Op.ALLOC
              and s.attr('extern') == 'carry_s0']
    every = walk_stmts(tuple(behind))
    defs = {t.id: s for s in every for t in s.target}
    stores = [s for s in every if s.op is Op.STORE
              and any(a.base is run.s for a in s.accesses)]
    assert stores and all(
        any(a.base is image for a in defs[s.args[1].id].accesses)
        for s in stores)


def test_a_lane_that_adds_another_lanes_row_keeps_the_temporary_in_memory():
    """A lane adds the next lane's row onto its own: what it reads, another
    lane writes, which no register of this lane holds."""
    run = Own(other=True)
    assert run.carry() == run.body
    assert any('another lane writes' in line for line in run.report)


def test_what_the_code_ahead_did_not_store_keeps_the_temporary_in_memory():
    """The store ahead of the loop leaves the last column out: the trips read
    it first from what an earlier statement wrote, which no register
    holds."""
    run = Own(partial=True)
    assert run.carry() == run.body
    assert any('does not store from registers' in line
               for line in run.report)


def test_a_store_ahead_of_more_than_the_trips_reach_keeps_it_in_memory():
    """The trips leave the last column alone, which the store ahead of the
    loop writes as well: a register of the trips' image for it would be one
    the trips do not have."""
    run = Own(extra=True)
    assert run.carry() == run.body
    assert any('do not reach' in line for line in run.report)


def test_a_temporary_the_trip_defines_anew_stays_in_memory():
    run = Own(mark=True)
    assert run.carry() == run.body
    assert any('a mark touches it' in line for line in run.report)


def test_around_a_back_edge_the_loop_reads_what_it_left():
    """The run stands in a loop of its own: around that loop's back edge the
    store ahead of the run and the run itself come again, and the run reads
    what the last trip left -- so it is stored behind the loop, and is not
    where that loop's body begins by defining it anew."""
    run = Own(outer=True)
    out = run.carry()
    inner = next(s for s in out if s.op is Op.FOR).regions[0].body
    _, _, behind = _split(inner)
    assert [a for a in _shared(behind, run.s) if a.writes]
    run = Own(outer=True, defines=True)
    out = run.carry()
    inner = next(s for s in out if s.op is Op.FOR).regions[0].body
    _, _, behind = _split(inner)
    assert not _shared(behind, run.s)


def test_a_register_that_would_hold_two_elements_keeps_it_in_memory():
    """An odd lane reads its second column where an even lane reads its
    first, and every lane writes its first: no register holds the same of
    what every lane reads there."""
    run = Own(parity=True)
    assert run.carry() == run.body
    assert any("two of a lane's elements" in line for line in run.report)


def test_a_store_ahead_of_elements_the_trips_only_read_stays():
    """The trips read the columns past the first and write nothing of them,
    which lane `l + 1` reads behind the loop: what the store ahead of the
    loop wrote there nothing writes again, so it stays."""
    run = Own(readonly=True, across=True)
    ahead, loop, _ = _split(run.carry())
    assert not _shared(loop.regions[0].body, run.s)
    assert [a for a in _shared(ahead, run.s) if a.writes]
    assert any('taken from registers ahead of the loop' in line
               and 'not stored there' not in line for line in run.report)


def test_a_register_a_loop_picks_that_is_not_unrolled_stays_in_memory():
    """A register picked by the hop of a loop that is not unrolled is one the
    compiler cannot name ahead of time, and it keeps that in memory as
    well."""
    run = Own(plain=True)
    assert run.carry() == run.body
    assert any('not unrolled' in line for line in run.report)
