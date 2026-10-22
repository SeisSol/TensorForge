# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An accumulation into global memory, added there atomically where
nothing reads the sum (`pir.atomics`), asked of bodies built by hand.

`Q += f` the way the generator builds it: the element's `Q` loaded into a
register array, the terms computed into another, their sum into a third,
the third stored.  The pass turns the load into a zero and the store into an
atomic add where that is the same thing -- and leaves the accumulation as
it is where anything reads the sum, the element, or a register the sum
passes through, which is when the two are not.  The merged runs of the
generator are asked in `test_rolling.py` and the corpus in
`tools/host/corpus_check.py`; here are the refusals, one at a time.
"""

from __future__ import annotations

import contextlib

from tensorforge.backend.pir import IRBuilder, dce, fold, verify
from tensorforge.backend.pir.atomics import accumulate_atomically
from tensorforge.backend.pir.carry import carry_trips
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, BufferType,
                                          Effect, MemSpace, Op, ScalarType,
                                          Uniformity, walk_stmts)
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32
LANES = 32
ROWS = 24
COLS = 4
REAL = ScalarType(F32)


class Acc:
    """`Q[l, j] += 2 * 3` for the lanes `l` below `ROWS`, through the
    register arrays `r0` (the bias), `t` (the terms) and `r2` (the sum).

    `read_after` reads `Q` once more behind the store; `sum_read` and
    `bias_read` read the sum's and the bias's registers for another output;
    `between` writes `Q` between the load and the store; `shared_element`
    has every lane store one element; `silent` stores the bias back with no
    term; `other_lane` stores each lane's sum into the next lane's element.
    `trips` puts the whole accumulation in a loop of that many trips, the
    way a closed trip does; `carried` loads the bias ahead of such a loop,
    adds onto it in its registers and stores it behind the loop, the way
    the carry leaves it, into registers the allocation zeroes unless
    `uninit`.  `guarded` puts the store under a guard spelled in text,
    `guarded_load` the load.  `narrow_load` loads the bias for four rows
    fewer than the store stores, `flagged_load` under a flag no lane
    decides; `split_store` stores the first two columns and the last two
    from two loops of their own.  `assigned` stores a number into the
    element before the load, under `guarded`'s kind of guard, which
    `'element'` makes an element's mask.  `unplaced` stores into a row the
    body takes from outside, which no flow can place.
    """

    def __init__(self, *, read_after=False, sum_read=False, bias_read=False,
                 between=False, shared_element=False, silent=False,
                 other_lane=False, trips=0, carried=False, uninit=False,
                 guarded=False, guarded_load=False, narrow_load=False,
                 flagged_load=False, split_store=False, assigned=False,
                 unplaced=False):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        q = self.q = b.decl_expr(
            'float *const __restrict__ glb_m0', '&m0[{0} * 96]',
            BufferType(F32, (ROWS * COLS,), MemSpace.GLOBAL), 'm0', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
            extern='glb_m0')
        p = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
            BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
        valid = b.op('lt', BOOL, lane, ROWS, hint='g')
        narrow = b.op('lt', BOOL, lane, ROWS - 4, hint='n')
        flag = b.extern_value('flag', BOOL, uniform=Uniformity.MULT,
                              hint='flag')

        def element(j, at=None):
            row = lane if at is None else at
            return b.op('add', INDEX, row, b.op('mul', INDEX, j, ROWS),
                        hint='a')

        def guard(on):
            if on == 'element':
                return b.If('active', guard='element')
            return (b.Block('if (active)') if on
                    else contextlib.nullcontext())

        def load_bias(bias):
            predicate = (narrow if narrow_load else flag if flagged_load
                         else None)
            with b.if_(valid):
                with b.for_(0, COLS, 1, unroll=True) as hop:
                    j = hop.induction
                    with guard(guarded_load):
                        value = b.load(q, element(j), predicate=predicate,
                                       other=None if predicate is None
                                       else 0.0)
                        b.store(bias, value, j)

        def add_terms(into, onto):
            terms = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='t',
                            extern='t', init='{}')
            with b.for_(0, COLS, 1, unroll=True) as hop:
                b.store(terms, b.op('mul', REAL, 2.0, 3.0), hop.induction)
            with b.if_(valid):
                with b.for_(0, COLS, 1, unroll=True) as hop:
                    j = hop.induction
                    b.store(into, b.op('add', REAL, b.load(onto, j),
                                       b.load(terms, j)), j)

        def store(src):
            if between:
                b.store(q, 1.0, element(0))
            target = None
            if shared_element:
                target = 0
            if other_lane:
                target = b.op('rem', INDEX, b.op('add', INDEX, lane, 1),
                              ROWS)
            if unplaced:
                target = b.extern_value('row', INDEX,
                                        uniform=Uniformity.MULT, hint='row')
            spans = (((0, 2), (2, COLS)) if split_store
                     else ((0, COLS),))
            with b.if_(valid):
                for lo, hi in spans:
                    with b.for_(lo, hi, 1, unroll=True) as hop:
                        j = hop.induction
                        value = b.load(src, j)
                        with guard(guarded):
                            b.store(q, value, element(j, target))

        def once():
            if assigned:
                with b.if_(valid):
                    with b.for_(0, COLS, 1, unroll=True) as hop:
                        with guard(guarded):
                            b.store(q, 1.0, element(hop.induction))
            bias = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r0',
                           extern='r0', init='{}')
            load_bias(bias)
            if silent:
                store(bias)
                return bias, bias
            total = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r2',
                            extern='r2', init='{}')
            add_terms(total, bias)
            store(total)
            return bias, total

        if carried:
            image = b.alloc(F32, (COLS,), MemSpace.REGISTER, hint='r0',
                            extern='r0', init='' if uninit else '{}')
            load_bias(image)
            with b.for_(0, 3, 1, rolled=True):
                add_terms(image, image)
            store(image)
            bias = total = image
        elif trips:
            with b.for_(0, trips, 1, rolled=True):
                bias, total = once()
        else:
            bias, total = once()
        if sum_read:
            b.store(p, b.load(total, 0), lane)
        if bias_read:
            b.store(p, b.load(bias, 0), lane)
        if read_after:
            b.store(p, b.load(q, element(0)), lane)
        self.body = b.finish()
        verify(self.body)

    def run(self, supported=True, carried=False):
        """The pass over the body -- over what the carry made of it first,
        with `carried`, as the pipeline runs them."""
        self.report = []
        body = self.body
        if carried:
            body = carry_trips(body, self.b.scratch, threads=LANES,
                               report=self.report)
            verify(body)
        out = accumulate_atomically(body, threads=LANES, arena='shrMem',
                                    supported=lambda v: supported,
                                    report=self.report)
        out = dce(fold(out))
        verify(out)
        return out


def _atomic(body):
    return [s for s in walk_stmts(body) if s.op == Op.STORE
            and s.effect & Effect.ATOMIC]


def _stores_of_q(body, q):
    """The stores into `q`: a store names the buffer by the value bound
    for it, its access by the buffer's name."""
    return [s for s in walk_stmts(body) if s.op == Op.STORE
            and s.accesses[0].space is MemSpace.GLOBAL and s.args[0] is q]


def _loads_of_q(body, q):
    return [s for s in walk_stmts(body) if s.op == Op.LOAD
            and s.args and s.args[0] is q]


def _stays(acc, why):
    out = acc.run()
    assert not _atomic(out), acc.report
    assert any(why in line for line in acc.report), acc.report
    return out


def test_an_accumulation_nothing_reads_is_added_atomically():
    acc = Acc()
    out = acc.run()
    assert len(_atomic(out)) == 1, acc.report
    assert not _loads_of_q(out, acc.q)
    # the bias is gone, and with it the add: the terms are what is stored
    adds = [s for s in walk_stmts(out) if s.op == 'add' and s.target
            and isinstance(s.target[0].type, ScalarType)
            and s.target[0].type.base is F32]
    assert not adds


def test_a_read_of_the_element_behind_the_store_keeps_it():
    _stays(Acc(read_after=True), 'something reads what it stores')


def test_a_read_of_the_sum_keeps_it():
    """The register the sum is stored from is read for another output: it
    wants the sum, bias and all."""
    _stays(Acc(sum_read=True), 'read elsewhere')


def test_a_read_of_the_bias_keeps_it():
    _stays(Acc(bias_read=True), 'read elsewhere')


def test_a_store_the_flow_cannot_place_keeps_it():
    """Into a row the body takes from outside, with the element read
    behind it: where the store writes, and so who reads what it wrote, no
    flow can tell."""
    _stays(Acc(unplaced=True, read_after=True), 'nothing the flow can place')


def test_a_write_between_the_load_and_the_store_keeps_it():
    _stays(Acc(between=True), 'written between')


def test_lanes_that_store_one_element_keep_it():
    """Every lane stores its sum into one element: as atomics, the element
    would get every lane's terms, where it gets the last lane's sum."""
    acc = Acc(shared_element=True)
    out = acc.run()
    assert not _atomic(out), acc.report


def test_a_store_into_another_lanes_element_keeps_it():
    _stays(Acc(other_lane=True), 'another element')


def test_a_store_of_what_was_loaded_goes():
    acc = Acc(silent=True)
    out = acc.run()
    assert not _atomic(out)
    assert not _stores_of_q(out, acc.q), acc.report
    assert not _loads_of_q(out, acc.q)


def test_trips_that_accumulate_through_memory_add_atomically_together():
    """A loop whose every trip loads the element, adds its terms and stores
    it: the next trip's load reads the store, and both go together."""
    acc = Acc(trips=3)
    out = acc.run()
    assert len(_atomic(out)) == 1, acc.report
    assert not _loads_of_q(out, acc.q)


def test_trips_the_carry_took_into_registers_add_once_behind_the_loop():
    """The same trips, carried first (`pir.carry`): the load ahead of the
    loop, the trips adding in its image, the store behind it -- and that
    store adds the sum of the trips, onto an image that starts from zero."""
    acc = Acc(trips=3)
    out = acc.run(carried=True)
    assert any('in registers over 3 trips' in line for line in acc.report)
    assert len(_atomic(out)) == 1, acc.report
    assert not _loads_of_q(out, acc.q)
    loop, = [s for s in out if s.op == Op.FOR]
    assert not _stores_of_q(loop.regions[0].body, acc.q)


def test_an_image_carried_over_the_trips_starts_from_zero():
    """The bias loaded ahead of the loop, the trips adding onto it in its
    registers, the sum stored behind it: the loop adds onto zeros."""
    acc = Acc(carried=True)
    out = acc.run()
    assert len(_atomic(out)) == 1, acc.report
    assert not _loads_of_q(out, acc.q)


def test_an_image_the_allocation_leaves_unset_starts_from_a_zero():
    """Registers the allocation does not zero hold the bias and nothing
    else before it: the load becomes a zero, which the trips add onto."""
    acc = Acc(carried=True, uninit=True)
    out = acc.run()
    assert len(_atomic(out)) == 1, acc.report
    assert not _loads_of_q(out, acc.q)
    assert any(s.op == Op.CONST and s.attr('value') == 0.0
               for s in walk_stmts(out))


def test_rows_its_load_leaves_to_an_else_value_keep_it():
    """Four rows store what the load left them -- a zero -- plus the terms:
    an assignment of the terms, which no add onto memory is."""
    _stays(Acc(narrow_load=True), 'else-value')


def test_a_load_under_a_flag_the_store_is_not_under_keeps_it():
    _stays(Acc(flagged_load=True), 'where its load did not run')


def test_one_load_feeding_two_stores_changes_for_both_or_neither():
    """The bias is loaded in one loop and stored back from two, and
    something reads the first column behind them: the load cannot turn into
    a zero for the columns of one store and not the other's."""
    acc = Acc(split_store=True, read_after=True)
    out = acc.run()
    assert not _atomic(out), acc.report
    assert any('runs where no chain of it does' in line
               for line in acc.report)


def test_a_type_the_target_cannot_add_atomically_keeps_it():
    acc = Acc()
    out = acc.run(supported=False)
    assert not _atomic(out)
    assert any('adds no such value' in line for line in acc.report)


def test_a_store_under_a_guard_its_load_is_not_under_is_added():
    """Where the store runs, the load ran: the load's guard is the store's
    or none."""
    acc = Acc(guarded=True)
    out = acc.run()
    assert len(_atomic(out)) == 1, acc.report


def test_a_load_under_a_guard_the_store_is_not_under_keeps_it():
    _stays(Acc(guarded_load=True), 'not certain')


def test_onto_a_store_under_an_elements_mask_it_is_added():
    """On AMD every store of an element stands under its mask and no load
    does: the bias reads what the store before it wrote where the mask
    holds, and where it does not, nothing is stored, atomic or not."""
    acc = Acc(assigned=True, guarded='element')
    out = acc.run()
    assert len(_atomic(out)) == 1, acc.report
    assert len(_stores_of_q(out, acc.q)) == 2


def test_onto_a_store_under_a_guard_of_unknown_value_it_is_kept():
    """The same under a guard spelled in text that is no mask: where it
    does not hold, the bias reads what memory held before, which the store
    behind it does not replace."""
    _stays(Acc(assigned=True, guarded=True), 'written between')


def test_a_body_past_the_budget_keeps_every_store(monkeypatch):
    """The accumulation of the first test, in a body too large for the flow
    to keep track of: no store is added atomically."""
    from tensorforge.backend.pir import flow
    monkeypatch.setattr(flow, 'BUDGET', 10)
    acc = Acc()
    out = acc.run()
    assert not _atomic(out), acc.report
    assert _loads_of_q(out, acc.q)
    assert any('more than 10 elements' in line for line in acc.report), \
        acc.report
