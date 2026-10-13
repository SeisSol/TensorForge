# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A loop whose trips are counted, as `wrap` takes one: any loop over
constant bounds with a trip (`ahead.counted`), and not only a merged run's.

The loop here is a reduction rolled by a count, the way `k_roll` writes one,
reading a slice of an operand per trip.  What `wrap` does to a merged run's
loop is in `test_pir_wrap.py`; what `move` takes out of a counted loop in
`test_pir_move.py`.
"""

from __future__ import annotations

from contextlib import ExitStack

import pytest

from tensorforge.backend.pir import emit, verify, walk
from tensorforge.backend.pir.ahead import counted
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (SIZE, BufferType, Effect, MemSpace,
                                          Op, Uniformity)
from tensorforge.backend.pir.wrap import wrap_loads
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.target import Target
from harness.placement import placed

F32 = Datatype.F32


def _slice(b, element, *trips, name='glb_m0'):
    """A pointer to the slice of `m0` an element reads on a trip -- of each
    loop in `trips`, outermost first."""
    text = '&m0[{0} * 64' + ''.join(
        f' + {{{n + 1}}} * {16 // (n + 1)}' for n in range(len(trips))) + ']'
    return b.decl_expr(
        f'const float *const __restrict__ {name}', text,
        BufferType(F32, (8,), MemSpace.GLOBAL, readonly=True), 'm0',
        args=(element, *trips), kind=Effect.READ, space=MemSpace.GLOBAL,
        hint=name, extern=name)


def _fill(b, src, dest, shared):
    """`dest` filled from `src`: a copy and its wait for a window, eight
    loads for a register array."""
    if shared:
        b.Comment('s0 = load{g>s}(glb_m0)')
        b.mark('defines', dest)
        lane = b.thread_id('x')
        token = b.copy_async(dest, src, dst_index=(lane,), src_index=(lane,))
        return token
    b.Comment('r0 = load{g>r}(glb_m0)')
    with b.for_(0, 8, 1, unroll=True) as i:
        b.store(dest, b.load(src, i.induction, hint='g'), i.induction)
    return None


class Trips:
    """A body around a counted loop that is no merged run."""

    def __init__(self, *, shared=False, unroll=4, lo=0, hi=4, transfer=True,
                 nested=False):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        out = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 64]',
            BufferType(F32, (64,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        self.dest = (b.alloc(F32, (64,), MemSpace.SHARED, hint='s',
                             extern='s0') if shared
                     else b.alloc(F32, (8,), MemSpace.REGISTER, hint='r',
                                  extern='r0'))
        if isinstance(hi, str):
            hi = b.extern_value(hi, SIZE, uniform=Uniformity.GRID, hint=hi)
        with ExitStack() as around:
            trips = ()
            if nested:
                trips = (around.enter_context(
                    b.for_(0, 2, 1, hint='kk')).induction,)
            with b.for_(lo, hi, 1, unroll=unroll, hint='kt') as loop:
                self.counter = loop.induction
                src = _slice(b, k, *trips, loop.induction)
                if transfer:
                    token = _fill(b, src, self.dest, shared)
                    b.store(out, 0.0, 1)
                    if token is not None:
                        b.Comment('wait(s0 = load{g>s}(glb_m0))')
                        b.wait(token)
                lane = b.thread_id('x')
                b.store(out, b.load(self.dest, lane, hint='u'),
                        loop.induction)
        self.body = b.finish()
        verify(self.body)

    def wrap(self, **kwargs):
        self.report = []
        out = wrap_loads(self.body, self.b.scratch, report=self.report,
                         **kwargs)
        verify(out)
        return out


def _the_loop(body, hint='kt'):
    return next(s for s, _ in walk(tuple(body)) if s.op is Op.FOR
                and s.regions[0].args[0].hint == hint)


def _bindings(stmts):
    return {s.attr('extern'): s for s, _ in walk(tuple(stmts))
            if s.op is Op.RAWEXPR and str(s.attr('extern', '')).endswith(
                'glb_m0')}


def _defs(body):
    return {t.id: s for s, _ in walk(tuple(body)) for t in s.target}


@pytest.mark.parametrize('unroll', [4, True, False])
def test_a_counted_loop_fetches_the_next_trips_slice_at_its_tail(unroll):
    """Unrolled by a count, all the way or not at all: the trips run in
    order, and the slice of the trip after is fetched once this trip is done
    with the buffer -- the first trip's ahead of the loop."""
    trips = Trips(unroll=unroll)
    out = trips.wrap()
    assert trips.report == ['+ r0 [reg] in a counted loop']
    loop = _the_loop(out)
    following = loop.attr('next')
    assert loop.attr('first') == 0
    inside = _bindings(loop.regions[0].body)
    assert set(inside) == {'wrap_glb_m0'}
    assert inside['wrap_glb_m0'].args[-1] is following
    # the trip after, held at the last one
    select = _defs(out)[following.id]
    assert select.op == 'select' and select.args[2] is trips.counter
    peel = _bindings(out[:out.index(loop)])
    assert set(peel) == {'peel_glb_m0'}
    assert peel['peel_glb_m0'].args[-1] == 0


def test_a_counted_loops_shared_copy_rides_the_loop_and_is_drained():
    trips = Trips(shared=True)
    out = trips.wrap()
    assert trips.report[0].startswith('+ s0 [shr]')
    assert trips.report[0].endswith(' in a counted loop')
    loop = _the_loop(out)
    assert loop.target, 'the loop carries the tail\'s copies'
    assert any(s.op is Op.WAIT for s in out[out.index(loop) + 1:])


def test_the_wrapped_counted_loop_renders():
    out = Trips().wrap()
    w = Writer()
    emit(placed(out), w, Target('sm_86', 'cuda'))
    src = w.get_src()
    head, _, rest = src.partition('for (')
    assert 'peel_glb_m0' in head and 'wrap_glb_m0' in rest


@pytest.mark.parametrize('bounds', [dict(lo=4, hi=4), dict(hi='trips')])
def test_a_loop_that_may_not_run_is_left_alone(bounds):
    """A first trip fetched ahead of a loop that has none reads a slice no
    trip does."""
    trips = Trips(**bounds)
    assert not counted(_the_loop(trips.body))
    assert trips.wrap() is trips.body
    assert trips.report == []


def test_a_counted_loop_without_a_transfer_is_passed_over_without_a_word():
    """A reduction holds no transfer of its own, and is no merged run, the
    loop written to stage operands in: nothing to report."""
    trips = Trips(transfer=False)
    assert trips.wrap() is trips.body
    assert trips.report == []


def test_a_transfers_own_loop_is_no_loop_to_wrap():
    """The hops of a loader are a loop over constant bounds, and the
    transfer: the load of the next hop is no transfer issued ahead."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    dest = b.alloc(F32, (8,), MemSpace.REGISTER, hint='r', extern='r0')
    _fill(b, _slice(b, k), dest, shared=False)
    b.load(dest, 3, hint='u')
    body = b.finish()
    assert counted(next(s for s in body if s.op is Op.FOR))
    report = []
    assert wrap_loads(body, b.scratch, report=report) is body
    assert report == []


def test_the_innermost_loop_first():
    """What the inner loop leaves ahead of itself -- the first trip's fetch
    -- is a transfer of the outer loop's body, which the outer loop then
    asks about in turn; the buffer is the inner loop's as well, so it
    declines."""
    trips = Trips(nested=True)
    out = trips.wrap()
    assert trips.report[0] == '+ r0 [reg] in a counted loop'
    assert trips.report[1].startswith('- r0 in a counted loop: the buffer is '
                                      'written by more than this transfer')
    outer = _the_loop(out, 'kk')
    assert _the_loop(outer.regions[0].body).attr('next') is not None
    assert outer.attr('next') is None


def test_a_loop_around_a_wrapped_one_keeps_a_second_stage():
    """The outer loop copies a window a trip, the inner one fills a
    register array a trip, and with two stages both move.  The inner loop
    is wrapped first, which makes the outer loop a statement other than the
    one its body holds -- which is not something around the loop that reads
    the window."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    out = b.decl_expr(
        'float *const __restrict__ glb_m1', '&m1[{0} * 64]',
        BufferType(F32, (64,), MemSpace.GLOBAL), 'm1', args=(k,),
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
        extern='glb_m1')
    window = b.alloc(F32, (64,), MemSpace.SHARED, hint='s', extern='s0')
    array = b.alloc(F32, (8,), MemSpace.REGISTER, hint='r', extern='r0')
    with b.for_(0, 2, 1, hint='kk') as outer:
        token = _fill(b, _slice(b, k, outer.induction, name='glb_m2'),
                      window, shared=True)
        b.Comment('wait(s0 = load{g>s}(glb_m0))')
        b.wait(token)
        with b.for_(0, 4, 1, unroll=4, hint='kt') as inner:
            _fill(b, _slice(b, k, outer.induction, inner.induction), array,
                  shared=False)
            b.store(out, b.load(array, b.thread_id('x'), hint='u'),
                    inner.induction)
        b.store(out, b.load(window, b.thread_id('x'), hint='v'),
                outer.induction)
    body = b.finish()
    verify(body)
    report = []
    verify(wrap_loads(body, b.scratch, report=report, stages=2))
    assert report[0] == '+ r0 [reg] in a counted loop', report
    assert '+ s0 [shr, 2 stages] in a counted loop' in report, report
