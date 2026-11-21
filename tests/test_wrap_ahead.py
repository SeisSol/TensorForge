# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A buffer's own request of the transfer into it (`IRBuilder.alloc`'s
`ahead`): `(stages, distance)`, which `pir.wrap` honors in place of its own
settings (`move_distance`, `enable_multibuffer`).

The transfer is counted against its buffer's distance -- 0 keeps it in its
trip, one past its place in its pipeline takes it -- and gets its buffer's
stages, a second putting a shared transfer at the head of the trip ahead.
Without a request the pass's settings hold, as before.  What is checked
here: what a request may be, where the builder refuses it, and on bodies of
one and two transfers which transfers move and into how many stages.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.backend.pir import verify, walk
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import AHEAD_STAGES, IRBuilder
from tensorforge.backend.pir.core import (BOOL, SIZE, IRError, MemSpace, Op,
                                          Participants, Uniformity)
from tensorforge.backend.pir.wrap import wrap_loads
from tensorforge.common.basic_types import Datatype
from test_pir_wrap import DST, SRC, _binding, _loop, _ops_before_loop

F32 = Datatype.F32


# -- the request -------------------------------------------------------------- #

def test_a_buffer_states_its_request():
    b = IRBuilder(fptype=F32, arena='shrMem')
    s = b.alloc(F32, (64,), MemSpace.SHARED, hint='s', ahead=(2, 3))
    r = b.alloc(F32, (8,), MemSpace.REGISTER, hint='r', ahead=(1, 0))
    plain = b.alloc(F32, (8,), MemSpace.SHARED, hint='p')
    allocs = {x.target[0].id: x for x in b.finish() if x.op is Op.ALLOC}
    assert allocs[s.id].attr('ahead') == (2, 3)
    assert allocs[r.id].attr('ahead') == (1, 0)
    assert allocs[plain.id].attr('ahead') is None
    assert AHEAD_STAGES == (1, 2)


@pytest.mark.parametrize('space,ahead,why', [
    (MemSpace.SHARED, (3, 1), 'a ring of more is not built'),
    (MemSpace.SHARED, (0, 1), 'one stage or two'),
    (MemSpace.REGISTER, (2, 1),
     "it is for a shared buffer's copies"),
    (MemSpace.SHARED, (1, -1), 'a count of transfers'),
    (MemSpace.SHARED, (1, True), 'a count of transfers'),
    (MemSpace.SHARED, (1,), '(stages, distance)'),
])
def test_a_request_wrap_does_not_build_is_refused(space, ahead, why):
    b = IRBuilder(fptype=F32, arena='shrMem')
    with pytest.raises(IRError, match=re.escape(why)):
        b.alloc(F32, (8,), space, hint='x', ahead=ahead)


# -- bodies ------------------------------------------------------------------- #

def _section(asks, distance=1, stages=1, registers=False):
    """A batch loop whose body copies its element's data into one buffer a
    request of `asks`, in order -- shared windows by asynchronous copies, or
    register arrays by loads (`registers`) -- each read back behind it.
    Wrapped at the pass's `distance` and `stages`: (body, report)."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    count = b.extern_value('numElements0', SIZE, hint='count')
    start = b.extern_value('start', SIZE, uniform=Uniformity.MULT,
                           hint='start')
    inside = b.op('lt', BOOL, start, count, hint='inside')
    first = b.op('select', SIZE, inside, start, 0, hint='batchId1',
                 escapes=True)
    space = MemSpace.REGISTER if registers else MemSpace.SHARED
    bufs = [b.alloc(F32, (64,), space, hint='s', extern=f's{j}', ahead=ask)
            for j, ask in enumerate(asks)]
    with b.for_('start', count, 'stride', hint='batchId0', index_type=SIZE,
                peel_index=first, uniform=Uniformity.MULT) as f:
        k = f.induction
        ahead = b.op('add', SIZE, k, 'stride', hint='ahead1')
        fits = b.op('lt', BOOL, ahead, count, hint='inbatch1')
        f._next_index = b.op('select', SIZE, fits, ahead, k,
                             hint='batchId1', escapes=True)
        out = _binding(b, k, base=DST, name='glb_m1', writable=True)
        lane = b.thread_id('x')
        for j, buf in enumerate(bufs):
            src = _binding(b, k, base=SRC, name=f'glb_m0_{j}' if j
                           else 'glb_m0')
            if registers:
                b.Comment(f's{j} = load{{g>r}}(glb_m0)')
                with b.for_(0, 8, 1, unroll=True) as i:
                    b.store(buf, b.load(src, i.induction, hint='g'),
                            i.induction)
                b.store(out, b.load(buf, 3, hint='u'), j)
            else:
                b.mark('defines', buf)
                tok = b.copy_async(buf, src, dst_index=(lane,),
                                   src_index=(lane,))
                b.store(out, 0.0, 16 + j)
                b.wait(tok)
                b.store(out, b.load(buf, lane, hint='u'), lane)
        b.barrier(Participants.MULT, threads=32)
    body = b.finish()
    verify(body)
    report = []
    out = wrap_loads(body, b.scratch, distance=distance, stages=stages,
                     report=report)
    verify(out)
    if not registers:
        _, diag = schedule_async(out)
        assert not diag, diag
    return out, report


def test_without_a_request_the_pass_decides():
    """Two shared transfers at the pass's distance 1: the first wraps."""
    out, report = _section([None, None])
    assert report == ['+ s0 [shr]']


def test_a_request_as_the_pass_asks_changes_nothing():
    """A request of the pass's own settings is the body without one."""
    plain, report = _section([None, None])
    asked, same = _section([(1, 1), (1, 1)])
    assert same == report
    ops = lambda body: [s.op for s, _ in walk(body)]
    assert ops(asked) == ops(plain)


def test_a_buffer_asks_for_its_transfer_beyond_the_pass_distance():
    """The second transfer, at place 1 of its pipeline, wraps where its
    buffer's distance reaches past it."""
    out, report = _section([None, (1, 2)])
    assert sorted(report) == ['+ s0 [shr]', '+ s1 [shr]']
    peel = [s for s in _ops_before_loop(out) if s.op is Op.COPY_ASYNC]
    assert len(peel) == 2


def test_a_buffer_keeps_its_transfer_in_its_trip():
    """Distance 0: the first transfer stays, and the second -- at place 1,
    past the pass's distance -- stays as well: nothing moves."""
    out, report = _section([(1, 0), None])
    assert '- loop: nothing to move' in report
    assert not any(line.startswith('+') for line in report)


def test_a_buffer_asks_for_a_second_stage():
    """The pass at one stage, the buffer asking for two: its copy goes to
    the head into the stage the trip does not read; the other keeps one."""
    out, report = _section([(2, 1), (1, 2)])
    assert sorted(report) == ['+ s0 [shr, 2 stages]', '+ s1 [shr]']
    windows = [s for s, _ in walk(out) if s.op is Op.ALLOC
               and s.attr('stages') == 2]
    assert windows and all(s.attr('ahead') is None for s in windows)
    loop = _loop(out)
    head = loop.regions[0].body
    first_copy = next(i for i, s in enumerate(head) if s.op is Op.COPY_ASYNC)
    first_read = next(i for i, s in enumerate(head) if s.op is Op.LOAD
                      and any(a.space is MemSpace.SHARED
                              for a in s.accesses))
    assert first_copy < first_read


def test_a_buffer_asks_for_one_stage_where_the_pass_gives_two():
    """The pass at two stages, the second buffer asking for one.  (A
    transfer of one stage ahead of one of two is scheduled with full drains
    in the loop, as where a buffer keeps one stage for a reason of its own:
    the wait for the first retires the second's older copy as well.)"""
    out, report = _section([None, (1, 2)], distance=2, stages=2)
    assert sorted(report) == ['+ s0 [shr, 2 stages]', '+ s1 [shr]']


def test_a_register_buffer_asks_as_well():
    out, report = _section([(1, 0), (1, 2)], registers=True)
    assert report == ['+ s1 [reg]']
