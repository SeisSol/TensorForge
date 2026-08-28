# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Moving a prefetch across the back edge, and the four times it must not.

`schedule.can_reorder` licenses swaps inside a body, and the census that came
with it said the schedule is already at their fixed point.  The distance that
is missing is one iteration away, and this is the pass that goes there.

Most of what follows is the refusals.  A transformation that changes what a
loop reads and when is only as good as the cases where it declines, and
"nothing happened" reads the same whether the pass was right or merely broken.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import emit, verify, walk
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, MemSpace, Op
from tensorforge.backend.pir.wrap import wrap_prefetch
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.vm import vm_factory


def _loop(*, double_buffer=True, read_before_wait=False, barrier=False,
          extra_issue=False):
    """A loop that fills a buffer and reads it in the same iteration."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 256))
    fill = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    read = (b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='t')
            if double_buffer else fill)
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (128,), MemSpace.GLOBAL, hint='o')
    lane = b.thread_id('x')
    nxt = {}
    with b.for_(0, 8, 1) as f:
        k = f.induction
        nxt[k.id] = b.op('add', INDEX, k, 1, hint='nk')
        tok = b.copy_async(fill, glb, dst_index=(lane,), src_index=(k,))
        if read_before_wait:
            b.store(out, b.load(read, lane, hint='early'), lane)
        if barrier:
            b.barrier()
        if extra_issue:
            # A *separate* transfer with its own wait.  Two issues retired by
            # one wait are one transfer in pieces and the pass moves them
            # together; two waits mean two transfers, and choosing between
            # them would be a schedule.
            second = b.copy_async(fill, glb, dst_index=(lane,), src_index=(k,))
            b.wait(second)
            b.wait(tok)
        else:
            b.wait(tok)
        b.store(out, b.load(read, lane, hint='u'), lane)
    body = b.finish()
    verify(body)
    return b, body, nxt


def _wrap(b, body, nxt):
    return wrap_prefetch(body, lambda ty, hint: b.value(ty, hint=hint), nxt)


def _moved(body) -> bool:
    """Did a loop gain a carried token?"""
    return any(s.op is Op.FOR and s.target for s, _ in walk(body))


def test_the_prefetch_moves_a_whole_iteration():
    b, body, nxt = _loop()
    wrapped = _wrap(b, body, nxt)
    verify(wrapped)
    assert _moved(wrapped)

    scheduled, diag = schedule_async(wrapped)
    assert not diag, diag
    waits = [s for s, _ in walk(scheduled) if s.op is Op.WAIT]
    assert [w.attr('prior') for w in waits] == [1, 0], (
        "inside the loop one transfer is in flight -- the one this iteration "
        "issued for the next; the drain afterwards waits for it")

    w = Writer()
    emit(scheduled, w, vm_factory('sm_86', 'cuda', 'float'))
    src = w.get_src()
    head, _, tail = src.partition('for (')
    assert '__pipeline_memcpy_async' in head, (
        f"the peeled transfer belongs before the loop:\n{src}")
    assert '__pipeline_wait_prior(1);' in tail, src
    assert src.rstrip().endswith('__pipeline_wait_prior(0);'), src


def test_it_refuses_a_single_buffered_destination():
    """The slot accounting, enforced rather than assumed.

    A transfer at distance `d` in a body of `n` slots needs `ceil((d+1)/n)`
    copies; this pass does `d = n`, which is the first value needing two. With
    one, the transfer issued for element k+1 lands in the buffer iteration k
    is reading, and the wait does not cover it -- it is for the previous
    transfer.
    """
    b, body, nxt = _loop(double_buffer=False)
    assert not _moved(_wrap(b, body, nxt))


def test_it_refuses_a_read_of_the_destination_before_the_wait():
    b, body, nxt = _loop(double_buffer=False, read_before_wait=True)
    assert not _moved(_wrap(b, body, nxt))


def test_it_refuses_to_cross_a_barrier():
    """Crossing a back edge is a stronger move than a swap, so it needs at
    least the same licence -- and `can_reorder` never crosses a barrier."""
    b, body, nxt = _loop(barrier=True)
    assert not _moved(_wrap(b, body, nxt))


def test_it_refuses_two_transfers_with_two_waits():
    """Picking among several is a schedule, not a rewrite."""
    b, body, nxt = _loop(extra_issue=True)
    assert not _moved(_wrap(b, body, nxt))


def test_it_moves_the_hops_of_one_transfer_together():
    """A macro copy is hops of 4, 2 and 1 elements plus a predicated tail,
    each its own `copy.async` and all retired by one wait.  That is one
    transfer in pieces, and the group is what the wait consumes."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 256))
    fill = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    read = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='t')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (128,), MemSpace.GLOBAL, hint='o')
    lane = b.thread_id('x')
    nxt = {}
    with b.for_(0, 8, 1) as f:
        k = f.induction
        nxt[k.id] = b.op('add', INDEX, k, 1, hint='nk')
        hops = [b.copy_async(fill, glb, dst_index=(lane,), src_index=(k,),
                             elems=n) for n in (4, 2, 1)]
        b.wait(hops[-1], *hops[:-1])
        b.store(out, b.load(read, lane, hint='u'), lane)
    body = b.finish()
    verify(body)

    wrapped = _wrap(b, body, nxt)
    verify(wrapped)
    loops = [s for s, _ in walk(wrapped) if s.op is Op.FOR]
    assert len(loops[0].target) == 3, "all three hops carried, or none"
    peeled = [s for s in wrapped if s.op is Op.COPY_ASYNC]
    assert len(peeled) == 3, "the prologue is the whole transfer, not a hop"


def test_it_refuses_a_loop_that_already_carries_a_token():
    """Applying it twice would put two iterations of distance behind buffers
    sized for one."""
    b, body, nxt = _loop()
    once = _wrap(b, body, nxt)
    assert _moved(once)
    twice = _wrap(b, once, nxt)
    loops = [s for s, _ in walk(twice) if s.op is Op.FOR]
    assert len(loops) == 1
    assert len(loops[0].target) == 1, "a second token was carried"
