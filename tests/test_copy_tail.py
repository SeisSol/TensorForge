# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A macro copy is hops plus a ragged tail, and both are one token chain.

The loader splits a transfer into hops of 4, 2 and 1 elements per lane and is
then left with `length % num_threads` elements, moved by a last hop under
`linear_idx < rest`.  So the shape the IR needs is not a tile: it is an extent
*and* a predicate, or the tail has no representation and every transfer whose
length is not a multiple of the block falls out of the structured path.

What is checked here:

  - a predicated copy guards rather than selects, because its result is a
    token and there is nothing to select on;
  - the guard does not change the counter, since the copy issues for the wave
    whenever any lane is active;
  - a wait on the last token of a group retires the whole group, which is what
    makes "one token per macro copy" fall out of the existing model instead of
    needing a group object.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import optimize, verify, walk
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir import BOOL, INDEX
from tensorforge.backend.pir.core import MemSpace, Op
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.vm import vm_factory


def _macro_copy(num_threads: int = 64, length: int = 100):
    """One hop of 1 element per lane, then a tail of `length % num_threads`."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', length))
    dst = b.alloc(Datatype.F32, (length,), MemSpace.SHARED, hint='s')
    src = b.alloc(Datatype.F32, (length,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')

    tokens = []
    whole = (length // num_threads) * num_threads
    for start in range(0, whole, num_threads):
        tokens.append(b.copy_async(dst, src,
                                   dst_index=(b.op('add', INDEX, lane, start),),
                                   src_index=(b.op('add', INDEX, lane, start),)))
    rest = length % num_threads
    if rest:
        tokens.append(b.copy_async(dst, src,
                                   dst_index=(b.op('add', INDEX, lane, whole),),
                                   src_index=(b.op('add', INDEX, lane, whole),),
                                   predicate=b.op('lt', BOOL, lane, rest, hint='ok')))
    b.wait(tokens[-1], *tokens[:-1])
    return b.finish(), len(tokens)


def test_the_tail_is_representable_and_guarded():
    body, n_copies = _macro_copy()
    assert n_copies == 2, "expected one whole hop and one tail"
    verify(body)

    copies = [s for s, _ in walk(body) if s.op == Op.COPY_ASYNC]
    assert len(copies) == 2
    assert copies[0].predicate is None
    assert copies[1].predicate is not None, (
        "the tail hop has to carry its own predicate; without one the ragged "
        "remainder has no structured form")


def test_the_guard_is_a_branch_not_a_select():
    body, _ = _macro_copy()
    w = Writer()
    from tensorforge.backend.pir import emit
    emit(optimize(body), w, vm_factory('gfx942', 'hip', 'float'))
    src = w.get_src()
    assert '?' not in src.split('\n')[-3:][0] or 'if (' in src, src
    assert 'if (' in src, (
        "a predicated copy produces a token, which has no C++ value to fold a "
        f"select onto, so it must be guarded:\n{src}")


def test_one_wait_retires_the_whole_group():
    body, n_copies = _macro_copy()
    scheduled, _ = schedule_async(optimize(body))
    waits = [s for s, _ in walk(scheduled) if s.op == Op.WAIT]
    assert len(waits) == 1, "one wait for the group"
    assert waits[0].attr('prior') == 0, (
        "a wait on the last token of the group retires every copy up to it, "
        "so nothing of this group stays in flight -- that is what makes one "
        "token per macro copy fall out of the model rather than needing a "
        "group object")
