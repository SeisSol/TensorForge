# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Would rotation help this loop?

The decision to allocate a second copy of a buffer has to be made before the
body exists -- `ShrMemOpt` sizes the arena long before any PIR pass runs.  So
something has to predict what `wrap_prefetch` will say, and the tempting thing
is a second predicate written to match it.

That is the failure this codebase keeps paying for: two spellings of one rule,
agreeing until one of them is edited.  `_can_swap` and `can_reorder` had
already drifted by the time they were noticed.

`assume_rotated` avoids it by asking the pass.  It drops exactly one refusal --
the one about *space* rather than legality -- and leaves every other check in
place, so a loop that would still be refused for a barrier, a second wait or
an unadvanceable index does not get a buffer it cannot use.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import verify, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, MemSpace, Op
from tensorforge.backend.pir.wrap import wrap_prefetch
from tensorforge.common.basic_types import Datatype


def _loop(*, single_buffer, barrier=False):
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 256))
    fill = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    read = fill if single_buffer else b.alloc(Datatype.F32, (128,),
                                              MemSpace.SHARED, hint='t')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (128,), MemSpace.GLOBAL, hint='o')
    lane = b.thread_id('x')
    nxt = {}
    with b.for_(0, 8, 1) as f:
        k = f.induction
        nxt[k.id] = b.op('add', INDEX, k, 1, hint='nk')
        tok = b.copy_async(fill, glb, dst_index=(lane,), src_index=(k,))
        if barrier:
            b.barrier()
        b.wait(tok)
        b.store(out, b.load(read, lane, hint='u'), lane)
    body = b.finish()
    verify(body)
    return b, body, nxt


def _moved(b, body, nxt, *, assume_rotated):
    out = wrap_prefetch(body, lambda ty, h: b.value(ty, hint=h), nxt,
                        assume_rotated=assume_rotated)
    return any(s.op is Op.FOR and s.target for s, _ in walk(out))


def test_rotation_is_what_the_single_buffered_loop_needs():
    b, body, nxt = _loop(single_buffer=True)
    assert not _moved(b, body, nxt, assume_rotated=False)
    assert _moved(b, body, nxt, assume_rotated=True), (
        "one buffer is the only thing stopping this loop, so assuming a "
        "second must be enough to accept it")


def test_it_changes_nothing_for_a_loop_that_already_has_two():
    b, body, nxt = _loop(single_buffer=False)
    assert _moved(b, body, nxt, assume_rotated=False)
    assert _moved(b, body, nxt, assume_rotated=True)


def test_it_does_not_excuse_anything_else():
    """A barrier is not a space problem, and rotation must not paper over it.

    This is the check that keeps `assume_rotated` from becoming "accept
    everything": a buffer allocated for a loop that gets refused anyway is
    shared memory spent on nothing.
    """
    b, body, nxt = _loop(single_buffer=True, barrier=True)
    assert not _moved(b, body, nxt, assume_rotated=False)
    assert not _moved(b, body, nxt, assume_rotated=True)
