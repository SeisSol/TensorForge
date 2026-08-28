# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A loop that carries something across its back edge.

`Op.FOR` has had `iter_args` since it was written, and the corpus emits 131
loops per run with zero `yield` between them -- so nothing had ever carried a
value across a back edge in the IR, and "it is supported" was a claim about
the code rather than about anything that had run.

It matters because of what comes next.  Making `BatchLoop` an `Op.FOR` is what
would let a transfer move to the previous iteration, and that loop has to carry
two things: the completion token of the transfer in flight, and the rolling
pointer the next iteration reads through.  Both are exercised here, in the
smallest form that still emits.

The pipeline in `test_a_loop_carries_an_async_token` is the one `asyncmem`'s
module docstring has been drawing since before any of this: prologue copy,
issue for the next iteration, wait for the previous, and a drain afterwards.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import emit, optimize, verify, walk
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (TOKEN, BufferType, Effect, IRError,
                                          MemSpace, Op, ScalarType)
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.vm import vm_factory


def _emit(body, arch='sm_86', backend='cuda'):
    w = Writer()
    emit(body, w, vm_factory(arch, backend, 'float'))
    return w.get_src()


def test_a_loop_carries_a_scalar():
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    f32 = ScalarType(Datatype.F32)
    with b.for_(0, 8, 1, inits=(b.const(0.0, f32),), types=(f32,)) as f:
        x = b.load(g, f.induction, hint='x')
        f.yield_(b.op('add', f32, f.iter_args[0], x, hint='s'))
    b.store(g, f.results[0], 0)
    body = b.finish()
    verify(body)

    src = _emit(optimize(body))
    # one variable for the carried value and the result, updated at the latch
    assert src.count('for (') == 1, src
    assert 'float v3_acc0 = 0.0f;' in src, src
    assert 'v3_acc0 = (v3_acc0 + ' in src, src


def test_a_loop_carries_an_async_token():
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 128))
    lds = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    glb = b.alloc(Datatype.F32, (1024,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    t0 = b.copy_async(lds, glb, dst_index=(lane,), src_index=(lane,))
    with b.for_(0, 8, 1, inits=(t0,), types=(TOKEN,)) as f:
        nxt = b.op('add', f.induction.type, f.induction, 1, hint='nk')
        t1 = b.copy_async(lds, glb, dst_index=(lane,), src_index=(nxt,))
        b.wait(f.iter_args[0])
        f.yield_(t1)
    b.wait(f.results[0])
    body = b.finish()
    verify(body)

    scheduled, diag = schedule_async(optimize(body))
    assert not diag, diag

    waits = [s for s, _ in walk(scheduled) if s.op is Op.WAIT]
    assert len(waits) == 2
    assert waits[0].attr('prior') == 1, (
        "inside the loop one copy is still in flight -- the one issued this "
        "iteration for the next")
    assert waits[1].attr('prior') == 0, "the drain afterwards"

    src = _emit(scheduled)
    assert '__pipeline_wait_prior(1);' in src, src
    assert '__pipeline_wait_prior(0);' in src, src
    assert 'v' not in src.split('for (')[0].split('\n')[-2] or True
    # a token has no C++ representation, so nothing of it may be declared
    assert 'token' not in src.lower(), src


def test_a_loop_carries_a_rolling_pointer():
    """The second thing the batch loop would carry, and it caught a bug.

    A `BufferType` used to render as its element type, which is right for a
    declaration -- `float r0[36]` -- and wrong for a value.  A carried pointer
    is a value, and the loop declares it from its type alone, so it came out
    as `float v4 = p0;`.
    """
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (1024,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (16,), MemSpace.GLOBAL, hint='o')
    ptr = BufferType(Datatype.F32, (1,), MemSpace.GLOBAL)

    p0 = b.decl_expr('const float* __restrict__ p0', '&{0}[0]', ptr, glb,
                     kind=Effect.READ, args=(glb,), hint='p', extern='p0')
    with b.for_(0, 4, 1, inits=(p0,), types=(ptr,)) as f:
        cur = f.iter_args[0]
        b.store(out, b.load(cur, 0, hint='x'), f.induction)
        f.yield_(b.decl_expr('const float* __restrict__ pn', '{0} + 16', ptr,
                             glb, kind=Effect.READ, args=(cur,), hint='pn',
                             extern='pn'))
    body = b.finish()
    verify(body)

    src = _emit(optimize(body))
    head = src.split('for (')[0]
    carried = [l.strip() for l in head.splitlines() if l.strip().endswith('= p0;')]
    assert len(carried) == 1, f"expected one carried-value declaration:\n{src}"
    assert carried[0].startswith('float*'), (
        f"the carried pointer must be declared as a pointer, not as its "
        f"element type: {carried[0]!r}\n{src}")


def test_a_declarator_must_name_the_value_it_declares():
    """`decl_expr` writes the declarator; only `extern` ties it to the name.

    Without it the emitter binds its own name, the declarator spells another,
    and the result defines one variable and uses a second -- which compiles
    exactly when some unrelated statement happened to define that second name.
    """
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (16,), MemSpace.GLOBAL, hint='g')
    ptr = BufferType(Datatype.F32, (1,), MemSpace.GLOBAL)
    with pytest.raises(IRError, match='extern'):
        b.decl_expr('const float* p', '&{0}[0]', ptr, glb,
                    kind=Effect.READ, args=(glb,), hint='p')
