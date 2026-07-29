# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Self-test / demo:  ``python -m tensorforge.backend.pir.demo``

Builds the shape of ``MultilinearInstruction._nonleading_dim``'s inner loop,
runs verify -> cse -> licm -> dce, and lowers the result into the real
``backend.writer.Writer``.  Doubles as a regression test for the passes.
"""

from __future__ import annotations

from tensorforge.common.basic_types import Datatype
from tensorforge.backend.writer import Writer
from tensorforge.backend import pir
from tensorforge.backend.pir import (BOOL, Effect, IRBuilder, MemSpace, Op,
                                     ScalarType, dump, emit, optimize, verify)

F32 = ScalarType(Datatype.F32)
K = 35


def build_contraction():
    """acc = sum_k A[tid, k] * B[k]  --- with deliberate CSE/LICM/DCE bait."""
    b = IRBuilder(fptype=Datatype.F32)

    A = b.alloc(Datatype.F32, (64, K), MemSpace.SHARED, hint='A')
    B = b.alloc(Datatype.F32, (K,), MemSpace.REGISTER, hint='B')
    tid = b.thread_id('x')                       # non-uniform
    zero = b.const(0.0)

    loop = b.for_(0, K, 1, inits=(zero,), types=(F32,), unroll=True, hint='k')
    with loop:
        k = loop.induction
        acc = loop.iter_args[0]
        scale = b.load(A, 0, 3, hint='scale')    # loop-invariant -> LICM
        a = b.load(A, tid, k, hint='a')
        v = b.load(B, k, hint='b')
        p = b.op('mul', F32, a, v, hint='p')
        q = b.op('mul', F32, a, v, hint='q')     # identical to p -> CSE
        b.op('mul', F32, a, a, hint='dead')      # unused -> DCE
        ps = b.op('mul', F32, q, scale, hint='ps')
        nxt = b.op('add', F32, acc, ps, hint='n')
        loop.yield_(nxt)

    b.store(A, loop.result, tid, 0)
    b.barrier()
    return b.finish()


def build_loop_with_write():
    """Same shape, but the loop writes A --- the invariant load must stay."""
    b = IRBuilder(fptype=Datatype.F32)
    A = b.alloc(Datatype.F32, (64, K), MemSpace.SHARED, hint='A')
    tid = b.thread_id('x')
    zero = b.const(0.0)

    loop = b.for_(0, K, 1, inits=(zero,), types=(F32,), hint='k')
    with loop:
        k = loop.induction
        acc = loop.iter_args[0]
        scale = b.load(A, 0, 3, hint='scale')
        nxt = b.op('add', F32, acc, scale, hint='n')
        b.store(A, nxt, tid, k)
        loop.yield_(nxt)
    return b.finish()


def build_divergent_barrier():
    b = IRBuilder(fptype=Datatype.F32)
    tid = b.thread_id('x')
    cond = b.op('lt', BOOL, tid, 9, hint='guard')     # non-uniform
    with b.if_(cond):
        b.barrier()
    return b.finish()


def build_speculative():
    """The replacement for the throw-away-Writer probe in multilinear.py."""
    b = IRBuilder(fptype=Datatype.F32)
    A = b.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='A')
    with b.speculative() as spec:
        b.load(A, 0, hint='probe')
        b.load(A, 1, hint='probe')
        spec.discard()                 # "this load would not have succeeded"
    b.load(A, 2, hint='kept')
    return b.finish()


def build_legacy():
    """Un-migrated call sites keep working and stay opaque."""
    b = IRBuilder(fptype=Datatype.F32)
    x = b.varalloc()
    b(f'float {x} = 0.0f;')
    with b.For(f'int i = 0; i < {K}; ++i', unroll=True):
        with b.If('i % 2 == 0'):
            b.Accumulate(f'{x}', 'i')
    b.Comment('legacy tail')
    return b.finish()


def build_select():
    """if/else producing a value, plus a predicated store."""
    b = IRBuilder(fptype=Datatype.F32)
    A = b.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='A')
    tid = b.thread_id('x')
    guard = b.op('lt', BOOL, tid, 9, hint='ok')

    sel = b.if_else(guard, (F32,))
    with sel.then():
        sel.yield_(b.load(A, tid, hint='hit'))
    with sel.otherwise():
        sel.yield_(b.const(0.0))

    st = b.store(A, sel.result, 0)
    return b.finish(), st


def _count(body, op):
    return sum(1 for s, _ in pir.walk(body) if s.op == op)


def main():
    body = build_contraction()
    verify(body)
    print('=== before ===')
    print(dump(body))

    stages = []
    opt = optimize(body, dump_hook=lambda n, b: stages.append((n, b)))
    verify(opt)
    print('\n=== after cse/licm/dce ===')
    print(dump(opt))

    # -- assertions ------------------------------------------------------- #
    loop_before = [s for s in body if s.op == Op.FOR][0]
    loop_after = [s for s in opt if s.op == Op.FOR][0]
    n_before = len(loop_before.regions[0].body)
    n_after = len(loop_after.regions[0].body)
    assert n_after < n_before, (n_before, n_after)
    # the invariant load left the loop
    assert not any(s.op == Op.LOAD and s.target[0].hint == 'scale'
                   for s in loop_after.regions[0].body)
    assert any(s.op == Op.LOAD and s.target[0].hint == 'scale' for s in opt)
    # duplicate multiply gone, dead multiply gone
    hints = {s.target[0].hint for s in loop_after.regions[0].body if s.op == 'mul'}
    assert hints == {'p', 'ps'}, hints          # 'q' folded away, 'dead' dropped

    # -- LICM must not fire across a conflicting write --------------------- #
    w = build_loop_with_write()
    verify(w)
    wo = optimize(w)
    wloop = [s for s in wo if s.op == Op.FOR][0]
    assert any(s.op == Op.LOAD for s in wloop.regions[0].body), \
        'load hoisted across a write to the same buffer'

    # -- uniformity ------------------------------------------------------- #
    d = build_divergent_barrier()
    diag = verify(d, strict=False)
    assert any('divergent' in m for m in diag), diag
    print('\n=== uniformity diagnostics ===')
    for m in diag:
        print(' ', m)

    # -- speculation ------------------------------------------------------ #
    sp = build_speculative()
    assert _count(sp, Op.LOAD) == 1, dump(sp)

    # -- legacy facade ---------------------------------------------------- #
    leg = build_legacy()
    verify(leg)
    assert _count(leg, Op.RAWBLOCK) == 2 and _count(leg, Op.RAWSTMT) == 3
    lw = Writer()
    emit(leg, lw)
    assert '#pragma unroll' in lw.get_src() and 'if (i % 2 == 0)' in lw.get_src()

    # -- if/else with results --------------------------------------------- #
    sel, _ = build_select()
    verify(sel)
    sw = Writer()
    emit(sel, sw)
    print('\n=== if/else with results ===')
    print(sw.get_src())

    # -- lowering --------------------------------------------------------- #
    writer = Writer()
    emit(opt, writer)
    print('\n=== generated C++ ===')
    print(writer.get_src())

    print('all checks passed')


if __name__ == '__main__':
    main()
