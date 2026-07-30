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
from tensorforge.backend.pir import (BOOL, INDEX, TOKEN, Effect, IRBuilder,
                                     MemSpace, Op, ScalarType, dump, emit,
                                     optimize, schedule_async, verify)
from tensorforge.common.vm.vm import vm_factory

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


TILES = 8
TILE = 64


def build_pipeline():
    """Double-buffered global -> LDS pipeline, the reason tokens exist.

    The copy issued in iteration k is waited in iteration k+1; the token rides
    through the loop's iter_args next to the accumulator.
    """
    b = IRBuilder(fptype=Datatype.F32)
    # leading dimension first (DataView convention): the thread index is the
    # fastest-varying axis, so lanes stay coalesced and the buffer index picks
    # the half of the double buffer.
    glb = b.alloc(Datatype.F32, (TILE, TILES), MemSpace.GLOBAL, hint='glb')
    lds = b.alloc(Datatype.F32, (TILE, 2), MemSpace.SHARED, hint='lds')
    tid = b.thread_id('x')
    zero = b.const(0.0)

    # prologue: fill buffer 0
    t0 = b.copy_async(lds, glb, dst_index=(tid, 0), src_index=(tid, 0))

    loop = b.for_(0, TILES - 1, 1, inits=(t0, zero), types=(TOKEN, F32), hint='k')
    with loop:
        k = loop.induction
        tok, acc = loop.iter_args
        nxt = b.op('add', INDEX, k, 1, hint='kn')
        par = b.op('rem', INDEX, nxt, 2, hint='par')
        cur = b.op('rem', INDEX, k, 2, hint='cur')
        # issue k+1 first, then wait for k: that is what buys the overlap
        t1 = b.copy_async(lds, glb, dst_index=(tid, par), src_index=(tid, nxt))
        b.wait(tok)
        b.barrier()
        x = b.load(lds, tid, cur, hint='x')
        acc2 = b.op('add', F32, acc, x, hint='sum')
        b.barrier()
        loop.yield_(t1, acc2)

    b.wait(loop.results[0])
    return b.finish(), loop


def build_reg_prefetch(depth=4):
    """Global -> register prefetch: issue `depth` loads, consume them in order.

    The classic AMD idiom --- each wait only needs the loads issued after it to
    stay in flight, so the counts run 3, 2, 1, 0.
    """
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (TILE, TILES), MemSpace.GLOBAL, hint='glb')
    tid = b.thread_id('x')
    toks = [b.load_async(glb, tid, i, hint=f'p{i}') for i in range(depth)]
    acc = b.const(0.0)
    for t in toks:
        v = b.wait(t)
        acc = b.op('add', F32, acc, v, hint='s')
    return b.finish()


def build_masked():
    """A predicated load must become a select, not a guard block."""
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (TILE, TILES), MemSpace.GLOBAL, hint='glb')
    tid = b.thread_id('x')
    ok = b.op('lt', BOOL, tid, 9, hint='ok')
    x = b.load(glb, tid, 0, hint='x', predicate=ok)
    b.op('mul', F32, x, x, hint='y')
    return b.finish()


def build_mixed():
    """One copy and one register load in flight at once."""
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (TILE, TILES), MemSpace.GLOBAL, hint='glb')
    lds = b.alloc(Datatype.F32, (TILE, 2), MemSpace.SHARED, hint='lds')
    tid = b.thread_id('x')
    tc = b.copy_async(lds, glb, dst_index=(tid, 0), src_index=(tid, 0))
    tl = b.load_async(glb, tid, 1, hint='r')
    b.wait(tc)
    b.wait(tl)
    return b.finish()


def build_carried_load():
    """Carrying a load.async token across the back edge must be rejected."""
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (TILE, TILES), MemSpace.GLOBAL, hint='glb')
    tid = b.thread_id('x')
    t0 = b.load_async(glb, tid, 0, hint='p')
    loop = b.for_(0, TILES - 1, 1, inits=(t0,), types=(TOKEN,), hint='k')
    with loop:
        k = loop.induction
        nxt = b.op('add', INDEX, k, 1, hint='kn')
        t1 = b.load_async(glb, tid, nxt, hint='p')
        b.wait(loop.iter_args[0])
        loop.yield_(t1)
    b.wait(loop.results[0])
    return b.finish()


def check_writer_parity():
    """The acceptance criterion for swapping `gen_code_inner` to `gen_ir`.

    The same legacy call sequence, once straight into a ``Writer`` and once
    through ``IRBuilder`` + ``emit``, must produce byte-identical source.  Until
    that holds, migrating an instruction is not a refactor but a rewrite: the
    diff on the generated kernel would hide whatever the change actually did.
    """
    import difflib
    from tensorforge.backend.writer import Writer as W

    def sequence(w):
        a = w.varalloc()
        w(f'float {a} = 0.0f;')
        w.Comment('accumulate')
        with w.Scope():
            with w.For('int i = 0; i < 8; ++i', unroll=True):
                b = w.varalloc()
                w(f'float {b} = data[i];')
                with w.If(f'{b} > 0'):
                    w(f'{a} += {b};')
            with w.While('cond'):
                w('spin();')
        w.Pragma('nounroll')
        w.new_line()
        with w.AnonymousScope():          # empty -> must elide entirely
            pass
        with w.Scope():
            with w.For('int j = 0; j < 4; ++j', unroll=True):
                pass                      # empty unrolled loop -> must elide
        w('done();')

    ref = W()
    sequence(ref)

    b = IRBuilder(fptype=Datatype.F32)
    sequence(b)
    body = b.finish()
    assert not verify(body, strict=False)
    got = W()
    emit(body, got)

    if ref.get_src() != got.get_src():
        raise AssertionError('writer parity lost:\n' + '\n'.join(
            difflib.unified_diff(ref.get_src().splitlines(),
                                 got.get_src().splitlines(),
                                 'Writer', 'IRBuilder', lineterm='')))
    return body


def check_linearized_loop():
    """`LinearizedLoop` is reached by no test case, so cover it here.

    Also the first place where the passes visibly earn their keep on migrated
    code: `threadIdx.x % blocksize` is loop-invariant and LICM hoists it.
    """
    from tensorforge.backend.symbol import Loop, LinearizedLoop
    from tensorforge.backend.writer import Writer as W
    from tensorforge.common.context import Context

    ctx = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    LinearizedLoop([Loop('k0', 0, 9, 1), Loop('k1', 2, 10, 2)],
                   blocksize=4).write(
        ctx, b, lambda idx: b(f'use({idx[0]._name}, {idx[1]._name});'))
    body = b.finish()
    assert not verify(body, strict=False)
    w = W()
    emit(optimize(body), w, ctx)
    src = w.get_src()
    # the lane offset left the loop, and no `/ 1`, `* 1` or `+ 0` survives
    assert src.index('threadIdx.x % 4') < src.index('for ('), src
    assert '/ 1' not in src and '* 1' not in src and '+ 0' not in src, src
    return src


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

    # -- async pipeline ---------------------------------------------------- #
    pipe, _ = build_pipeline()
    verify(pipe)
    pipe, adiag = schedule_async(pipe)
    assert not adiag, adiag
    waits = [s for s, _ in pir.walk(pipe) if s.op == Op.WAIT]
    assert [x.attr('prior') for x in waits] == [1, 0], \
        [x.attr('prior') for x in waits]
    print('\n=== async pipeline (prior counts derived) ===')
    print(dump(pipe))

    for arch, backend in (('sm_80', 'cuda'), ('gfx942', 'hip'), ('sm_70', 'cuda')):
        pw = Writer()
        emit(pipe, pw, vm_factory(arch, backend, 'float'))
        print(f'\n--- {arch}/{backend} ---')
        print(pw.get_src())

    # an unwaited copy is reported
    b2 = IRBuilder(fptype=Datatype.F32)
    g2 = b2.alloc(Datatype.F32, (4,), MemSpace.GLOBAL, hint='g')
    l2 = b2.alloc(Datatype.F32, (4,), MemSpace.SHARED, hint='l')
    b2.copy_async(l2, g2, dst_index=(0,), src_index=(0,))
    leaked = b2.finish()
    assert any('never' in m for m in verify(leaked, strict=False))

    # -- global -> register prefetch --------------------------------------- #
    regs = build_reg_prefetch()
    verify(regs)
    regs, rdiag = schedule_async(regs)
    assert not rdiag, rdiag
    priors = [x.attr('prior') for x, _ in pir.walk(regs) if x.op == Op.WAIT]
    assert priors == [3, 2, 1, 0], priors
    for arch, backend in (('gfx942', 'hip'), ('sm_80', 'cuda')):
        rw = Writer()
        emit(regs, rw, vm_factory(arch, backend, 'float'))
        print(f'\n=== register prefetch, {arch}/{backend} ===')
        print(rw.get_src())

    # -- one counter or two ------------------------------------------------ #
    mix = build_mixed()
    verify(mix)
    mix, _ = schedule_async(mix)
    waits = {x.attr('counter'): x for x, _ in pir.walk(mix) if x.op == Op.WAIT}
    # the copy is waited first, with the register load still in flight: same
    # class -> 0, any class -> 1.  AMD must use the latter or it under-waits.
    assert waits['copy'].attr('prior') == 0
    assert waits['copy'].attr('prior_unified') == 1
    mw = Writer()
    emit(mix, mw, vm_factory('gfx942', 'hip', 'float'))
    assert 'vmcnt(1)' in mw.get_src(), mw.get_src()
    mw = Writer()
    emit(mix, mw, vm_factory('sm_80', 'cuda', 'float'))
    assert '__pipeline_wait_prior(0);' in mw.get_src(), mw.get_src()

    # -- masked load ------------------------------------------------------- #
    msk = build_masked()
    verify(msk)
    kw = Writer()
    emit(msk, kw, vm_factory('gfx942', 'hip', 'float'))
    src = kw.get_src()
    assert '?' in src and 'if (' not in src, src
    print('\n=== masked load ===')
    print(src)

    # -- carried load token is rejected ------------------------------------ #
    car = build_carried_load()
    assert any('back edge' in m for m in verify(car, strict=False))

    lin = check_linearized_loop()
    print('\n=== LinearizedLoop (kein Testfall deckt ihn ab) ===')
    print(lin)

    # -- writer parity ----------------------------------------------------- #
    legacy_body = check_writer_parity()
    raw = sum(1 for x, _ in pir.walk(legacy_body) if x.op in Op.RAW)
    total = sum(1 for _ in pir.walk(legacy_body))
    print(f'\n=== writer parity: identical; {raw}/{total} nodes still raw ===')

    # -- lowering --------------------------------------------------------- #
    writer = Writer()
    emit(opt, writer)
    print('\n=== generated C++ ===')
    print(writer.get_src())

    print('all checks passed')


if __name__ == '__main__':
    main()
