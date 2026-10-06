# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The passes over a body as a pipeline the pass manager runs.

What the manager adds over calling the functions in a row is what the
functions cannot say about themselves: which facts a pass needs, which it
leaves behind, and which pass a verifier finding came from.  These pin those,
and that a body is finished once on its way to the emitter.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions import abstract_instruction
from tensorforge.backend.passmanager import PassManager
from tensorforge.backend.pir import BodyContext, WrapPrefetch, optimize, standard_pipeline, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, MemSpace, Op
from tensorforge.backend.pir.passes import flatten_scopes
from tensorforge.backend.pir.pipeline import Rewrite
from tensorforge.backend.pir.wrap import wrap_prefetch
from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import GenerationError


def _loop(scoped: bool = False):
    """A loop that fills one buffer and reads another, the transfer inside an
    anonymous scope where `scoped` -- as a loader opens one."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 256))
    fill = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    read = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='t')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='o')
    lane = b.thread_id('x')
    with b.for_(0, 8, 1) as loop:
        k = loop.induction
        # What the loop calls the next element; read by the store as well, so
        # that the cleanup in front of the prefetch keeps it.
        loop._next_index = b.op('add', INDEX, k, 1, hint='nk')
        if scoped:
            with b.AnonymousScope():
                token = b.copy_async(fill, glb, dst_index=(lane,),
                                     src_index=(k,))
                b.wait(token)
        else:
            token = b.copy_async(fill, glb, dst_index=(lane,), src_index=(k,))
            b.wait(token)
        b.store(out, b.load(read, lane, hint='u'), loop._next_index)
    return b, b.finish()


def _prefetch(b):
    return WrapPrefetch(wrap_prefetch,
                        lambda ty, hint, quals=(): b.value(ty, hint=hint,
                                                           quals=quals))


def _moved(body) -> bool:
    """Did a loop gain a carried token?"""
    return any(s.op is Op.FOR and s.target for s, _ in walk(body))


def test_a_requirement_no_earlier_pass_provides_is_refused_on_registration():
    b, _ = _loop()
    with pytest.raises(GenerationError, match=r"requires \['flat'\]"):
        PassManager().add(_prefetch(b))


def test_a_fact_the_transform_in_between_does_not_preserve_is_gone():
    """Registered after the pass that provides it, refused when it runs: a
    transform that does not say it keeps a fact invalidates it."""
    b, body = _loop()
    pm = PassManager()
    pm.add(Rewrite('flatten', flatten_scopes, provides=('flat',)))
    pm.add(Rewrite('reorders', lambda body: body))
    pm.add(_prefetch(b))
    with pytest.raises(GenerationError, match='invalidated by an earlier'):
        pm.run(BodyContext(body))


def test_a_pass_behind_the_scheduler_fails_the_run():
    """The emitter reads the wait counts; a pass after the scheduler may
    have moved what they count."""
    _, body = _loop()
    pm = standard_pipeline()
    pm.add(Rewrite('late', lambda body: body))
    with pytest.raises(GenerationError, match='scheduled'):
        pm.run(BodyContext(body))


def test_the_prefetch_sees_the_transfer_a_loader_scoped():
    """Behind the cleanup: a transfer inside an anonymous scope is one the
    prefetch pass cannot move, and `flatten` is what takes the scope away."""
    b, body = _loop(scoped=True)
    alone = wrap_prefetch(body, lambda ty, hint, quals=(): b.value(ty, hint=hint))
    assert not _moved(alone)

    piped = optimize(body, prefetch=_prefetch(b))
    assert _moved(piped)


def test_the_scheduler_counts_the_order_the_prefetch_left():
    """Ahead of the scheduler: inside the loop one transfer is in flight --
    the one this iteration issued for the next -- and the drain after the
    loop waits for it."""
    b, body = _loop()
    piped = optimize(body, prefetch=_prefetch(b))
    waits = [s for s, _ in walk(piped) if s.op is Op.WAIT]
    assert [w.attr('prior') for w in waits] == [1, 0]


def test_a_finding_names_the_pass_that_introduced_it(capsys):
    b = IRBuilder(fptype=Datatype.F32)
    value = b.op('add', INDEX, b.thread_id('x'), 1, hint='a')
    b(f'use({value});', value, accesses=())
    body = b.finish()

    def drop_definitions(body):
        return tuple(s for s in body if s.text is not None)

    pm = PassManager(debug='1')
    pm.add(Rewrite('drops', drop_definitions))
    pm.run(BodyContext(body, where='test'))
    printed = capsys.readouterr().out
    assert 'pir: test: after drops: rawstmt: raw text names' in printed, printed


class _Hardware:
    max_reg_per_thread = 1000


class _Lexic:
    simd_mode = False


class _VM:
    def get_hw_descr(self):
        return _Hardware()

    def get_lexic(self):
        return _Lexic()


class _Context:
    def get_vm(self):
        return _VM()


@pytest.mark.parametrize('materialized', [False, True])
def test_a_body_that_fits_is_finished_once(materialized, monkeypatch):
    """Measured where a broadcast was materialized, and the measurement is
    of the finished body -- which is then the one emitted."""
    context = _Context()
    finished = []
    monkeypatch.setattr(abstract_instruction.pir, 'pressure',
                        lambda body, **kwargs: 10)

    def attempt():
        context.materialized_broadcast = materialized
        return 'builder', ('body',)

    def finish(builder, body):
        finished.append(body)
        return body

    _, body = abstract_instruction._fused_if_over_budget(context, attempt,
                                                         finish)
    assert finished == [('body',)]
    assert body == ('body',)


def test_a_body_over_budget_is_built_again_and_that_one_is_finished(monkeypatch):
    context = _Context()
    attempts = []
    finished = []

    def attempt():
        attempts.append(getattr(context, 'force_fused_broadcast', False))
        context.materialized_broadcast = True
        return 'builder', (f'body{len(attempts)}',)

    def finish(builder, body):
        finished.append(body)
        return body

    monkeypatch.setattr(abstract_instruction.pir, 'pressure',
                        lambda body, **kwargs: 10_000)
    _, body = abstract_instruction._fused_if_over_budget(context, attempt,
                                                         finish)
    assert attempts == [False, True]
    assert finished == [('body1',), ('body2',)]
    assert body == ('body2',)
