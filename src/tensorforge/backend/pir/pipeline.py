# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: the passes over one body, run by the pass manager.

Every pass in `passes` is a function ``body -> body``; what this module adds
is what the functions cannot say about themselves.  Which facts a pass leaves
behind and which it needs is declared once, on the pass, and checked by
`PassManager` when the pipeline is assembled and again when it runs:

``flat``       no anonymous scope hides a statement from the passes
               (`flatten_scopes`).  The prefetch pass needs it: a transfer
               two rawblocks deep is one it cannot see.
``scheduled``  every `wait` counts what is outstanding in the final issue
               order (`schedule_async`).  Anything that moves a statement
               invalidates it, and the emitter relies on it -- so the pipeline
               delivers it, and a pass registered behind the scheduler is
               refused at the end of the run rather than emitted with stale
               counts.

Under `ir_debug` the body is verified after every pass, and a finding is
reported with the name of the pass that introduced it; with `dump` in the
option each pass's output is printed as well.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from tensorforge.backend.passmanager import Pass, PassContext, PassManager

from .asyncmem import schedule_async
from .core import Stmt, dump
from .passes import (converge_crosslane, cse, dce, flatten_scopes, fold,
                     if_convert, licm, load_cse, verify)


class BodyContext(PassContext):
    """One body, and what is known about it."""

    def __init__(self, body: Tuple[Stmt, ...], *, explicit_simd: bool = False,
                 where: str = ''):
        super().__init__()
        self.body = body
        #: What built the body, for the findings reported about it.
        self.where = where
        #: Whether the lane is in the type, where a guard over a lane-varying
        #: condition is a mask with no branch to lower it to.
        self.explicit_simd = explicit_simd
        #: What `schedule_async` could not determine statically.
        self.diagnostics: List[str] = []
        self._reported: frozenset = frozenset()

    def check(self, stage: str, debug: str) -> None:
        found = verify(self.body, strict=False)
        prefix = f'pir: {self.where}: ' if self.where else 'pir: '
        for finding in found:
            if finding not in self._reported:
                print(prefix + (finding if stage == 'build'
                                else f'after {stage}: {finding}'))
        self._reported = frozenset(found)
        if 'dump' in debug:
            print(f'// pir after {stage}\n{dump(self.body)}')


class Rewrite(Pass):
    """A function ``body -> body`` as a pass.

    A fact in `provides` is one the function establishes of its output; a
    fact in `preserves` is one it leaves standing if it held before.
    """

    is_transform = True

    def __init__(self, name: str, fn: Callable[[Tuple[Stmt, ...]], Tuple[Stmt, ...]],
                 *, requires: Sequence[str] = (), provides: Sequence[str] = (),
                 preserves: Sequence[str] = (),
                 when: Optional[Callable[[BodyContext], bool]] = None):
        self.name = name
        self.requires = tuple(requires)
        self.provides = tuple(provides)
        self.preserves = tuple(preserves)
        self._fn = fn
        self._when = when

    def enabled(self, pc: BodyContext) -> bool:
        return self._when is None or self._when(pc)

    def run(self, pc: BodyContext) -> None:
        pc.body = self._fn(pc.body)
        for fact in self.provides:
            pc.put(fact, True)


class ScheduleAsync(Pass):
    """Place the commits and count what every `wait` waits for."""

    name = 'async'
    provides = ('scheduled',)
    preserves = ('flat',)
    is_transform = True

    def run(self, pc: BodyContext) -> None:
        pc.body, found = schedule_async(pc.body)
        pc.diagnostics.extend(found)
        pc.put('scheduled', True)


class WrapPrefetch(Pass):
    """Move each loop's transfer one iteration earlier (`wrap.wrap_prefetch`).

    `wrap` is the pass itself or a stand-in with its signature -- the rotation
    query asks its questions through one (`Generator._rotation_targets`).
    `make_value` mints the values the moved transfer carries, and `report`
    collects the reasons a loop was left alone.
    """

    name = 'wrap'
    requires = ('flat',)
    preserves = ('flat',)
    is_transform = True

    def __init__(self, wrap, make_value, report: Optional[List[str]] = None):
        self._wrap = wrap
        self._make_value = make_value
        self._report = report

    def run(self, pc: BodyContext) -> None:
        pc.body = self._wrap(pc.body, self._make_value, report=self._report)


def standard_pipeline(debug: str = '',
                      prefetch: Optional[WrapPrefetch] = None) -> PassManager:
    """The passes every body goes through, in their order.

    ``fold`` runs first: it turns expressions into constants and removes
    identity operations, which gives ``cse`` more equal keys to merge and
    ``licm`` fewer statements to consider.  It runs a second time after
    ``licm``, because hoisting can bring two constants into the same scope.

    ``if_convert`` runs only under the explicit vector, and there it is not an
    optimization.  A guard over a lane-varying condition is a mask in that
    model and there is no branch to lower it to, so the conversion is the
    only legal path rather than a trade of one shared brace for several.  It
    runs before ``fold``, so that the predicates it attaches take part in the
    same simplification as everything else.

    ``load_cse`` runs after ``cse`` and before ``licm``: it removes the loads
    that would otherwise be hoisting candidates, so ``licm`` sees fewer
    statements.  It wants ``cse`` at its fixed point in front of it, since a
    load is keyed on the *value* its address is, not on the expression that
    spells it.  A second run after ``cse2`` finds 32 more loads out of ~2900
    on ``local_flux`` at order 6, which does not pay for another sweep of the
    whole body, so it is not in the pipeline.

    `schedule.hoist_issues` and `schedule.sink_waits` are deliberately *not*
    here.  Both are correct, and on the corpus they move nothing but comments
    between them, on 15 of 232 outputs, and the mean issue-to-wait distance
    goes 7.7 to 8.1 statements entirely through comments changing places.
    The schedule the macro layer produces is already at the fixed point of
    those two greedy moves --- the wait sits immediately before the read that
    needs it, and the issue sits immediately after the pointer binding it
    reads.

    That is worth knowing rather than working around.  The distance that is
    missing is not reachable by any local swap: more than half the transfers
    have five statements or fewer of cover, and getting more means moving an
    issue across the loop back edge, which is a different transformation with
    a distance parameter and a prologue -- `prefetch`, where it is asked for,
    which goes behind everything that cleans the body up: the transfers sit
    in the anonymous scopes the loaders open until ``flatten`` has removed
    them.

    ``schedule_async`` runs last on purpose: the wait counts depend on the
    final issue order, so anything that may still move statements has to have
    happened already.
    """
    pm = PassManager(debug=debug, delivers=('scheduled',))
    pm.add(Rewrite('flatten', flatten_scopes, provides=('flat',)))
    pm.add(Rewrite('if_convert',
                   lambda body: if_convert(body, sink_into_loops=True),
                   preserves=('flat',), when=lambda pc: pc.explicit_simd))
    for name, fn in (('converge', converge_crosslane), ('fold', fold),
                     ('cse', cse), ('loads', load_cse), ('licm', licm),
                     ('fold2', fold), ('cse2', cse), ('dce', dce)):
        pm.add(Rewrite(name, fn, preserves=('flat',)))
    if prefetch is not None:
        pm.add(prefetch)
    pm.add(ScheduleAsync())
    return pm


def optimize(body: Tuple[Stmt, ...], *, explicit_simd: bool = False,
             debug: str = '', diagnostics: Optional[List[str]] = None,
             prefetch: Optional[WrapPrefetch] = None,
             where: str = '') -> Tuple[Stmt, ...]:
    """`body` through the standard pipeline (`standard_pipeline`).

    ``diagnostics`` collects what the scheduler could not determine;
    ``prefetch`` adds the prefetch pass ahead of the scheduler; ``where``
    names what built the body in the findings `debug` reports.
    """
    pc = BodyContext(body, explicit_simd=explicit_simd, where=where)
    standard_pipeline(debug, prefetch).run(pc)
    if diagnostics is not None:
        diagnostics.extend(pc.diagnostics)
    return pc.body
