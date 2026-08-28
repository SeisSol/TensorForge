# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: moving a prefetch across the back edge.

`schedule.can_reorder` licenses swaps within a body, and the measurement that
came with it said the schedule is already at their fixed point: the wait sits
immediately before the read that needs it, the issue immediately after the
binding it reads, and more than half the transfers have five statements or
fewer of cover.  The distance that is missing is not reachable by any local
move.  It is one iteration away.

So this is the transformation the whole chain was for.  A transfer issued for
element `k` and waited in the same iteration becomes a transfer issued for
element `k+1` and waited in the *next* iteration::

    %t0 = copy.async  ...                 # peeled, for element 0
    %tn = for %k iter(%t = %t0) {
            %t1 = copy.async ... %next    # for element k+1
            wait %t                       # what k-1 issued -- prior = 1
            ... compute on it ...
            yield %t1
          }
    wait %tn                              # drain

The accounting is the one from the very first slot census.  A body of `n`
compute slots carrying a transfer at distance `d` needs `ceil((d + 1) / n)`
copies of its buffer, so `d <= n - 1` is free: one copy, no rotation, no stage
index.  This pass does `d = n`, one whole iteration, which is the first value
that needs two -- and that is why it takes the destination buffer as something
the caller has already double-buffered, rather than pretending the copies are
free.

What it refuses, and why each one would be wrong:

* **A transfer whose token is not waited in the same region.**  Then the pass
  does not know what it would be moving away from.
* **A destination read before the wait.**  The read would see the value the
  *next* element's transfer is landing into.
* **Anything between the issue and the wait that `can_reorder` will not let
  the issue cross.**  Crossing the back edge is a stronger move than a swap,
  so it needs at least the same licence.
* **A loop that already carries the token.**  Applying this twice would build
  a distance of two iterations behind one buffer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .core import (Op, Region, Stmt, TokenType, Value, accesses_conflict,
                   walk)
from .passes import substitute
from .schedule import can_reorder


class Refusal(Exception):
    """Why a loop was left alone.  Carried rather than logged: the caller
    asked for a transformation and is entitled to the reason it did not
    happen."""


def _sole_async(region: Region) -> Tuple[int, Stmt, Optional[int]]:
    """The one async issue, whether it is the loop's or its guard's.

    Returns ``(index, statement, guard_index)``, where `guard_index` is the
    position of the `Op.IF` it was found inside, or None if it was a direct
    statement of the loop.

    Looking inside one guard is not a convenience.  The batch loop's guard is
    *per element* -- `flags0[batchId0]` -- so element k being masked says
    nothing about k+1, and a prefetch for k+1 that sits under k's mask breaks
    the chain for every element after a masked one.  A transfer moved across
    the back edge therefore has to leave the guard, which is the same rule
    `_split_guard` already applies to barriers and for the same reason: what
    the next iteration depends on cannot be conditional on this one.
    """
    direct = [s for s in region.body
              if s.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC)]
    guards = [(i, s) for i, s in enumerate(region.body) if s.op is Op.IF]
    inside = []
    for gi, g in guards:
        for s in g.regions[0].body:
            if s.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC):
                inside.append((gi, s))
    if direct and inside:
        raise Refusal('transfers on both sides of the guard; which of them is '
                      'the transfer is then a choice rather than a fact')
    if len(guards) > 1:
        raise Refusal('more than one guard in the body; which one a transfer '
                      'must leave is then a choice rather than a fact')
    group = direct or [s for _, s in inside]
    if not group:
        raise Refusal('no async transfer in the body')
    guard_at = inside[0][0] if inside else None

    # One wait for all of them, or they are not one transfer.  A macro copy is
    # split into hops of 4, 2 and 1 elements per lane plus a predicated tail,
    # each its own `copy.async`, and `LoadWait` retires the lot with a single
    # `wait` naming every token.  So "several issues" is not several transfers
    # -- it is one transfer in pieces, and the group is exactly what that wait
    # consumes.  If the tokens go to different waits, they are independent
    # transfers and choosing between them would be a schedule.
    scope = (region.body[guard_at].regions[0].body if guard_at is not None
             else region.body)
    tokens = {t.id for s in group for t in s.target}
    waits = [w for w in scope if w.op is Op.WAIT
             and any(isinstance(a, Value) and a.id in tokens for a in w.args)]
    if len(waits) != 1:
        raise Refusal(f'{len(group)} transfers retired by {len(waits)} waits; '
                      f'a group is what one wait consumes')
    if not tokens <= {a.id for a in waits[0].args if isinstance(a, Value)}:
        raise Refusal('the wait does not name every token of the group')
    return group, waits[0], guard_at


def _its_wait(region: Region, token: Value) -> Tuple[int, Stmt]:
    for i, s in enumerate(region.body):
        if s.op is Op.WAIT and any(isinstance(a, Value) and a.id == token.id
                                   for a in s.args):
            return i, s
    raise Refusal('the token is not waited in the same region')


def wrap_prefetch(body: Tuple[Stmt, ...], make_value,
                  next_index: Optional[Dict[int, Value]] = None,
                  report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """Move each loop's single async issue one iteration earlier.

    ``next_index`` maps an operand id to the value that names the *next*
    element -- the loop's lookahead binding.  The pass does not invent it: a
    clamped successor index is a property of how the loop is traversed, which
    the loop knows and the IR does not.

    ``make_value(type, hint)`` mints the carried token, the loop result and
    the prologue's.  A factory rather than a builder, because the statements
    this produces go where the pass puts them, not where a builder's cursor
    happens to be.
    """
    out: List[Stmt] = []
    for s in body:
        if s.op is not Op.FOR:
            out.append(s)
            continue
        mapping = next_index
        if mapping is None:
            nxt = s.attr('next')
            mapping = ({s.induction.id: nxt} if nxt is not None else None)
        if mapping is None:
            if report is not None:
                report.append('loop does not name its successor index')
            out.append(s)
            continue
        try:
            out.extend(_wrap_one(s, make_value, mapping))
        except Refusal as why:
            if report is not None:
                report.append(str(why))
            out.append(s)
    return tuple(out)


def _wrap_one(loop: Stmt, make_value,
              next_index: Dict[int, Value]) -> Sequence[Stmt]:
    region = loop.regions[0]
    if any(isinstance(a.type, TokenType) for a in region.args):
        raise Refusal('this loop already carries a token')

    group, wait, guard_at = _sole_async(region)
    scope = (region.body if guard_at is None
             else region.body[guard_at].regions[0].body)
    i_wait = next(i for i, s in enumerate(scope) if s is wait)
    first = min(i for i, s in enumerate(scope) if s in group)
    if i_wait < first:
        raise Refusal('the wait precedes the issue; nothing to stretch')

    if guard_at is not None:
        # Leaving the guard means the group may no longer use anything the
        # guard defines -- including its condition, which it must not depend
        # on: the whole point is that a masked element still prefetches.
        inner = region.body[guard_at].regions[0]
        inside_defs = {t.id for s in inner.body if s not in group
                       for t in s.target}
        inside_defs |= {a.id for a in inner.args}
        if any(isinstance(a, Value) and a.id in inside_defs
               for s in group for a in s.args):
            raise Refusal('the transfer reads something the guard defines, so '
                          'it cannot be issued outside it')

    between = [s for s in scope[first:i_wait] if s not in group]
    for s in between:
        if not all(can_reorder(g, s) for g in group):
            raise Refusal('the issue may not cross a statement between it '
                          'and its wait, so it may not cross the back edge '
                          'either')

    # The destination must not be read anywhere in the body.
    #
    # This is the slot accounting from the first census, enforced rather than
    # assumed.  A transfer at distance `d` in a body of `n` slots needs
    # `ceil((d + 1) / n)` copies of its buffer; this pass does `d = n`, one
    # whole iteration, which is the first value that needs two.  With one
    # copy, the transfer this iteration issues for element k+1 lands in the
    # buffer iteration k is reading -- a race that the wait does not cover,
    # because the wait is for the *previous* transfer.
    #
    # So a single-buffered destination is refused, not silently accepted.
    # Rotating the buffer is a separate transformation with its own cost, and
    # a pass that quietly assumed someone else had done it would be wrong in
    # exactly the cases where nobody had.
    dst_writes = [a for g in group for a in g.accesses if a.writes]
    scan = list(region.body)
    if guard_at is not None:
        scan += list(region.body[guard_at].regions[0].body)
    for s in scan:
        if s in group or s.op is Op.WAIT or s.op is Op.IF:
            continue
        for a in s.accesses:
            if a.writes:
                continue
            if any(accesses_conflict(a, w) for w in dst_writes):
                raise Refusal(
                    'the destination is read in the same iteration, so one '
                    'copy of it is not enough for a distance of one '
                    'iteration: ceil((d+1)/n) with d = n is 2')

    # The group moves to the *next* element.
    rewritten = [substitute((g,), next_index)[0] for g in group]
    if all(not isinstance(a, Value) or a.id not in next_index
           for g in group for a in g.args):
        raise Refusal('the transfer does not name an index this loop knows '
                      'how to advance')

    tokens = [g.target[0] for g in group]
    carried = [make_value(t.type, 'cp') for t in tokens]
    swap = dict(zip((t.id for t in tokens), carried))
    new_wait = replace(wait, args=tuple(
        swap.get(a.id, a) if isinstance(a, Value) else a for a in wait.args))

    if guard_at is None:
        body = [s for s in region.body if s not in group]
        body[[i for i, s in enumerate(body) if s is wait][0]] = new_wait
        at = min(i for i, s in enumerate(body)
                 if s is new_wait or s is wait)
        body[at:at] = rewritten
    else:
        guard = region.body[guard_at]
        inner = [s for s in guard.regions[0].body if s not in group]
        inner[[i for i, s in enumerate(inner) if s is wait][0]] = new_wait
        # The group lands in the loop's region, before the guard: outside it,
        # because the next element's transfer must not be conditional on this
        # element's mask.
        body = list(region.body)
        body[guard_at] = replace(guard, regions=(
            replace(guard.regions[0], body=tuple(inner)),))
        body[guard_at:guard_at] = rewritten

    term = region.terminator
    yielded = (term.args if term is not None else ()) + tuple(tokens)
    body = [s for s in body if s.op is not Op.YIELD]
    body.append(Stmt(op=Op.YIELD, args=tuple(yielded)))

    # The prologue: the same transfers, for the element the loop starts on.
    # Not `induction - 1` and not `0` -- the loop's own `lo`, which is where
    # the first iteration would have issued from.
    lo = loop.loop_bounds[0]
    peel = [replace(substitute((g,), {loop.induction.id: lo})[0],
                    target=(make_value(g.target[0].type, 'cp'),))
            for g in group]

    results = [make_value(t.type, 'cp') for t in tokens]
    new_loop = replace(
        loop,
        target=loop.target + tuple(results),
        args=loop.args + tuple(p.target[0] for p in peel),
        regions=(replace(region, args=region.args + tuple(carried),
                         body=tuple(body)),))
    drain = Stmt(op=Op.WAIT, args=tuple(results), pure=False, movable=True,
                 effect=wait.effect, accesses=wait.accesses)
    return list(peel) + [new_loop, drain]
