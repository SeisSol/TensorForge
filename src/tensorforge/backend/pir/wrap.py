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


def _sole_async(region: Region) -> Tuple[int, Stmt]:
    issues = [(i, s) for i, s in enumerate(region.body)
              if s.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC)]
    if len(issues) != 1:
        raise Refusal(f'{len(issues)} async issues in the body; this pass '
                      f'moves one, and picking among several is a schedule '
                      f'rather than a rewrite')
    return issues[0]


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

    i_issue, issue = _sole_async(region)
    token = issue.target[0]
    i_wait, wait = _its_wait(region, token)
    if i_wait < i_issue:
        raise Refusal('the wait precedes the issue; nothing to stretch')

    for s in region.body[i_issue + 1:i_wait]:
        if not can_reorder(issue, s):
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
    dst_writes = [a for a in issue.accesses if a.writes]
    for s in region.body:
        if s is issue or s.op is Op.WAIT:
            continue
        for a in s.accesses:
            if a.writes:
                continue
            if any(accesses_conflict(a, w) for w in dst_writes):
                raise Refusal(
                    'the destination is read in the same iteration, so one '
                    'copy of it is not enough for a distance of one '
                    'iteration: ceil((d+1)/n) with d = n is 2')

    # The issue moves to the *next* element.
    rewritten = substitute((issue,), next_index)[0]
    if rewritten is issue or all(
            not isinstance(a, Value) or a.id not in next_index
            for a in issue.args):
        raise Refusal('the issue does not name an index this loop knows how '
                      'to advance')

    carried = make_value(token.type, 'cp')
    body = list(region.body)
    body[i_issue] = rewritten
    body[i_wait] = replace(wait, args=tuple(
        carried if isinstance(a, Value) and a.id == token.id else a
        for a in wait.args))

    term = region.terminator
    yielded = (term.args if term is not None else ()) + (token,)
    body = [s for s in body if s.op is not Op.YIELD]
    body.append(Stmt(op=Op.YIELD, args=tuple(yielded)))

    # The prologue: the same transfer, for the element the loop starts on.
    # Not `induction - 1` and not `0` -- the loop's own `lo`, which is where
    # the first iteration would have issued from.
    lo = loop.loop_bounds[0]
    peel = replace(substitute((issue,), {loop.induction.id: lo})[0],
                   target=(make_value(token.type, 'cp'),))

    result = make_value(token.type, 'cp')
    new_loop = replace(
        loop,
        target=loop.target + (result,),
        args=loop.args + (peel.target[0],),
        regions=(replace(region, args=region.args + (carried,),
                         body=tuple(body)),))
    drain = Stmt(op=Op.WAIT, args=(result,), pure=False, movable=True,
                 effect=wait.effect, accesses=wait.accesses)
    return [peel, new_loop, drain]
