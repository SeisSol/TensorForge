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

from .core import (Effect, Op, Region, Stmt, TokenType, Value,
                   accesses_conflict, walk)
from .passes import substitute
from .schedule import (_defines, _touches_fixed, _uses, can_reorder,
                       is_wall, may_cross, touches)


def _why_not(group: Sequence[Stmt], fixed: Stmt) -> str:
    """Name what blocks a crossing, because "a rawblock" is not a reason.

    The refusal that cost the most to diagnose said only that a `rawblock` was
    in the way.  It was in the way because it *writes the same shared buffer*
    the transfer does -- which, once said out loud, is not an obstacle at all
    but a symptom: a macro copy is split into hops, some of them direct
    statements of the guard and the rest inside an unrolled loop, and the
    group only ever collected the direct ones.  The block is the same
    transfer, and the pass was refusing to move a transfer past itself.
    """
    if not fixed.movable:
        detail = 'immovable'
    elif fixed.effect & (Effect.BARRIER | Effect.UNKNOWN):
        detail = 'barrier or unknown effect'
    else:
        detail = 'undescribable subtree'
    tf = _touches_fixed(fixed)
    if tf is None:
        return detail
    for g in group:
        tm = touches(g)
        for x in tm or ():
            for y in tf:
                if accesses_conflict(x, y):
                    base = getattr(y.base, 'name', None) or getattr(
                        y.base, 'hint', y.base)
                    return f'both touch {base}'
    if any(_defines(g) & _uses(fixed) or _defines(fixed) & _uses(g)
           for g in group):
        return 'def-use'
    return detail


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
    def _carries_async(st):
        return any(x.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC)
                   for x, _ in walk((st,)))

    guards = [(i, g) for i, g in enumerate(region.body) if g.op is Op.IF]
    if len(guards) > 1:
        raise Refusal('more than one guard in the body; which one a transfer '
                      'must leave is then a choice rather than a fact')
    direct = [st for st in region.body
              if st.op is not Op.IF and _carries_async(st)]
    inside = ([st for st in guards[0][1].regions[0].body if _carries_async(st)]
              if guards else [])
    if direct and inside:
        raise Refusal('transfers on both sides of the guard; which of them is '
                      'the transfer is then a choice rather than a fact')
    group = direct or inside
    if not group:
        raise Refusal('no async transfer in the body')
    guard_at = guards[0][0] if inside else None

    # The unit is a *section*, not a set of statements.
    #
    # A macro copy is split into hops of 4, 2 and 1 elements per lane plus a
    # predicated tail.  Some hops are direct statements; the rest are inside
    # an unrolled loop.  Collecting only the direct ones left the loop
    # standing between a transfer and its wait, writing the same buffer -- so
    # the pass refused to move a transfer past itself, and said so as `both
    # touch s0`.
    #
    # Widening the group to reach *into* the loop would be wrong: pulling
    # statements out of a loop changes how many times they run.  What crosses
    # the back edge is every top-level statement whose subtree issues, the
    # loop included and whole.
    scope = (region.body[guard_at].regions[0].body if guard_at is not None
             else region.body)
    tokens = {t.id for st in group for x, _ in walk((st,)) for t in x.target
              if x.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC)}
    waits = [w for w in scope if w.op is Op.WAIT
             and any(isinstance(a, Value) and a.id in tokens for a in w.args)]
    if len(waits) != 1:
        raise Refusal(f'{len(tokens)} transfers retired by {len(waits)} '
                      f'waits; a group is what one wait consumes')
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
                  report: Optional[List[str]] = None,
                  assume_rotated: bool = False) -> Tuple[Stmt, ...]:
    """Move each loop's single async issue one iteration earlier.

    ``next_index`` maps an operand id to the value that names the *next*
    element -- the loop's lookahead binding.  The pass does not invent it: a
    clamped successor index is a property of how the loop is traversed, which
    the loop knows and the IR does not.

    ``assume_rotated`` drops the one refusal that is about *space* rather than
    about legality: with two copies of the destination, the transfer for k+1
    no longer lands in the buffer k is reading.  It exists so the question
    "would rotation help here" can be asked of the pass itself rather than of
    a second predicate written to imitate it -- because the decision to
    allocate two copies has to be made before the body exists, and a copy of
    these criteria kept elsewhere is a copy that drifts.

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
            out.extend(_wrap_one(s, make_value, mapping, assume_rotated))
        except Refusal as why:
            if report is not None:
                report.append(str(why))
            out.append(s)
    return tuple(out)


def _wrap_one(loop: Stmt, make_value,
              next_index: Dict[int, Value],
              assume_rotated: bool = False) -> Sequence[Stmt]:
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
        if not all(may_cross(g, s) for g in group):
            raise Refusal(f'the issue may not cross a `{s.op}` between it and '
                          f'its wait ({_why_not(group, s)}), so it may not '
                          f'cross the back edge either')

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
    # The whole subtree, not the top level.  The compute reads its operand
    # inside nested loops, so a one-level scan found no read of the
    # destination and accepted a transfer that fills the buffer the current
    # element is still reading -- the exact race this check exists to refuse,
    # slipping through because the read was two regions down.
    scan = [] if assume_rotated else [st for st, _ in walk(region.body)]
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

    # The destination has to exist before the loop, or the prologue cannot
    # name it.  `s4 = &localShrMem0[..]` is declared inside the loop body, and
    # the peeled transfer is emitted before it: the peel then writes through a
    # value whose `extern` name is bound later, so the generated code uses
    # `v33_s4` above the line that declares `s4`.  It renders and does not
    # compile, which is the failure mode `test_syntax` exists for and the
    # corpus alone would not show.
    #
    # Declaring the window ahead of the loop is the fix, and it is the same
    # move that took the address bindings and the windows out of the guard --
    # one scope further out.  Until then this declines rather than emitting
    # something that cannot build.
    defined_in_loop = {t.id for st in region.body for t in st.target}
    if guard_at is not None:
        defined_in_loop |= {t.id for st in region.body[guard_at].regions[0].body
                            for t in st.target}
    if any(isinstance(a, Value) and a.id in defined_in_loop
           for g in group for a in g.args[:1]):
        raise Refusal('the destination is declared inside the loop, so a '
                      'peeled transfer would name it before it exists')

    # The group moves to the *next* element -- and so does everything it
    # reads that names the element.
    #
    # A transfer reads through `glb_m2`, a pointer already offset by
    # `batchId0`, so the index does not appear in the transfer's operands at
    # all: it appears in the binding.  Substituting into the group alone finds
    # nothing to substitute.  What has to move is the backward slice: every
    # statement the group transitively reads that mentions the induction, with
    # the successor index put in its place.  That slice *is* the rolling
    # pointer the old macro-level pipeline built by hand.
    # The whole subtree, since a section member may be a hop loop and the
    # index it reads is an operand of a statement inside it, not of the block.
    def _names_index(st):
        return any(isinstance(a, Value) and a.id in next_index
                   for x, _ in walk((st,)) for a in x.args)

    slice_ = _index_slice(region, group, next_index)
    if not slice_ and not any(_names_index(g) for g in group):
        raise Refusal('the transfer does not name an index this loop knows '
                      'how to advance')
    advanced, advance_map = _advance(slice_, next_index, make_value)
    rewritten = [substitute((g,), {**next_index, **advance_map})[0]
                 for g in group]

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
        body[at:at] = advanced + rewritten
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
        body[guard_at:guard_at] = advanced + rewritten

    term = region.terminator
    yielded = (term.args if term is not None else ()) + tuple(tokens)
    body = [s for s in body if s.op is not Op.YIELD]
    body.append(Stmt(op=Op.YIELD, args=tuple(yielded)))

    # The prologue: the same transfers, for the element the loop starts on.
    # Not `induction - 1` and not `0` -- the loop's own `lo`, which is where
    # the first iteration would have issued from.
    lo = loop.loop_bounds[0]
    peel_slice, peel_map = _advance(slice_, {loop.induction.id: lo},
                                    make_value)
    peel = peel_slice + [
        replace(substitute((g,), {loop.induction.id: lo, **peel_map})[0],
                target=(make_value(g.target[0].type, 'cp'),))
        for g in group]

    results = [make_value(t.type, 'cp') for t in tokens]
    new_loop = replace(
        loop,
        target=loop.target + tuple(results),
        args=loop.args + tuple(p.target[0] for p in peel[-len(group):]),
        regions=(replace(region, args=region.args + tuple(carried),
                         body=tuple(body)),))
    drain = Stmt(op=Op.WAIT, args=tuple(results), pure=False, movable=True,
                 effect=wait.effect, accesses=wait.accesses)
    return list(peel) + [new_loop, drain]


def _index_slice(region: Region, group: Sequence[Stmt],
                 next_index: Dict[int, Value]) -> List[Stmt]:
    """The statements the group reads that lead back to the element index.

    Backwards from the group's operands, through the loop's own region only:
    anything inside the guard has already been refused, and anything outside
    the loop does not depend on the iteration.  A statement joins the slice if
    it defines something the slice reads *and* it, or something it reads,
    names an index the loop can advance.

    Only pure, movable statements with no writes.  The slice is *duplicated*,
    not moved -- the current element still needs its own copy -- so a member
    that wrote anything would write it twice.
    """
    by_def = {t.id: s for s in region.body for t in s.target}
    want = {a.id for g in group for x, _ in walk((g,))
            for a in x.args if isinstance(a, Value)}
    slice_: List[Stmt] = []
    seen = set()
    changed = True
    while changed:
        changed = False
        for s in region.body:
            if id(s) in seen or not s.target:
                continue
            if not any(t.id in want for t in s.target):
                continue
            if s.has_side_effects or not s.movable or s.regions:
                continue
            if any(a.writes for a in s.accesses):
                continue
            seen.add(id(s))
            slice_.append(s)
            want |= {a.id for a in s.args if isinstance(a, Value)}
            changed = True
    # Keep only what actually leads to the index; a binding that does not
    # mention it is the same for every element and must not be duplicated.
    keep = []
    reaches = set(next_index)
    for s in reversed(slice_):
        if any(isinstance(a, Value) and a.id in reaches for a in s.args):
            keep.append(s)
            reaches |= {t.id for t in s.target}
    order = {id(s): i for i, s in enumerate(region.body)}
    return sorted(keep, key=lambda s: order[id(s)])


def _advance(slice_: Sequence[Stmt], mapping: Dict[int, Value],
             make_value) -> Tuple[List[Stmt], Dict[int, Value]]:
    """Clone the slice with the index replaced; return the clones and a map.

    The clones drop `decl` and `extern`.  Those carry a declarator the caller
    wrote with a name in it, and a second statement declaring `glb_m2` would
    be a redefinition rather than a second pointer.  Without them the emitter
    names the value itself and renders the type, which for a buffer is a
    plain pointer -- `const` and `__restrict__` are lost on the clone, which
    costs optimisation and not correctness.
    """
    out: List[Stmt] = []
    sub = dict(mapping)
    for s in slice_:
        fresh = tuple(make_value(t.type, t.hint or 'adv') for t in s.target)
        clone = replace(substitute((s,), sub)[0], target=fresh,
                        attrs=tuple(a for a in s.attrs
                                    if a[0] not in ('decl', 'extern')))
        out.append(clone)
        sub.update({t.id: f for t, f in zip(s.target, fresh)})
    return out, {k: v for k, v in sub.items() if k not in mapping}
