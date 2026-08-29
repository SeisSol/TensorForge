# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: moving statements past each other.

Every pass so far rewrote statements in place or deleted them.  This one is the
first that changes their *order*, which is the whole reason the last several
commits went to the trouble of making buffers into values, pointer bindings
into definitions and transfers into `copy.async`: order is exactly what cannot
be changed safely while a body is full of raw text whose effects are unknown.

`can_reorder(a, b)` is the predicate, and it is deliberately one function
rather than a rule spread across the passes that need it.  Two adjacent
statements may swap when none of these holds:

* **`b` uses what `a` defines.**  The def-use edge, which is why the migration
  mattered: a read through `glb_m1` used to be text, so nothing connected it
  to the binding above it, and any reorder had to assume the worst.
* **Their accesses conflict.**  `accesses_conflict` is read-after-read free
  and compares alias roots, so a window is the buffer it is a window into.
  An access with `base=None` conflicts with everything in its space --- which
  is what a raw statement declares, so raw text still pins, just locally
  instead of globally.
* **Either is immovable, carries an unknown effect, or is a barrier.**
  A barrier orders *all* threads, not just this one's memory, so no access
  analysis can license moving across it.
* **Either has regions.**  A loop or conditional is not opaque to this pass in
  principle, but its body may touch anything its header does not say, and the
  cheap thing to do is treat it as a wall.  Widening that is a later pass with
  its own test, not a flag here.

`sink_waits` and `hoist_issues` are the first two users, one in each
direction, and they are in the pipeline nowhere.  That is a measurement, not
an oversight: between them they move nothing but comments on the corpus --- 15
of 232 outputs differ, all of them in comment placement, and the mean
issue-to-wait distance goes 7.7 to 8.1 statements entirely through comments
changing places.  The macro layer already emits the wait immediately before
the read that needs it and the issue immediately after the pointer binding it
reads, so the schedule is already at the fixed point of both greedy moves.

The distance that is missing is not reachable by a local swap.  More than half
the transfers have five statements or fewer of cover; getting more means
moving an issue *across the loop back edge*, which is a different
transformation --- it has a distance parameter, it needs a prologue, and it
changes how many copies of a buffer are live.  This module is what that
transformation will check its moves against, and `overlap` is what will show
whether it worked.

So what this module does *not* do is decide how far anything should move.
"As late as legal" has no free parameter; a modulo schedule has several, and
they belong on top of this predicate rather than inside it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .core import Effect, Op, Stmt, Value, accesses_conflict, walk

#: Effects that no access analysis can reason across.
_WALL = Effect.BARRIER | Effect.UNKNOWN


def _defines(s: Stmt) -> Set[int]:
    return {t.id for t in s.target}


def _uses(s: Stmt) -> Set[int]:
    ids = {v.id for v in s.operands()}
    for r in s.regions:
        for inner, _ in walk(r.body):
            ids |= {v.id for v in inner.operands()}
    return ids


def is_wall(s: Stmt) -> bool:
    """Nothing crosses this statement.

    A region is no longer one by itself.  It was, and the docstring said the
    cheap thing was to treat it as a wall until something needed better --
    which turned out to be six of the corpus's loops, where the statement
    between a transfer and its wait is a `rawblock` and nothing else.

    What replaces the blanket rule is `touches`: a region's accesses are the
    union of its body's, and it is a wall only when something *inside* it is.
    That keeps the conservatism where it belongs -- a barrier or an unknown
    effect anywhere in the subtree still stops everything -- and drops it
    where a loop merely reads and writes buffers it declares.
    """
    return bool(s.effect & _WALL) or not s.movable or touches(s) is None


def touches(s: Stmt) -> Optional[Tuple]:
    """Every access in this statement's subtree, or None if it is a wall.

    None means "assume the worst": something in there carries an unknown
    effect, orders threads, or refuses to move, and no union of accesses can
    describe what crossing it would mean.
    """
    if s.effect & _WALL or not s.movable:
        return None
    out = list(s.accesses)
    for r in s.regions:
        for inner, _ in walk(r.body):
            if inner.effect & _WALL or not inner.movable:
                return None
            out.extend(inner.accesses)
    return tuple(out)


def may_cross(mover: Stmt, fixed: Stmt) -> bool:
    """May ``mover`` be emitted on the other side of ``fixed``?

    Directional, and the direction is the point.  `can_reorder` asks whether
    two statements may *swap*, so both have to be movable -- and a raw block
    is not, because its head is text whose semantics the IR cannot read.  But
    a transfer moving past a loop does not move the loop: the block stays
    exactly where it is, and whether it could have moved is not a question
    anyone asked.

    That distinction cost six of the corpus's loops, where the only thing
    between a transfer and its wait is a `rawblock` that nobody wanted to
    move.  What still matters about the fixed statement is what it *touches*,
    and `touches` says that for a whole subtree or says nothing at all.
    """
    if not mover.movable or mover.effect & _WALL:
        return False
    tm, tf = touches(mover), _touches_fixed(fixed)
    if tm is None or tf is None:
        return False
    if _defines(mover) & _uses(fixed) or _defines(fixed) & _uses(mover):
        return False
    return not any(accesses_conflict(x, y) for x in tm for y in tf)


def _touches_fixed(s: Stmt) -> Optional[Tuple]:
    """`touches` for a statement that is not being asked to move.

    Same walk, without the movability test: an immovable statement can still
    be described, and a description is all that crossing it needs.  A barrier
    or an unknown effect anywhere inside still returns None, because those are
    about what crossing *means* rather than about who moves.
    """
    if s.effect & _WALL:
        return None
    out = list(s.accesses)
    for r in s.regions:
        for inner, _ in walk(r.body):
            if inner.effect & _WALL:
                return None
            out.extend(inner.accesses)
    return tuple(out)


def can_reorder(a: Stmt, b: Stmt) -> bool:
    """May ``b``, which currently follows ``a``, be emitted before it?"""
    ta, tb = touches(a), touches(b)
    if ta is None or tb is None:
        return False
    if _defines(a) & _uses(b):
        return False                    # b reads what a wrote
    if _defines(b) & _uses(a):
        return False                    # a reads what b will write
    if _defines(a) & _defines(b):
        return False
    for x in ta:
        for y in tb:
            if accesses_conflict(x, y):
                return False
    return True


# --------------------------------------------------------------------------- #
# Sinking waits
# --------------------------------------------------------------------------- #

def sink_waits(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Move every `wait` as late as legality allows.

    A transfer overlaps with whatever sits between its issue and its wait, so
    the distance between them is the thing worth maximising and the only thing
    this pass changes.  It stops at the first statement it may not cross;
    ``can_reorder`` decides that, and for a `wait` the binding constraint is
    usually the first read of what the transfer wrote --- which is the answer
    one wants, since that read is the reason to wait at all.
    """
    out: List[Stmt] = []
    for s in body:
        if s.regions:
            s = replace(s, regions=tuple(replace(r, body=sink_waits(r.body))
                                         for r in s.regions))
        out.append(s)

    changed = True
    while changed:
        changed = False
        for i, s in enumerate(out):
            if s.op is not Op.WAIT and s.op != 'wait':
                continue
            j = i
            while j + 1 < len(out) and can_reorder(out[j], out[j + 1]):
                out[j], out[j + 1] = out[j + 1], out[j]
                j += 1
            if j != i:
                changed = True
    return tuple(out)


def hoist_issues(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Move every async issue as early as legality allows.

    The mirror of `sink_waits`, and on the current corpus the one that has
    something to do.  Sinking waits moves almost nothing, because the macro
    layer already emits the wait immediately before the read that needs it ---
    which is the right place and leaves nothing to win.  The distance is short
    at the other end: the issue sits where its instruction sits, which is
    wherever the transfer happened to be scheduled at macro level.

    Same predicate, opposite direction, and the same stopping rule: an issue
    rises until it meets something it may not cross, which for a `copy.async`
    is usually the write of the address it reads.
    """
    out: List[Stmt] = []
    for s in body:
        if s.regions:
            s = replace(s, regions=tuple(replace(r, body=hoist_issues(r.body))
                                         for r in s.regions))
        out.append(s)

    for i in range(len(out)):
        if out[i].op not in (Op.COPY_ASYNC, Op.LOAD_ASYNC):
            continue
        j = i
        while j > 0 and can_reorder(out[j - 1], out[j]):
            out[j - 1], out[j] = out[j], out[j - 1]
            j -= 1
    return tuple(out)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def overlap(body: Tuple[Stmt, ...]) -> Dict[int, int]:
    """Statements between each async issue and the wait that retires it.

    The number this pass exists to raise, reported per token so a change can
    be attributed rather than admired in aggregate.
    """
    issued: Dict[int, int] = {}
    spans: Dict[int, int] = {}
    for pos, (s, _) in enumerate(walk(body)):
        if s.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC):
            for t in s.target:
                issued[t.id] = pos
        elif s.op is Op.WAIT:
            for v in s.operands():
                if v.id in issued:
                    spans[v.id] = pos - issued[v.id] - 1
    return spans
