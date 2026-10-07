# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: moving statements past each other.

Rewriting statements in place or deleting them leaves their *order* alone;
this module is about changing it.  Order is exactly what cannot be changed
safely while a body is full of raw text whose effects are unknown, which is why
buffers are values, pointer bindings definitions and transfers `copy.async`.

`can_reorder(a, b)` is the predicate, and it is deliberately one function
rather than a rule spread across the passes that need it.  Two adjacent
statements may swap when none of these holds:

* **`b` uses what `a` defines.**  The def-use edge, which is why a pointer
  binding is a definition: a read through `glb_m1` written as text would
  connect to nothing, not even the binding above it, and any reorder would
  have to assume the worst.
* **Their accesses conflict.**  `accesses_conflict` is read-after-read free
  and compares alias roots, so a window is the buffer it is a window into.
  An access with `base=None` conflicts with everything in its space --- which
  is what a raw statement declares, so raw text still pins, just locally
  instead of globally.
* **Either is immovable, carries an unknown effect, or is a barrier.**
  A barrier orders *all* threads, not just this one's memory, so no access
  analysis can license moving across it.
* **Either holds such a statement in its regions.**  A region is not a wall by
  itself: its accesses are the union of its body's (`touches`), and it is a
  wall only when something *inside* it is.  A barrier or an unknown effect
  anywhere in the subtree still stops everything; a loop that merely reads
  and writes buffers it declares does not.

`sink_waits` and `hoist_issues` apply it greedily, one in each direction, and
they are in the pipeline nowhere.  That is a measurement, not an oversight:
behind `move`, which issues each transfer early under a distance, they find
almost nothing --- 21 of 505 outputs differ on the corpus for five targets,
all but three in where a comment stands, and those three by a statement.
`move` leaves the issue right behind what stopped it and the wait in front of
the read that needs it, so the schedule is at the fixed point of both greedy
moves but for those.

The distance that is missing is not reachable by a local swap.  More than half
the transfers have five statements or fewer of cover; getting more means
moving an issue *across the loop back edge*, which is a different
transformation --- it has a distance parameter, it needs a prologue, and it
changes how many copies of a buffer are live.  This module is what that
transformation checks its moves against, and `overlap` is what shows whether
it works.

So what this module does *not* do is decide how far anything should move.
"As late as legal" has no free parameter; a modulo schedule has several, and
they belong on top of this predicate rather than inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .core import (Effect, MemSpace, Op, Stmt, accesses_conflict, walk,
                   walk_stmts)

#: Effects that no access analysis can reason across.
_WALL = Effect.BARRIER | Effect.UNKNOWN


def _defines(s: Stmt) -> Set[int]:
    return {t.id for t in s.target}


def _uses(s: Stmt) -> Set[int]:
    ids = {v.id for v in s.operands()}
    for r in s.regions:
        for inner in walk_stmts(r.body):
            ids |= {v.id for v in inner.operands()}
    return ids


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
        for inner in walk_stmts(r.body):
            if inner.effect & _WALL or not inner.movable:
                return None
            out.extend(inner.accesses)
    return tuple(out)


@dataclass(frozen=True)
class Footprint:
    """What crossing a statement is about: what its subtree touches -- None
    where it is a wall (`_touches_fixed`) -- and the values it defines and
    reads.  Of several statements moved as one, the union, which crosses
    exactly what every one of them crosses."""
    touches: Optional[Tuple]
    defines: FrozenSet[int]
    uses: FrozenSet[int]

    @staticmethod
    def of(s: Stmt) -> 'Footprint':
        return Footprint(_touches_fixed(s), frozenset(_defines(s)),
                         frozenset(_uses(s)))

    def __or__(self, other: 'Footprint') -> 'Footprint':
        touches = (None if self.touches is None or other.touches is None
                   else self.touches + other.touches)
        return Footprint(touches, self.defines | other.defines,
                         self.uses | other.uses)

    @cached_property
    def kinds(self):
        """The touches by what an access can conflict with: reads (1) and
        writes (2) per `(space, base)`, per space, and of an unknown space.

        `accesses_conflict` asks the same of every pair, and one access
        meets few of a statement's: a write in another space or of another
        named buffer never conflicts.  So the pairs are looked up rather
        than walked -- the answer is the same.
        """
        by_base: Dict[tuple, int] = {}
        by_space: Dict[MemSpace, int] = {}
        unknown = 0
        for a in self.touches or ():
            bit = 2 if a.writes else 1
            if a.space is MemSpace.UNKNOWN:
                unknown |= bit
                continue
            key = (a.space, None if a.base is None else id(a.base))
            by_base[key] = by_base.get(key, 0) | bit
            by_space[a.space] = by_space.get(a.space, 0) | bit
        return by_base, by_space, unknown


def crosses(mover: Footprint, fixed: Footprint) -> bool:
    """`may_cross`, on footprints: for a caller that asks about the same
    statements many times."""
    if mover.touches is None or fixed.touches is None:
        return False
    if mover.defines & fixed.uses or fixed.defines & mover.uses:
        return False
    by_base, by_space, unknown = fixed.kinds
    for a in mover.touches:
        # What conflicts with it: anything, where it writes; a write where
        # it only reads.
        need = 3 if a.writes else 2
        if unknown & need:
            return False
        if a.space is MemSpace.UNKNOWN:
            if any(bits & need for bits in by_space.values()):
                return False
            continue
        if a.base is None:
            if by_space.get(a.space, 0) & need:
                return False
            continue
        if (by_base.get((a.space, id(a.base)), 0)
                | by_base.get((a.space, None), 0)) & need:
            return False
    return True


def may_cross(mover: Stmt, fixed: Stmt) -> bool:
    """May ``mover`` be emitted on the other side of ``fixed``?

    Directional, and the direction is the point.  `can_reorder` asks whether
    two statements may *swap*, so both have to be movable -- and a raw block
    is not, because its head is text whose semantics the IR cannot read.  But
    a transfer moving past a loop does not move the loop: the block stays
    exactly where it is, and whether it could have moved is not a question
    anyone asked.

    The distinction matters wherever the only thing between a transfer and
    its wait is a `rawblock` that nobody wants to move.  What still matters
    about the fixed statement is what it *touches*, and `touches` says that
    for a whole subtree or says nothing at all.
    """
    # Describable, not movable.  Since the mover may be a whole section --
    # a hop loop moved as a unit -- it is a `rawblock`, and a raw block is
    # `movable=False` because its head is text a pass cannot read.  That is an
    # answer to "may a pass relocate this on its own", which is not the
    # question here: the caller has already decided to move it, and asks only
    # what crossing costs.  What still has to hold is that both sides can be
    # described, which is what `touches` answers for a subtree.
    return crosses(Footprint.of(mover), Footprint.of(fixed))


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
        for inner in walk_stmts(r.body):
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
    the distance between them is the thing worth maximizing and the only thing
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

    The mirror of `sink_waits`, a copy at a time and without a distance: an
    issue rises past whatever it may swap with -- other transfers, pointer
    bindings it does not read -- where `move` stops a whole transfer.

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
