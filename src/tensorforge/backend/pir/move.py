# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: a transfer issued ahead of where it was written.

A builder writes each transfer in front of the statement that first reads
what it fills, so it has nothing to overlap with: the wait -- or, for a
register array, the first read -- comes right behind the issue.  This pass
moves the transfer up, towards the start of the statement list it stands in,
past whatever it does not depend on, and leaves the wait where it is::

    before:   [ l1 w1 c1  l2 w2 c2 ]
    after:    [ l1 l2 w1 c1  w2 c2 ]        (distance 1)

*How far* is `distance`, the number of transfers one may pass: at 1 it stops
below the transfer above it, so every transfer is issued one transfer ahead
of where it was written; at 2 it passes that one and stops below the next.
The transfer above counts where it was written, not where it is going: the
transfers move from the bottom up, and each stops below its predecessor's
place before the predecessor leaves it.

*What stops it* sooner:

* a statement it may not cross (`schedule.may_cross`): one that defines a
  value it reads, writes what it reads or touches its buffer, or says nothing
  about what it touches.  A barrier stops a transfer into shared memory,
  whose order the barrier is about, and not one into registers;
* a pointer binding.  What the bindings read is where the element's operands
  are, and for an operand behind a pointer array that is a load another
  transfer's address waits on: a transfer issued ahead of the binding delays
  that load and gains nothing over issuing behind it.

*What it takes along*: the declaration of the register array it fills, and
the arithmetic its addresses are computed from, where those stand above it --
a transfer that stopped for them would hardly move.

Each statement list on its own, the innermost first: a transfer does not
leave the loop, the guard or the scope it was written in.  Across the batch
loop's back edge is `wrap`'s.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from .core import BufferType, MemSpace, Op, Stmt
from .schedule import Footprint, crosses
from .transfers import Transfer, glue, neutral, transfers, writes


def move_loads(body: Tuple[Stmt, ...], distance: int = 1) -> Tuple[Stmt, ...]:
    """`body` with every transfer moved up, `distance` transfers ahead at
    most."""
    if distance < 1:
        raise ValueError(f'move distance must be >= 1, got {distance}')
    return _list(body, distance, {})


def _reads_global(s: Stmt, memo: Dict[int, bool]) -> bool:
    """Does anything in `s` read global memory -- could a transfer be in
    there?"""
    found = memo.get(id(s))
    if found is None:
        found = memo[id(s)] = any(
            a.space is MemSpace.GLOBAL and not writes(a)
            for a in s.accesses) or any(_reads_global(x, memo)
                                        for r in s.regions for x in r.body)
    return found


def _list(stmts: Sequence[Stmt], distance: int,
          memo: Dict[int, bool]) -> Tuple[Stmt, ...]:
    if not any(_reads_global(s, memo) for s in stmts):
        return tuple(stmts)
    found = transfers(stmts)
    pieces = {id(x) for t in found for x in t.stmts}
    out: List[Stmt] = []
    changed = False
    for s in stmts:
        # The regions of a transfer are its own hop loops, with nothing in
        # them to move.
        if s.regions and id(s) not in pieces:
            regions = tuple(replace(r, body=_list(r.body, distance, memo))
                            for r in s.regions)
            if any(n.body is not r.body for n, r in zip(regions, s.regions)):
                s = replace(s, regions=regions)
                changed = True
        out.append(s)
    index: Dict[int, int] = {id(x): i for i, t in enumerate(found)
                             for x in t.stmts}
    # What crossing a statement takes, asked of the same statements by every
    # transfer below them.
    prints: Dict[int, Footprint] = {}
    for i in reversed(range(len(found))):
        moved = _hoist(out, found[i], i, index, distance, prints)
        if moved is not None:
            out = moved
            changed = True
    return tuple(out) if changed else tuple(stmts)


def _binding(s: Stmt) -> bool:
    """A pointer bound by the body: a declaration of a buffer that is not
    one of its own allocations."""
    return s.op is not Op.ALLOC and any(isinstance(v.type, BufferType)
                                        for v in s.target)


def _hoist(out: List[Stmt], t: Transfer, i: int, index: Dict[int, int],
           distance: int, prints: Dict[int, Footprint]):
    """`out` with transfer `t` -- the `i`-th of its list -- as far up as it
    may go, or None where it stays."""
    def footprint(s: Stmt) -> Footprint:
        found = prints.get(id(s))
        if found is None:
            found = prints[id(s)] = Footprint.of(s)
        return found

    mine = {id(x) for x in t.stmts}
    at = [j for j, x in enumerate(out) if id(x) in mine]
    first, last = at[0], at[-1]
    # The arithmetic between its pieces goes with them.
    unit = out[first:last + 1]
    moving = footprint(unit[0])
    for x in unit[1:]:
        moving = moving | footprint(x)
    taken: List[Stmt] = []
    passed = set()
    register = t.dest.type.space is MemSpace.REGISTER
    land = first
    j = first - 1
    while j >= 0:
        # The statement above, and the comments leading it, which stay with
        # it.
        k = j
        while k >= 0 and neutral(out[k]):
            k -= 1
        if k < 0:
            break
        x = out[k]
        lead = k
        while lead > 0 and neutral(out[lead - 1]):
            lead -= 1
        other = index.get(id(x))
        if other is not None and other < i and other not in passed:
            if len(passed) + 1 >= distance:
                break
            passed.add(other)
        if _binding(x):
            break
        if (x.op is Op.ALLOC and x.target
                and x.target[0].id == t.dest.id) or (
                    glue(x) and bool(footprint(x).defines & moving.uses)):
            taken.insert(0, x)
            moving = moving | footprint(x)
            j = k - 1
            continue
        if not (register and x.op is Op.BARRIER and not x.regions) and not (
                crosses(moving, footprint(x))):
            break
        land = lead
        j = lead - 1
    gone = {id(x) for x in taken} | {id(x) for x in unit}
    moved = ([x for x in out[:land] if id(x) not in gone] + taken + unit
             + [x for x in out[land:] if id(x) not in gone])
    if all(a is b for a, b in zip(moved, out)):
        return None
    return moved
