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

*Which transfers count* is the pipeline (`transfers.PIPELINES`): only those
of its own.  By default every memory path is one -- global to shared memory
and global to registers -- so a copy into shared memory passes a load into
registers without counting it, and the other way round.  On NVIDIA the two
retire on counters of their own, and with one pipeline a load into registers
written between two copies would keep the second copy behind the wait for
the first: one copy in flight, the pipeline of copies drained at every wait.

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
leave the guard or the scope it was written in, nor a loop -- unless the loop
is a merged run's and the transfer is the same on every trip of it
(`_out_of_loop`): then it runs once, ahead of the loop, and goes up the list
there.  Across the batch loop's back edge is `wrap`'s.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from .core import (BufferType, Effect, MemSpace, Op, Stmt, accesses_conflict,
                   walk_stmts)
from .schedule import Footprint, crosses
from .transfers import Transfer, glue, memory_path, neutral, transfers, writes


def move_loads(body: Tuple[Stmt, ...], distance: int = 1,
               pipeline=memory_path) -> Tuple[Stmt, ...]:
    """`body` with every transfer moved up, `distance` transfers of its own
    pipeline ahead at most.  `pipeline` keys a transfer to its pipeline
    (`transfers.PIPELINES`)."""
    if distance < 1:
        raise ValueError(f'move distance must be >= 1, got {distance}')
    return _list(body, distance, {}, pipeline)


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


def _list(stmts: Sequence[Stmt], distance: int, memo: Dict[int, bool],
          pipeline=memory_path) -> Tuple[Stmt, ...]:
    if not any(_reads_global(s, memo) for s in stmts):
        return tuple(stmts)
    found = transfers(stmts)
    pieces = {id(x) for t in found for x in t.stmts}
    out: List[Stmt] = []
    changed = False
    left = False
    for s in stmts:
        # The regions of a transfer are its own hop loops, with nothing in
        # them to move.
        if s.regions and id(s) not in pieces:
            regions = tuple(replace(r, body=_list(r.body, distance, memo,
                                                   pipeline))
                            for r in s.regions)
            if any(n.body is not r.body for n, r in zip(regions, s.regions)):
                s = replace(s, regions=regions)
                changed = True
            leaving, s = _out_of_loop(s)
            if leaving:
                out.extend(leaving)
                changed = left = True
        out.append(s)
    if left:
        # What left a loop is a transfer of this list now, and goes up it
        # like the ones written here.
        found = transfers(out)
    index: Dict[int, int] = {id(x): i for i, t in enumerate(found)
                             for x in t.stmts}
    # What crossing a statement takes, asked of the same statements by every
    # transfer below them.
    prints: Dict[int, Footprint] = {}
    keys = [pipeline(t) for t in found]
    for i in reversed(range(len(found))):
        moved = _hoist(out, found[i], i, index, distance, prints, keys)
        if moved is not None:
            out = moved
            changed = True
    return tuple(out) if changed else tuple(stmts)


def _counted(s: Stmt) -> bool:
    """A merged run's loop (`VariantLoop`): rolled, over constant bounds, at
    least one trip.  What it does once a trip it does at least once, so a
    transfer that is the same on every trip can be done once, ahead of it.

    Only a run's: it is the loop whose body the macro layer writes for every
    trip alike, operands the trips share included, and the one this is
    measured on.
    """
    if s.op is not Op.FOR or not s.regions or not s.attr('rolled'):
        return False
    lo, hi, step = s.loop_bounds
    return (all(isinstance(x, int) for x in (lo, hi, step)) and step > 0
            and lo < hi)


def _out_of_loop(loop: Stmt) -> Tuple[List[Stmt], Stmt]:
    """The transfers of `loop` that fill the same buffer from the same data on
    every trip, taken out to run once ahead of it -- and `loop` without them.

    A merged run's body (`VariantLoop`) stages the operands every iteration
    shares as well as the one its counter selects, and the first are the same
    copy every trip.  Such a transfer leaves when

    * nothing it reads is computed in the loop: the induction, a value the
      loop carries, a binding the counter selects -- except arithmetic that
      reads nothing of the loop itself, which leaves with it;
    * nothing in the loop writes what it reads, and nothing there says
      nothing about what it touches;
    * it alone writes its buffer, and nothing ahead of it in the loop reads
      the buffer;
    * for a shared buffer, the loop holds no barrier: one orders what the
      other threads did to shared memory around the copy.

    The waits that retire its copies leave with it and stand right behind
    it, so that nothing the hardware carries out on its own is in flight
    across the loop; the transfer then goes up the list ahead of the loop as
    any other.
    """
    if not _counted(loop):
        return [], loop
    region = loop.regions[0]
    body = region.body
    found = transfers(body)
    if not found:
        return [], loop
    every = list(walk_stmts(body))
    if any(x.effect & Effect.UNKNOWN for x in every):
        return [], loop
    inside = {a.id for a in region.args}
    defs: Dict[int, Stmt] = {}
    for x in every:
        for v in x.target:
            inside.add(v.id)
            defs[v.id] = x
        for r in x.regions:
            inside.update(a.id for a in r.args)
    order = {id(x): i for i, x in enumerate(body)}
    barrier = any(x.effect & Effect.BARRIER for x in every)
    taken: Dict[int, Stmt] = {}

    def along(vid: int, need: Dict[int, Stmt]) -> bool:
        """Whether the value `vid` the transfer reads is available ahead of
        the loop -- defined outside it, or by arithmetic of the body that
        is, and then collected into `need`."""
        if vid not in inside:
            return True
        x = defs.get(vid)
        if x is not None and (id(x) in need or id(x) in taken):
            return True
        if x is None or id(x) not in order or not glue(x):
            return False
        need[id(x)] = x
        return all(along(v.id, need) for v in x.operands())

    for t in found:
        mine = {id(x) for x in t.stmts}
        own = {v.id for x in walk_stmts(tuple(t.stmts)) for v in x.target}
        own |= {a.id for x in walk_stmts(tuple(t.stmts))
                for r in x.regions for a in r.args}
        need: Dict[int, Stmt] = {}
        alloc = defs.get(t.dest.id)
        if (alloc is not None and alloc.op is Op.ALLOC
                and id(alloc) in order):
            # Its declaration, which a buffer filled ahead of the loop needs
            # ahead of the loop as well.
            need[id(alloc)] = alloc
        if not all(along(v.id, need)
                   for x in walk_stmts(tuple(t.stmts)) for v in x.operands()
                   if v.id not in own):
            continue
        if t.shared and barrier:
            continue
        tokens = {v.id for v in t.tokens()}
        waits = [x for x in body if x.op is Op.WAIT and x.waited is not None
                 and x.waited.id in tokens]
        # And the comments leading each, which say what it waits for.
        for i in sorted(order[id(x)] for x in waits):
            j = i - 1
            while j >= 0 and neutral(body[j]) and id(body[j]) not in mine:
                waits.append(body[j])
                j -= 1
        leaving = mine | set(need) | {id(x) for x in waits}
        reads = [a for x in walk_stmts(tuple(t.pieces)) for a in x.accesses
                 if a.space is MemSpace.GLOBAL and not writes(a)]
        start = min(order[i] for i in mine)
        legal = True
        for x in body:
            if id(x) in leaving or id(x) in taken:
                continue
            for y in walk_stmts((x,)):
                for a in y.accesses:
                    if a.base is not None and getattr(a.base, 'id', None) == t.dest.id:
                        if writes(a) or order[id(x)] < start:
                            legal = False
                    if writes(a) and any(accesses_conflict(r, a) for r in reads):
                        legal = False
            if not legal:
                break
        if not legal:
            continue
        taken.update((i, x) for i, x in need.items())
        taken.update((id(x), x) for x in t.stmts)
        taken.update((id(x), x) for x in waits)
    if not taken:
        return [], loop
    leaving = [x for x in body if id(x) in taken]
    rest = tuple(x for x in body if id(x) not in taken)
    return leaving, replace(loop, regions=(replace(region, body=rest),)
                            + tuple(loop.regions[1:]))


def _binding(s: Stmt) -> bool:
    """A pointer bound by the body: a declaration of a buffer that is not
    one of its own allocations."""
    return s.op is not Op.ALLOC and any(isinstance(v.type, BufferType)
                                        for v in s.target)


def _hoist(out: List[Stmt], t: Transfer, i: int, index: Dict[int, int],
           distance: int, prints: Dict[int, Footprint],
           keys: Sequence = ()):
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
        if (other is not None and other < i and other not in passed
                and (not keys or keys[other] == keys[i])):
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
