# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: what a trip of a counted loop reads from global memory, read
into registers as a transfer of its own.

A rolled reduction (`k_roll`) reads its operator where it lies: a load from
global memory a trip, and the products of the trip right behind it, which
take its value at once.  At order 6 that is one element of the operator and
nine products with it -- a few dozen cycles of arithmetic behind a load of
several hundred.  `wrap` issues a transfer a trip ahead in a loop whose trips
are counted (`wrap._runs`), and here there is nothing for it to issue: an
operand read in place is no transfer, its load produces the value the
products take, and no buffer stands between the two that could be filled
ahead of time.

So this makes one (`trip_reads`): the loads go to the head of the trip, into
a register array declared ahead of the loop, a slot each, and where they
stood the products read the slot.  That array is a transfer's buffer like
any other, and `wrap` fills it for the trip after, behind its last read in
this one, and for the first trip ahead of the loop::

    before:   for k { a = A[f(k)]  ... a ... }
    after:    t[0] = A[f(lo)]  for k { a = t[0]  ... a ...  t[0] = A[f(k + 1)] }

Which loops: those that stay loops, a reduction `k_roll` rolls or a merged
run; one unrolled whole is not one to the compiler, which schedules the loads
of all its trips together already.

Which loads: the loop's own statements -- not those of a loop or a branch
inside it -- that read global memory and nothing else, from a buffer bound
outside the loop and written nowhere in it, whose lanes read different
elements, and whose index and predicate are computed from the counter, from
values bound outside the loop, and by arithmetic of the loop's own: what
`wrap` computes again for another trip.  A load the lanes all read alike is
left in place: on AMD it comes out of the scalar cache into a scalar
register, and a slot of a register array is one per lane.  A load that is a
transfer's already, or half of one, belongs to the transfer.  And no more of
them than `TRIP_SLOTS`: a trip that reads more is left as it is.

Only scalars: a slot is one element.  And only where `wrap` runs behind this,
which is the reason for the array -- without it the loads would stand at the
head of the trip and wait there, as they waited where they were.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .ahead import counted
from .core import (INDEX, MemSpace, ScalarType, Stmt, Uniformity, Value,
                   walk_stmts)
from .transfers import fetch, glue, opaque, transfers, writes

#: The most loads a trip's register array takes.  A trip that reads more
#: keeps the memory busy on its own -- its loads are independent and issue
#: back to back, each behind the others' latency -- and an array as wide
#: holds as many registers through the whole trip.  A merged run reads all
#: its member reads in place a trip: SeisSol's `projectToDR` at order 8,
#: merged on gfx942, 360 doubles, more than the registers of a lane.  A
#: rolled reduction reads one to four elements a trip.
TRIP_SLOTS = 16


def trip_reads(body: Tuple[Stmt, ...], scratch,
               report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with what each counted loop in it reads from global memory read
    into a register array at the head of the trip, wherever the loop stands.

    `scratch` makes the builder for the array, the slot reads and the slot
    writes (`IRBuilder.scratch`).  `report` collects a line per loop: `+`,
    the loads and the array; `-` and why a loop that reads global memory has
    none of its loads taken.
    """
    out, changed = _visit(body, scratch, report)
    return out if changed else body


def _visit(stmts: Sequence[Stmt], scratch,
           report: Optional[List[str]]) -> Tuple[Tuple[Stmt, ...], bool]:
    out: List[Stmt] = []
    changed = False
    pieces = None
    for s in stmts:
        if not s.regions:
            out.append(s)
            continue
        regions = []
        for r in s.regions:
            inner, inner_changed = _visit(r.body, scratch, report)
            regions.append(replace(r, body=inner) if inner_changed else r)
            changed = changed or inner_changed
        if any(n is not r for n, r in zip(regions, s.regions)):
            s = replace(s, regions=tuple(regions))
        if counted(s) and s.attr('unroll') is not True:
            # A loop unrolled whole is no loop to the compiler, which
            # schedules every trip's loads together already.
            if pieces is None:
                pieces = {id(x) for t in transfers(stmts) for x in t.stmts}
            if id(s) not in pieces:
                # Not the hops of a loader, which are a transfer themselves.
                taken = _gather(s, scratch, report)
                if taken is not None:
                    declare, s = taken
                    out.extend(declare)
                    changed = True
        out.append(s)
    return tuple(out), changed


def _key(base) -> tuple:
    """An access's buffer as something to compare: a value by its id, an
    operand by its name."""
    if isinstance(base, Value):
        return ('value', base.id)
    return ('operand', base if isinstance(base, (str, int)) else id(base))


def _written(stmts: Sequence[Stmt]) -> Tuple[set, bool]:
    """The global buffers something in `stmts` writes, and whether something
    there writes memory it does not name."""
    bases, unknown = set(), False
    for x in walk_stmts(tuple(stmts)):
        for a in x.accesses:
            if not writes(a):
                continue
            if opaque(a):
                unknown = True
            elif a.space is MemSpace.GLOBAL:
                bases.add(_key(a.base))
    return bases, unknown


def _slice(x: Stmt, defs: Dict[int, Stmt], inside: set,
           counter: Value) -> Optional[List[Stmt]]:
    """The arithmetic of the loop that `x`'s operands are computed by, or
    None where one of them is computed otherwise in the loop -- by a load, a
    call, a statement with an effect."""
    found: Dict[int, Stmt] = {}
    todo = [v for v in x.operands()[1:]]
    other = x.attr('other')
    if isinstance(other, Value):
        todo.append(other)
    while todo:
        v = todo.pop()
        if v.id == counter.id or v.id not in inside:
            continue
        d = defs.get(v.id)
        if d is None or not glue(d):
            return None
        if id(d) in found:
            continue
        found[id(d)] = d
        todo.extend(d.operands())
        other = d.attr('other')
        if isinstance(other, Value):
            todo.append(other)
    return list(found.values())


def _gather(loop: Stmt, scratch,
            report: Optional[List[str]]) -> Optional[Tuple[Tuple[Stmt, ...],
                                                         Stmt]]:
    """`(declaration, loop)` with the loads `loop`'s trips take read into a
    register array at the head of the trip, or None where it has none."""
    region = loop.regions[0]
    stmts = region.body
    counter = region.args[0]
    inside = {t.id for x in walk_stmts(stmts) for t in x.target}
    inside |= {a.id for r in loop.regions for a in r.args}
    defs = {t.id: x for x in stmts for t in x.target}
    held = {id(x) for t in transfers(stmts) for x in t.stmts}
    written, unknown = _written(stmts)
    found: List[Tuple[Stmt, List[Stmt]]] = []
    why = []
    for x in stmts:
        if not fetch(x) or id(x) in held:
            continue
        value = x.target[0]
        base = x.args[0]
        if not isinstance(value.type, ScalarType) or value.type.is_vector:
            why.append('a vector')
            continue
        if value.uniformity is not Uniformity.LANE:
            why.append('the lanes read one element')
            continue
        if not isinstance(base, Value) or base.id in inside:
            why.append('a buffer bound in the loop')
            continue
        if unknown or any(_key(a.base) in written for a in x.accesses):
            why.append('a buffer written in the loop')
            continue
        deps = _slice(x, defs, inside, counter)
        if deps is None:
            why.append('an index the loop computes by more than arithmetic')
            continue
        found.append((x, deps))
    if not found:
        if why and report is not None:
            report.append(f'- trip reads: {sorted(set(why))[0]}')
        return None
    if len(found) > TRIP_SLOTS:
        if report is not None:
            report.append(f'- trip reads: {len(found)} loads a trip, more '
                          f'than an array takes ({TRIP_SLOTS})')
        return None
    # One array per element type, since a slot is one element.
    by_type: Dict[object, List[Tuple[Stmt, List[Stmt]]]] = {}
    for x, deps in found:
        by_type.setdefault(x.target[0].type.base, []).append((x, deps))
    declare: List[Stmt] = []
    moved = {id(d) for _, deps in found for d in deps}
    moved |= {id(x) for x, _ in found}
    head: List[Stmt] = [d for d in stmts if id(d) in moved
                        and not any(d is x for x, _ in found)]
    for elem, group in by_type.items():
        b = scratch()
        buf = b.alloc(elem, (len(group),), MemSpace.REGISTER, hint='trip',
                      init='{}')
        declare.extend(b.finish())
        fill = scratch()
        fill.mark('defines', buf)
        for slot, (x, _) in enumerate(group):
            value = x.target[0]
            fetched = fill.value(value.type, hint='tripRead',
                                 uniform=value.uniformity,
                                 layout=value.layout)
            fill.emit(replace(x, target=(fetched,)))
            fill.store(buf, fetched, fill.const(slot, INDEX))
        head.extend(fill.finish())
        # The slots are read right behind the fill, each into the value its
        # load produced, so that what read the load reads the slot.  All of
        # them there and not each where its load stood: the last read of the
        # array is where `wrap` fills it for the next trip, and there the
        # fill has the whole trip ahead of it.
        for slot, (x, _) in enumerate(group):
            read = scratch()
            read.load(buf, read.const(slot, INDEX), type_=x.target[0].type,
                      hint='trip')
            built = read.finish()
            head.extend(built[:-1])
            head.append(replace(built[-1], target=(x.target[0],)))
        if report is not None:
            report.append(f'+ trip reads: {len(group)} into {buf}')
    rest = [x for x in stmts if id(x) not in moved]
    body = tuple(head) + tuple(rest)
    return tuple(declare), replace(
        loop, regions=(replace(region, body=body),) + tuple(loop.regions[1:]))
