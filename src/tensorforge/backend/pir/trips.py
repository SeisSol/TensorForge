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

The same reads can go through shared memory instead (`trip_shards`): the run
of the operand a trip's lanes read, copied by the lanes of the
multiplication between them with the asynchronous copy, into a window of the
multiplication's own arena, and read there.  The copies are the transfer
then, and `wrap` issues them a trip ahead -- with a second stage at the head
of the trip before.  The barrier between the copies and the reads is the
multiplication's: no other multiplication reads the window, and none has to
arrive at it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .ahead import counted
from .core import (INDEX, Access, Effect, MemSpace, ScalarType, Stmt,
                   Uniformity, Value, walk_stmts)
from .shards import _Indices, _NotStatic
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
    out, changed = _visit(body, lambda loop: _gather(loop, scratch, report))
    return out if changed else body


def _visit(stmts: Sequence[Stmt], take) -> Tuple[Tuple[Stmt, ...], bool]:
    """`stmts` with `take(loop)` -- the declarations to put ahead of a
    counted loop and the loop rewritten, or None -- applied to every counted
    loop in them, the innermost first."""
    out: List[Stmt] = []
    changed = False
    pieces = None
    for s in stmts:
        if not s.regions:
            out.append(s)
            continue
        regions = []
        for r in s.regions:
            inner, inner_changed = _visit(r.body, take)
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
                taken = take(s)
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


def _candidates(loop: Stmt, what: str, report: Optional[List[str]]
                ) -> List[Tuple[Stmt, List[Stmt]]]:
    """The loads of `loop`'s own statements a trip reads in place and that
    could be read for another trip, each with the arithmetic of the loop its
    operands are computed by.  `what` names the reading in the line `report`
    gets where there is none."""
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
    if not found and why and report is not None:
        report.append(f'- {what}: {sorted(set(why))[0]}')
    return found


def _gather(loop: Stmt, scratch,
            report: Optional[List[str]]) -> Optional[Tuple[Tuple[Stmt, ...],
                                                         Stmt]]:
    """`(declaration, loop)` with the loads `loop`'s trips take read into a
    register array at the head of the trip, or None where it has none."""
    found = _candidates(loop, 'trip reads', report)
    if not found:
        return None
    if len(found) > TRIP_SLOTS:
        if report is not None:
            report.append(f'- trip reads: {len(found)} loads a trip, more '
                          f'than an array takes ({TRIP_SLOTS})')
        return None
    region = loop.regions[0]
    stmts = region.body
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


# --------------------------------------------------------------------------- #
# A shard a trip, in the multiplication's shared memory
# --------------------------------------------------------------------------- #

def trip_shards(body: Tuple[Stmt, ...], scratch, *, threads: int,
                report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with what each counted loop in it reads from global memory
    copied into shared memory of the multiplication's own, a shard a trip,
    and read there.

    `threads` is how many lanes a multiplication spans, `threadIdx.x` the
    lane: they copy a trip's shard between them, an element each, with the
    asynchronous copy.  The copies are a transfer into a window of the
    multiplication's arena, which `wrap` issues a trip ahead -- with a second
    stage into the stage the trip before does not read, at its head -- and
    the barrier between the copies and the reads is the multiplication's
    (`barriers.place_barriers`): no other multiplication reads the window,
    and none has to arrive.
    """
    indices = _Indices(body, {'x': threads})
    out, changed = _visit(
        body, lambda loop: _staged(loop, scratch, report, threads, indices))
    return out if changed else body


def _trip_span(x: Stmt, counter: Value, trips: range, threads: int,
               indices) -> Optional[Tuple[int, int, int]]:
    """`(first, step, count)`: the elements a trip's lanes read with `x` are
    `count` of them from `first + step * (trip - trips.start)` on, for every
    trip of `trips` -- or None where they are not a run that moves by a step.

    The lanes are the multiplication's, `x` the thread's index; a predicate
    whose value they and the counter decide holds lanes out, and one they do
    not decide leaves every lane in, which copies more than is read.
    """
    index = x.args[1]
    try:
        leaves = indices.leaves(index)
        cond = x.predicate
        if cond is not None:
            reach = indices.leaves(cond)
            if reach <= {'x', counter.id}:
                leaves = leaves | reach
            else:
                cond = None
    except _NotStatic:
        return None
    if not leaves <= {'x', counter.id} or 'x' not in leaves:
        return None
    spans = []
    for trip in trips:
        lo = hi = None
        for lane in range(threads):
            env = {'x': lane, counter.id: trip}
            memo: Dict[int, int] = {}
            if cond is not None and not indices.evaluate(cond, env, memo):
                continue
            i = indices.evaluate(index, env, memo)
            lo = i if lo is None else min(lo, i)
            hi = i if hi is None else max(hi, i)
        if lo is None:
            return None
        spans.append((lo, hi))
    first, last = spans[0]
    count = last - first + 1
    step = spans[1][0] - first if len(spans) > 1 else 0
    for k, (lo, hi) in enumerate(spans):
        if lo != first + k * step or hi - lo + 1 != count:
            return None
    return first, step, count


def _staged(loop: Stmt, scratch, report: Optional[List[str]], threads: int,
            indices) -> Optional[Tuple[Tuple[Stmt, ...], Stmt]]:
    """`(declaration, loop)` with the loads `loop`'s trips take copied into
    shared memory a shard a trip and read there, or None where it has none.
    The loads of one operand whose runs overlap or meet, moving by one step,
    read one shard."""
    found = _candidates(loop, 'trip shards', report)
    if not found:
        return None
    region = loop.regions[0]
    stmts = region.body
    counter = region.args[0]
    lo, hi, step = loop.loop_bounds
    trips = range(lo, hi, step)
    runs: List[list] = []
    for x, deps in found:
        span = _trip_span(x, counter, trips, threads, indices)
        if span is None:
            continue
        first, move, count = span
        for run in runs:
            if (run[0] is x.args[0] and run[2] == move
                    and first <= run[1] + run[3] and run[1] <= first + count):
                end = max(run[1] + run[3], first + count)
                run[1] = min(run[1], first)
                run[3] = end - run[1]
                run[4].append((x, deps))
                break
        else:
            runs.append([x.args[0], first, move, count, [(x, deps)]])
    if not runs:
        if report is not None:
            report.append('- trip shards: no load whose lanes read a run '
                          'moving by a step')
        return None
    declare: List[Stmt] = []
    taken = {id(x) for run in runs for x, _ in run[4]}
    moved = {id(d) for run in runs for _, deps in run[4] for d in deps}
    moved |= taken
    head: List[Stmt] = [d for d in stmts if id(d) in moved
                        and id(d) not in taken]
    for operand, first, move, count, loads in runs:
        hops = -(-count // threads)
        b = scratch()
        # Room for every lane of every hop: the lanes past the run copy its
        # last element again, behind it, which no read reaches -- a copy
        # without a predicate, which is what `wrap` moves.
        window = b.alloc(operand.type.elem, (hops * threads,),
                         MemSpace.SHARED, hint='tripShard')
        declare.extend(b.finish())
        copy = scratch()
        # Where the trip's run starts in the operand: `first`, moved by
        # `move` a trip.
        start = counter
        if lo:
            start = copy.op('sub', INDEX, start, lo, hint='trip')
        if move != 1:
            start = copy.op('mul', INDEX, start, move, hint='trip')
        if first:
            start = copy.op('add', INDEX, start, first, hint='shardStart')
        copy.mark('defines', window)
        lane = copy.thread_id('x')
        tokens = []
        for hop in range(hops):
            at = (lane if hop == 0 else
                  copy.op('add', INDEX, lane, hop * threads, hint='hop'))
            within = (copy.op('min', INDEX, at, count - 1, hint='hop')
                      if (hop + 1) * threads > count else at)
            source = copy.op('add', INDEX, start, within, hint='hop')
            tokens.append(copy.copy_async(window, operand, dst_index=(at,),
                                          src_index=(source,), elems=1,
                                          hint='shard'))
        copy.wait(*tokens)
        # Read through the operand as its loads were: against the buffer
        # they named, which this body knows the binding to stand for.
        head.extend(_rebase(copy.finish(), operand,
                            loads[0][0].accesses[0].base))
        # The shard is read right behind its wait, every load of it, for the
        # reason the slots of `trip_reads` are: the last read is where the
        # copy for the next trip goes with one stage.
        for x, _ in loads:
            read = scratch()
            local = read.op('sub', INDEX, x.args[1], start, hint='shard')
            head.extend(read.finish())
            head.append(replace(
                x, args=(window, local),
                accesses=(Access(Effect.READ, MemSpace.SHARED, window),),
                attrs=tuple((k, v) for k, v in x.attrs
                            if k != 'nontemporal')))
        if report is not None:
            report.append(f'+ trip shards: {len(loads)} into {window}, '
                          f'{count} a trip')
    body = tuple(head) + tuple(x for x in stmts if id(x) not in moved)
    return tuple(declare), replace(
        loop, regions=(replace(region, body=body),) + tuple(loop.regions[1:]))


def _rebase(stmts: Sequence[Stmt], value: Value, base) -> List[Stmt]:
    """`stmts` with their accesses of `value` recorded against `base`."""
    out = []
    for x in stmts:
        if any(a.base is value for a in x.accesses):
            x = replace(x, accesses=tuple(
                replace(a, base=base) if a.base is value else a
                for a in x.accesses))
        out.append(x)
    return out
