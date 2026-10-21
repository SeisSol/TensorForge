# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: what the trips of a merged run hand each other through memory,
handed over in registers instead.

A merged run whose trips are closed (`Options.closed_trips`) touches an
accumulator twice a trip: its head reads it into a register image, and its
tail stores the image the trip computed from that.  Memory is all that
crosses the back edge, which is what makes the body the same on every trip
-- and it is a load and a store of the whole accumulator per trip.  Where the
next trip only reads back what this one stored, the load goes ahead of the
loop, the store behind it, and the tail copies its image into the head's::

    for k:                        I = load H             (once)
      I = load H                  for k:
      ...                           ...
      J = f(I, ...)                 J = f(I, ...)
      store H <- J                  I <- J
                                  store H <- I           (once)

An atomic store reads nothing: it adds what the trip computed onto memory.
The trips add into an image of zeros instead, and one atomic store behind
the loop adds the sum::

    for k:                        A = 0                  (once)
      ...                         for k:
      J = f(...)                    ...
      atomic H += J                 J = f(...)
                                    A <- A + J
                                  atomic H += A          (once)

which is the order the list written out adds in as well: its terms into
registers, and one atomic update.

A loop that does not qualify keeps its load and its store, which costs a
round trip through memory per trip and never a wrong value.  What has to
hold, besides a loop that runs at least once and is a merged run's
(`ahead.counted`, `rolled`):

* nothing in the loop but the load and the store touches `H`: whatever names
  it is one of the two, nothing hands `H`'s binding on -- to a table of
  pointers, say -- and nothing in the loop says nothing about what it
  touches;
* every lane reads back what it stored, into the slot it stored it from.
  Evaluated for every lane and every hop of the inner loops (`_Indices`):
  each element the store writes from the tail's image is one the load reads,
  by the same lane, into the same slot of the head's image; no other lane
  stores what a lane loads; and what the store writes besides -- the zeros
  past the rows a product defines -- is nothing the load reads;
* the head's image is written by the load alone, and read neither ahead of
  the load nor behind the store;
* the load needs nothing the loop computes, and the store nothing but the
  tail's image, so that one can stand ahead of the loop and the other
  behind it.

For an atomic store: nothing else in the loop touches `H`, and each slot of
the tail's image is added once a trip.

Where every slot of the head's image is read before the trip first writes
that slot of the tail's, the tail is computed in the head's image itself,
and the copy goes (`_fused`): an accumulation slot by slot, as the list
written out has it.

And where the code ahead of the loop stored `H` from registers just before
-- a closed trip stores what the body touches ahead of the loop, and an
accumulator the list began in front of the run is one -- the load that now
stands there reads that image instead of memory, and the store goes as well
where the one behind the loop writes all of it again (`_forward`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .ahead import Clone, counted
from .core import (Access, Effect, MemSpace, Op, Stmt, Value,
                   collect_accesses, collect_effect, defined_within, may_alias,
                   walk_stmts)
from .shards import _Indices, _NotStatic
from .transfers import neutral, tag


def carry_trips(body: Tuple[Stmt, ...], scratch, *, threads: int,
                report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with what the trips of its merged runs hand each other through
    memory kept in registers, where that is shown to be the same (see the
    module).  `threads` is how many lanes a multiplication spans,
    `threadIdx.x` the lane; `scratch` makes the builder new statements are
    built with; `report` collects what was carried and why the rest was
    not."""
    taken = {str(s.attr('extern')) for s in walk_stmts(body)
             if s.attr('extern') is not None}
    constants = {t.id: s.attr('value') for s in walk_stmts(body)
                 if s.op is Op.CONST for t in s.target}
    carrier = _Carrier(scratch, _Indices(body, {'x': threads}), threads,
                       report, taken, constants, _addresses(body))
    out, changed = carrier.stmts(body)
    return out if changed else body


class _Refused(Exception):
    """Why a buffer stays where it is."""


#: The value a predicated load leaves where its predicate does not hold.
_ZERO = object()


@dataclass
class _Event:
    """A load or a store, and what it stands under: the conditions of the
    branches around it, each with the side it is on, and the inner loops'
    trips by induction."""
    stmt: Stmt
    conds: Tuple[Tuple[object, bool], ...]
    loops: Dict[int, range]


@dataclass
class _Unit:
    """Statements of a list that touch the carried buffer, and the register
    image they move it through."""
    #: `load` fills the image from the buffer, `store` stores the image into
    #: it, `atomic` adds the image onto it, and `fill` stores numbers into
    #: it and no image (`_filled`).
    kind: str
    #: Positions in the statement list.
    at: List[int]
    #: The image's buffer, as its accesses name it -- and as its loads and
    #: stores spell it.  None for a `fill`.
    image: object
    operand: object
    events: List[_Event]


@dataclass
class _Carried:
    """A buffer the trips now hand each other in registers, with what the
    load ahead of the loop reads (`held`: per lane, slot to element) and what
    the store behind it writes (`stored`: per lane, the elements), and the
    statements of that load by id."""
    access: Access
    head: _Unit
    held: Dict[int, Dict[int, object]]
    stored: Dict[int, Set[int]]
    load: Set[int]


class _Carrier:
    """One body's walk: its merged runs' loops, innermost first, each with
    the buffers its trips store from registers tried one at a time."""

    def __init__(self, scratch, indices: _Indices, threads: int,
                 report: Optional[List[str]], taken: Set[str],
                 constants: Dict[int, object], addresses: Set[int]):
        self._scratch = scratch
        self._indices = indices
        self._threads = threads
        self._report = report
        self._taken = taken
        #: What the body's `const` statements define, by value.
        self._constants = constants
        #: The values the body uses other than as the buffer of an access:
        #: a pointer put into a table, say (`_addresses`).
        self._addresses = addresses
        self._b = None

    def _constant(self, x) -> bool:
        """Is `x` a number, written out or defined by a `const`?"""
        if isinstance(x, Value):
            x = self._constants.get(x.id)
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _say(self, line: str) -> None:
        if self._report is not None:
            self._report.append(line)

    def _fresh(self, v: Value) -> Value:
        if self._b is None:
            self._b = self._scratch()
        return self._b.value(v.type, hint=v.hint, uniform=v.uniformity,
                             layout=v.layout, quals=v.quals)

    # -- walking ------------------------------------------------------------- #

    def stmts(self, stmts: Sequence[Stmt]) -> Tuple[Tuple[Stmt, ...], bool]:
        out: List[Stmt] = []
        changed = False
        for s in stmts:
            if s.regions:
                regions, inner = [], False
                for r in s.regions:
                    body, c = self.stmts(r.body)
                    regions.append(replace(r, body=body) if c else r)
                    inner |= c
                if inner:
                    s = replace(s, regions=tuple(regions))
                    changed = True
            if counted(s) and s.attr('rolled'):
                ahead, s, behind, carried = self._loop(s)
                if ahead or behind:
                    changed = True
                for c in carried:
                    ahead = self._forward(out, ahead, c)
                out.extend(ahead)
                out.append(s)
                out.extend(behind)
            else:
                out.append(s)
        return tuple(out), changed

    def _loop(self, loop: Stmt):
        ahead: List[Stmt] = []
        behind: List[Stmt] = []
        carried: List[_Carried] = []
        tried: Set[int] = set()
        while True:
            body = loop.regions[0].body
            found = None
            for access in _stored(body):
                if id(access.base) in tried:
                    continue
                tried.add(id(access.base))
                if not _stored_from_registers(body, access):
                    continue
                name = getattr(access.base, 'name', str(access.base))
                try:
                    found = self._carry(loop, access)
                except _Refused as why:
                    self._say(f'carry: {name} stays in memory: {why}')
                    continue
                except _NotStatic as why:
                    self._say(f'carry: {name} stays in memory: an index or a '
                              f'condition not known per lane ({why})')
                    continue
                self._say(f'carry: {name} in registers over '
                          f'{len(range(*loop.loop_bounds))} trips')
                break
            if found is None:
                return ahead, loop, behind, carried
            before, body, after, record = found
            ahead.extend(before)
            behind[:0] = after
            if record is not None:
                carried.append(record)
            loop = replace(loop, regions=(replace(loop.regions[0],
                                                  body=tuple(body)),))

    # -- one buffer ---------------------------------------------------------- #

    def _carry(self, loop: Stmt, access: Access):
        stmts = list(loop.regions[0].body)
        probe = Access(Effect.READ, access.space, access.base)
        touching = [i for i, s in enumerate(stmts)
                    if any(may_alias(a, probe)
                           for a in collect_accesses((s,)))]
        units = self._units(stmts, touching, access)
        kinds = {u.kind for u in units}
        if kinds == {'atomic'}:
            return self._atomic(loop, stmts, units, access)
        loads = [u for u in units if u.kind == 'load']
        stores = [u for u in units if u.kind == 'store']
        if not loads or not stores or kinds - {'load', 'store'}:
            raise _Refused('not a load at the head of the trip and a store '
                           'at its tail')
        if max(i for u in loads for i in u.at) > min(i for u in stores
                                                     for i in u.at):
            raise _Refused('a store ahead of a load')
        if len({_key(u.image) for u in loads}) != 1 \
                or len({_key(u.image) for u in stores}) != 1:
            raise _Refused('more than one register image')
        return self._loaded(loop, stmts, loads, stores, access)

    def _units(self, stmts: List[Stmt], touching: List[int],
               access: Access) -> List[_Unit]:
        """The statements at `touching`, grouped into the transfers they are
        part of (`transfers.tag`), as units of the loop that move `access`'s
        buffer through one register image."""
        groups: List[List[int]] = []
        done: Set[int] = set()
        for i in touching:
            if i in done:
                continue
            t = tag(stmts[i])
            group = ([j for j, s in enumerate(stmts) if tag(s) == t
                      and not neutral(s)] if t is not None else [i])
            if i not in group:
                group.append(i)
            group.sort()
            done.update(group)
            groups.append(group)
        if any(j not in touching and _touches_memory(stmts[j])
               for group in groups for j in group):
            raise _Refused('a transfer of it moves something else as well')
        return [self._unit(stmts, group, access) for group in groups]

    def _unit(self, stmts: List[Stmt], at: List[int],
              access: Access) -> _Unit:
        inner = tuple(stmts[i] for i in at)
        if collect_effect(inner) & (Effect.BARRIER | Effect.ASYNC
                                    | Effect.UNKNOWN):
            raise _Refused('a statement that synchronizes or says nothing '
                           'about what it touches')
        reads_h = writes_h = atomic_h = False
        images: Dict[object, Tuple[object, bool, bool]] = {}
        for x in walk_stmts(inner):
            for a in x.accesses:
                if a.base is access.base and a.space is access.space:
                    if a.kind & Effect.ATOMIC:
                        atomic_h = True
                    elif a.kind & Effect.WRITE:
                        writes_h = True
                    else:
                        reads_h = True
                    if x.op not in (Op.LOAD, Op.STORE) and not (
                            x.op is Op.RAWSTMT and a.kind & Effect.ATOMIC):
                        raise _Refused(f'a {x.op} touches it')
                elif a.space is MemSpace.REGISTER and a.base is not None \
                        and x.op in (Op.LOAD, Op.STORE):
                    key = _key(a.base)
                    was = images.get(key, (None, False, False))
                    images[key] = (a.base if was[0] is None else was[0],
                                   was[1] or bool(a.kind & Effect.READ),
                                   was[2] or a.writes)
                else:
                    raise _Refused(f'it is moved together with {a!r}')
        if len(images) != 1:
            raise _Refused('not one register image to move it through')
        (image, reads_i, writes_i), = images.values()
        operand = next(x.args[0] for x in walk_stmts(inner)
                       if x.op in (Op.LOAD, Op.STORE)
                       and any(_key(a.base) == _key(image)
                               for a in x.accesses))
        events = _events(inner)
        if reads_h and writes_i and not (writes_h or atomic_h or reads_i):
            return _Unit('load', at, image, operand, events)
        if writes_h and reads_i and not (reads_h or atomic_h or writes_i):
            return _Unit('store', at, image, operand, events)
        if atomic_h and reads_i and not (reads_h or writes_h or writes_i):
            return _Unit('atomic', at, image, operand, events)
        raise _Refused('a statement that both reads and writes it')

    # -- a load at the head, a store at the tail ---------------------------- #

    def _loaded(self, loop: Stmt, stmts: List[Stmt], loads: List[_Unit],
                stores: List[_Unit], access: Access):
        head, tail = loads[0], stores[0]
        if _key(head.image) == _key(tail.image):
            raise _Refused('the trip changes the image it loaded into')
        mine = {i for u in loads + stores for i in u.at}
        first = min(i for u in loads for i in u.at)
        last = max(i for u in stores for i in u.at)
        self._alone(loop, stmts, mine, access, loads + stores)

        # The head's image: written by the load alone, read only between the
        # load and the store.
        alloc_at = None
        for i, s in enumerate(stmts):
            if i in mine:
                continue
            if s.op is Op.ALLOC and any(_key(t) == _key(head.image)
                                        for t in s.target):
                alloc_at = i
                continue
            for a in collect_accesses((s,)):
                if _key(a.base) != _key(head.image):
                    continue
                if a.writes:
                    raise _Refused('the trip writes the image the load fills')
                if i < first or i > last:
                    raise _Refused('the image the load fills is read outside '
                                   'the trip it is loaded for')
        if alloc_at is not None and alloc_at > first:
            raise _Refused('the image is declared behind its load')
        if alloc_at is None and any(
                x.op is Op.ALLOC and any(_key(t) == _key(head.image)
                                         for t in x.target)
                for x in walk_stmts(tuple(stmts))):
            raise _Refused('the image is declared in a branch of the trip')
        inside = set(defined_within(tuple(stmts)))
        inside |= {a.id for a in loop.regions[0].args}
        _needs_only(tuple(stmts[i] for u in loads for i in u.at), inside,
                    'the load needs what the trip computes',
                    allowed={_key(head.operand), _key(head.image)})
        _needs_only(tuple(stmts[i] for u in stores for i in u.at), inside,
                    'the store needs what the trip computes',
                    allowed={_key(tail.operand), _key(tail.image)})

        held = self._head_map(loads)
        stored = self._check_tail(stores, held)

        ahead = ([stmts[alloc_at]] if alloc_at is not None else [])
        led = {tag(stmts[i]) for u in loads for i in u.at} - {None}
        ahead += [s for i, s in enumerate(stmts)
                  if i in {j for u in loads for j in u.at}
                  or (neutral(s) and tag(s) in led)]
        dropped = {id(s) for s in ahead}
        body: List[Stmt] = []
        copies: Set[int] = set()
        for i, s in enumerate(stmts):
            if id(s) in dropped:
                continue
            unit = next((u for u in stores if i in u.at), None)
            if unit is not None:
                copied = self._copy_back(s, access, unit, head)
                copies.update(id(x) for x in copied)
                body.extend(copied)
                continue
            body.append(s)
        body = self._fused(body, copies, head, tail)
        behind = self._rebased(
            Clone(self._fresh, 'carry', self._taken).stmts(
                [stmts[i] for u in stores for i in u.at]),
            tail, head.operand, head.image)
        loaded = {id(stmts[i]) for u in loads for i in u.at}
        return ahead, body, behind, _Carried(access, head, held, stored,
                                             loaded)

    def _fused(self, body: List[Stmt], copies: Set[int], head: _Unit,
               tail: _Unit) -> List[Stmt]:
        """`body` computing the tail's image in the head's, where that is the
        same: then the copy at the end of the trip copies every slot onto
        itself, and goes.

        It is the same where every slot of the head's image is read only
        ahead of the trip's first write of that slot of the tail's, and every
        slot of the tail's is written before it is read -- under one
        condition, so that a lane that skips the write skips the reads as
        well.  An accumulation `J = f(I)` slot by slot is that, and in one
        image it is two images fewer of registers live at once than a copy
        between two, as far as anything counting slots can tell.  Else
        `body` as it is.
        """
        events: List[Tuple[str, int, tuple]] = []
        try:
            self._sequence([s for s in body if id(s) not in copies],
                           head, tail, {}, (), events)
        except (_Refused, _NotStatic, KeyError):
            return body
        first: Dict[int, int] = {}
        for at, (what, slot, _) in enumerate(events):
            if what == 'write tail':
                first.setdefault(slot, at)
        for at, (what, slot, _) in enumerate(events):
            if what == 'write head':
                return body
            if what == 'read head' and slot in first and at > first[slot]:
                return body
            if what == 'read tail' and (slot not in first
                                         or at < first[slot]):
                return body
        if len({conds for what, _, conds in events
                if what.endswith('tail')}) > 1:
            return body
        out = []
        for s in body:
            if id(s) in copies:
                continue
            if s.op is Op.ALLOC and any(_key(t) == _key(tail.image)
                                        for t in s.target):
                continue
            out.extend(self._rebased([s], tail, head.operand, head.image,
                                     stores=True))
        if any(_key(a.base) == _key(tail.image)
               for a in collect_accesses(tuple(out))):
            return body
        return out

    def _sequence(self, stmts: Sequence[Stmt], head: _Unit, tail: _Unit,
                  env: Dict, conds: tuple, events: List) -> None:
        """The accesses of `stmts` to the two images, in the order a lane
        makes them, every inner loop run through: `(what, slot, conds)`."""
        images = {_key(head.image): 'head', _key(tail.image): 'tail'}
        for s in stmts:
            if not any(_key(a.base) in images
                       for a in collect_accesses((s,))):
                continue
            if s.op is Op.FOR:
                if not counted(s) or s.loop_inits:
                    raise _Refused('a loop of an unknown count')
                for trip in range(*s.loop_bounds):
                    self._sequence(s.regions[0].body, head, tail,
                                   {**env, s.induction.id: trip}, conds,
                                   events)
                continue
            if s.op is Op.IF:
                self._sequence(s.regions[0].body, head, tail, env,
                               conds + ((s.cond, True),), events)
                if len(s.regions) > 1:
                    self._sequence(s.regions[1].body, head, tail, env,
                                   conds + ((s.cond, False),), events)
                continue
            if s.regions or s.op not in (Op.LOAD, Op.STORE):
                raise _Refused(f'a {s.op} touches the image')
            index = s.args[1] if s.op is Op.LOAD else s.args[2]
            if len(s.args) != (2 if s.op is Op.LOAD else 3):
                raise _Refused('an image of more than one index')
            leaves = self._indices.leaves(index)
            if leaves - set(env):
                raise _Refused('a slot not known for every hop')
            slot = self._indices.evaluate(index, env, {})
            for a in s.accesses:
                which = images.get(_key(a.base))
                if which is not None:
                    events.append((('write ' if a.writes else 'read ')
                                   + which, slot, conds))

    def _head_map(self, loads: List[_Unit]):
        """Per lane, the slot of the head's image each element goes into:
        `{lane: {slot: element or _ZERO}}`."""
        held: Dict[int, Dict[int, object]] = {}
        for unit in loads:
            producers = {t.id: e for e in unit.events
                         if e.stmt.op is Op.LOAD for t in e.stmt.target}
            for e in unit.events:
                s = e.stmt
                if s.op is not Op.STORE:
                    continue
                value, index = _store_parts(s)
                if not isinstance(value, Value) or value.id not in producers:
                    raise _Refused('the image is filled with something not '
                                   'loaded from the buffer')
                src = producers[value.id]
                loops = {**src.loops, **e.loops}
                src_index = _load_index(src.stmt)
                for lane, env, memo in self._domain(
                        (index, src_index, src.stmt.predicate,
                         s.predicate, *(c for c, _ in e.conds + src.conds)),
                        loops):
                    if not self._active(e, env, memo):
                        continue
                    slot = self._indices.evaluate(index, env, memo)
                    element = (self._indices.evaluate(src_index, env, memo)
                               if self._active(src, env, memo) else _ZERO)
                    lane_held = held.setdefault(lane, {})
                    if slot in lane_held:
                        raise _Refused('the load writes a slot twice')
                    lane_held[slot] = element
        return held

    def _check_tail(self, stores: List[_Unit], held) -> None:
        """Every element the store writes from the tail's image is one the
        load reads into that slot, by the same lane; nothing else it writes
        is read; no lane stores what another loads.  What it writes from the
        image, per lane, is the answer."""
        owner: Dict[int, int] = {}
        constants: Set[int] = set()
        written: Dict[int, Set[int]] = {}
        for unit in stores:
            producers = {t.id: e for e in unit.events
                         if e.stmt.op is Op.LOAD for t in e.stmt.target}
            for e in unit.events:
                s = e.stmt
                if s.op is not Op.STORE:
                    continue
                value, index = _store_parts(s)
                if self._constant(value):
                    for lane, env, memo in self._domain(
                            (index, s.predicate, *(c for c, _ in e.conds)),
                            e.loops):
                        if self._active(e, env, memo):
                            constants.add(
                                self._indices.evaluate(index, env, memo))
                    continue
                if not isinstance(value, Value) or value.id not in producers:
                    raise _Refused('the store writes something other than the '
                                   'image')
                src = producers[value.id]
                if src.stmt.predicate is not None:
                    raise _Refused('a predicated read of the image')
                loops = {**src.loops, **e.loops}
                src_index = _load_index(src.stmt)
                for lane, env, memo in self._domain(
                        (index, src_index, s.predicate,
                         *(c for c, _ in e.conds + src.conds)), loops):
                    if not self._active(e, env, memo):
                        continue
                    element = self._indices.evaluate(index, env, memo)
                    slot = self._indices.evaluate(src_index, env, memo)
                    if owner.setdefault(element, lane) != lane:
                        raise _Refused('two lanes store one element')
                    written.setdefault(lane, set()).add(element)
                    if held.get(lane, {}).get(slot, _ZERO) != element:
                        raise _Refused('the store writes an element the load '
                                       'does not read into that slot')
        for lane, slots in held.items():
            for element in slots.values():
                if element is _ZERO:
                    continue
                if owner.get(element, lane) != lane:
                    raise _Refused('a lane loads what another stores')
                if element in constants:
                    raise _Refused('the load reads what the store fills in')
        return written

    def _copy_back(self, s: Stmt, access: Access, tail: _Unit,
                   head: _Unit) -> List[Stmt]:
        """`s`, a statement of the store, as the copy of the tail's image
        into the head's: what it stores from the image goes into the slot it
        was read from, and what it fills in goes nowhere."""
        if s.regions:
            regions = []
            for r in s.regions:
                inner: List[Stmt] = []
                for x in r.body:
                    inner.extend(self._copy_back(x, access, tail, head))
                regions.append(replace(r, body=tuple(inner)))
            if s.op in (Op.FOR, Op.IF) and not s.target and not any(
                    _has_effect(x) for r in regions for x in r.body):
                return []
            return [replace(s, regions=tuple(regions))]
        if s.op is not Op.STORE or not any(
                a.base is access.base for a in s.accesses):
            return [s]
        value, _ = _store_parts(s)
        if self._constant(value):
            return []
        slot = self._slot_of(value, s, tail)
        return [replace(s, args=(head.operand, value, slot),
                        accesses=(Access(Effect.WRITE, MemSpace.REGISTER,
                                         head.image),),
                        attrs=())]

    def _slot_of(self, value: Value, s: Stmt, tail: _Unit):
        for e in tail.events:
            if e.stmt.op is Op.LOAD and any(t.id == value.id
                                            for t in e.stmt.target):
                return _load_index(e.stmt)
        raise _Refused(f'no read of the image for {s.op}')

    # -- the store ahead of the loop ------------------------------------------ #

    def _forward(self, out: List[Stmt], ahead: List[Stmt],
                 c: _Carried) -> List[Stmt]:
        """`ahead` reading what the statements in front of the loop stored
        from a register image, where that is what the load ahead of the loop
        reads back -- and those stores taken out of `out` where the store
        behind the loop writes all of it again.

        Which is how a loop whose trips are closed meets the code ahead of
        it: whatever that left in registers and the body touches is stored
        ahead of the loop (`Generator._emit_closed_loop`), and the carry has
        just put the load of it there.  An accumulator the list began ahead
        of the run -- SeisSol's Taylor expansion sums its first term there --
        would go to memory and come straight back.  Else `ahead` as it is."""
        name = getattr(c.access.base, 'name', str(c.access.base))
        try:
            ahead = self._forwarded(out, ahead, c)
        except _Refused as why:
            self._say(f'carry: {name} is loaded from memory ahead of the '
                      f'loop: {why}')
        except _NotStatic as why:
            self._say(f'carry: {name} is loaded from memory ahead of the '
                      f'loop: an index or a condition not known per lane '
                      f'({why})')
        return ahead

    def _forwarded(self, out: List[Stmt], ahead: List[Stmt],
                   c: _Carried) -> List[Stmt]:
        access = c.access
        probe = Access(Effect.READ, access.space, access.base)
        units: List[_Unit] = []
        for i in range(len(out) - 1, -1, -1):
            if not any(may_alias(a, probe)
                       for a in collect_accesses((out[i],))):
                continue
            try:
                unit = self._unit(out, [i], access)
            except _Refused:
                unit = self._filled(out, i, access)
            if unit is None or unit.kind not in ('store', 'fill'):
                break
            units.append(unit)
        stores = [u for u in units if u.kind == 'store']
        if not stores:
            raise _Refused('nothing ahead of the loop stores it from '
                           'registers')
        if len({_key(u.image) for u in stores}) != 1:
            raise _Refused('the store ahead of the loop reads more than one '
                           'register image')
        source = stores[0]
        first = min(i for u in units for i in u.at)
        mine = {i for u in units for i in u.at}
        for i in range(first, len(out)):
            if i in mine:
                continue
            if any(a.writes and _key(a.base) == _key(source.image)
                   for a in collect_accesses((out[i],))):
                raise _Refused('the image is written again ahead of the loop')

        # What the store ahead of the loop writes, per lane: element to slot,
        # and what it fills in.
        slots: Dict[int, Dict[int, int]] = {}
        filled: Set[int] = set()
        for unit in units:
            producers = {t.id: e for e in unit.events
                         if e.stmt.op is Op.LOAD for t in e.stmt.target}
            for e in unit.events:
                if e.stmt.op is not Op.STORE:
                    continue
                value, index = _store_parts(e.stmt)
                constant = self._constant(value)
                src = None if constant else producers.get(getattr(value, 'id',
                                                                  None))
                if not constant and (src is None
                                     or src.stmt.predicate is not None):
                    raise _Refused('the store ahead of the loop writes '
                                   'something other than its image')
                operands = (index, e.stmt.predicate,
                            *(cond for cond, _ in e.conds))
                if src is not None:
                    operands += (_load_index(src.stmt),
                                 *(cond for cond, _ in src.conds))
                loops = {**(src.loops if src is not None else {}), **e.loops}
                for lane, env, memo in self._domain(operands, loops):
                    if not self._active(e, env, memo):
                        continue
                    element = self._indices.evaluate(index, env, memo)
                    if constant:
                        filled.add(element)
                        continue
                    slot = self._indices.evaluate(_load_index(src.stmt), env,
                                                  memo)
                    if slots.setdefault(lane, {}).setdefault(element,
                                                             slot) != slot:
                        raise _Refused('the store ahead of the loop writes an '
                                       'element twice')
        for lane, held in c.held.items():
            for slot, element in held.items():
                if element is _ZERO:
                    continue
                if element in filled:
                    raise _Refused('the load reads what the store ahead of '
                                   'the loop fills in')
                if slots.get(lane, {}).get(element) != slot:
                    raise _Refused('the load reads what the store ahead of '
                                   'the loop does not write, or not from that '
                                   'slot')

        forwarded = self._from_image(ahead, c, source)
        # Behind the loop, the store writes every element the one ahead of it
        # wrote from its image, and nothing in between reads one: that store
        # writes nothing anybody reads.  What it fills in stays.
        dead = all(set(written) <= c.stored.get(lane, set())
                   for lane, written in slots.items())
        if dead:
            for i in sorted((i for u in stores for i in u.at), reverse=True):
                del out[i]
        name = getattr(access.base, 'name', str(access.base))
        self._say(f'carry: {name} taken from registers ahead of the loop'
                  + (', and not stored there' if dead else ''))
        return forwarded

    def _filled(self, stmts: List[Stmt], i: int,
                access: Access) -> Optional[_Unit]:
        """`stmts[i]` as what a store fills in -- constants into the buffer
        and nothing else, the zeros past the rows a product defines -- or
        None."""
        s = stmts[i]
        if collect_effect((s,)) & (Effect.BARRIER | Effect.ASYNC
                                   | Effect.UNKNOWN):
            return None
        for x in walk_stmts((s,)):
            for a in x.accesses:
                if not (a.base is access.base and a.space is access.space
                        and a.writes and not a.kind & Effect.ATOMIC
                        and x.op is Op.STORE
                        and self._constant(_store_parts(x)[0])):
                    return None
        return _Unit('fill', [i], None, None, _events((s,)))

    def _from_image(self, ahead: List[Stmt], c: _Carried,
                    source: _Unit) -> List[Stmt]:
        """`ahead` with the load in it reading the slots of `source`'s image
        where it read the buffer.

        A slot is read where the head's image is written, at the index it is
        written at, and the read of the buffer goes: the index may be
        computed behind that read -- an unrolled copy computes the address it
        reads and then the slot it writes."""
        mine = [s for s in ahead if id(s) in c.load]
        loads = {t.id: s for s in walk_stmts(tuple(mine))
                 if s.op is Op.LOAD
                 and any(a.base is c.access.base for a in s.accesses)
                 for t in s.target}
        fills: Dict[int, Stmt] = {}
        uses: Dict[int, int] = {}
        for s in walk_stmts(tuple(ahead)):
            for v in s.operands():
                if v.id in loads:
                    uses[v.id] = uses.get(v.id, 0) + 1
            if s.op is Op.STORE and any(_key(a.base) == _key(c.head.image)
                                        for a in s.accesses):
                value, _ = _store_parts(s)
                if getattr(value, 'id', None) in loads:
                    fills[id(s)] = loads[value.id]
        if not loads:
            raise _Refused('nothing ahead of the loop loads it')
        filling = {load.target[0].id for load in fills.values()}
        for vid in loads:
            if vid not in filling:
                raise _Refused('a load of it fills no slot')
            if uses.get(vid) != 1:
                raise _Refused('a load of it is read elsewhere than into '
                               'the image')
        dropped = {id(load) for load in loads.values()}

        def rewrite(stmts) -> List[Stmt]:
            out: List[Stmt] = []
            for s in stmts:
                if id(s) in dropped:
                    continue
                if s.regions:
                    out.append(replace(s, regions=tuple(
                        replace(r, body=tuple(rewrite(r.body)))
                        for r in s.regions)))
                    continue
                load = fills.get(id(s))
                if load is not None:
                    _, slot = _store_parts(s)
                    out.append(replace(
                        load, args=(source.operand, slot),
                        accesses=(Access(Effect.READ, MemSpace.REGISTER,
                                         source.image),),
                        attrs=()))
                out.append(s)
            return out

        return [s for x in ahead
                for s in (rewrite((x,)) if id(x) in c.load else (x,))]

    # -- an atomic store at the tail ----------------------------------------- #

    def _atomic(self, loop: Stmt, stmts: List[Stmt], units: List[_Unit],
                access: Access):
        tail = units[0]
        if len({_key(u.image) for u in units}) != 1:
            raise _Refused('more than one register image')
        mine = {i for u in units for i in u.at}
        self._alone(loop, stmts, mine, access, units)
        inside = set(defined_within(tuple(stmts)))
        inside |= {a.id for a in loop.regions[0].args}
        _needs_only(tuple(stmts[i] for i in sorted(mine)), inside,
                    'the store needs what the trip computes',
                    allowed={_key(tail.operand), _key(tail.image)})
        alloc = next((s for s in walk_stmts(tuple(stmts))
                      if s.op is Op.ALLOC
                      and any(_key(t) == _key(tail.image) for t in s.target)),
                     None)
        if alloc is None or not isinstance(tail.image, Value):
            raise _Refused('the image is not declared in the trip')
        # A guard of the update spelled in text goes behind the loop with it,
        # and has to mean the same there.
        counter = loop.attr('extern')
        if counter and any(
                isinstance(cond, str)
                and re.search(rf'\b{re.escape(str(counter))}\b', cond)
                for u in units for e in u.events for cond, _ in e.conds):
            raise _Refused('a guard of the update reads the trip')
        # Each slot of the image added once a trip.
        slots: Set[int] = set()
        for unit in units:
            producers = {t.id: e for e in unit.events
                         if e.stmt.op is Op.LOAD for t in e.stmt.target}
            for e in unit.events:
                s = e.stmt
                if not (s.op is Op.RAWSTMT or s.op is Op.STORE):
                    continue
                value = s.args[0] if s.op is Op.RAWSTMT else s.args[1]
                if not isinstance(value, Value) or value.id not in producers:
                    raise _Refused('the update adds something other than the '
                                   'image')
                if getattr(value.type, 'length', None) is not None:
                    raise _Refused('a vector update')
                src = producers[value.id]
                index = _load_index(src.stmt)
                trips = self._indices.leaves(index)
                if trips - set(src.loops):
                    raise _Refused('a slot not known for every hop')
                for env in self._indices.domain(trips, src.loops):
                    slot = self._indices.evaluate(index, env, {})
                    if slot in slots:
                        raise _Refused('a slot of the image added twice')
                    slots.add(slot)

        image = self._fresh(tail.image)
        name = Clone(self._fresh, 'carry', self._taken).rename(
            str(alloc.attr('extern', tail.image.hint or 'acc')))
        attrs = [(k, v) for k, v in alloc.attrs
                 if k not in ('extern', 'init', 'decl')]
        attrs += [('extern', name), ('init', '{}')]
        declare = replace(alloc, target=(image,), attrs=tuple(attrs))
        body: List[Stmt] = []
        for i, s in enumerate(stmts):
            unit = next((u for u in units if i in u.at), None)
            if unit is not None:
                body.extend(self._accumulate(s, access, unit, image))
            else:
                body.append(s)
        behind = self._rebased(
            Clone(self._fresh, 'carry', self._taken).stmts(
                [stmts[i] for i in sorted(mine)]),
            tail, image, image)
        return [declare], body, behind, None

    def _accumulate(self, s: Stmt, access: Access, tail: _Unit,
                    image: Value) -> List[Stmt]:
        """`s`, a statement of the atomic store, as the sum into `image`: an
        update of memory becomes an update of the slot its value was read
        from."""
        if s.regions:
            regions = []
            for r in s.regions:
                inner: List[Stmt] = []
                for x in r.body:
                    inner.extend(self._accumulate(x, access, tail, image))
                regions.append(replace(r, body=tuple(inner)))
            return [replace(s, regions=tuple(regions))]
        if not any(a.base is access.base for a in s.accesses):
            return [s]
        value = s.args[0] if s.op is Op.RAWSTMT else s.args[1]
        slot = self._slot_of(value, s, tail)
        b = self._scratch()
        was = b.load(image, slot, type_=value.type, hint='sum')
        now = b.op('add', value.type, was, value, hint='sum')
        b.store(image, now, slot)
        return list(b.finish())

    # -- shared --------------------------------------------------------------- #

    def _alone(self, loop: Stmt, stmts: List[Stmt], mine: Set[int],
               access: Access, units: List[_Unit]) -> None:
        """Nothing in the loop but `units` reaches the buffer.  What names it
        is a unit (`_units`), and one that is no load or store of it is
        refused there; what reaches it unnamed is a binding of it handed on
        -- into a table of pointers, say -- or a statement that says nothing
        about what it touches."""
        bindings = {x.args[0].id for u in units for i in u.at
                    for x in walk_stmts((stmts[i],))
                    if x.op in (Op.LOAD, Op.STORE)
                    and isinstance(x.args[0], Value)
                    and any(a.base is access.base for a in x.accesses)}
        if bindings & self._addresses:
            raise _Refused('its binding is handed on, to a table of pointers '
                           'or the like, and may reach it from anywhere')
        if any(collect_effect((s,)) & Effect.UNKNOWN
               for i, s in enumerate(stmts) if i not in mine):
            raise _Refused('the trip holds a statement that says nothing '
                           'about what it touches')

    def _domain(self, operands, loops) -> Iterator[Tuple[int, Dict, Dict]]:
        """`(lane, env, memo)` for every lane and every combination of the
        inner loops' trips that `operands` reach."""
        leaves: Set = set()
        for x in operands:
            if x is not None:
                leaves |= self._indices.leaves(x)
        leaves.discard('x')
        for lane in range(self._threads):
            for env in self._indices.domain(frozenset(leaves), loops):
                env = dict(env)
                env['x'] = lane
                yield lane, env, {}

    def _active(self, e: _Event, env, memo) -> bool:
        for cond, side in e.conds:
            if bool(self._indices.evaluate(cond, env, memo)) != side:
                return False
        if e.stmt.predicate is not None and not self._indices.evaluate(
                e.stmt.predicate, env, memo):
            return False
        return True

    def _rebased(self, stmts: Sequence[Stmt], tail: _Unit, operand,
                 image, stores: bool = False) -> List[Stmt]:
        """`stmts` reading `operand` -- an image as its loads and stores
        spell it, accessed as `image` -- where they read the tail's image,
        and writing it where they write the tail's if `stores`."""
        out = []
        for s in stmts:
            if s.regions:
                s = replace(s, regions=tuple(
                    replace(r, body=tuple(self._rebased(r.body, tail,
                                                         operand, image,
                                                         stores)))
                    for r in s.regions))
            elif (s.op is Op.LOAD or (stores and s.op is Op.STORE)) and any(
                    _key(a.base) == _key(tail.image) for a in s.accesses):
                kind = Effect.READ if s.op is Op.LOAD else Effect.WRITE
                s = replace(s, args=(operand,) + tuple(s.args[1:]),
                            accesses=(Access(kind, MemSpace.REGISTER,
                                             image),))
            out.append(s)
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _addresses(body: Sequence[Stmt]) -> Set[int]:
    """The values `body` uses other than as the buffer an access goes
    through -- a binding put into a table of pointers, or spelled into raw
    text -- by id.  What reaches a buffer from there is nothing the accesses
    that name it can tell."""
    out: Set[int] = set()
    for s in walk_stmts(tuple(body)):
        args = s.args
        if s.op in (Op.LOAD, Op.STORE, Op.PREFETCH, Op.LOAD_ASYNC):
            args = args[1:]
        elif s.op is Op.COPY_ASYNC:
            args = args[2:]
        out.update(v.id for v in args if isinstance(v, Value))
        if s.predicate is not None:
            out.add(s.predicate.id)
    return out


def _key(base):
    """A buffer's identity: a value by its id, anything else as itself."""
    return ('v', base.id) if isinstance(base, Value) else ('o', id(base))


def _stored(body: Sequence[Stmt]) -> List[Access]:
    """The buffers in global or shared memory that `body` writes, one access
    each, in the order they are first written."""
    out: List[Access] = []
    for a in collect_accesses(tuple(body)):
        if (a.writes and a.space in (MemSpace.GLOBAL, MemSpace.SHARED)
                and a.base is not None
                and not any(b.base is a.base for b in out)):
            out.append(a)
    return out


def _stored_from_registers(body: Sequence[Stmt], access: Access) -> bool:
    """Does a statement of `body` write `access`'s buffer and read a register
    image -- the store at the tail of a trip it would take, if any?  What
    is not one of those, nothing here carries, and nothing reports."""
    for s in body:
        found = collect_accesses((s,))
        if any(a.writes and a.base is access.base for a in found) and any(
                a.space is MemSpace.REGISTER and not a.writes for a in found):
            return True
    return False


def _touches_memory(s: Stmt) -> bool:
    return any(a.space is not MemSpace.REGISTER
               for a in collect_accesses((s,)))


_SIDE = int(Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.ASYNC
            | Effect.UNKNOWN)


def _has_effect(s: Stmt) -> bool:
    """Does `s` do anything a statement after it could observe?"""
    return bool(int(collect_effect((s,))) & _SIDE)


def _store_parts(s: Stmt):
    """`(value, index)` of a store into a flat buffer."""
    if len(s.args) != 3:
        raise _Refused('a store with more than one index')
    return s.args[1], s.args[2]


def _load_index(s: Stmt):
    if len(s.args) != 2:
        raise _Refused('a load with more than one index')
    return s.args[1]


def _events(stmts: Sequence[Stmt]) -> List[_Event]:
    """Every load and store in `stmts`, with what it stands under."""
    out: List[_Event] = []

    def visit(xs, conds, loops):
        for x in xs:
            if x.op is Op.FOR:
                if not counted(x) or x.loop_inits:
                    raise _NotStatic('a loop of an unknown count')
                lo, hi, step = x.loop_bounds
                visit(x.regions[0].body, conds,
                      {**loops, x.induction.id: range(lo, hi, step)})
            elif x.op is Op.IF:
                visit(x.regions[0].body, conds + ((x.cond, True),), loops)
                if len(x.regions) > 1:
                    visit(x.regions[1].body, conds + ((x.cond, False),),
                          loops)
            elif (x.op is Op.RAWBLOCK and len(x.regions) == 1
                  and (x.text or '').startswith('if (')):
                # A guard spelled in text: which lanes take it is not known
                # here, and whatever has to know refuses (`_domain`).
                visit(x.regions[0].body, conds + ((x.text, True),), loops)
            elif x.regions:
                raise _NotStatic(f'a {x.op} whose runs are not known')
            elif x.op in (Op.LOAD, Op.STORE) or (x.op is Op.RAWSTMT
                                                  and x.accesses):
                out.append(_Event(x, conds, loops))

    visit(stmts, (), {})
    return out


def _needs_only(stmts: Tuple[Stmt, ...], inside: Set[int], why: str,
                allowed=frozenset()) -> None:
    """`stmts` read no value the loop defines but those they define
    themselves -- and the buffers `allowed` names, which the rewrite
    replaces."""
    own = defined_within(stmts)
    for x in walk_stmts(stmts):
        for v in x.operands():
            if v.id in own or _key(v) in allowed:
                continue
            if v.id in inside:
                raise _Refused(why)
        for a in x.accesses:
            if isinstance(a.base, Value) and a.base.id in inside \
                    and _key(a.base) not in allowed and a.base.id not in own:
                raise _Refused(why)
