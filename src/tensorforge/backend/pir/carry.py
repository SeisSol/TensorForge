# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: what the trips of a merged run hand each other through memory,
handed over in registers instead.

A merged run's trips hand each other nothing but memory
(`Generator._emit_closed_loop`), so the run touches an accumulator twice a
trip: its head reads it into a register image, and its tail stores the
image the trip computed from that.  Memory is all that crosses the back
edge, which is what makes the body the same on every trip -- and it is a
load and a store of the whole accumulator per trip.  Where the next trip
only reads back what this one stored, the load goes ahead of the loop, the
store behind it, and the tail copies its image into the head's::

    for k:                        I = load H             (once)
      I = load H                  for k:
      ...                           ...
      J = f(I, ...)                 J = f(I, ...)
      store H <- J                  I <- J
                                  store H <- I           (once)

Whether the one store behind the loop then adds its sum atomically is not
this pass's question: the body asks it afterwards, where it can see who reads
the sum (`pir.atomics`).

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

A guard spelled in text -- a batch element's, on AMD -- is not known per
lane, and what stands under one is taken to run wherever the rest of its
conditions say.  Which is the same where the guard means the same over the
whole loop -- it names nothing the trip computes, its counter included --
and every guard of the load is one of the store's (`_guarded`): where one of
the store's does not hold, no trip stores and none copies its image back, so
each reads what the first one loaded, as each did from memory.

Where every slot of the head's image is read before the trip first writes
that slot of the tail's, the tail is computed in the head's image itself,
and the copy goes (`_fused`): an accumulation slot by slot, as the list
written out has it.  Under a guard spelled in text, only where nothing but
the images reads their values (`_closed`): where the guard does not hold,
the trips sum in place what no trip stores and nothing else reads.

And where the code ahead of the loop stored `H` from registers just before
-- a closed trip stores what the body touches ahead of the loop, and an
accumulator the list began in front of the run is one -- the load that now
stands there reads that image instead of memory, and the store goes as well
where the one behind the loop writes all of it again (`_forward`).  A store
there that pads what it writes with numbers in the same statement goes
behind the loop instead, reading the carried image.  The code behind the
loop that loads `H` straight back reads the carried image
(`_forward_behind`).

A buffer in shared memory need not be loaded whole at the head of a trip and
stored whole at its tail.  Where every lane reads of it only what it wrote
itself, the trips read and write it anywhere, in the middle of a computation
as well -- a temporary of the multiplication's own arena that a trip adds
onto and the next reads -- and every element a lane reaches is a register of
an image of its own, one register per element, the same for every lane
(`_private`)::

    store H[e(l, 0)] <- R[0]      P[0] <- R[0]           (the code ahead)
    for k:                        for k:
      a = H[e(l, 0)] + ...          a = P[0] + ...
      store H[e(l, 0)] <- a         P[0] <- a
                                  store H[e(l, 0)] <- P[0]

with the store behind the loop where anything behind it reads `H` before
something defines it anew, and the store ahead of it where anything still
reads what it wrote.

The same holds outside of a merged run's trips: a buffer of the arena that
every lane reads only what it wrote itself lives in a register image of
every lane's own, over the statements that touch it -- the innermost list
around all of them, or a loop around that where the trips hand it on --
and the arena loses it (`keep_own`).  Over those statements and not across
them: what a lane reads of it there, the lane wrote there before, and
nothing ahead of them -- the last element of the batch, say -- hands
anything in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .ahead import Clone, counted
from .core import (Access, BufferType, Effect, MemSpace, Op, Stmt, Value,
                   collect_accesses, collect_effect, defined_within, may_alias,
                   walk_stmts)
from .flow import body_masks
from .shards import _Indices, _NotStatic
from .transfers import neutral, tag


def carry_trips(body: Tuple[Stmt, ...], scratch, *, threads: int,
                arena: Optional[str] = None,
                report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with what the trips of its merged runs hand each other through
    memory kept in registers, where that is shown to be the same (see the
    module).  `threads` is how many lanes a multiplication spans,
    `threadIdx.x` the lane; `arena` is the multiplication's own arena in
    shared memory, whose buffers no lane of another multiplication reaches,
    and None where the buffers of none are to be kept per lane (`_private`);
    `scratch` makes the builder new statements are built with; `report`
    collects what was carried and why the rest was not."""
    taken = {str(s.attr('extern')) for s in walk_stmts(body)
             if s.attr('extern') is not None}
    constants = {t.id: s.attr('value') for s in walk_stmts(body)
                 if s.op is Op.CONST for t in s.target}
    carrier = _Carrier(scratch, _Indices(body, {'x': threads}), threads,
                       report, taken, constants, _addresses(body),
                       _own(body, arena), body, body_masks(body, arena))
    out, changed = carrier.stmts(body)
    return out if changed else body


def keep_own(body: Tuple[Stmt, ...], scratch, *, threads: int,
             arena: Optional[str] = None,
             report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with each buffer of `arena`, the multiplication's own, that
    every lane reads only what it wrote itself kept in a register image of
    every lane's own instead (see the module); the arguments are those of
    `carry_trips`, and `report` collects what was kept and why the rest was
    not."""
    if arena is None:
        return body
    # What every buffer asks of the body, asked once.  Keeping one changes
    # nothing another asks: the names a new image may not take, which it
    # adds to (`Clone.rename`); what an index of another buffer computes;
    # what the body hands on as a pointer.
    owned = _own(body, arena)
    taken = {str(s.attr('extern')) for s in walk_stmts(body)
             if s.attr('extern') is not None}
    constants = {t.id: s.attr('value') for s in walk_stmts(body)
                 if s.op is Op.CONST for t in s.target}
    indices = _Indices(body, {'x': threads})
    addresses = _addresses(body)
    touched = _Touched(set(owned))
    # Whether a buffer is kept, and the registers its accesses take, is
    # asked of the body as it came: keeping another changes no access of
    # it, no loop or branch around one, and nothing a statement around them
    # says of what it touches.  The body is rewritten once, for all of them.
    plans: List[Tuple['_Carrier', '_Kept']] = []
    for alloc in list(owned.values()):
        name = str(alloc.attr('extern') or alloc.target[0])
        keeper = _Carrier(scratch, indices, threads, report, taken,
                          constants, addresses, owned, body,
                          touched=touched)
        try:
            plan = keeper._kept(body, alloc)
        except _Refused as why:
            keeper._say(f'own: {name} stays in shared memory: {why}')
            continue
        except _NotStatic as why:
            keeper._say(f'own: {name} stays in shared memory: an index or a '
                        f'condition not known per lane ({why})')
            continue
        keeper._say(f'own: {name} in {plan.size} registers of every lane')
        plans.append((keeper, plan))
    return _keep(body, plans, touched) if plans else body


@dataclass
class _Kept:
    """A buffer `keep_own` keeps: the list its image is declared at the head
    of (`stmts`, as the body holds it), the register each access takes
    (`fits`), how many a lane holds (`size`), the inductions of the loops
    around its accesses, and the statements that go with the buffer -- its
    declaration and the marks of it."""
    alloc: Stmt
    key: tuple
    stmts: Tuple[Stmt, ...]
    fits: Dict[int, Tuple[int, Dict[int, int]]]
    size: int
    inductions: Dict[int, Value]
    gone: Set[int]


def _keep(body: Tuple[Stmt, ...], plans, touched: '_Touched'
          ) -> Tuple[Stmt, ...]:
    """`body` with the buffers of `plans` kept: each image declared at the
    head of its list -- the last kept first -- each access of the buffer
    one of its image, and the buffer's declaration and marks gone.  The
    values the new statements define are made in the order of `plans`, the
    image first, then the accesses in the order of the body."""
    heads: Dict[int, Tuple[Tuple[Stmt, ...], List[Stmt]]] = {}
    instead: Dict[int, List[Stmt]] = {}
    gone: Set[int] = set()
    keys: Set[tuple] = set()
    for keeper, plan in plans:
        b = keeper._scratch()
        extern = Clone(keeper._fresh, 'own', keeper._taken).rename(
            str(plan.alloc.attr('extern', 'own')))
        base = plan.alloc.target[0]
        image = b.alloc(base.type.elem, (plan.size,), MemSpace.REGISTER,
                        hint='own', extern=extern, init='{}')
        held = heads.setdefault(id(plan.stmts), (plan.stmts, []))[1]
        held[:0] = b.finish()
        for x in _accesses_of(plan.stmts, plan.key, touched):
            fit = plan.fits.get(id(x))
            if fit is not None:
                instead[id(x)] = keeper._privatized_one(
                    x, plan.fits, image, plan.inductions)
        gone |= plan.gone
        keys.add(plan.key)

    def rewrite(xs: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        head = heads.get(id(xs))
        out: List[Stmt] = list(head[1]) if head is not None \
            and head[0] is xs else []
        changed = bool(out)
        for x in xs:
            if id(x) in gone:
                changed = True
                continue
            new = instead.get(id(x))
            if new is not None:
                out.extend(new)
                changed = True
                continue
            if x.regions and touched.reaches(x) & keys:
                regions = tuple(replace(r, body=rewrite(r.body))
                                for r in x.regions)
                if any(n.body is not o.body
                       for n, o in zip(regions, x.regions)):
                    x = replace(x, regions=regions)
                    changed = True
            out.append(x)
        return tuple(out) if changed else xs

    return rewrite(tuple(body))


def _accesses_of(stmts: Sequence[Stmt], key, touched: '_Touched'
                 ) -> Iterator[Stmt]:
    """The statements in `stmts` that access the buffer `key` names and hold
    none, in the order of the body."""
    for i in touched.where(stmts, key, accessed=True):
        x = stmts[i]
        if x.regions:
            for r in x.regions:
                yield from _accesses_of(r.body, key, touched)
        else:
            yield x


class _Refused(Exception):
    """Why a buffer stays where it is."""


class _Trip:
    """What each statement of a list touches, asked of it once for the list
    as it stands: a loop's trip, which `_Carrier._loop` asks of every buffer
    it stores, or the code behind it.

    `touching` is where the statements are that may reach a buffer
    (`may_alias`), `transfer` where those of a transfer are, and
    `stored_from_registers` whether a statement writes a buffer and reads a
    register image.  `declared` holds the declarations in the list, and
    `unsaid` whether it holds a statement that says nothing about what it
    touches."""

    def __init__(self, stmts: Sequence[Stmt]):
        self.stmts: List[Stmt] = list(stmts)
        #: Positions of the statements that access a buffer, by its space
        #: and identity; of those that access a space through no buffer, by
        #: the space; and of those that access a space nobody knows.
        self._of: Dict[Tuple[MemSpace, int], List[int]] = {}
        self._unnamed: Dict[MemSpace, List[int]] = {}
        self._anywhere: List[int] = []
        #: Positions of the statements of a transfer, by its tag, but text
        #: that touches nothing (`neutral`).
        self._transfers: Dict[int, List[int]] = {}
        #: The buffers a statement that reads a register image writes.
        self._from_registers: Set[int] = set()
        self.declared: Set[int] = set()
        self.unsaid = False
        for i, s in enumerate(self.stmts):
            t = tag(s)
            if t is not None and not neutral(s):
                self._transfers.setdefault(t, []).append(i)
            keys: Set[Optional[Tuple[MemSpace, Optional[int]]]] = set()
            written: Set[int] = set()
            registers = False
            for x in walk_stmts((s,)):
                if x.op is Op.ALLOC:
                    self.declared.add(id(x))
                if x.effect & Effect.UNKNOWN:
                    self.unsaid = True
                for a in x.accesses:
                    if a.space is MemSpace.UNKNOWN:
                        keys.add(None)
                    else:
                        keys.add((a.space, None if a.base is None
                                  else id(a.base)))
                    if a.writes and a.base is not None:
                        written.add(id(a.base))
                    elif a.space is MemSpace.REGISTER and not a.writes:
                        registers = True
            for key in keys:
                if key is None:
                    self._anywhere.append(i)
                elif key[1] is None:
                    self._unnamed.setdefault(key[0], []).append(i)
                else:
                    self._of.setdefault(key, []).append(i)
            if registers:
                self._from_registers |= written

    def touching(self, access: Access) -> List[int]:
        """Where the statements are that may reach the buffer `access`
        names, in order: those that access it, those that access its space
        through no buffer, and those that access a space nobody knows."""
        found = set(self._anywhere)
        found.update(self._unnamed.get(access.space, ()))
        found.update(self._of.get((access.space, id(access.base)), ()))
        return sorted(found)

    def transfer(self, t: int) -> List[int]:
        """Where the statements of transfer `t` are, in order."""
        return self._transfers.get(t, [])

    def stored_from_registers(self, access: Access) -> bool:
        """Does a statement write `access`'s buffer and read a register
        image -- the store at the tail of a trip it would take, if any?
        What is not one of those, nothing here carries, and nothing
        reports."""
        return id(access.base) in self._from_registers


#: How many lists `_Touched` keeps what it found in.
_LISTS = 64


class _Touched:
    """Which of the buffers `keys` names each statement and those in it
    reach, asked once of each statement for as long as it is part of the
    body: `keep_own` asks it of every own buffer, and keeping one changes
    only the statements around it.

    `accessed` is what their accesses reach; `reaches` adds what they name
    as an operand and what they declare -- what `_Carrier._kept` looks for.
    `unsaid` is whether a list holds an access that says nothing of where
    it goes."""

    def __init__(self, keys: Set[tuple]):
        self._keys = keys
        self._of: Dict[int, Tuple[Stmt, frozenset, frozenset]] = {}
        self._unsaid: Dict[int, Tuple[Tuple[Stmt, ...], bool]] = {}
        self._said: Dict[int, Tuple[Stmt, bool]] = {}
        #: Per list, by buffer: the positions of the statements that reach
        #: it, and of those that access it.
        self._where: Dict[int, Tuple[Sequence[Stmt], Dict, Dict]] = {}

    def _get(self, x: Stmt) -> Tuple[Stmt, frozenset, frozenset]:
        known = self._of.get(id(x))
        if known is not None and known[0] is x:
            return known
        accessed = {_key(a.base) for a in x.accesses
                    if a.base is not None} & self._keys
        reaches = set(accessed)
        reaches.update(k for k in (('v', v.id) for v in x.args
                                   if isinstance(v, Value))
                       if k in self._keys)
        if x.op is Op.ALLOC and x.target and _key(x.target[0]) in self._keys:
            reaches.add(_key(x.target[0]))
        for r in x.regions:
            for y in r.body:
                _, a, b = self._get(y)
                accessed |= a
                reaches |= b
        known = (x, frozenset(accessed), frozenset(reaches))
        self._of[id(x)] = known
        return known

    def accessed(self, x: Stmt) -> frozenset:
        return self._get(x)[1]

    def reaches(self, x: Stmt) -> frozenset:
        return self._get(x)[2]

    def where(self, xs: Sequence[Stmt], key,
              accessed: bool = False) -> Sequence[int]:
        """The positions in the list `xs` of the statements that reach
        `key` -- with `accessed`, that access it -- the first first."""
        known = self._where.get(id(xs))
        if known is None or known[0] is not xs:
            reached: Dict = {}
            touched: Dict = {}
            for i, x in enumerate(xs):
                _, a, r = self._get(x)
                for k in r:
                    reached.setdefault(k, []).append(i)
                for k in a:
                    touched.setdefault(k, []).append(i)
            known = (xs, reached, touched)
            self._where[id(xs)] = known
            if len(self._where) > _LISTS:
                # A list a buffer was kept in is gone from the body: what
                # it held is asked of the lists that took its place.
                del self._where[next(iter(self._where))]
        return (known[2] if accessed else known[1]).get(key, ())

    def unsaid(self, stmts: Tuple[Stmt, ...]) -> bool:
        known = self._unsaid.get(id(stmts))
        if known is not None and known[0] is stmts:
            return known[1]
        found = any(self._unsaid_in(x) for x in stmts)
        self._unsaid[id(stmts)] = (stmts, found)
        if len(self._unsaid) > _LISTS:
            del self._unsaid[next(iter(self._unsaid))]
        return found

    def _unsaid_in(self, x: Stmt) -> bool:
        known = self._said.get(id(x))
        if known is not None and known[0] is x:
            return known[1]
        found = (any(a.space is MemSpace.UNKNOWN or (
            a.space is MemSpace.SHARED and a.base is None)
            for a in x.accesses)
            or any(self._unsaid_in(y) for r in x.regions for y in r.body))
        self._said[id(x)] = (x, found)
        return found


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
    #: it, and `fill` stores numbers into it and no image (`_filled`).
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
    """A buffer the trips now hand each other in registers: a load ahead of
    the loop and a store behind it.

    The load reads into `image` -- spelled `operand` by its loads and
    stores -- what `held` says (per lane, slot to element), the store behind
    the loop writes `stored` (per lane, the elements) from it, and `load`
    holds the statements of the load by id.  `behind` holds the statements
    behind the loop that write the buffer, by id."""
    access: Access
    operand: object
    image: object
    held: Dict[int, Dict[int, object]]
    stored: Dict[int, Set[int]]
    load: Set[int]
    behind: Set[int]
    #: The guards spelled in text the store stands under (`_guarded`), and
    #: those the load stands under, every one of them one of the store's.
    guards: frozenset = frozenset()
    loaded: frozenset = frozenset()
    #: Whether the images reach nothing but themselves and the store
    #: (`_closed`), and whether the trips compute in the head's image
    #: (`_fused`).
    closed: bool = False
    fused: bool = False


class _Carrier:
    """One body's walk: its merged runs' loops, innermost first, each with
    the buffers its trips store from registers tried one at a time."""

    def __init__(self, scratch, indices: _Indices, threads: int,
                 report: Optional[List[str]], taken: Set[str],
                 constants: Dict[int, object], addresses: Set[int],
                 own: Dict[tuple, Stmt], root: Tuple[Stmt, ...],
                 masks: frozenset = frozenset(),
                 touched: Optional['_Touched'] = None):
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
        #: The buffers a lane may keep to itself, by key, with their
        #: declarations (`_own`).
        self._own = own
        #: The body as it came, and the loop being carried, by id: what is
        #: read behind the loop is asked of them (`_read_behind`).
        self._root = root
        #: The element masks the body keeps the contract of (`flow.
        #: kept_masks`): where one does not hold, nothing a lane computes is
        #: kept, so a read there may be taken from wherever the lane's value
        #: would be where it holds.
        self._masks = masks
        #: Which of the own buffers each statement reaches (`_kept`).
        self._touched = touched if touched is not None else _Touched(set())
        self._here: Optional[int] = None
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
        behind_loops: List[_Carried] = []
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
                ahead, s, behind, carried = self._loop(s, out)
                if ahead or behind:
                    changed = True
                for c in carried:
                    ahead, behind = self._forward(out, ahead, behind, c)
                out.extend(ahead)
                out.append(s)
                out.extend(behind)
                behind_loops.extend(carried)
            else:
                out.append(s)
        for c in behind_loops:
            out = self._forward_behind(out, c)
        return tuple(out), changed

    def _loop(self, loop: Stmt, out: List[Stmt]):
        """`(ahead, loop, behind, carried)` for one merged run's loop, with
        `out` the statements ahead of it in its list -- which the buffers a
        lane keeps to itself take their first values from, and lose their
        stores to, where those stores go (`_private`)."""
        ahead: List[Stmt] = []
        behind: List[Stmt] = []
        carried: List[_Carried] = []
        tried: Set[int] = set()
        self._here = id(loop)
        while True:
            body = loop.regions[0].body
            trip = _Trip(body)
            found = None
            for access in _stored(body):
                if id(access.base) in tried:
                    continue
                tried.add(id(access.base))
                if not trip.stored_from_registers(access):
                    continue
                name = getattr(access.base, 'name', str(access.base))
                try:
                    found = self._carry(loop, trip, access, out)
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

    def _carry(self, loop: Stmt, trip: '_Trip', access: Access,
               out: List[Stmt]):
        touching = trip.touching(access)
        try:
            return self._shaped(loop, trip, touching, access)
        except _Refused as why:
            if (access.space is not MemSpace.SHARED
                    or _key(access.base) not in self._own):
                raise
            try:
                return self._private(loop, trip, touching, access, out)
            except _Refused as other:
                if str(other) == str(why):
                    raise
                raise _Refused(f"{why}; and it is not each lane's own: "
                               f"{other}") from other

    def _shaped(self, loop: Stmt, trip: '_Trip', touching: List[int],
                access: Access):
        """A load at the head and a store at the tail."""
        units = self._units(trip, touching, access)
        kinds = {u.kind for u in units}
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
        return self._loaded(loop, trip.stmts, loads, stores, access)

    def _units(self, trip: '_Trip', touching: List[int],
               access: Access) -> List[_Unit]:
        """The statements at `touching`, grouped into the transfers they are
        part of (`transfers.tag`), as units of the loop that move `access`'s
        buffer through one register image."""
        stmts = trip.stmts
        groups: List[List[int]] = []
        done: Set[int] = set()
        for i in touching:
            if i in done:
                continue
            t = tag(stmts[i])
            group = list(trip.transfer(t)) if t is not None else [i]
            if i not in group:
                group.append(i)
            group.sort()
            done.update(group)
            groups.append(group)
        inside = set(touching)
        if any(j not in inside and _touches_memory(stmts[j])
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
        reads_h = writes_h = False
        images: Dict[object, Tuple[object, bool, bool]] = {}
        for x in walk_stmts(inner):
            for a in x.accesses:
                if a.base is access.base and a.space is access.space:
                    if a.kind & Effect.ATOMIC:
                        raise _Refused('an atomic update of it')
                    if a.kind & Effect.WRITE:
                        writes_h = True
                    else:
                        reads_h = True
                    if x.op not in (Op.LOAD, Op.STORE):
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
        if reads_h and writes_i and not (writes_h or reads_i):
            return _Unit('load', at, image, operand, events)
        if writes_h and reads_i and not (reads_h or writes_i):
            return _Unit('store', at, image, operand, events)
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

        guards = self._guarded(loop, stmts, access, loads, stores)
        held = self._head_map(loads, text=True)
        stored, handed = self._check_tail(stores, held, text=True)

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
        closed = self._closed(body, copies, head, tail)
        copied = body
        # An element's mask the body keeps is no guard here: where it does
        # not hold, what the trips sum in place is nothing anyone keeps.
        if closed or not {g for g in guards
                          if not (g[1] and g[0] in self._masks)}:
            body = self._fused(body, copies, head, tail, handed)
        behind = self._rebased(
            Clone(self._fresh, 'carry', self._taken).stmts(
                [stmts[i] for u in stores for i in u.at]),
            tail, head.operand, head.image)
        loaded = {id(stmts[i]) for u in loads for i in u.at}
        under = frozenset(g for u in loads for e in u.events
                          for g in _spelled((e,)))
        return ahead, body, behind, _Carried(
            access, head.operand, head.image, held, stored, loaded,
            {id(x) for x in behind}, guards, under, closed, body is not copied)

    def _guarded(self, loop: Stmt, stmts: List[Stmt], access: Access,
                 loads: List[_Unit], stores: List[_Unit]) -> frozenset:
        """The guards spelled in text the store stands under -- one set for
        all of its statements that write the buffer -- where every guard of
        the load is one of them and none names what the trip computes.

        Such a guard is not known per lane, and the trips are taken to run
        it.  Where it holds, they do what the per-lane account says.  Where
        a guard of the store does not, no trip stores and none copies its
        image back, so every trip reads what the first one loaded -- as each
        loaded it from memory, where nothing stored it either.  That needs
        the load to run wherever the store does, and the guard to mean the
        same over the whole loop, ahead of it and behind it."""
        def spelled(e):
            return _spelled((e,))

        stored = {spelled(e) for u in stores for e in u.events
                  if e.stmt.op is Op.STORE
                  and any(a.base is access.base for a in e.stmt.accesses)}
        if len(stored) > 1:
            raise _Refused('the statements of the store stand under '
                           'different guards spelled in text')
        guards = next(iter(stored), frozenset())
        if any(not spelled(e) <= guards for u in loads for e in u.events):
            raise _Refused('the load stands under a guard spelled in text '
                           'the store does not')
        texts = {cond for cond, _ in guards}
        texts |= {cond for u in loads for e in u.events
                  for cond, _ in spelled(e)}
        if not texts:
            return guards
        names = {str(x.attr('extern')) for x in walk_stmts(tuple(stmts))
                 if x.attr('extern') is not None}
        if loop.attr('extern') is not None:
            names.add(str(loop.attr('extern')))
        for text in texts:
            for name in names:
                if re.search(rf'\b{re.escape(name)}\b', text):
                    raise _Refused('a guard spelled in text names what the '
                                   'trip computes')
        return guards

    def _fused(self, body: List[Stmt], copies: Set[int], head: _Unit,
               tail: _Unit, handed: Dict[int, Set[int]]) -> List[Stmt]:
        """`body` computing the tail's image in the head's, where that is the
        same: then the copy at the end of the trip copies every slot onto
        itself, and goes.

        It is the same where every lane reads every slot of the head's image
        only ahead of its first write of that slot of the tail's, and every
        slot of the tail's only behind that write -- the order a lane makes
        them in, every inner loop run through, a branch where it takes it
        (`_sequence`) -- and where the slots a lane writes of the tail's are
        the slots the copy hands on of it (`handed`): one it writes and does
        not hand on is the next trip's input still.  An accumulation
        `J = f(I)` slot by slot is that, rows past a lane's end or not, and
        in one image it is two images fewer of registers live at once than a
        copy between two, as far as anything counting slots can tell.  Else
        `body` as it is.
        """
        events: List[Tuple[str, int, tuple, Dict, Stmt]] = []
        try:
            self._sequence([s for s in body if id(s) not in copies],
                           head, tail, {}, (), events)
            for lane in range(self._threads):
                order: List[Tuple[str, int]] = []
                for what, slot, conds, env, _ in events:
                    full = {**env, 'x': lane}
                    self._known(tuple(c for c, _ in conds), full)
                    if all(bool(self._indices.evaluate(c, full, {})) == side
                           for c, side in conds):
                        order.append((what, slot))
                first: Dict[int, int] = {}
                for at, (what, slot) in enumerate(order):
                    if what == 'write tail':
                        first.setdefault(slot, at)
                for at, (what, slot) in enumerate(order):
                    if what == 'write head':
                        return body
                    if what == 'read head' and slot in first \
                            and at > first[slot]:
                        return body
                    if what == 'read tail' and (slot not in first
                                                 or at < first[slot]):
                        return body
                if set(first) != handed.get(lane, set()):
                    return body
        except (_Refused, _NotStatic, KeyError):
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

    def _closed(self, body: List[Stmt], copies: Set[int], head: _Unit,
                tail: _Unit) -> bool:
        """Whether `body`, the trip with its copy at the end (`copies`),
        reads the images for nothing but the tail's slots: a read of the
        head's image goes, through arithmetic alone, into stores of the
        tail's, and only the copy reads the tail's.  Then what the images
        hold reaches nothing but the store behind the loop."""
        every = list(walk_stmts(tuple(x for x in body if id(x) not in copies)))
        users: Dict[int, List[Stmt]] = {}
        for x in every:
            for v in x.operands():
                users.setdefault(v.id, []).append(x)
        work: List[Value] = []
        for x in every:
            for a in x.accesses:
                if _key(a.base) == _key(tail.image) and not a.writes:
                    return False
                if _key(a.base) == _key(head.image) and not a.writes:
                    if x.op is not Op.LOAD:
                        return False
                    work.extend(x.target)
        seen: Set[int] = set()
        while work:
            v = work.pop()
            if v.id in seen:
                continue
            seen.add(v.id)
            for y in users.get(v.id, []):
                if y.op is Op.STORE:
                    if (len(y.args) != 3 or y.args[1] is not v
                            or y.args[2] is v or y.predicate is v
                            or not any(_key(a.base) == _key(tail.image)
                                       for a in y.accesses)):
                        return False
                    continue
                if (y.op not in Op.ARITH and y.op != Op.MATH) or y.regions \
                        or y.accesses or y.effect != Effect.NONE:
                    return False
                work.extend(y.target)
        return True

    def _sequence(self, stmts: Sequence[Stmt], head: _Unit, tail: _Unit,
                  env: Dict, conds: tuple, events: List) -> None:
        """The accesses of `stmts` to the two images, in the order a lane
        makes them, every inner loop run through: `(what, slot, conds,
        hops, stmt)`, with `conds` what the access stands under and `hops`
        the inner loops' trips it is made at."""
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
            under = conds + (((s.predicate, True),) if s.predicate is not None
                             else ())
            for a in s.accesses:
                which = images.get(_key(a.base))
                if which is not None:
                    events.append((('write ' if a.writes else 'read ')
                                   + which, slot, under, env, s))

    def _head_map(self, loads: List[_Unit], text: bool = False):
        """Per lane, the slot of the head's image each element goes into:
        `{lane: {slot: element or _ZERO}}`.  With `text`, a guard spelled in
        text is taken to hold (`_active`)."""
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
                    if not self._active(e, env, memo, text):
                        continue
                    slot = self._indices.evaluate(index, env, memo)
                    element = (self._indices.evaluate(src_index, env, memo)
                               if self._active(src, env, memo, text)
                               else _ZERO)
                    lane_held = held.setdefault(lane, {})
                    if slot in lane_held:
                        raise _Refused('the load writes a slot twice')
                    lane_held[slot] = element
        return held

    def _check_tail(self, stores: List[_Unit], held, text: bool = False):
        """Every element the store writes from the tail's image is one the
        load reads into that slot, by the same lane; nothing else it writes
        is read; no lane stores what another loads.  What it writes from the
        image, per lane, is the answer -- with the slots of the image it
        writes them from, per lane.  With `text`, a guard spelled in text is
        taken to hold (`_active`)."""
        owner: Dict[int, int] = {}
        constants: Set[int] = set()
        written: Dict[int, Set[int]] = {}
        handed: Dict[int, Set[int]] = {}
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
                        if self._active(e, env, memo, text):
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
                    if not self._active(e, env, memo, text):
                        continue
                    element = self._indices.evaluate(index, env, memo)
                    slot = self._indices.evaluate(src_index, env, memo)
                    if owner.setdefault(element, lane) != lane:
                        raise _Refused('two lanes store one element')
                    written.setdefault(lane, set()).add(element)
                    handed.setdefault(lane, set()).add(slot)
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
        return written, handed

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
                 behind: List[Stmt], c: _Carried):
        """`(ahead, behind)` taking what the statements in front of the loop
        stored from a register image from that image, where that is what the
        carry reads back -- and those stores out of `out`, or behind the
        loop, where nothing needs them where they are.

        Which is how a loop whose trips are closed meets the code ahead of
        it: whatever that computed and the body touches is stored ahead of
        the loop, behind the operation that computed it, and the carry has
        just put the load of it there.  An accumulator the list began ahead
        of the run -- SeisSol's Taylor expansion sums its first term there --
        would go to memory and come straight back.  Else as they are."""
        name = getattr(c.access.base, 'name', str(c.access.base))
        try:
            return self._forwarded(out, ahead, behind, c)
        except _Refused as why:
            self._say(f'carry: {name} is loaded from memory ahead of the loop: '
                      f'{why}')
        except _NotStatic as why:
            self._say(f'carry: {name} is loaded from memory ahead of the loop: '
                      f'an index or a condition not known per lane ({why})')
        return ahead, behind

    def _last_stores(self, out: List[Stmt], access: Access) -> List[_Unit]:
        """The statements at the end of `out` that store `access`'s buffer,
        from a register image or as numbers, last first: back to the first
        that does anything else to it."""
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
        return units

    def _forwarded(self, out: List[Stmt], ahead: List[Stmt],
                   behind: List[Stmt], c: _Carried):
        access = c.access
        units = self._last_stores(out, access)
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
        written, filled, mixed, guards = self._writes(out, units, source)
        at = sorted(i for u in stores for i in u.at)
        name = getattr(access.base, 'name', str(access.base))

        # Under a guard spelled in text, the load takes what the store ahead
        # wrote, and the store goes, only where the two hold for the same
        # elements -- and where the guard does not hold, what the trips then
        # compute from those registers is read by nothing else.
        if guards != c.guards or (guards and not c.closed):
            raise _Refused('the store ahead of the loop and the one behind it '
                           'stand under different guards spelled in text, or '
                           'the trips read the image for more than the store')

        for lane, held in c.held.items():
            for slot, element in held.items():
                if element is _ZERO:
                    continue
                if element in filled:
                    raise _Refused('the load reads what the store ahead of '
                                   'the loop fills in')
                if written.get(lane, {}).get(element) != slot:
                    raise _Refused('the load reads what the store ahead of '
                                   'the loop does not write, or not from that '
                                   'slot')
        forwarded = self._from_image(ahead, c.load, c.image, access.base,
                                     source.operand, source.image)
        # Behind the loop, the store writes every element the one ahead of it
        # wrote from its image, and nothing in between reads one: that store
        # writes nothing anybody reads -- and goes, where it writes nothing
        # else.  One that pads what it writes with numbers in the same
        # statement still has those to write: it goes behind the loop
        # instead, reading the carried image, in place of the store there.
        dead = all(set(w) <= c.stored.get(lane, set())
                   for lane, w in written.items())
        if dead and not mixed:
            for i in reversed(at):
                del out[i]
            self._say(f'carry: {name} taken from registers ahead of the '
                      f'loop, and not stored there')
            return forwarded, behind
        if dead and self._movable(out, at, forwarded, behind):
            moved = self._reread([out[i] for i in at], source.image,
                                 c.operand, c.image)
            for i in reversed(at):
                del out[i]
            # The store behind the loop writes nothing the moved one does not.
            spare = all(c.stored.get(lane, set()) <= set(written.get(lane, {}))
                        for lane in c.stored)
            kept = [x for x in behind if not (spare and id(x) in c.behind)]
            self._say(f'carry: {name} taken from registers ahead of the '
                      f'loop, and stored behind it instead')
            return forwarded, kept + moved
        self._say(f'carry: {name} taken from registers ahead of the loop')
        return forwarded, behind

    def _movable(self, out: List[Stmt], at: List[int], *others) -> bool:
        """Whether the statements of `out` at `at` can stand anywhere behind
        where they are: nothing else reads what they define."""
        defined = set(defined_within(tuple(out[i] for i in at)))
        rest = [x for i, x in enumerate(out) if i not in set(at)]
        for group in (rest, *others):
            for x in walk_stmts(tuple(group)):
                if any(v.id in defined for v in x.operands()):
                    return False
        return True

    def _writes(self, out: List[Stmt], units: List[_Unit], source: _Unit):
        """What the store ahead of the loop writes, per lane: `{lane:
        {element: slot}}` from `source`'s image; the elements it fills in
        with numbers, of any lane; whether a statement that stores from the
        image fills some in as well; and the guards spelled in text it
        stands under, which hold or not for every lane alike.  A value is a
        load of the image, a number, or a `select` between the two over a
        condition known per lane -- the padding of a store that writes its
        whole column."""
        written: Dict[int, Dict[int, int]] = {}
        filled: Set[int] = set()
        mixed = False
        owner: Dict[int, int] = {}
        guards = _spelled(e for u in units for e in u.events
                          if e.stmt.op is Op.STORE)
        for unit in units:
            stmts = tuple(out[i] for i in unit.at)
            defs = {t.id: x for x in walk_stmts(stmts) for t in x.target}
            producers = {t.id: e for e in unit.events
                         if e.stmt.op is Op.LOAD for t in e.stmt.target}
            for e in unit.events:
                if e.stmt.op is not Op.STORE:
                    continue
                value, index = _store_parts(e.stmt)
                form = self._form(value, defs, producers, source)
                operands = [index, e.stmt.predicate,
                            *(cond for cond, _ in e.conds)]
                loops = dict(e.loops)
                for leaf in _leaves(form):
                    if isinstance(leaf, _Event):
                        operands += [_load_index(leaf.stmt),
                                     *(cond for cond, _ in leaf.conds)]
                        loops = {**leaf.loops, **loops}
                    else:
                        operands.append(leaf)
                for lane, env, memo in self._domain(tuple(operands), loops):
                    if not self._active(e, env, memo, text=True):
                        continue
                    element = self._indices.evaluate(index, env, memo)
                    pick = form
                    if pick[0] == 'select':
                        pick = (pick[2] if self._indices.evaluate(
                            pick[1], env, memo) else pick[3])
                    if pick[0] == 'const':
                        filled.add(element)
                        mixed |= unit.kind == 'store'
                        continue
                    load = pick[1]
                    if not self._active(load, env, memo, text=True):
                        raise _Refused('the store ahead of the loop stores a '
                                       'value it did not read')
                    slot = self._indices.evaluate(_load_index(load.stmt), env,
                                                  memo)
                    if owner.setdefault(element, lane) != lane:
                        raise _Refused('two lanes store one element ahead of '
                                       'the loop')
                    if written.setdefault(lane, {}).setdefault(
                            element, slot) != slot:
                        raise _Refused('the store ahead of the loop writes an '
                                       'element twice')
        return written, filled, mixed, guards

    def _form(self, value, defs, producers, source: _Unit):
        """What a value stored ahead of the loop is: `('const',)`, `('image',
        load)` for a load of `source`'s image, or `('select', cond, a, b)`
        between two of those."""
        if self._constant(value):
            return ('const',)
        vid = getattr(value, 'id', None)
        load = producers.get(vid)
        if load is not None:
            if load.stmt.predicate is not None or not any(
                    _key(a.base) == _key(source.image)
                    for a in load.stmt.accesses):
                raise _Refused('the store ahead of the loop writes something '
                               'other than its image')
            return ('image', load)
        d = defs.get(vid)
        if d is not None and d.op == 'select' and d.predicate is None:
            cond, a, b = d.args
            forms = [self._form(x, defs, producers, source) for x in (a, b)]
            if any(f[0] == 'select' for f in forms):
                raise _Refused('a select between selects')
            return ('select', cond, *forms)
        raise _Refused('the store ahead of the loop writes something other '
                       'than its image')

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

    def _from_image(self, stmts: List[Stmt], load: Set[int], image,
                    buffer, operand, source) -> List[Stmt]:
        """`stmts` with the load of `buffer` into `image` -- its statements
        by id, `load` -- reading the slots of `source`, spelled `operand`,
        where it read the buffer.

        A slot is read where the image is written, at the index it is
        written at, and the read of the buffer goes: the index may be
        computed behind that read -- an unrolled copy computes the address it
        reads and then the slot it writes."""
        mine = [s for s in stmts if id(s) in load]
        loads = {t.id: s for s in walk_stmts(tuple(mine))
                 if s.op is Op.LOAD
                 and any(a.base is buffer for a in s.accesses)
                 for t in s.target}
        fills: Dict[int, Stmt] = {}
        uses: Dict[int, int] = {}
        for s in walk_stmts(tuple(stmts)):
            for v in s.operands():
                if v.id in loads:
                    uses[v.id] = uses.get(v.id, 0) + 1
            if s.op is Op.STORE and any(_key(a.base) == _key(image)
                                        for a in s.accesses):
                value, _ = _store_parts(s)
                if getattr(value, 'id', None) in loads:
                    fills[id(s)] = loads[value.id]
        if not loads:
            raise _Refused('no load of it to take from registers')
        filling = {x.target[0].id for x in fills.values()}
        for vid in loads:
            if vid not in filling:
                raise _Refused('a load of it fills no slot')
            if uses.get(vid) != 1:
                raise _Refused('a load of it is read elsewhere than into '
                               'the image')
        dropped = {id(x) for x in loads.values()}

        def rewrite(xs) -> List[Stmt]:
            out: List[Stmt] = []
            for s in xs:
                if id(s) in dropped:
                    continue
                if s.regions:
                    out.append(replace(s, regions=tuple(
                        replace(r, body=tuple(rewrite(r.body)))
                        for r in s.regions)))
                    continue
                read = fills.get(id(s))
                if read is not None:
                    _, slot = _store_parts(s)
                    out.append(replace(
                        read, args=(operand, slot),
                        accesses=(Access(Effect.READ, MemSpace.REGISTER,
                                         source),),
                        attrs=()))
                out.append(s)
            return out

        return [s for x in stmts
                for s in (rewrite((x,)) if id(x) in load else (x,))]

    # -- the load behind the loop -------------------------------------------- #

    def _forward_behind(self, out: List[Stmt], c: _Carried) -> List[Stmt]:
        """`out` with the first load of the buffer behind the loop taking
        what it reads from the carried image, where the store behind the
        loop has just stored it from there -- the code behind a run reading
        the accumulator it ends with.  Else `out` as it is."""
        name = getattr(c.access.base, 'name', str(c.access.base))
        try:
            out = self._forwarded_behind(out, c)
        except _Refused as why:
            self._say(f'carry: {name} is loaded from memory behind the loop: '
                      f'{why}')
        except _NotStatic as why:
            self._say(f'carry: {name} is loaded from memory behind the loop: '
                      f'an index or a condition not known per lane ({why})')
        return out

    def _forwarded_behind(self, out: List[Stmt], c: _Carried) -> List[Stmt]:
        stored = [i for i, x in enumerate(out) if id(x) in c.behind]
        if not stored:
            return out
        access = c.access
        probe = Access(Effect.READ, access.space, access.base)
        behind = _Trip(out)
        touching = [i for i in behind.touching(access) if i > max(stored)]
        if not touching:
            return out
        unit = self._units(behind, touching, access)[0]
        if unit.kind != 'load':
            return out
        first = min(unit.at)
        for i in range(max(stored) + 1, first):
            if any(a.writes and (_key(a.base) == _key(c.image)
                                 or may_alias(a, probe))
                   for a in collect_accesses((out[i],))):
                raise _Refused('something between writes the image or the '
                               'buffer')
        # Where a guard of the store behind the loop does not hold, memory
        # holds what it held ahead of the loop -- and so does the image the
        # trips copy into, where they copy under that guard and the load
        # ahead of the loop ran: the load reads the image where it runs only
        # under the store's guards, or under the load's of an image the
        # trips did not compute in place.
        # An element's mask the body keeps is taken to hold for the load as
        # well: where it does not, nothing the lane computes from what the
        # load reads is kept.
        masked = {g for g in c.guards if g[1] and g[0] in self._masks}
        for e in unit.events:
            under = _spelled((e,)) | masked
            if not (c.guards <= under or (not c.fused and c.loaded <= under)):
                raise _Refused('the store behind the loop stands under a '
                               'guard spelled in text the load does not')
        for lane, held in self._head_map([unit], text=True).items():
            for slot, element in held.items():
                if element is _ZERO:
                    continue
                if (element not in c.stored.get(lane, set())
                        or c.held.get(lane, {}).get(slot) != element):
                    raise _Refused('it reads what the store behind the loop '
                                   'does not write, or not from that slot')
        out = self._from_image(out, {id(out[i]) for i in unit.at}, unit.image,
                               access.base, c.operand, c.image)
        self._say(f'carry: {getattr(access.base, "name", str(access.base))} '
                  f'taken from registers behind the loop')
        return out

    # -- a buffer each lane keeps to itself ---------------------------------- #

    def _private(self, loop: Stmt, trip: '_Trip', touching: List[int],
                 access: Access, out: List[Stmt]):
        """`access`'s buffer in a register image of every lane's own over the
        trips: an element a lane writes in the trip, no other lane reads or
        writes there, and every element a lane reaches is one register of
        the image, whichever access reaches it -- the same register for
        every lane (`_reach`), one of the hops of the loops around the
        access (`_affine`).  Shared memory alone, of the multiplication's
        own arena (`_own`): no lane of another multiplication reaches it.

        What the trips read before they write it, the image takes ahead of
        the loop from the registers the code there stored it from (`_seed`);
        what they write goes to memory behind the loop, where anything
        behind it reads the buffer (`_read_behind`).  The stores ahead go
        where nothing reads what they wrote any more: the loop does not, and
        behind it the buffer is defined anew or written again."""
        stmts = trip.stmts
        base = access.base
        alloc = self._own[_key(base)]
        if id(alloc) in trip.declared:
            raise _Refused('it is declared in the trip')
        probe = Access(Effect.READ, access.space, base)
        nests = tuple(stmts[i] for i in touching)
        reached: List[Stmt] = []
        for x in walk_stmts(nests):
            mine = [a for a in x.accesses if may_alias(a, probe)]
            if not mine:
                continue
            if any(a.base is not base for a in mine):
                raise _Refused('it is reached through another binding')
            if x.op not in (Op.LOAD, Op.STORE) or any(
                    a.kind & Effect.ATOMIC for a in mine):
                raise _Refused(f'a {x.op} touches it')
            if len(x.args) != (2 if x.op is Op.LOAD else 3):
                raise _Refused('an access of more than one index')
            value = x.target[0] if x.op is Op.LOAD else x.args[1]
            if getattr(getattr(value, 'type', None), 'length',
                       None) is not None:
                raise _Refused('a vector access')
            reached.append(x)
        named = {x.args[0].id for x in reached
                 if isinstance(x.args[0], Value)}
        if isinstance(base, Value):
            named.add(base.id)
        if named & self._addresses:
            raise _Refused('its binding is handed on, to a table of pointers '
                           'or the like, and may reach it from anywhere')
        if trip.unsaid:
            raise _Refused('the trip holds a statement that says nothing '
                           'about what it touches')

        events = {id(e.stmt): e for e in _events(nests)
                  if any(a.base is base for a in e.stmt.accesses)}
        trace: List[Tuple[Stmt, Dict]] = []
        _trace(nests, {}, events, trace)
        steps, register = self._reach(trace, events)
        if not register:
            raise _Refused('no lane reaches it')
        size = max(register.values()) + 1
        fits = self._fits(trace, register, size, _unrolled(nests))
        exposed, written, own = _uses(steps, register)

        b = self._scratch()
        extern = Clone(self._fresh, 'carry', self._taken).rename(
            str(alloc.attr('extern', 'own')))
        image = b.alloc(base.type.elem, (size,), MemSpace.REGISTER,
                        hint='own', extern=extern, init='{}')
        declare = list(b.finish())

        units, seed, seeded = self._seed(out, access, own, exposed, image,
                                         size)
        read_behind = self._read_behind(base)
        behind: List[Stmt] = []
        if read_behind:
            behind = self._stored_behind(loop, stmts, touching, trace, steps,
                                         register, fits, image)

        inductions = _inductions(nests)
        inside = set(touching)
        body: List[Stmt] = []
        for i, s in enumerate(stmts):
            if i in inside:
                body.extend(self._privatized([s], fits, image, inductions))
            else:
                body.append(s)

        # The stores ahead of the loop: the loop reads nothing they wrote
        # any more, and behind it the buffer is defined anew, or written
        # again wherever they wrote it.
        at = sorted(i for u in units for i in u.at)
        name = getattr(base, 'name', str(base))
        again = all(elements <= set(written.get(lane, {}).values())
                    for lane, elements in seeded.items())
        if not units:
            pass
        elif (again or not read_behind) and self._movable(out, at):
            for i in reversed(at):
                del out[i]
            self._say(f'carry: {name} taken from registers ahead of the '
                      f'loop, and not stored there')
        else:
            self._say(f'carry: {name} taken from registers ahead of the loop')
        if not read_behind:
            self._say(f'carry: {name} not stored behind the loop: nothing '
                      f'behind it reads it')
        return declare + seed, body, behind, None

    def _kept(self, body: Tuple[Stmt, ...], alloc: Stmt) -> _Kept:
        """How `body` keeps the buffer `alloc` declares in a register image
        of every lane's own, over the innermost list around every statement
        that touches it -- or the loop around that list, where its trips
        hand the buffer on -- and without the buffer (`_keep`).
        Refused where a statement other than a load or a store of one
        element, through the buffer's own binding, touches it; where a lane
        reaches an element another lane writes (`_reach`); and where a lane
        reads an element there that it did not write there before."""
        base = alloc.target[0]
        key = _key(base)
        if base.id in self._addresses:
            raise _Refused('its binding is handed on, to a table of pointers '
                           'or the like, and may reach it from anywhere')
        touching: List[Tuple[int, ...]] = []
        marks: Set[int] = set()
        touched = self._touched
        reaches = touched.reaches

        def visit(xs, path):
            for i in touched.where(xs, key):
                x = xs[i]
                if x is alloc:
                    continue
                if x.regions:
                    for j, r in enumerate(x.regions):
                        visit(r.body, path + ((i, j),))
                    continue
                mine = [a for a in x.accesses
                        if a.base is not None and _key(a.base) == key]
                named = any(isinstance(v, Value) and v.id == base.id
                            for v in x.args)
                if not mine and not named:
                    continue
                if x.op is Op.MARK:
                    if x.attr('mark') not in ('defines', 'clears', 'cleared'):
                        raise _Refused(f"a mark {x.attr('mark')} of it")
                    marks.add(id(x))
                    continue
                if x.op not in (Op.LOAD, Op.STORE):
                    raise _Refused(f'a {x.op} touches it')
                if any(a.base is not base for a in mine):
                    raise _Refused('it is reached through another binding')
                if any(a.kind & Effect.ATOMIC for a in mine):
                    raise _Refused('an atomic update of it')
                if len(x.args) != (2 if x.op is Op.LOAD else 3):
                    raise _Refused('an access of more than one index')
                value = x.target[0] if x.op is Op.LOAD else x.args[1]
                if getattr(getattr(value, 'type', None), 'length',
                           None) is not None:
                    raise _Refused('a vector access')
                touching.append(path)

        visit(body, ())
        if not touching:
            raise _Refused('nothing reads or writes it')
        scope = touching[0]
        for path in touching[1:]:
            n = 0
            while n < min(len(scope), len(path)) and scope[n] == path[n]:
                n += 1
            scope = scope[:n]
        accessed = touched.accessed

        def where(xs):
            return touched.where(xs, key, accessed=True)

        while True:
            stmts = _region(body, scope)
            if touched.unsaid(stmts):
                raise _Refused('a statement around it says nothing about '
                               'what it touches')
            owner = _region_owner(body, scope)
            try:
                events = {id(e.stmt): e
                          for e in _events_of(stmts, key, where)}
                trace: List[Tuple[Stmt, Dict]] = []
                _trace_of(stmts, {}, events, _around(stmts, events, where),
                          trace, where)
                steps, register = self._reach(trace, events)
                exposed, _, _ = _uses(steps, register)
                if exposed:
                    raise _Refused('a lane reads what it did not write there '
                                   'before')
                break
            except (_Refused, _NotStatic):
                # The trips of a loop around them may hand it on, or say
                # what a hop of it is: asked over the loop.
                if (owner is None or owner.op is not Op.FOR
                        or not counted(owner)):
                    raise
                scope = scope[:-1]
        size = max(register.values()) + 1
        # Its loops, where it is: what is not around one of its accesses
        # has nothing to say about the registers they take.
        around = [stmts[i] for i in where(stmts)]
        fits = self._fits(trace, register, size, _unrolled(around))
        return _Kept(alloc, key, stmts, fits, size, _inductions(around),
                     marks | {id(alloc)})

    def _reach(self, trace: List[Tuple[Stmt, Dict]],
               events: Dict[int, _Event]):
        """What the accesses of `trace` reach, lane by lane: `(steps,
        register)`, with `steps` per lane the list of `(node, writes,
        element)` in the order the lane makes them -- a node an access at
        the hops of the loops around it -- and `register` a node's register:
        one per element of a lane, the same for every lane.  Refused where
        an element a lane writes another lane reaches, or where a node
        reaches two elements of a lane that two others reach apart."""
        parent: Dict = {}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        steps: Dict[int, List] = {}
        for lane in range(self._threads):
            first: Dict[int, object] = {}
            for x, env in trace:
                e = events[id(x)]
                full = {**env, 'x': lane}
                index = x.args[1] if x.op is Op.LOAD else x.args[2]
                self._known((index, x.predicate, *(c for c, _ in e.conds)),
                            full)
                memo: Dict = {}
                if not self._active(e, full, memo):
                    continue
                element = self._indices.evaluate(index, full, memo)
                node = (id(x), tuple(sorted(env.items())))
                parent.setdefault(node, node)
                if element in first:
                    one, other = find(first[element]), find(node)
                    if one != other:
                        parent[other] = one
                else:
                    first[element] = node
                steps.setdefault(lane, []).append(
                    (node, x.op is Op.STORE, element))
        touched: Dict[int, Set[int]] = {}
        written: Set[int] = set()
        for lane, seq in steps.items():
            held: Dict = {}
            for node, writes, element in seq:
                if held.setdefault(find(node), element) != element:
                    raise _Refused("a register would hold two of a lane's "
                                   "elements")
                touched.setdefault(element, set()).add(lane)
                if writes:
                    written.add(element)
        if any(len(touched[e]) > 1 for e in written):
            raise _Refused('a lane reaches an element another lane writes')
        slot: Dict = {}
        for x, env in trace:
            node = (id(x), tuple(sorted(env.items())))
            if node in parent:
                slot.setdefault(find(node), len(slot))
        return steps, {node: slot[find(node)] for node in parent}

    def _fits(self, trace: List[Tuple[Stmt, Dict]], register: Dict,
              size: int,
              unrolled: Set[int]) -> Dict[int, Tuple[int, Dict[int, int]]]:
        """Per access of `trace`, by id, its register as an affine function
        of the hops of the loops around it (`_affine`) -- one inside the
        image at every hop, whether a lane takes it or not, and of the hops
        of unrolled loops alone (`unrolled`, their inductions): a register
        the compiler cannot name ahead of time is one it keeps in memory."""
        points: Dict[int, List] = {}
        for x, env in trace:
            node = (id(x), tuple(sorted(env.items())))
            found = points.setdefault(id(x), [])
            if node in register:
                found.append((env, register[node]))
        fits = {sid: _affine(found) for sid, found in points.items()}
        for x, env in trace:
            c, coef = fits[id(x)]
            if set(coef) - unrolled:
                raise _Refused('a register picked by the hop of a loop that '
                               'is not unrolled')
            if not 0 <= c + sum(a * env[k] for k, a in coef.items()) < size:
                raise _Refused('a register outside the image')
        return fits

    def _seed(self, out: List[Stmt], access: Access, own, exposed, image,
              size: int):
        """`(units, seed, seeded)`: the stores at the end of `out` that the
        trips read first what they wrote -- every element a lane reads in
        the trip before it writes it there -- and `seed`, the same stores
        into the registers of `image` instead (`own`: per lane, element to
        register), behind them and ahead of the loop; `seeded` holds per
        lane the elements they write.  Refused where they do not write all
        of that, or write an element the trips do not reach."""
        if not exposed:
            return [], [], {}
        units = self._last_stores(out, access)
        if not units:
            raise _Refused('nothing ahead of the loop stores it from '
                           'registers')
        units.reverse()
        for u in units:
            if u.image is None:
                continue
            for i in range(min(u.at), len(out)):
                if i not in u.at and any(
                        a.writes and _key(a.base) == _key(u.image)
                        for a in collect_accesses((out[i],))):
                    raise _Refused('an image it is stored from ahead of the '
                                   'loop is written again there')
        base = access.base
        seeded: Dict[int, Set[int]] = {}
        seed: List[Stmt] = []
        for u in units:
            nests = tuple(out[i] for i in sorted(u.at))
            events = {id(e.stmt): e for e in u.events
                      if e.stmt.op is Op.STORE
                      and any(a.base is base for a in e.stmt.accesses)}
            trace: List[Tuple[Stmt, Dict]] = []
            _trace(nests, {}, events, trace)
            register: Dict = {}
            for x, env in trace:
                e = events[id(x)]
                node = (id(x), tuple(sorted(env.items())))
                for lane in range(self._threads):
                    full = {**env, 'x': lane}
                    index = x.args[2]
                    self._known((index, x.predicate,
                                 *(c for c, _ in e.conds)), full)
                    memo: Dict = {}
                    if not self._active(e, full, memo):
                        continue
                    element = self._indices.evaluate(index, full, memo)
                    r = own.get(lane, {}).get(element)
                    if r is None:
                        raise _Refused('the code ahead of the loop stores '
                                       'what the trips do not reach')
                    if register.setdefault(node, r) != r:
                        raise _Refused('the code ahead of the loop stores '
                                       'into registers that differ between '
                                       'lanes')
                    seeded.setdefault(lane, set()).add(element)
            fits = self._fits(trace, register, size, _unrolled(nests))
            seed += self._privatized(list(nests), fits, image,
                                     _inductions(nests))
        for lane, first in exposed.items():
            if not set(first.values()) <= seeded.get(lane, set()):
                raise _Refused('the trips read first what the code ahead of '
                               'the loop does not store from registers')
        seed = Clone(self._fresh, 'carry', self._taken).stmts(seed,
                                                              named=True)
        return units, seed, seeded

    def _privatized(self, stmts: Sequence[Stmt], fits, image,
                    inductions: Dict[int, Value]) -> List[Stmt]:
        """`stmts` with each access of `fits` an access of the register of
        `image` that its fit gives."""
        out: List[Stmt] = []
        for s in stmts:
            out += self._privatized_one(s, fits, image, inductions)
        return out

    def _privatized_one(self, s: Stmt, fits, image,
                        inductions: Dict[int, Value]) -> List[Stmt]:
        """`s` with each access of `fits` in it an access of the register of
        `image` that its fit gives: what computes the register, and the
        access."""
        if s.regions:
            return [replace(s, regions=tuple(
                replace(r, body=tuple(
                    y for x in r.body for y in self._privatized_one(
                        x, fits, image, inductions)))
                for r in s.regions))]
        fit = fits.get(id(s))
        if fit is None:
            return [s]
        b = self._scratch()
        slot = _slot(b, fit, inductions)
        out = list(b.finish())
        if s.op is Op.LOAD:
            out.append(replace(
                s, args=(image, slot),
                accesses=(Access(Effect.READ, MemSpace.REGISTER, image),),
                attrs=tuple((k, v) for k, v in s.attrs
                            if k in ('extern', 'other'))))
        else:
            out.append(replace(
                s, args=(image, s.args[1], slot),
                accesses=(Access(Effect.WRITE, MemSpace.REGISTER, image),),
                attrs=()))
        return out

    def _stored_behind(self, loop: Stmt, stmts: List[Stmt],
                       touching: List[int], trace, steps, register, fits,
                       image) -> List[Stmt]:
        """The stores of the trip again, behind the loop, each storing the
        register of `image` its fit gives in place of what it stored -- as
        few of them as write every register a lane writes."""
        writes: Dict[int, Set] = {}
        for lane, seq in steps.items():
            for node, wrote, _ in seq:
                if wrote:
                    writes.setdefault(node[0], set()).add(
                        (lane, register[node]))
        order: List[Stmt] = []
        for x, _ in trace:
            if x.op is Op.STORE and all(y is not x for y in order):
                order.append(x)
        covered: Set = set()
        kept: List[Stmt] = []
        for x in reversed(order):
            pairs = writes.get(id(x), set())
            if pairs <= covered:
                continue
            covered |= pairs
            kept.append(x)
        kept.reverse()
        where = {id(y): i for i in touching for y in walk_stmts((stmts[i],))}
        inside = set(defined_within(tuple(stmts)))
        inside |= {a.id for a in loop.regions[0].args}
        out: List[Stmt] = []
        for x in kept:
            top = stmts[where[id(x)]]
            cut = self._cut(top, x, fits[id(x)], image, _inductions((top,)))
            out += Clone(self._fresh, 'carry', self._taken).stmts(cut,
                                                                  named=True)
        _needs_only(tuple(out), inside, 'the store behind the loop needs what '
                    'the trip computes')
        return out

    def _cut(self, top: Stmt, x: Stmt, fit, image,
             inductions: Dict[int, Value]) -> List[Stmt]:
        """`top` cut down to the store `x` and what it needs -- the loops
        and branches around it, what computes their conditions and its index
        and predicate -- with `x` storing the register of `image` that `fit`
        gives instead of what it stored."""
        parent: Dict[int, Stmt] = {}

        def index(xs, owner):
            for y in xs:
                if owner is not None:
                    parent[id(y)] = owner
                for r in y.regions:
                    index(r.body, y)

        index((top,), None)
        defs = {t.id: y for y in walk_stmts((top,)) for t in y.target}
        need: Set[int] = set()
        work = [x]
        while work:
            y = work.pop()
            if id(y) in need:
                continue
            need.add(id(y))
            if id(y) in parent:
                work.append(parent[id(y)])
            operands = (tuple(y.args[2:]) + ((y.predicate,) if y.predicate
                                             is not None else ())
                        if y is x else y.operands())
            for v in operands:
                if isinstance(v, Value) and v.id in defs:
                    work.append(defs[v.id])
        b = self._scratch()
        value = b.load(image, _slot(b, fit, inductions), hint='own')
        loaded = list(b.finish())

        def keep(xs) -> List[Stmt]:
            kept: List[Stmt] = []
            for y in xs:
                if id(y) not in need:
                    continue
                if y is x:
                    kept += loaded
                    kept.append(replace(y, args=(y.args[0], value)
                                        + tuple(y.args[2:])))
                    continue
                if y.regions:
                    y = replace(y, regions=tuple(
                        replace(r, body=tuple(keep(r.body)))
                        for r in y.regions))
                kept.append(y)
            return kept

        return keep((top,))

    def _read_behind(self, base) -> bool:
        """Whether anything may read `base`, a buffer in shared memory,
        behind the loop being carried before something defines it anew
        (`_read_after`)."""
        return _read_after(self._root, self._here, _key(base))

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

    def _known(self, operands, env) -> None:
        """Refuses (`_NotStatic`) an operand that the lane and the hops of
        `env` do not decide -- a guard spelled in text among them."""
        for x in operands:
            if x is None:
                continue
            if isinstance(x, str):
                raise _NotStatic(x)
            if self._indices.leaves(x) - set(env):
                raise _NotStatic('a value neither a lane nor a hop decides')

    def _domain(self, operands, loops) -> Iterator[Tuple[int, Dict, Dict]]:
        """`(lane, env, memo)` for every lane and every combination of the
        inner loops' trips that `operands` reach -- a guard spelled in text
        is no operand here, and `_active` answers for it."""
        leaves: Set = set()
        for x in operands:
            if x is not None and not isinstance(x, str):
                leaves |= self._indices.leaves(x)
        leaves.discard('x')
        for lane in range(self._threads):
            for env in self._indices.domain(frozenset(leaves), loops):
                env = dict(env)
                env['x'] = lane
                yield lane, env, {}

    def _active(self, e: _Event, env, memo, text: bool = False) -> bool:
        """Whether `e` runs for the lane and the hops of `env`.  A guard
        spelled in text is not known per lane: with `text`, the caller has
        answered for it and it is taken to hold (`_guarded`), and else it is
        refused."""
        for cond, side in e.conds:
            if isinstance(cond, str):
                if text:
                    continue
                raise _NotStatic(cond)
            if bool(self._indices.evaluate(cond, env, memo)) != side:
                return False
        if e.stmt.predicate is not None and not self._indices.evaluate(
                e.stmt.predicate, env, memo):
            return False
        return True

    def _reread(self, stmts: Sequence[Stmt], was, operand,
                image) -> List[Stmt]:
        """`stmts` reading `image`, spelled `operand`, where they read the
        image `was`."""
        out = []
        for s in stmts:
            if s.regions:
                s = replace(s, regions=tuple(
                    replace(r, body=tuple(self._reread(r.body, was, operand,
                                                       image)))
                    for r in s.regions))
            elif s.op is Op.LOAD and any(_key(a.base) == _key(was)
                                         for a in s.accesses):
                s = replace(s, args=(operand,) + tuple(s.args[1:]),
                            accesses=(Access(Effect.READ, MemSpace.REGISTER,
                                             image),))
            out.append(s)
        return out

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
        if s.op is Op.MARK:
            # Names the buffers it says something about, and hands them to
            # nobody.
            continue
        args = s.args
        if s.op in (Op.LOAD, Op.STORE, Op.PREFETCH, Op.LOAD_ASYNC):
            args = args[1:]
        elif s.op is Op.COPY_ASYNC:
            args = args[2:]
        out.update(v.id for v in args if isinstance(v, Value))
        if s.predicate is not None:
            out.add(s.predicate.id)
    return out


def _spelled(events) -> frozenset:
    """The guards spelled in text the statements of `events` stand under,
    with the side of each."""
    return frozenset((cond, side) for e in events for cond, side in e.conds
                     if isinstance(cond, str))


def _leaves(form) -> List[object]:
    """The loads and conditions a value form of `_Carrier._form` reads."""
    if form[0] == 'image':
        return [form[1]]
    if form[0] == 'select':
        return [form[1], *_leaves(form[2]), *_leaves(form[3])]
    return []


def _own(body: Sequence[Stmt], arena: Optional[str]) -> Dict[tuple, Stmt]:
    """The buffers of `arena` in shared memory that a lane may keep to
    itself, by key, with their declarations: those that are buffers of
    their own -- one window, of one stage -- since an access names its
    window, and through another window of the buffer it would reach the
    same bytes unseen."""
    if arena is None:
        return {}
    allocs = [s for s in walk_stmts(tuple(body))
              if s.op is Op.ALLOC and s.target
              and isinstance(s.target[0].type, BufferType)
              and s.target[0].type.space is MemSpace.SHARED]
    windows: Dict[int, int] = {}
    for s in allocs:
        if s.attr('identity') is not None:
            key = id(s.attr('identity'))
            windows[key] = windows.get(key, 0) + 1
    out: Dict[tuple, Stmt] = {}
    for s in allocs:
        if s.attr('arena') != arena or (s.attr('stages') or 1) > 1:
            continue
        if s.attr('identity') is not None \
                and windows[id(s.attr('identity'))] > 1:
            continue
        out[_key(s.target[0])] = s
    return out


def _uses(steps, register):
    """Per lane, from the `steps` of `_Carrier._reach`: `(exposed, written,
    own)` -- the registers the trip reads before it writes them, with the
    element each holds; those it writes, likewise; and every element it
    reaches, with its register."""
    exposed: Dict[int, Dict[int, int]] = {}
    written: Dict[int, Dict[int, int]] = {}
    own: Dict[int, Dict[int, int]] = {}
    for lane, seq in steps.items():
        done: Set[int] = set()
        for node, writes, element in seq:
            r = register[node]
            own.setdefault(lane, {})[element] = r
            if writes:
                done.add(r)
                written.setdefault(lane, {})[r] = element
            elif r not in done:
                exposed.setdefault(lane, {}).setdefault(r, element)
    return exposed, written, own


def _trace(xs: Sequence[Stmt], env: Dict, events: Dict[int, _Event],
           out: List[Tuple[Stmt, Dict]]) -> None:
    """The statements of `events` among `xs`, each with the hops of the
    loops around it, in the order a lane makes them -- through every
    branch, which `_Carrier._active` decides per lane."""
    for x in xs:
        if x.op is Op.FOR:
            if not counted(x) or x.loop_inits:
                raise _NotStatic('a loop of an unknown count')
            for trip in range(*x.loop_bounds):
                _trace(x.regions[0].body, {**env, x.induction.id: trip},
                       events, out)
        elif x.regions:
            for r in x.regions:
                _trace(r.body, env, events, out)
        elif id(x) in events:
            out.append((x, env))


def _affine(points) -> Tuple[int, Dict[int, int]]:
    """`(c, {induction: a})` such that `c` plus the sum of `a` times the hop
    of each induction is the register of every point `(hops, register)` --
    refused where the registers are no such function of the hops."""
    if not points:
        return 0, {}
    keys = sorted(points[0][0])
    env0, r0 = points[0]
    coef: Dict[int, int] = {}
    for k in keys:
        coef[k] = 0
        for env, r in points[1:]:
            if env[k] != env0[k] and all(env[j] == env0[j]
                                         for j in keys if j != k):
                step = env[k] - env0[k]
                if (r - r0) % step:
                    raise _Refused('registers that are not affine in the '
                                   'hops of the loops around the access')
                coef[k] = (r - r0) // step
                break
    c = r0 - sum(coef[k] * env0[k] for k in keys)
    for env, r in points:
        if c + sum(coef[k] * env[k] for k in keys) != r:
            raise _Refused('registers that are not affine in the hops of the '
                           'loops around the access')
    return c, {k: a for k, a in coef.items() if a}


def _slot(b, fit, inductions: Dict[int, Value]):
    """The register `fit` gives, as an operand: a number, or arithmetic over
    the inductions built with `b`."""
    c, coef = fit
    terms = []
    for k, a in sorted(coef.items()):
        induction = inductions[k]
        terms.append(induction if a == 1 else
                     b.op('mul', induction.type, induction, a, hint='slot'))
    if not terms:
        return c
    out = terms[0]
    for t in terms[1:]:
        out = b.op('add', out.type, out, t, hint='slot')
    if c:
        out = b.op('add', out.type, out, c, hint='slot')
    return out


def _inductions(stmts: Sequence[Stmt]) -> Dict[int, Value]:
    """The inductions of the loops among `stmts`, by id."""
    return {x.induction.id: x.induction for x in walk_stmts(tuple(stmts))
            if x.op is Op.FOR}


def _unrolled(stmts: Sequence[Stmt]) -> Set[int]:
    """The inductions of the unrolled loops among `stmts`, by id."""
    return {x.induction.id for x in walk_stmts(tuple(stmts))
            if x.op is Op.FOR and x.attr('unroll')}


def _read_after(root: Sequence[Stmt], target: Optional[int], key) -> bool:
    """Whether anything may read the buffer in shared memory that `key`
    names behind the statement `target` (by id) of `root`, before a
    `mark defines` of it says it holds nothing anybody wants (`allocate`).

    Behind it in its list, then behind each statement around it -- and
    around every back edge of a loop it stands in, from the start of the
    loop's body down to `target` again, which reached again counts as a
    read.  Past the end of the body nothing reads shared memory.  Which
    branch a lane takes is not asked: a read in either is a read, and a
    `mark defines` in a branch or a loop is none."""
    path = _path(tuple(root), target) if target is not None else None
    if path is None:
        return True
    for level in range(len(path) - 1, -1, -1):
        stmts, i = path[level]
        first = _decides(stmts[i + 1:], key)
        if first is not None:
            return first
        if level == 0:
            return False
        holder = path[level - 1][0][path[level - 1][1]]
        if _repeats(holder) and _again(path, level, key):
            return True
    return False


def _again(path, level: int, key) -> bool:
    """Around the back edge of the loop whose body is `path[level]`'s list:
    whether the way from the start of that body down `path` to its last
    statement reads the buffer `key` names before it defines it anew --
    reaching that statement again counts as reading it."""
    for stmts, i in path[level:]:
        first = _decides(stmts[:i], key)
        if first is not None:
            return first
    return True


def _path(stmts: Tuple[Stmt, ...], target: int):
    """`[(list, index)]` from `stmts` down to the statement `target`."""
    for i, s in enumerate(stmts):
        if id(s) == target:
            return [(stmts, i)]
        for r in s.regions:
            found = _path(tuple(r.body), target)
            if found is not None:
                return [(stmts, i)] + found
    return None


def _decides(stmts: Sequence[Stmt], key) -> Optional[bool]:
    """True where the first of `stmts` that decides about the buffer `key`
    names reads it, False where it defines it anew, None where none does.
    A statement that says nothing about what it touches reads it."""
    for s in stmts:
        if s.op is Op.MARK and s.attr('mark') == 'defines' and any(
                isinstance(a, Value) and _key(a) == key for a in s.args):
            return False
        if s.op is Op.ALLOC and s.target and _key(s.target[0]) == key:
            return False
        for x in walk_stmts((s,)):
            if x.effect & Effect.UNKNOWN:
                return True
            for a in x.accesses:
                if a.space is MemSpace.UNKNOWN:
                    return True
                if (a.space is MemSpace.SHARED and a.kind & Effect.READ
                        and (a.base is None or _key(a.base) == key)):
                    return True
    return None


def _repeats(s: Stmt) -> bool:
    """Whether `s` may run what it holds more than once: a loop, or a raw
    block whose head is one."""
    if s.op in (Op.FOR, Op.WHILE):
        return True
    if s.op is not Op.RAWBLOCK:
        return False
    head = (s.text or '').lstrip()
    return bool(re.match(r'(?:#pragma[^\n]*\n\s*)?(?:for|while)\b', head))


def _region(body: Sequence[Stmt], path) -> Tuple[Stmt, ...]:
    """The statement list `path` leads to: a pair per level, the statement
    and the region of it."""
    xs = tuple(body)
    for i, j in path:
        xs = xs[i].regions[j].body
    return xs


def _region_owner(body: Sequence[Stmt], path) -> Optional[Stmt]:
    """The statement whose region `path` leads to, or None at the top."""
    if not path:
        return None
    return _region(body, path[:-1])[path[-1][0]]


def _touches(x: Stmt, key) -> bool:
    return any(a.base is not None and _key(a.base) == key for a in x.accesses)


def _events_of(stmts: Sequence[Stmt], key, where=None) -> List[_Event]:
    """The loads and stores in `stmts` that touch the buffer `key` names,
    with what they stand under, as `_events` has them.  What does not hold
    one of them is not asked: a loop of its own that carries values, or a
    loop the trips of which nothing counts.  `where` gives the positions in
    a list of the statements that access the buffer, where known
    (`_Touched.where`)."""
    out: List[_Event] = []

    def visit(xs, conds, loops):
        for x in ((xs[i] for i in where(xs)) if where is not None else xs):
            if x.regions:
                if where is None and not any(
                        _touches(y, key) for y in walk_stmts((x,))):
                    continue
                if x.op is Op.FOR:
                    if not counted(x):
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
                    visit(x.regions[0].body, conds + ((x.text, True),),
                          loops)
                else:
                    raise _NotStatic(f'a {x.op} whose runs are not known')
            elif x.op in (Op.LOAD, Op.STORE) and _touches(x, key):
                out.append(_Event(x, conds, loops))

    visit(stmts, (), {})
    return out


def _around(stmts: Sequence[Stmt], events: Dict[int, _Event],
            where=None) -> Set[int]:
    """The statements of `stmts` that are among `events` or hold one, by
    id; with `where` (`_Touched.where`), looking only at the positions of a
    list that access the buffer of the events."""
    out: Set[int] = set()

    def visit(xs) -> bool:
        found = False
        for x in ((xs[i] for i in where(xs)) if where is not None else xs):
            here = id(x) in events
            for r in x.regions:
                here = visit(r.body) or here
            if here:
                out.add(id(x))
                found = True
        return found

    visit(stmts)
    return out


def _trace_of(xs: Sequence[Stmt], env: Dict, events: Dict[int, _Event],
              around: Set[int], out: List[Tuple[Stmt, Dict]],
              where=None) -> None:
    """`_trace` over the statements of `around` alone; `where`, as for
    `_around`, gives the positions of a list to look at."""
    for x in ((xs[i] for i in where(xs)) if where is not None else xs):
        if id(x) not in around:
            continue
        if x.op is Op.FOR:
            for trip in range(*x.loop_bounds):
                _trace_of(x.regions[0].body, {**env, x.induction.id: trip},
                          events, around, out, where)
        elif x.regions:
            for r in x.regions:
                _trace_of(r.body, env, events, around, out, where)
        elif id(x) in events:
            out.append((x, env))


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
