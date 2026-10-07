# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: where the threads of a multiplication have to meet.

Shared memory is where lanes hand values to each other, and a barrier is what
makes the hand-over safe.  Where one is needed is a question about the *final*
order of the accesses, so it is answered once the body is in that order --
behind every pass that moves a statement -- and for the body as a whole.
Every pass in front reasons about one thread, and may move an access past a
point where another lane writes, because no barrier stands there yet; the
barriers then go where the order those passes arrived at needs them.

The accesses name a buffer and no index, so the rules are about buffers, and
they assume the worst about lanes -- what one lane writes, another reads:

* **read after write**: a read of a buffer written since the last barrier.
  An asynchronous copy writes its buffer at the wait that retires it, and the
  wait makes it visible to the issuing lane only, so the wait is a write too,
  and a barrier between the issue and the wait fences nothing of it.
* **write after read**: a write over memory read since the last barrier, the
  buffer's own or another's that the allocation placed on the same bytes.
  The wait is no such write: whatever reads the buffer between the issue and
  the wait reads a copy in flight, and no barrier in front of the wait would
  change that.
* **a store over a clearing store**, and a clearing store over a store: the
  zeros an assignment owes go out on the clearing nest's lanes, and a store
  into the same buffer on either side of them writes some of those cells from
  other lanes (`mark clears` in front of the clearing store, `mark cleared`
  behind it).  Two stores are otherwise no reason: the cells two stores of
  one buffer share are the cells something reads in between, and that read
  is the reason -- but zeros nothing computed are read as what the buffer
  holds, and only the order of the writes says they are.
* **a one-lane store**: a value without axes, stored by its owner lane and
  read back by all of them (`handoff`).  The barrier then also owes the
  compiler a fence where the rendezvous costs no instruction
  (`Lexic.handoff_fence`), which the emitter supplies for a barrier marked so.

A barrier goes in front of the first statement that needs it, at the
innermost level of the structure where every thread it waits for arrives:
that level is the entry uniformity (`passes._entry_uniformity`), and a block
whose head the IR cannot read is entered per lane unless it says otherwise.
A hazard between what happened before a construct and the first thing inside
it is met in front of the construct; one around a loop's back edge -- what
the end of an iteration leaves for the beginning of the next -- inside the
body, from a fixed point over it.

Barriers already in the body stay where they are and count: those an
instruction places around its own staging, the block barrier a preload ends
in, the one a persistent loop ends in.  A barrier spelled for one
multiplication meets that multiplication only, so what the other
multiplications of the block wrote to the block-wide arena is fenced by a
block barrier alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional,
                    Sequence, Tuple)

from .core import (BufferType, Effect, MemSpace, Op, Stmt, Uniformity, Value,
                   walk_stmts)

#: The two kinds of shared memory a window is part of: one copy per
#: multiplication, and one for the whole block.
MULT = 'mult'
BLOCK = 'block'

#: Where nothing says: anywhere, in any arena.
_ANYWHERE = (None, None, None)

#: The buffer a statement touches that does not say which: any of them.
_OPAQUE = ('opaque',)


@dataclass(frozen=True)
class Arena:
    """Where the windows of a body sit.

    ``windows`` maps an arena name -- the `arena` an `Op.ALLOC` names -- to the
    shared memory it is part of and where it starts there, in elements.  A
    window into an arena not listed may overlap anything.
    """
    windows: Mapping[str, Tuple[str, int]]

    def extents(self, body: Sequence[Stmt]) -> Dict[int, Tuple]:
        """value id -> ``(root, lo, hi)`` for every shared buffer allocated
        in `body`; ``lo`` and ``hi`` are None where the offset is no number."""
        out: Dict[int, Tuple] = {}
        for s in walk_stmts(body):
            if s.op != Op.ALLOC or not s.target:
                continue
            v = s.target[0]
            if getattr(v.type, 'space', None) != MemSpace.SHARED:
                continue
            root, base = self.windows.get(s.attr('arena'), (None, 0))
            offset = s.attr('offset', 0)
            span = v.type.volume
            if isinstance(offset, int):
                out[v.id] = (root, base + offset, base + offset + span)
                continue
            # A rotating window, `576 + (stage % 2) * 96`: somewhere in its
            # two stages, from the leading constant on.
            m = re.match(r'\s*(\d+)\s*\+', str(offset))
            out[v.id] = ((root, base + int(m.group(1)),
                          base + int(m.group(1)) + 2 * span)
                         if m else (root, None, None))
        return out


@dataclass
class _State:
    """What is outstanding since the last barrier.

    ``writes``: buffer key -> (whether one lane wrote it, its root);
    ``cleared``: buffer key -> root, for what a clearing store wrote;
    ``reads``: buffer key -> where the buffer sits.
    """

    writes: Dict[Any, Tuple[bool, str]]
    cleared: Dict[Any, str]
    reads: Dict[Any, Tuple]

    @staticmethod
    def empty() -> '_State':
        return _State({}, {}, {})

    def copy(self) -> '_State':
        return _State(dict(self.writes), dict(self.cleared), dict(self.reads))

    def is_empty(self) -> bool:
        return not (self.writes or self.cleared or self.reads)

    def merge(self, other: '_State') -> '_State':
        writes = dict(self.writes)
        for k, (handoff, root) in other.writes.items():
            mine = writes.get(k)
            writes[k] = (handoff or (mine is not None and mine[0]), root)
        cleared = dict(self.cleared)
        cleared.update(other.cleared)
        reads = dict(self.reads)
        reads.update(other.reads)
        return _State(writes, cleared, reads)

    def covers(self, other: '_State') -> bool:
        return (all(k in self.writes and (self.writes[k][0] or not h)
                    for k, (h, _) in other.writes.items())
                and set(other.cleared) <= set(self.cleared)
                and set(other.reads) <= set(self.reads))

    def fence(self, scope: Uniformity) -> '_State':
        """What is still outstanding past a barrier of `scope`: a block-wide
        one settles everything, a narrower one what its multiplication did to
        its own memory."""
        if scope >= Uniformity.BLOCK:
            return _State.empty()
        return _State({k: v for k, v in self.writes.items() if v[1] == BLOCK},
                      {k: v for k, v in self.cleared.items() if v == BLOCK},
                      {k: v for k, v in self.reads.items() if v[0] == BLOCK})


def _scope_of(s: Stmt) -> Uniformity:
    scope = s.attr('scope')
    return scope if isinstance(scope, Uniformity) else Uniformity.BLOCK


def _overlap(a: Tuple, b: Tuple) -> bool:
    ra, la, ha = a
    rb, lb, hb = b
    if ra is not None and rb is not None and ra != rb:
        return False
    if None in (la, ha, lb, hb):
        return True
    return la < hb and lb < ha


def _binds(s: Stmt) -> bool:
    """Does `s` compute an address rather than touch memory?

    A window or a pointer is declared with an access to what it is a view of,
    so that no pass moves a write through the underlying buffer past the
    binding.  The binding itself reads nothing a lane could have written.
    """
    return any(isinstance(t.type, BufferType) for t in s.target)


def _loops(s: Stmt) -> bool:
    """Is `s` a loop -- a counted or queried one, or a raw block whose head
    iterates?"""
    if s.op in (Op.FOR, Op.WHILE):
        return True
    if s.op != Op.RAWBLOCK:
        return False
    head = (s.text or '').lstrip()
    return bool(re.match(r'(?:#pragma[^\n]*\n\s*)?(?:for|while)\b', head))


def _may_skip(s: Stmt) -> bool:
    """Can control pass `s` without entering any of its regions?"""
    return _loops(s) or s.op == Op.RAWBLOCK or (
        s.op == Op.IF and len(s.regions) == 1)


class _Pass:
    def __init__(self, extents, identity, handoff, make_barrier, arrival):
        self.extents = extents
        self.identity = identity
        self.handoff = handoff
        self.make_barrier = make_barrier
        self.arrival = arrival
        #: Hazards inside a block no barrier may be placed in, which nothing
        #: inside it fences either -- counted rather than raised: the pass has
        #: nothing to add there, and whether the block's author meant its
        #: lanes to meet some other way is not something it can see.
        self.unfenced = 0

    # -- what a statement does ------------------------------------------- #

    def key(self, base) -> Any:
        if isinstance(base, Value):
            return self.identity.get(base.id, ('v', base.id))
        return ('o', id(base))

    def extent(self, base) -> Tuple:
        if isinstance(base, Value):
            return self.extents.get(base.id, _ANYWHERE)
        return _ANYWHERE

    def accesses(self, s: Stmt):
        """`(reads, writes)` of a statement without regions, as
        `(key, extent, root)`, over shared memory and one-lane stores.

        A statement that does not say what it touches reads and writes any
        buffer of the multiplication."""
        reads, writes = [], []
        if s.op in (Op.ALLOC, Op.MARK) or _binds(s):
            return reads, writes
        for a in s.accesses:
            if a.space == MemSpace.UNKNOWN or (a.space == MemSpace.SHARED
                                               and a.base is None):
                reads.append((_OPAQUE, _ANYWHERE, MULT))
                writes.append((_OPAQUE, _ANYWHERE, MULT))
                continue
            if a.base is None:
                continue
            key = self.key(a.base)
            if a.space != MemSpace.SHARED and key not in self.handoff:
                continue
            ext = self.extent(a.base)
            entry = (key, ext, ext[0] or MULT)
            if a.kind & (Effect.WRITE | Effect.ATOMIC):
                writes.append(entry)
            elif a.kind & Effect.READ:
                reads.append(entry)
        return reads, writes

    def conflict(self, s: Stmt, state: _State) -> Tuple[bool, bool, bool]:
        """`(needed, handoff, block)` for a statement without regions."""
        if s.op == Op.MARK:
            return self._clears(s, state)
        reads, writes = self.accesses(s)
        needed = handoff = block = False
        for key, _, root in reads:
            if key == _OPAQUE:
                hits = list(state.writes.values())
            else:
                hits = [h for h in (state.writes.get(key),
                                    state.writes.get(_OPAQUE))
                        if h is not None]
            for hit in hits:
                needed = True
                handoff = handoff or hit[0]
                block = block or hit[1] == BLOCK
        if s.op == Op.WAIT:
            return needed, handoff, block
        for key, ext, root in writes:
            if key == _OPAQUE:
                cleared = list(state.cleared.values())
            else:
                cleared = [state.cleared[key]] if key in state.cleared else []
            for where in cleared:
                needed = True
                block = block or where == BLOCK
            for rext in state.reads.values():
                if _overlap(ext, rext):
                    needed = True
                    block = block or root == BLOCK
        return needed, handoff, block

    def _clears(self, s: Stmt, state: _State) -> Tuple[bool, bool, bool]:
        """A clearing store waits for every store into its buffer since the
        last barrier."""
        needed = block = False
        if s.attr('mark') == 'clears':
            for a in s.args:
                if not isinstance(a, Value):
                    continue
                for hit in (state.writes.get(self.key(a)),
                            state.writes.get(_OPAQUE)):
                    if hit is not None:
                        needed = True
                        block = block or hit[1] == BLOCK
        return needed, False, block

    def transfer(self, s: Stmt, state: _State) -> _State:
        """`state` past a statement without regions."""
        if s.effect & Effect.BARRIER:
            return state.fence(_scope_of(s))
        if s.op == Op.MARK:
            if s.attr('mark') != 'cleared':
                return state
            out = state.copy()
            for a in s.args:
                if isinstance(a, Value):
                    out.cleared[self.key(a)] = self.extent(a)[0] or MULT
            return out
        reads, writes = self.accesses(s)
        if not reads and not writes:
            return state
        out = state.copy()
        for key, ext, root in writes:
            out.writes[key] = (key in self.handoff, root)
        for key, ext, root in reads:
            out.reads[key] = ext
        return out

    def barrier(self, handoff: bool, block: bool, state: _State):
        return (self.make_barrier(handoff, block),
                state.fence(Uniformity.BLOCK if block else self.arrival))

    # -- the hazards a construct meets on the way in ---------------------- #

    def entry_conflict(self, s: Stmt, state: _State):
        """Whether something inside `s` meets what was outstanding before it,
        ahead of any barrier inside that settles it: `(needed, handoff,
        block)`.  Only what came before counts -- a hazard among the
        statements inside is the inside's to meet."""
        found = [False, False, False]

        def visit(body, st) -> Optional[_State]:
            for inner in body:
                if inner.regions:
                    outs = [visit(r.body, st.copy()) for r in inner.regions]
                    outs = [o for o in outs if o is not None]
                    if _may_skip(inner):
                        outs.append(st)
                    if not outs:
                        return None
                    st = outs[0]
                    for o in outs[1:]:
                        st = st.merge(o)
                    continue
                if inner.effect & Effect.BARRIER:
                    st = st.fence(_scope_of(inner))
                    if st.is_empty():
                        return None
                    continue
                n, h, b = self.conflict(inner, st)
                if n:
                    found[0] = True
                    found[1] = found[1] or h
                    found[2] = found[2] or b
            return st

        for r in s.regions:
            visit(r.body, state.copy())
        return tuple(found)

    # -- a construct nobody may place a barrier in ------------------------ #

    def summary(self, s: Stmt, state: _State) -> _State:
        """`state` past `s`, in order and with a loop's body walked twice, so
        that what it carries around its back edge is in the answer."""
        def visit(body, st):
            for inner in body:
                if inner.regions:
                    outs = []
                    for r in inner.regions:
                        o = visit(r.body, st.copy())
                        if _loops(inner):
                            o = visit(r.body, o.merge(st))
                        outs.append(o)
                    merged = outs[0]
                    for o in outs[1:]:
                        merged = merged.merge(o)
                    st = merged.merge(st) if _may_skip(inner) else merged
                    continue
                if not inner.effect & Effect.BARRIER:
                    if self.conflict(inner, st)[0]:
                        self.unfenced += 1
                st = self.transfer(inner, st)
            return st
        return visit((s,), state)

    # -- placement -------------------------------------------------------- #

    def place(self, body: Tuple[Stmt, ...], state: _State,
              level: Uniformity) -> Tuple[Tuple[Stmt, ...], _State]:
        from .passes import _entry_uniformity
        out: List[Stmt] = []
        for s in body:
            if s.op == Op.YIELD:
                out.append(s)
                continue
            if not s.regions:
                if not s.effect & Effect.BARRIER:
                    needed, handoff, block = self.conflict(s, state)
                    if needed:
                        stmt, state = self.barrier(handoff, block, state)
                        out.append(stmt)
                state = self.transfer(s, state)
                out.append(s)
                continue
            needed, handoff, block = self.entry_conflict(s, state)
            inner = min(level, _entry_uniformity(s))
            if inner < self.arrival:
                if needed:
                    stmt, state = self.barrier(handoff, block, state)
                    out.append(stmt)
                state = self.summary(s, state)
                out.append(s)
                continue
            if not needed:
                s, state = self.place_region(s, state, inner)
                out.append(s)
                continue
            # Met inside, by a barrier the construct needs there anyway --
            # what an iteration leaves for the next one is fenced at the same
            # place the entry is -- or in front of it, once, where meeting it
            # inside would take a barrier of its own.
            inside, after = self.place_region(s, state, inner)
            stmt, fenced = self.barrier(handoff, block, state)
            outside, behind = self.place_region(s, fenced, inner)
            if _count(inside) <= _count(outside):
                out.append(inside)
                state = after
                continue
            out.extend((stmt, outside))
            state = behind
        return tuple(out), state

    def place_region(self, s: Stmt, state: _State,
                     level: Uniformity) -> Tuple[Stmt, _State]:
        regions = []
        ends: List[_State] = []
        for r in s.regions:
            if not _loops(s):
                body, end = self.place(r.body, state.copy(), level)
                regions.append(replace(r, body=body))
                ends.append(end)
                continue
            # Around the back edge: what an iteration leaves outstanding at
            # its end is outstanding at the beginning of the next one.
            carried = _State.empty()
            for _ in range(8):
                body, end = self.place(r.body, state.merge(carried), level)
                if carried.covers(end):
                    break
                carried = carried.merge(end)
            regions.append(replace(r, body=body))
            ends.append(end)
        merged = ends[0]
        for e in ends[1:]:
            merged = merged.merge(e)
        if _may_skip(s):
            merged = merged.merge(state)
        return replace(s, regions=tuple(regions)), merged


def _count(s: Stmt) -> int:
    """Barriers in the regions of `s`."""
    inside = tuple(x for r in s.regions for x in r.body)
    return sum(1 for inner in walk_stmts(inside)
               if inner.effect & Effect.BARRIER)


def place_barriers(body: Tuple[Stmt, ...], *, arena: Arena,
                   make_barrier: Callable[[bool, bool], Stmt],
                   arrival: Uniformity,
                   handoff: FrozenSet = frozenset(),
                   report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with a barrier wherever the threads of a multiplication have to
    meet before an access.

    ``make_barrier(handoff, block)`` builds one: whether it also carries a
    one-lane store across, and whether it has to reach the whole block.
    ``arrival`` is who a multiplication's barrier waits for, which is how
    uniformly a block has to be entered for one to sit in it.  ``handoff``
    are the buffers one lane stores and all of them read, by identity.
    """
    identity: Dict[int, Any] = {}
    for s in walk_stmts(body):
        if s.op == Op.ALLOC and s.target and s.attr('identity') is not None:
            identity[s.target[0].id] = ('o', id(s.attr('identity')))
    work = _Pass(arena.extents(body), identity,
                 {('o', id(h)) for h in handoff}, make_barrier, arrival)
    out, _ = work.place(tuple(body), _State.empty(), Uniformity.GRID)
    if report is not None and work.unfenced:
        report.append(f'{work.unfenced} access(es) to shared memory meet an '
                      f'earlier one inside a block no barrier may be placed '
                      f'in, and nothing there fences them')
    return out
