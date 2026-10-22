# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: which store every load of a body reads, lane by lane.

A body is a program every lane of a multiplication runs, and what a load
reads is decided by the stores before it -- of the same lane, for a register
array, and of any lane, for shared or global memory.  `Flow` runs the body
once over the lanes of one multiplication, statement by statement, and keeps
for every element of every buffer the store that wrote it last.  What comes
out is the relation the passes want and the IR does not state: for every
execution of a load (`Inst`), the store it reads (`Flow.reach`); for every
execution of a store, who reads what it wrote (`Flow.readers`), whether a
later store replaces it unread, and whether it is what the body leaves
behind (`Flow.final`).

Executions, not statements.  A statement in a counted loop runs once a trip,
and its trips read and write different elements: the loop is run trip by
trip, with the induction as a number (`Inst.hops`).  A loop whose bounds are
not numbers -- the batch loop -- is run once, as one element's iteration;
what an iteration hands the next through registers or shared memory is
added afterwards (`_wrap`).  An address is evaluated for every lane from
what defines it: thread indices, inductions and constants (`shards._Indices`).
An access whose address is anything else is placed nowhere, and it reaches
every element of its buffer: a read of all of them, a write that may have
replaced any of them.

Conditions come in two kinds.  One that is a function of the lane -- the
rows a multiplication's lanes hold -- is evaluated, and only the lanes it
holds for run what it guards.  Any other one -- an element's flag, a guard
spelled in text -- is the same value wherever the body tests it, and is
carried as a named condition (`Inst.conds`): a store under it is certain to
a load under it as well and possible to any other.  A load whose element a
store may or may not have written reads either; such a load reads nothing
for certain, and both stores stay read.

An element's mask is a condition of the second kind with a contract of its
own (`elementmask`): where it does not hold, the element is past the batch
or masked off, its lanes compute what nobody keeps, and every write that
could keep it -- to global memory -- stands under the mask.  Asked with
`masks`, a mask is taken to hold, as it does for every element anything is
kept of -- where the body keeps the contract (`_masks`): its global writes
stand under one, what it writes to shared memory others read is no value
of the element, and no branch it takes turns on what memory holds.  A
statement that emits nothing and only says what others reach through a name
the IR does not follow (`IRBuilder.touches`) writes nothing of its own; the
statements it speaks for are asked where they stand.

Asked with `values`, a store's execution records where its value came from
(`Flow.origin`), and a load's what it may read instead of its element
(`Flow.forward`): the value the lane's own store put there, a number, or a
register a load read it from and that still holds it -- and for a load of
memory, the register an earlier load by the same lanes put the element in,
where both found the same write there.  A register array declared zeroed
(`{}`) holds zeros until something stores into it, as a store of the number
would leave it.

What `Flow` keeps grows with the elements it places, lane by lane and
execution by execution -- with `values`, some 300 bytes each.  A body that
places more than `BUDGET` is not run (`TooLarge`), and a pass that asks
leaves it as it is.

Nothing here changes the body.  A pass asks, and decides what it may do
with the answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

from .core import Effect, MemSpace, Op, Stmt, Value, walk_stmts
from .shards import _Indices, _NotStatic

#: What a cell held before the body wrote it.
INITIAL = -1

#: A write by a lane of another multiplication, to shared memory the lanes
#: of more than one share: possible to every read of it.
EXTERNAL = -2

#: The spaces whose elements are placed: registers, which every lane has its
#: own of, and the memory the lanes of a multiplication share.
_PLACED = (MemSpace.REGISTER, MemSpace.SHARED, MemSpace.GLOBAL)

#: How many writes of uncertain standing a cell keeps apart before they are
#: taken as one.
_DEPTH = 8

#: The elements one body may place, summed over its executions and lanes,
#: before `Flow` gives up on it.  The largest body of the corpus places
#: 263 000 -- the merged derivative at order 6 in double, on sm_86, which
#: the flow holds in 92 MB -- and SeisSol's damage step written out at
#: order 5, every temporary in memory, 1.2 million.
BUDGET = 1_000_000


class TooLarge(Exception):
    """The body places more elements than `Flow` keeps track of (`BUDGET`)."""

#: The one empty set of writers every record without any shares: a
#: `frozenset()` of its own is 216 bytes, and a body reads a few hundred
#: thousand elements nothing in it wrote.
EMPTY: FrozenSet[int] = frozenset()


#: The flags as numbers: tested on every access, where the flag arithmetic
#: of `Effect` is the cost of the test.
_READ = int(Effect.READ)
_WRITTEN = int(Effect.WRITE | Effect.ATOMIC)
_READ_OR_ATOMIC = int(Effect.READ | Effect.ATOMIC)


@dataclass(frozen=True, slots=True)
class Inst:
    """One execution of `stmt`: at `hops`, the inductions of the counted
    loops around it, outermost first, for the lanes in `mask`, under the
    named conditions `conds`."""

    stmt: Stmt
    hops: Tuple[Tuple[int, Optional[int]], ...]
    mask: int
    conds: FrozenSet[Tuple[Any, bool]]

    def env(self) -> Dict[int, int]:
        return {k: v for k, v in self.hops if v is not None}


@dataclass(slots=True)
class Reach:
    """What one execution of a load reads, for the lanes in `mask`: the
    store `writer` wrote it for certain (`INITIAL`: nothing in the body did),
    or one of `maybe`, which it may read as well.  `own`: the writer is the
    lane that reads."""

    mask: int
    writer: Optional[int]
    maybe: FrozenSet[int] = EMPTY
    own: bool = True
    #: The element read.
    cell: Tuple = ()

    @property
    def certain(self) -> bool:
        return self.writer is not None and not self.maybe


@dataclass(slots=True)
class _Layer:
    """A write still visible in a cell: who, under which conditions, for
    which lanes (registers), or by which lane (memory)."""

    writer: int
    conds: FrozenSet[Tuple[Any, bool]]
    mask: int = 0
    lane: int = -1


class Flow:
    """The loads and stores of `body`, run over `threads` lanes.

    `insts` holds every execution; `by_stmt` lists them for a statement.
    For a load's execution, `reach` says what it read, lane group by lane
    group.  For a store's execution, `readers` lists the executions of loads
    that read it, `replaced` whether a later store took its place in every
    element before anything read it, `final` which of its elements are still
    what memory holds at the end of the body.  `opaque` names the
    statements whose accesses could not be placed: what they write, every
    read of the buffer may read, and their readers are not listed.
    `cells_of` gives the elements an execution touched, lane by lane.
    """

    def __init__(self, body: Sequence[Stmt], threads: int,
                 arena: Optional[str] = None, masks: bool = False,
                 values: bool = False, before: bool = True,
                 budget: Optional[int] = None, fetches: bool = True):
        self.body = tuple(body)
        self.threads = threads
        self._values = values
        #: Whether `forward` lists what the lanes loaded before as well as
        #: what they stored (`_fetches`).
        self._fetches_too = fetches
        self._before = before
        #: What may still be placed before the body counts as too large.
        self._limit = BUDGET if budget is None else budget
        self._budget = self._limit
        #: The shared buffers no lane of another multiplication reaches: those
        #: of the multiplication's own arena.
        self._own: Set[int] = {
            id(x.target[0]) for x in walk_stmts(self.body)
            if x.op == Op.ALLOC and x.target and arena is not None
            and x.attr('arena') == arena}
        self.full = (1 << threads) - 1
        self._ix = _Indices(self.body, {'x': threads})
        self.insts: List[Inst] = []
        self.by_stmt: Dict[int, List[int]] = defaultdict(list)
        self._key: Dict[Tuple[int, Tuple], int] = {}
        self.reach: Dict[int, List[Reach]] = {}
        self.readers: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        #: For a store's execution, per lane and element: the write the
        #: element held right before it, and whether that was certain.  Not
        #: kept without `before`.
        self.before: Dict[int, Dict[Tuple[int, Tuple], Tuple[Optional[int], bool]]] = {}
        #: Executions of stores some element of which is what the body
        #: leaves behind.
        self.final: Set[int] = set()
        #: Executions of stores read by what the analysis cannot place.
        self.escaped: Set[int] = set()
        self.opaque: Set[int] = set()
        #: Per execution, the elements it touched: the lane groups and the
        #: cells side by side, which `cells_of` pairs up.  A cell is
        #: `(buffer key, address)`.
        self._cells: Dict[int, Tuple[Tuple[int, ...], Tuple[Tuple, ...]]] = {}
        self.defs: Dict[int, Stmt] = {}
        self.uses: Dict[int, List[Stmt]] = defaultdict(list)
        self.loops: Dict[int, Tuple[Stmt, ...]] = {}
        self.parents: Dict[int, Tuple[Stmt, ...]] = {}
        #: The values that bind a buffer of memory, by the name their
        #: declaration gives them.
        self._bound: Dict[str, Value] = {}
        self._index(self.body, ())
        #: The buffers a load, a store or a copy places an element of: what
        #: anything else touches of another buffer -- an accumulator a call
        #: adds into in place -- is nothing those read.
        self._placed: Set[int] = {
            id(a.base) for x in walk_stmts(self.body)
            if x.op in (Op.LOAD, Op.STORE, Op.LOAD_ASYNC, Op.COPY_ASYNC)
            for a in x.accesses if a.base is not None}
        self._reg: Dict[Tuple, List[_Layer]] = {}
        self._mem: Dict[Tuple, List[_Layer]] = {}
        #: The cells of both tables by buffer, `(space, base)`.
        self._cells_of: Dict[Tuple, Set[Tuple]] = defaultdict(set)
        #: Buffers written by what could not be placed: every read of them
        #: may read that write.
        self._smeared: Dict[Tuple, List[int]] = defaultdict(list)
        #: The same for a write whose buffer nothing names: every buffer of
        #: its space.
        self._smeared_space: Dict[Any, List[int]] = defaultdict(list)
        self._fresh: Dict[int, int] = {}
        #: Each set of writers a record holds, once (`_frozen`).
        self._sets: Dict[FrozenSet[int], FrozenSet[int]] = {}
        #: Each element, once (`_place_one`): every access to it names the
        #: same cell.
        self._interned: Dict[Tuple, Tuple] = {}
        #: For a store's execution, lane group by lane group, where what it
        #: stored came from: `('reg', cell, writer)`, a register a load read
        #: it from, holding what `writer` wrote; `('const', number)`; or
        #: `('value', value)`.  With `values` only.
        self.origin: Dict[int, List[Tuple[int, Tuple]]] = {}
        #: For a load's execution, lane group by lane group, what it reads
        #: for certain besides its element: `('reg', cell, writer)`, a
        #: register that holds still what `writer` wrote; `('const',
        #: number)`; `('value', value)`.  With `values` only, and only where
        #: the store was the lane's own, or where an earlier load by the
        #: same lanes found the same write and put it in that register.
        self.forward: Dict[int, List[Tuple[int, Tuple]]] = {}
        #: For a load's execution under a predicate of the lanes: the lanes
        #: it would run for without it.  With `values` only.
        self.whole: Dict[int, int] = {}
        #: For a load's execution that `forward` gives a register for: the
        #: numbers each such register holds for certain where the load
        #: stands, number -> lanes, among the lanes of `whole` -- what reading
        #: the register in lanes that take a number, or that the predicate
        #: leaves out, would find (`_numbers`).  With `values` only.
        self.numbers: Dict[int, Dict[Tuple, Dict[Any, int]]] = {}
        #: The masks taken to hold (`masks`).
        self.masks: FrozenSet = self._masks() if masks else frozenset()
        self._forwarded: Dict[int, List[Tuple[int, Tuple, Tuple[Reach, ...]]]] = {}
        #: Per element of memory and lane group: the last load of it whose
        #: value went into a register, and what that load found there
        #: (`_fetches`).
        self._fetched: Dict[Tuple[Tuple, int], Tuple[int, Reach]] = {}
        #: Per execution of a load of memory: the registers its value went
        #: into, `(mask, cell, store)`.
        self._held: Dict[int, List[Tuple[int, Tuple, int]]] = defaultdict(list)
        self._run(self.body, {}, self.full, frozenset())
        # What a loop's next run hands back (`_wrap`) can make a read
        # uncertain after the fact; what it would read instead goes with it.
        for n, found in self._forwarded.items():
            kept = [(m, o) for m, o, rs in found
                    if all(r.certain for r in rs)]
            if kept:
                self.forward[n] = kept
        for cell, layers in self._mem.items():
            if cell[0][0] is MemSpace.GLOBAL:
                for layer in layers:
                    self.final.add(layer.writer)

    def cells_of(self, n: int) -> List[Tuple[int, Tuple]]:
        """The elements execution `n` touched, `(lanes, cell)` by lane
        group."""
        entry = self._cells.get(n)
        return list(zip(*entry)) if entry else []

    def _frozen(self, writers) -> FrozenSet[int]:
        """`writers` as the one set every record with these writers holds:
        the reads of a buffer something unplaced wrote all may read it, and
        a set apiece is a few hundred bytes each."""
        if not writers:
            return EMPTY
        key = frozenset(writers)
        return self._sets.setdefault(key, key)

    # -- static facts -------------------------------------------------------- #

    def _index(self, stmts: Sequence[Stmt], parents: Tuple[Stmt, ...]):
        loops = tuple(p for p in parents if p.op == Op.FOR)
        for s in stmts:
            self.parents[id(s)] = parents
            self.loops[id(s)] = loops
            if (s.op == Op.RAWEXPR and s.target and s.attr('decl') is not None
                    and s.attr('extern')):
                self._bound.setdefault(s.attr('extern'), s.target[0])
            for t in s.target:
                self.defs[t.id] = s
            for v in s.operands():
                self.uses[v.id].append(s)
            for r in s.regions:
                for a in r.args:
                    self.defs[a.id] = s
                self._index(r.body, parents + (s,))

    def _masks(self) -> FrozenSet:
        """The element masks the body keeps the contract of
        (`kept_masks`)."""
        return kept_masks(self.body, self.defs, self._own)

    def binding(self, base) -> Any:
        """The value a buffer of memory is reached through: `base` itself
        where it is one, the value its symbol's name binds where it is a
        symbol -- a store names the symbol, a load the value."""
        if isinstance(base, Value):
            return base
        name = getattr(base, 'name', None)
        return self._bound.get(name, base) if name is not None else base

    def inst(self, stmt: Stmt, env: Dict[int, int]) -> Optional[int]:
        """The execution of `stmt` at the trips `env` names."""
        hops = tuple(env.get(l.induction.id)
                     for l in self.loops.get(id(stmt), ()))
        return self._key.get((id(stmt), hops))

    def executions(self, stmt: Stmt) -> List[int]:
        return self.by_stmt.get(id(stmt), [])

    def lanes(self, mask: int) -> Iterator[int]:
        lane = 0
        while mask:
            if mask & 1:
                yield lane
            mask >>= 1
            lane += 1

    # -- running ------------------------------------------------------------- #

    def _new(self, s: Stmt, env: Dict[int, int], mask: int, conds) -> int:
        hops = tuple((l.induction.id, env.get(l.induction.id))
                     for l in self.loops.get(id(s), ()))
        n = len(self.insts)
        self.insts.append(Inst(s, hops, mask, conds))
        self.by_stmt[id(s)].append(n)
        self._key[(id(s), tuple(v for _, v in hops))] = n
        return n

    def _static(self, operand, env) -> bool:
        if not isinstance(operand, Value):
            return isinstance(operand, (bool, int))
        try:
            leaves = self._ix.leaves(operand)
        except _NotStatic:
            return False
        return all(l == 'x' or l in env for l in leaves)

    def _lanes_where(self, cond, env, mask) -> int:
        out = 0
        memo: Dict[int, int] = {}
        uniform = 'x' not in self._ix.leaves(cond)
        if uniform:
            return mask if self._ix.evaluate(cond, dict(env), memo) else 0
        for lane in self.lanes(mask):
            if self._ix.evaluate(cond, {**env, 'x': lane}, {}):
                out |= 1 << lane
        return out

    def _run(self, stmts: Sequence[Stmt], env: Dict[int, int], mask: int,
             conds) -> None:
        for s in stmts:
            if not mask:
                return
            if s.op == Op.FOR and s.regions:
                self._loop(s, env, mask, conds)
                continue
            if s.op == Op.IF and s.regions:
                cond = s.cond
                if isinstance(cond, Value) and self._static(cond, env):
                    held = self._lanes_where(cond, env, mask)
                    self._run(s.regions[0].body, env, held, conds)
                    if len(s.regions) > 1:
                        self._run(s.regions[1].body, env, mask & ~held, conds)
                    continue
                name = cond.id if isinstance(cond, Value) else repr(cond)
                if name in self.masks:
                    self._run(s.regions[0].body, env, mask, conds)
                    continue
                self._run(s.regions[0].body, env, mask,
                          conds | {(name, True)})
                if len(s.regions) > 1:
                    self._run(s.regions[1].body, env, mask,
                              conds | {(name, False)})
                continue
            if s.op == Op.RAWBLOCK and s.regions:
                inner = conds if s.text in self.masks \
                    else conds | {(s.text, True)}
                self._run(s.regions[0].body, env, mask, inner)
                continue
            if s.regions:
                # A loop that is not counted, or a region the IR gives no
                # more meaning: it may not run, and it may run again.
                self._uncounted(s, env, mask, conds)
                continue
            self._stmt(s, env, mask, conds)

    def _loop(self, s: Stmt, env, mask, conds) -> None:
        lo, hi, step = s.loop_bounds
        ind = s.induction
        if all(isinstance(x, int) and not isinstance(x, bool)
               for x in (lo, hi, step)) and step:
            for hop in range(lo, hi, step):
                self._run(s.regions[0].body, {**env, ind.id: hop}, mask, conds)
            return
        self._uncounted(s, env, mask, conds)

    def _uncounted(self, s: Stmt, env, mask, conds) -> None:
        """A region run once, as one of any number of runs.

        Under a condition of its own, since it may not run at all: what it
        writes is possible to what follows.  And what one run hands the next
        through a register or shared memory is added once the run is over:
        a read of an element the region writes, made before the region wrote
        it, may read what the run before left there (`_wrap`).
        """
        first = len(self.insts)
        inner = conds | {(('runs', id(s)), True)}
        for r in s.regions:
            self._run(r.body, env, mask, inner)
        self._wrap(s, first)

    def _wrap(self, s: Stmt, first: int) -> None:
        local = {t.id for x in walk_stmts(tuple(b for r in s.regions
                                                for b in r.body))
                 for t in x.target}
        for r in s.regions:
            for a in r.args:
                local.add(a.id)
        last: Dict[Tuple, Set[int]] = defaultdict(set)
        for n in range(first, len(self.insts)):
            for _, cell in self.cells_of(n):
                if self._writes(self.insts[n].stmt) and not self._iteration(
                        cell, local):
                    last[cell].add(n)
        if not last:
            return
        for n in range(first, len(self.insts)):
            if n not in self.reach:
                continue
            for k, (m, cell) in enumerate(self.cells_of(n)):
                writers = last.get(cell)
                if not writers:
                    continue
                for reach in self.reach[n]:
                    if not reach.mask & m:
                        continue
                    if reach.writer is not None and reach.writer >= first \
                            and not reach.maybe:
                        continue
                    others = frozenset(w for w in writers)
                    reach.maybe = reach.maybe | others
                    for w in others:
                        self.readers[w].append((n, reach.mask & m))

    @staticmethod
    def _iteration(cell, local: Set[int]) -> bool:
        """Is the element one an iteration has to itself: in a buffer bound
        inside the region, which every run binds anew?"""
        key = cell[0]
        return len(key) > 2 and key[2] in local

    @staticmethod
    def _writes(s: Stmt) -> bool:
        return any(int(a.kind) & _WRITTEN for a in s.accesses)

    # -- one statement ------------------------------------------------------- #

    def _stmt(self, s: Stmt, env, mask, conds) -> None:
        if s.op == Op.ALLOC and s.target:
            n = self._new(s, env, mask, conds)
            buf = s.target[0]
            space = getattr(buf.type, 'space', None)
            if space in _PLACED:
                zeros = s.attr('init') == '{}'
                self._reset(space, id(buf), n if zeros else None)
                if zeros and self._values and space is MemSpace.REGISTER:
                    self.origin[n] = [(mask, ('const', 0))]
            return
        if s.op == Op.MARK and s.attr('mark') == 'defines':
            n = self._new(s, env, mask, conds)
            for a in s.accesses:
                if a.base is not None:
                    self._reset(a.space, id(a.base), None)
            return
        if s.op == Op.MARK and s.attr('mark') != 'zeros':
            # What a later pass is told about the order of the writes
            # (`clears`, `cleared`): no element changes.
            return
        if not s.accesses or s.op in (Op.WAIT, Op.PREFETCH, Op.COMMIT_ASYNC):
            # A wait carries the accesses of the copies it completes, which
            # are placed where they are issued.
            return
        if s.op == Op.RAWEXPR and s.attr('decl') is not None:
            # A binding: it computes where a buffer is, and reads none of it.
            return
        whole = mask
        if s.predicate is not None:
            pred = s.predicate
            if isinstance(pred, Value) and self._static(pred, env):
                mask = self._lanes_where(pred, env, mask)
                if not mask:
                    return
            else:
                name = pred.id if isinstance(pred, Value) else repr(pred)
                conds = conds | {(name, True)}
        n = self._new(s, env, mask, conds)
        placed = self._place(s, env, mask)
        if placed is None:
            self.opaque.add(id(s))
            self._smear(n, s, conds)
            return
        self._cells[n] = (tuple(m for m, _, _ in placed),
                          tuple(cell for _, cell, _ in placed))
        read = [p for p in placed if p[2] in ('read', 'rmw')]
        written = [p for p in placed if p[2] in ('write', 'rmw')]
        if read:
            self.reach[n] = self._read(n, read, conds)
            if self._values and s.op == Op.LOAD:
                found = self._forwards(n, conds)
                taken = 0
                for m, _, _ in found:
                    taken |= m
                if self._fetches_too:
                    found += self._fetches(n, conds, taken)
                self._forwarded[n] = found
                if whole != mask:
                    self.whole[n] = whole
                cells = {o[1] for _, o, _ in found if o[0] == 'reg'}
                if cells:
                    self.numbers[n] = {c: self._numbers(c, whole, conds)
                                       for c in cells}
        if written:
            self._write(n, written, conds)
            if self._values and s.op in (Op.STORE, Op.MARK):
                self.origin[n] = self._origin(s, env, mask)
                self._holding(s, n, env, written)

    def _origin(self, s: Stmt, env, mask: int) -> List[Tuple[int, Tuple]]:
        """Where the value `s` stores came from, lane group by lane group."""
        v = s.args[1]
        if not isinstance(v, Value):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return [(mask, ('const', v))]
            return []
        if getattr(v.type, 'length', None) is not None:
            # A vector covers several elements, none of which it is.
            return []
        d = self.defs.get(v.id)
        if d is not None and d.op == Op.CONST:
            return [(mask, ('const', d.attr('value')))]
        if (d is not None and d.op == Op.LOAD and d.predicate is not None
                and isinstance(d.predicate, Value)
                and self._static(d.predicate, env)):
            # A load under a predicate of the lanes gives the lanes it leaves
            # out its else-value.
            other = d.attr('other')
            if other is None:
                other = 0
            inside = self._lanes_where(d.predicate, env, mask)
            if (inside != mask and isinstance(other, (int, float))
                    and not isinstance(other, bool)):
                out = [(mask & ~inside, ('const', other))]
                if inside:
                    out += self._loaded(v, d, env, inside)
                return out
        return self._loaded(v, d, env, mask)

    def _loaded(self, v: Value, d: Optional[Stmt], env,
                mask: int) -> List[Tuple[int, Tuple]]:
        """Where `v`, defined by `d`, came from for the lanes of `mask`: the
        register a load of registers read, where it read what one store
        wrote for certain, or the value itself."""
        if (d is not None and d.op == Op.LOAD and len(d.accesses) == 1
                and d.accesses[0].space is MemSpace.REGISTER):
            nd = self.inst(d, env)
            out: List[Tuple[int, Tuple]] = []
            for r in self.reach.get(nd, ()) if nd is not None else ():
                hit = r.mask & mask
                if not hit:
                    continue
                if r.certain and r.writer is not None and r.writer >= 0:
                    out.append((hit, ('reg', r.cell, r.writer)))
                else:
                    out.append((hit, ('value', v)))
            if out:
                return out
        return [(mask, ('value', v))]

    def _forwards(self, n: int, conds) -> List[Tuple[int, Tuple, Tuple[Reach, ...]]]:
        """What the load's execution `n` reads for certain, besides its
        element, lane group by lane group: what the lane's own store put
        there, where that is a value, a number, or a register that holds
        it still."""
        out: List[Tuple[int, Tuple, Tuple[Reach, ...]]] = []
        for r in self.reach.get(n, ()):
            if not r.certain or r.writer is None or r.writer < 0 or not r.own:
                continue
            for m, o in self.origin.get(r.writer, ()):
                hit = m & r.mask
                if not hit:
                    continue
                if o[0] == 'reg':
                    _, cell, writer = o
                    if self._holds(cell, writer, hit, conds):
                        out.append((hit, ('reg', cell, writer), (r,)))
                else:
                    out.append((hit, o, (r,)))
        return out

    def _fetches(self, n: int, conds, taken: int) -> List[Tuple[int, Tuple, Tuple[Reach, ...]]]:
        """What the load's execution `n` of memory reads for certain because
        an earlier load by the same lanes read it: the element unchanged in
        between -- both found the same write there -- and the register that
        earlier load's value went into holding it still.  Not for the lanes
        in `taken`, which read what their own store put there, and take
        that."""
        out: List[Tuple[int, Tuple, Tuple[Reach, ...]]] = []
        for r in self.reach.get(n, ()):
            if (not r.certain or not r.cell or r.mask & taken
                    or r.cell[0][0] is MemSpace.REGISTER):
                continue
            last = self._fetched.get((r.cell, r.mask))
            if last is None:
                continue
            earlier, found = last
            if not found.certain or found.writer != r.writer:
                continue
            for m, cell, store in self._held.get(earlier, ()):
                if m & r.mask == r.mask and self._holds(cell, store, r.mask,
                                                         conds):
                    out.append((r.mask, ('reg', cell, store), (r, found)))
                    break
        return out

    def _holding(self, s: Stmt, n: int, env, written) -> None:
        """Where the store into registers `s` puts what a load of memory
        read: the registers, for the load (`_held`), and the load, for the
        elements it read (`_fetched`)."""
        v = s.args[1]
        if not isinstance(v, Value) or getattr(v.type, 'length', None) is not None:
            return
        d = self.defs.get(v.id)
        if (d is None or d.op != Op.LOAD or len(d.accesses) != 1
                or d.accesses[0].space is MemSpace.REGISTER):
            return
        load = self.inst(d, env)
        if load is None:
            return
        held = [(m, cell, n) for m, cell, _ in written
                if cell[0][0] is MemSpace.REGISTER]
        if not held:
            return
        self._held[load].extend(held)
        for r in self.reach.get(load, ()):
            if r.cell:
                self._fetched[(r.cell, r.mask)] = (load, r)

    def _numbers(self, cell, lanes: int, conds) -> Dict[Any, int]:
        """The numbers `cell`, a register, holds for certain for the lanes of
        `lanes`, as a read under `conds` would find them now: number ->
        lanes.  A lane whose last write stands under a condition the read
        does not, or wrote anything but a number, holds none."""
        out: Dict[Any, int] = {}
        if (self._smeared.get(cell[0][:2])
                or self._smeared_space.get(cell[0][0])):
            return out
        left = lanes

        def found(writer: int, hit: int) -> None:
            for m, o in self.origin.get(writer, ()):
                if o[0] == 'const' and m & hit:
                    out[o[1]] = out.get(o[1], 0) | (m & hit)

        for layer in reversed(self._reg.get(cell, ())):
            hit = left & layer.mask
            if not hit:
                continue
            left &= ~hit
            if layer.conds <= conds:
                found(layer.writer, hit)
            if not left:
                return out
        fresh = self._fresh.get(cell[0][1])
        if left and fresh is not None:
            found(fresh, left)
        return out

    def _holds(self, cell, writer: int, lanes: int, conds) -> bool:
        """Does `cell`, a register, hold what `writer` wrote for every lane
        of `lanes`, as a read under `conds` would find it now?"""
        if (self._smeared.get(cell[0][:2])
                or self._smeared_space.get(cell[0][0])):
            return False
        left = lanes
        for layer in reversed(self._reg.get(cell, ())):
            hit = left & layer.mask
            if not hit:
                continue
            if not layer.conds <= conds or layer.writer != writer:
                return False
            left &= ~hit
            if not left:
                return True
        return not left or self._fresh.get(cell[0][1], INITIAL) == writer

    def _reset(self, space, base: int, fresh: Optional[int]) -> None:
        table = self._reg if space is MemSpace.REGISTER else self._mem
        for cell in self._cells_of.pop((space, base), ()):
            table.pop(cell, None)
        if fresh is not None:
            self._fresh[base] = fresh
        else:
            self._fresh.pop(base, None)
        self._smeared.pop((space, base), None)

    def _place(self, s: Stmt, env, mask):
        """`[(mask, cell, kind)]` for the elements `s` touches, or None where
        they cannot be told."""
        if s.op == Op.LOAD and len(s.accesses) == 1:
            a = s.accesses[0]
            index = s.args[1:]
            width = _width(s.target[0] if s.target else None, a, s.args[0])
            kind = 'read'
        elif s.op in (Op.STORE, Op.MARK) and len(s.accesses) == 1:
            # A promise of a zero (`mark zeros`) as the store of that zero.
            a = s.accesses[0]
            index, width = s.args[2:], _width(s.args[1], a, s.args[0])
            kind = 'rmw' if a.kind & Effect.ATOMIC else 'write'
        elif s.op == Op.LOAD_ASYNC and len(s.accesses) == 1:
            a = s.accesses[0]
            index = s.load_index
            width = _width(s.target[0] if s.target else None, a, s.load_base)
            kind = 'read'
            return self._place_one(s.load_base, a, index, width, kind, env,
                                   mask)
        elif s.op == Op.COPY_ASYNC:
            return self._place_copy(s, env, mask)
        else:
            return None
        return self._place_one(s.args[0], a, index, width, kind, env, mask)

    def _place_one(self, binding, a, index, width, kind, env, mask):
        if a.space not in _PLACED or a.base is None:
            return None
        if not all(self._static(i, env) for i in index):
            return None
        key = self._buffer(a, binding, env)
        if key is None:
            return None
        uniform = (a.space is MemSpace.REGISTER
                   and not any('x' in self._ix.leaves(i) for i in index
                               if isinstance(i, Value)))
        out = []
        interned = self._interned
        if uniform:
            for addr in _span(self._address(index, env), width):
                cell = (key, addr)
                out.append((mask, interned.setdefault(cell, cell), kind))
        else:
            for lane in self.lanes(mask):
                here = {**env, 'x': lane}
                for addr in _span(self._address(index, here), width):
                    cell = (key, addr)
                    out.append((1 << lane, interned.setdefault(cell, cell),
                                kind))
        self._budget -= len(out)
        if self._budget < 0:
            raise TooLarge(f'the body places more than {self._limit} '
                           f'elements')
        return out

    def _place_copy(self, s: Stmt, env, mask):
        """A copy reads its source and writes its destination, lane by lane;
        its value is none a pass could name."""
        reads = [a for a in s.accesses if a.kind & Effect.READ]
        writes = [a for a in s.accesses if a.kind & Effect.WRITE]
        if len(reads) != 1 or len(writes) != 1:
            return None
        elems = s.attr('elems', 1) or 1
        src = self._place_one(s.copy_src, reads[0], s.copy_src_index, elems,
                              'read', env, mask)
        dst = self._place_one(s.copy_dst, writes[0], s.copy_dst_index, elems,
                              'write', env, mask)
        if src is None or dst is None:
            return None
        return src + dst

    def _address(self, index, env) -> Tuple[int, ...]:
        memo: Dict[int, int] = {}
        return tuple(int(self._ix.evaluate(i, env, memo)) for i in index)

    def _buffer(self, a, binding, env) -> Optional[Tuple]:
        """The identity of the buffer an access lands in.

        A register array is its allocation.  Memory is the buffer and the
        binding it is read through, at the trips of the loops the binding is
        computed in: a binding a loop computes points somewhere else every
        trip -- a table's member, the next element.
        """
        if a.space is MemSpace.REGISTER:
            return (a.space, id(a.base))
        binding = self.binding(binding)
        if not isinstance(binding, Value):
            return (a.space, id(a.base), None, ())
        d = self.defs.get(binding.id)
        hops = ()
        if d is not None:
            hops = tuple(env.get(l.induction.id)
                         for l in self.loops.get(id(d), ()))
        return (a.space, id(a.base), binding.id, hops)

    def _layers(self, cell) -> List[_Layer]:
        table = self._reg if cell[0][0] is MemSpace.REGISTER else self._mem
        layers = table.get(cell)
        if layers is None:
            layers = table[cell] = []
            self._cells_of[cell[0][:2]].add(cell)
        return layers

    def _shared_with_others(self, cell) -> bool:
        return cell[0][0] is MemSpace.SHARED and cell[0][1] not in self._own

    def _read(self, n: int, placed, conds) -> List[Reach]:
        out: List[Reach] = []
        for m, cell, _ in placed:
            layers = self._layers(cell)
            smeared = frozenset(self._smeared.get(cell[0][:2], ())) | \
                frozenset(self._smeared_space.get(cell[0][0], ()))
            if self._shared_with_others(cell):
                smeared = smeared | {EXTERNAL}
            if cell[0][0] is MemSpace.REGISTER:
                out.extend(self._read_register(n, m, cell, layers, conds,
                                               smeared))
            else:
                lane = next(self.lanes(m))
                out.append(self._read_memory(n, m, lane, cell, layers, conds,
                                             smeared))
        return out

    def _read_register(self, n, mask, cell, layers, conds, smeared):
        out: List[Reach] = []
        left = mask
        maybe: Set[int] = set(smeared)
        for layer in reversed(layers):
            hit = left & layer.mask
            if not hit:
                continue
            if layer.conds <= conds:
                out.append(Reach(hit, layer.writer, self._frozen(maybe),
                                 cell=cell))
                self.readers[layer.writer].append((n, hit))
                left &= ~hit
                if not left:
                    break
            else:
                maybe.add(layer.writer)
                self.readers[layer.writer].append((n, hit))
        if left:
            fresh = self._fresh.get(cell[0][1], INITIAL)
            out.append(Reach(left, fresh, self._frozen(maybe), cell=cell))
        return out

    def _read_memory(self, n, mask, lane, cell, layers, conds, smeared):
        maybe: Set[int] = set(smeared)
        for layer in reversed(layers):
            self.readers[layer.writer].append((n, mask))
            if layer.conds <= conds:
                return Reach(mask, layer.writer, self._frozen(maybe),
                             own=layer.lane == lane, cell=cell)
            maybe.add(layer.writer)
        return Reach(mask, INITIAL, self._frozen(maybe), own=False, cell=cell)

    def _write(self, n: int, placed, conds) -> None:
        before: Dict[Tuple[int, Tuple], Tuple[Optional[int], bool]] = {}
        for m, cell, _ in placed:
            layers = self._layers(cell)
            register = cell[0][0] is MemSpace.REGISTER
            for lane in self.lanes(m) if self._before else ():
                prior, certain = INITIAL, True
                for layer in reversed(layers):
                    if register and not layer.mask & (1 << lane):
                        continue
                    if layer.conds <= conds:
                        prior = layer.writer
                        break
                    certain = False
                if prior == INITIAL and register:
                    prior = self._fresh.get(cell[0][1], INITIAL)
                if (self._smeared.get(cell[0][:2])
                        or self._smeared_space.get(cell[0][0])
                        or self._shared_with_others(cell)):
                    certain = False
                before[(lane, cell)] = (prior, certain)
            kept: List[_Layer] = []
            for layer in layers:
                if layer.conds >= conds:
                    if register:
                        layer.mask &= ~m
                        if not layer.mask:
                            continue
                    else:
                        continue
                kept.append(layer)
            if register:
                kept.append(_Layer(n, conds, mask=m))
            else:
                kept.append(_Layer(n, conds, lane=next(self.lanes(m))))
            if len(kept) > _DEPTH:
                kept = kept[-_DEPTH:]
            table = self._reg if register else self._mem
            table[cell] = kept
            if self._shared_with_others(cell):
                self.escaped.add(n)
        if self._before:
            self.before[n] = before

    def _smear(self, n: int, s: Stmt, conds) -> None:
        """An access nothing can place: it reads every element of its buffer
        and may have written any of them."""
        for a in s.accesses:
            spaces = (list(_PLACED) if a.space is MemSpace.UNKNOWN
                      else [a.space])
            base = None if a.base is None or a.space is MemSpace.UNKNOWN \
                else id(a.base)
            if base is not None and base not in self._placed:
                continue
            kind = int(a.kind)
            for space in spaces:
                if space not in _PLACED:
                    continue
                table = self._reg if space is MemSpace.REGISTER else self._mem
                if kind & _READ_OR_ATOMIC:
                    keys = ([(space, base)] if base is not None else
                            [k for k in self._cells_of if k[0] is space])
                    for key in keys:
                        for cell in self._cells_of.get(key, ()):
                            for layer in table.get(cell, ()):
                                self.readers[layer.writer].append(
                                    (n, self.full))
                                self.escaped.add(layer.writer)
                    smeared = ([self._smeared.get((space, base), ())]
                               if base is not None else
                               [w for k, w in self._smeared.items()
                                if k[0] is space])
                    for writers in smeared:
                        self.escaped.update(writers)
                    self.escaped.update(self._smeared_space.get(space, ()))
                    fresh = ([self._fresh[base]] if base in self._fresh
                             else list(self._fresh.values()) if base is None
                             else [])
                    for w in fresh:
                        self.readers[w].append((n, self.full))
                        self.escaped.add(w)
                if kind & _WRITTEN:
                    if base is None:
                        self._smeared_space[space].append(n)
                    else:
                        self._smeared[(space, base)].append(n)


def _stands_in(x: Stmt) -> bool:
    """Is `x` a statement that emits nothing and only says what others
    reach through a name the IR does not follow (`IRBuilder.touches`)?"""
    return (x.op == Op.RAWSTMT and not x.text and bool(x.accesses)
            and not x.target and not x.args)


def kept_masks(body: Sequence[Stmt], defs: Dict[int, Stmt],
               own: Set[int]) -> FrozenSet:
    """The element masks of `body` -- the text of a raw block, the condition
    of an `if`, that says `guard='element'` -- where the body keeps their
    contract, and none where it does not.  `defs` is what defines each value
    of the body, by id; `own` holds the shared buffers no lane of another
    multiplication reaches, by id.

    Kept, where every write to global memory stands under a mask; where
    every write to shared memory that lanes of other multiplications read
    stores a number, or what a load read from global memory the body never
    writes, which the mask changes nothing of; and where no condition of a
    branch or a loop reads memory.  A lane under a mask that does not hold
    then leaves nothing behind but what every lane would, and reaches every
    barrier the others do.

    Not asked: a statement that only says what others reach (`_stands_in`)
    -- a merged run's members, which its reads and writes through the
    pointer its counter selects reach.  Those writes are statements of
    their own, under a mask or not."""
    masks: Set = set()
    for x in walk_stmts(body):
        if x.attr('guard') != 'element':
            continue
        if x.op == Op.RAWBLOCK and x.text:
            masks.add(x.text)
        elif x.op == Op.IF:
            cond = x.cond
            masks.add(cond.id if isinstance(cond, Value) else repr(cond))
    if not masks:
        return frozenset()
    written = {id(a.base) for x in walk_stmts(body)
               for a in x.accesses if a.space is MemSpace.GLOBAL
               and a.kind & (Effect.WRITE | Effect.ATOMIC)}
    loads = _MemoryReads(defs)

    def kept(stmts, under: bool) -> bool:
        for x in stmts:
            if x.op == Op.RAWBLOCK and x.regions:
                inner = under or x.text in masks
                if not all(kept(r.body, inner) for r in x.regions):
                    return False
                continue
            if x.op == Op.IF and x.regions:
                cond = x.cond
                name = cond.id if isinstance(cond, Value) else repr(cond)
                if name in masks:
                    if not kept(x.regions[0].body, True):
                        return False
                    if len(x.regions) > 1 and not kept(x.regions[1].body,
                                                       under):
                        return False
                    continue
                if loads.reads(cond):
                    return False
                if not all(kept(r.body, under) for r in x.regions):
                    return False
                continue
            if x.regions:
                heads = list(x.operands())
                if any(loads.reads(v) for v in heads):
                    return False
                if not all(kept(r.body, under) for r in x.regions):
                    return False
                continue
            if x.op == Op.EXIT and loads.reads(x.exit_cond):
                return False
            if _stands_in(x):
                continue
            for a in x.accesses:
                if not a.kind & (Effect.WRITE | Effect.ATOMIC):
                    continue
                if a.space in (MemSpace.GLOBAL, MemSpace.UNKNOWN):
                    if not under:
                        return False
                elif (a.space is MemSpace.SHARED
                      and id(a.base) not in own
                      and not _element_free(x, written, defs)):
                    return False
        return True

    return frozenset(masks) if kept(body, False) else frozenset()


def body_masks(body: Sequence[Stmt], arena: Optional[str]) -> FrozenSet:
    """`kept_masks` of a body as it stands: the definitions read off it, and
    the buffers of `arena` -- the multiplication's own -- as those no lane of
    another multiplication reaches."""
    defs: Dict[int, Stmt] = {}
    own: Set[int] = set()
    for x in walk_stmts(body):
        for t in x.target:
            defs[t.id] = x
        for r in x.regions:
            for a in r.args:
                defs[a.id] = x
        if (x.op == Op.ALLOC and x.target and arena is not None
                and x.attr('arena') == arena):
            own.add(id(x.target[0]))
    return kept_masks(body, defs, own)


def _element_free(x: Stmt, written: Set[int], defs: Dict[int, Stmt]) -> bool:
    """Does `x`, a write to shared memory, store what no element's mask
    changes: a number, or what a load read from a buffer of global memory
    the body never writes?"""
    if x.op == Op.COPY_ASYNC:
        src = [a for a in x.accesses if a.kind & Effect.READ]
        return (len(src) == 1 and src[0].space is MemSpace.GLOBAL
                and id(src[0].base) not in written)
    if x.op != Op.STORE or len(x.args) < 2:
        return False
    v = x.args[1]
    if not isinstance(v, Value):
        return True
    d = defs.get(v.id)
    if d is None:
        return False
    if d.op == Op.CONST:
        return True
    return (d.op == Op.LOAD and len(d.accesses) == 1
            and d.accesses[0].space is MemSpace.GLOBAL
            and id(d.accesses[0].base) not in written)


class _MemoryReads:
    """Whether a value is computed from what a load read -- of memory or of
    registers, which memory may have filled."""

    def __init__(self, defs: Dict[int, Stmt]):
        self._defs = defs
        self._memo: Dict[int, bool] = {}

    def reads(self, v) -> bool:
        if not isinstance(v, Value):
            return False
        if v.id in self._memo:
            return self._memo[v.id]
        self._memo[v.id] = False
        d = self._defs.get(v.id)
        out = False
        if d is not None:
            if d.regions:
                # An induction is what the bounds give; any other value a
                # region defines carries what its body computed.
                if d.op == Op.FOR and v is d.induction:
                    out = any(self.reads(x) for x in d.operands())
                else:
                    out = True
            elif d.op in (Op.LOAD, Op.LOAD_ASYNC) or any(
                    a.kind & Effect.READ for a in d.accesses):
                out = True
            else:
                out = any(self.reads(x) for x in d.operands())
        self._memo[v.id] = out
        return out


def _width(v, access, binding) -> int:
    """How many elements of its buffer a value covers: its own width over
    the width of the buffer's element."""
    t = getattr(v, 'type', None)
    length = getattr(t, 'length', None) or 1
    buffer = access.base if access.space is MemSpace.REGISTER else binding
    elem = getattr(getattr(buffer, 'type', None), 'elem', None)
    per = getattr(elem, 'length', None) or 1
    return max(1, length // per)


def _span(addr: Tuple[int, ...], width: int) -> List[Tuple[int, ...]]:
    """The elements a `width`-wide access at `addr` covers: the last index
    runs on."""
    if width <= 1 or not addr:
        return [addr]
    return [addr[:-1] + (addr[-1] + k,) for k in range(width)]
