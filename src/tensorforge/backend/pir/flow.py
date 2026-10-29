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

Under the explicit vector (`simd`) one work-item is the whole multiplication
and a value spread over its lanes is one vector; an access carries no lane
in its index and reaches an element a lane: lane `l`'s at the index plus
`l`, in memory, and in a register array at the same element of every vector
the array is read and written as -- the lane's own, as under SPMD.  What is
not one element a lane of every lane, a vector of a register array that
starts elsewhere, or part of a vector is placed nowhere (`_place_vector`).

An access is placed once for all the lanes that make it.  A register is an
element each lane holds its own of: one cell, its writes and reads lane
groups of it.  Memory is kept in blocks of 64 elements along the last index
(`_mem`): an access names the elements each of its lanes reaches
(`_Cells`), a write leaves in every block it lands in the elements it wrote
there (`_Piece`), and a read finds, element by element, the last write it
stands under -- the lanes that read alike, the lane's own store or one lane's
for all of them, in one `Reach`.  What `Flow` keeps grows with the accesses
it places, execution by execution.  A body that places more than `BUDGET`
is not run (`TooLarge`), and a pass that asks leaves it as it is.

Nothing here changes the body.  A pass asks, and decides what it may do
with the answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .core import (BufferType, Effect, MemSpace, Op, Stmt, Uniformity, Value,
                   walk_stmts)
from .shards import _Indices, _NotStatic

#: What a cell held before the body wrote it.
INITIAL = -1

#: A write by a lane of another multiplication, to shared memory the lanes
#: of more than one share: possible to every read of it.
EXTERNAL = -2

#: The spaces whose elements are placed: registers, which every lane has its
#: own of, and the memory the lanes of a multiplication share.
_PLACED = (MemSpace.REGISTER, MemSpace.SHARED, MemSpace.GLOBAL)

#: How many writes of uncertain standing an element keeps apart before they
#: are taken as one.
_DEPTH = 8

#: The accesses one body may place, summed over its executions, before
#: `Flow` gives up on it: an element of a register, or of memory every lane
#: reads, one each; the elements of memory the lanes of an access reach, one
#: a row.  SeisSol's damage step in double, every operation storing its
#: result, places 0.42 million at order 6, which the flow of the forward
#: pass holds in 0.8 GB, and 1.7 million at order 8, in 2.8 GB.
BUDGET = 2_000_000


class TooLarge(Exception):
    """The body places more accesses than `Flow` keeps track of
    (`BUDGET`)."""

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
    #: The element read: a register, every lane's; in memory, each lane's
    #: (`_At`), which `cell_at` names.
    cell: Any = ()
    #: The lane that wrote it, for a read of memory another lane wrote; -1
    #: for the lane's own store, and where none did.
    by: int = -1

    @property
    def certain(self) -> bool:
        return self.writer is not None and not self.maybe

    def cell_at(self, lane: int) -> Tuple:
        """The element lane `lane` read, `(buffer key, address)`."""
        cell = self.cell
        return cell if isinstance(cell, tuple) else cell.at(lane)


@dataclass(slots=True)
class _Layer:
    """A write still visible in a register: who, under which conditions,
    for which lanes."""

    writer: int
    conds: FrozenSet[Tuple[Any, bool]]
    mask: int = 0


@dataclass(slots=True)
class _Piece:
    """What a write left visible in a block of memory: who, under which
    conditions, the elements it wrote there (`bits`, one a position in the
    block) that no later write has taken, and the lanes that wrote them
    (`cells`)."""

    writer: int
    conds: FrozenSet[Tuple[Any, bool]]
    cells: '_Cells'
    bits: int


class _Cells:
    """The elements of memory one access reaches, lane by lane, in one row
    of its buffer: lane `lanes[i]` at `starts[i]` along the last index and
    the `width - 1` after it, the other indices `lead` (`None` for a buffer
    of no index, its one element at 0) -- or, where `starts` is a number,
    every lane at that one, a read no lane tells apart (`broadcast`).
    What the flow asks of them it asks once (`blocks`, `readers`,
    `lane_of`)."""

    __slots__ = ('key', 'lead', 'lanes', 'starts', 'width', 'mask',
                 'broadcast', '_blocks', '_lanes_in', '_readers', '_last',
                 '_alone', '_start', '_at', '_lanes_at')

    def __init__(self, key: Tuple, lead: Optional[Tuple[int, ...]],
                 lanes: Tuple[int, ...], starts, width: int, mask: int):
        self.key = key
        self.lead = lead
        self.lanes = lanes
        self.starts = starts
        self.width = width
        self.mask = mask
        self.broadcast = isinstance(starts, int)
        self._blocks: Optional[Dict[int, int]] = None
        self._lanes_in: Optional[Dict[int, int]] = None
        self._readers: Optional[Dict[int, List[Tuple[int, int]]]] = None
        self._last: Optional[Dict[int, int]] = None
        self._alone = True
        self._start: Optional[Dict[int, int]] = None
        self._at: Optional[Dict[int, _At]] = None
        self._lanes_at: Optional[Dict[int, Dict[int, int]]] = None

    def each(self):
        """`(lane, k, element)` for every element a lane reaches, `k` past
        its start, lane by lane in the order the lanes run."""
        w = self.width
        if self.broadcast:
            s = self.starts
            for lane in self.lanes:
                for k in range(w):
                    yield lane, k, s + k
            return
        for lane, s in zip(self.lanes, self.starts):
            for k in range(w):
                yield lane, k, s + k

    def blocks(self) -> Dict[int, int]:
        """The elements, block by block of 64: block -> bits."""
        out = self._blocks
        if out is not None:
            return out
        out = {}
        lanes_in = self._lanes_in = {}
        w = self.width
        if self.broadcast:
            for e in range(self.starts, self.starts + w):
                b = e >> 6
                out[b] = out.get(b, 0) | (1 << (e & 63))
                lanes_in[b] = self.mask
        else:
            alone = True
            for lane, s in zip(self.lanes, self.starts):
                bit = 1 << lane
                for e in range(s, s + w):
                    b = e >> 6
                    held = out.get(b, 0)
                    one = 1 << (e & 63)
                    if held & one:
                        alone = False
                    out[b] = held | one
                    lanes_in[b] = lanes_in.get(b, 0) | bit
            self._alone = alone
        self._blocks = out
        return out

    def alone(self) -> bool:
        """Does every element have one lane that reaches it?"""
        self.blocks()
        return self._alone and not self.broadcast

    def lanes_in(self, block: int, bits: int) -> int:
        """The lanes that reach an element of `bits` in `block`."""
        if self.broadcast:
            return self.mask
        if bits == self.blocks()[block]:
            return self._lanes_in[block]
        out = 0
        readers = self.readers()
        base = block << 6
        while bits:
            low = bits & -bits
            bits ^= low
            for lanes, _ in readers[base + low.bit_length() - 1]:
                out |= lanes
        return out

    def readers(self) -> Dict[int, List[Tuple[int, int]]]:
        """Per element, `(lanes, k)`: the lanes that reach it `k` past their
        start."""
        out = self._readers
        if out is None and self.broadcast:
            out = self._readers = {self.starts + k: [(self.mask, k)]
                                   for k in range(self.width)}
        if out is None and self.alone():
            out = self._readers = {e: [(1 << lane, k)]
                                   for lane, k, e in self.each()}
        if out is None:
            out = {}
            for lane, k, e in self.each():
                at = out.setdefault(e, [])
                for i, (lanes, j) in enumerate(at):
                    if j == k:
                        at[i] = (lanes | (1 << lane), k)
                        break
                else:
                    at.append((1 << lane, k))
            self._readers = out
        return out

    def lanes_at(self, block: int) -> List[Tuple[int, int]]:
        """`(k, lanes)`: the lanes whose element `k` past their start is in
        `block`."""
        found = self._lanes_at
        if found is None:
            found = self._lanes_at = {}
            for lane, k, e in self.each():
                at = found.setdefault(e >> 6, {})
                at[k] = at.get(k, 0) | (1 << lane)
        return list(found.get(block, {}).items())

    def lane_of(self, e: int) -> int:
        """The lane whose write of element `e` memory holds: the last."""
        last = self._last
        if last is None:
            last = self._last = {e2: lane for lane, _, e2 in self.each()}
        return last[e]

    def alike(self, other: '_Cells') -> bool:
        """Does each lane reach the elements here it reaches in `other`, and
        every element one lane only?  Both in the same row."""
        return (self.width == other.width
                and (self.lanes is other.lanes or self.lanes == other.lanes)
                and self.starts == other.starts and self.alone())

    def cell(self, lane: int, k: int) -> Tuple:
        """`(key, address)` of the element lane `lane` reaches `k` past its
        start."""
        if self.lead is None:
            return (self.key, ())
        if self.broadcast:
            s = self.starts
        else:
            start = self._start
            if start is None:
                start = self._start = dict(zip(self.lanes, self.starts))
            s = start[lane]
        return (self.key, self.lead + (s + k,))

    def at(self, k: int) -> '_At':
        found = self._at
        if found is None:
            found = self._at = {}
        out = found.get(k)
        if out is None:
            out = found[k] = _At(self, k)
        return out

    def expanded(self) -> List[Tuple[int, Tuple]]:
        """`(lanes, cell)` element by element: a broadcast's for all its
        lanes, the others lane by lane."""
        if self.broadcast:
            return [(self.mask, self.cell(-1, k)) for k in range(self.width)]
        return [(1 << lane, self.cell(lane, k)) for lane, k, _ in self.each()]


class _At:
    """The element each lane of a `Reach` read: `k` past its start in
    `cells`."""

    __slots__ = ('cells', 'k')

    def __init__(self, cells: _Cells, k: int):
        self.cells = cells
        self.k = k

    def at(self, lane: int) -> Tuple:
        return self.cells.cell(lane, self.k)


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
                 budget: Optional[int] = None, fetches: bool = True,
                 exchange: bool = False, simd: bool = False):
        self.body = tuple(body)
        self.threads = threads
        #: Whether the lane is in the type rather than the index: one
        #: work-item, `threads` lanes wide (`_place_vector`).
        self.simd = simd
        self._values = values
        #: Whether `forward` lists what another lane stored -- a value, or
        #: from a register that lane holds it in still (`_forwards`).
        self._exchange = exchange
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
        self._ix = _Indices(self.body, {'x': threads}, simd=simd)
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
        #: Per execution, what it placed: `(lanes, cell, kind)`, a cell of a
        #: register `(buffer key, address)`, of memory `_Cells`; `cells_of`
        #: gives them element by element.
        self._cells: Dict[int, List[Tuple[int, Any, str]]] = {}
        self.defs: Dict[int, Stmt] = {}
        #: Which values are the same on every lane (`_Alike`), once asked.
        self._alike: Optional[_Alike] = None
        self.uses: Dict[int, List[Stmt]] = defaultdict(list)
        self.loops: Dict[int, Tuple[Stmt, ...]] = {}
        #: The inductions of `loops`, by id.
        self._inductions: Dict[int, Tuple[int, ...]] = {}
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
        #: Per cell of a register, the writes visible in it.
        self._reg: Dict[Tuple, List[_Layer]] = {}
        #: Per block of memory, `(buffer key, lead, block)`: the writes
        #: visible in it, the first first.
        self._mem: Dict[Tuple, List[_Piece]] = {}
        #: The cells of `_reg` and the blocks of `_mem` by buffer, `(space,
        #: base)`.
        self._cells_of: Dict[Tuple, Set[Tuple]] = defaultdict(set)
        self._blocks_of: Dict[Tuple, Set[Tuple]] = defaultdict(set)
        #: A reach a loop's next run tells lanes apart in (`_wrap`), by id:
        #: the reach and the reaches its lanes went to -- what a record made
        #: before holds of it, `_parts` and `_part` find there.
        self._split: Dict[int, Tuple[Reach, List[Reach]]] = {}
        self._root: Dict[int, Reach] = {}
        #: Buffers written by what could not be placed: every read of them
        #: may read that write.
        self._smeared: Dict[Tuple, List[int]] = defaultdict(list)
        #: The same for a write whose buffer nothing names: every buffer of
        #: its space.
        self._smeared_space: Dict[Any, List[int]] = defaultdict(list)
        self._fresh: Dict[int, int] = {}
        #: Each set of writers a record holds, once (`_frozen`).
        self._sets: Dict[FrozenSet[int], FrozenSet[int]] = {}
        #: Each cell of a register, once (`_place_one`): every access to it
        #: names the same cell.
        self._interned: Dict[Tuple, Tuple] = {}
        #: The lanes of each mask asked of `lanes`, once.
        self._lane_sets: Dict[int, Tuple[int, ...]] = {}
        #: Per statement, what placing it asks on every execution (`_plan`).
        self._plans: Dict[int, Tuple] = {}
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
        #: Per element of memory and lane: the loads of it whose value went
        #: into a register, the first first, and what each found there
        #: (`_fetches`).
        self._fetched: Dict[Tuple[Tuple, int], List[Tuple[int, Reach]]] = \
            defaultdict(list)
        #: Per execution of a load of memory: the registers its value went
        #: into, `(mask, cell, store)`.
        self._held: Dict[int, List[Tuple[int, Tuple, int]]] = defaultdict(list)
        self._run(self.body, {}, self.full, frozenset())
        # What a loop's next run hands back (`_wrap`) can make a read
        # uncertain after the fact, for some of its lanes; what they would
        # read instead goes with it.
        for n, found in self._forwarded.items():
            kept = []
            for m, o, rs in found:
                for r in rs:
                    for part in self._parts(r):
                        if not part.certain:
                            m &= ~part.mask
                if m:
                    kept.append((m, o))
            if kept:
                self.forward[n] = kept
        for where, pieces in self._mem.items():
            if where[0][0] is MemSpace.GLOBAL:
                for piece in pieces:
                    self.final.add(piece.writer)

    def cells_of(self, n: int) -> List[Tuple[int, Tuple]]:
        """The elements execution `n` touched, `(lanes, cell)`: a register's
        and what every lane reads of memory for the lanes, the rest of
        memory lane by lane."""
        out: List[Tuple[int, Tuple]] = []
        for m, cell, _ in self._cells.get(n, ()):
            if isinstance(cell, _Cells):
                out.extend(cell.expanded())
            else:
                out.append((m, cell))
        return out

    def _parts(self, r: Reach) -> List[Reach]:
        """The reaches the lanes of `r`, and of the reach it came from, went
        to: `r`, where no loop told them apart (`_wrap`)."""
        found = self._split.get(id(self._root.get(id(r), r)))
        return found[1] if found is not None else [r]

    def _part(self, r: Reach, lane: int) -> Reach:
        """The reach lane `lane` of `r` went to."""
        for part in self._parts(r):
            if part.mask >> lane & 1:
                return part
        return r

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
        inductions = tuple(l.induction.id for l in loops)
        for s in stmts:
            self.parents[id(s)] = parents
            self.loops[id(s)] = loops
            self._inductions[id(s)] = inductions
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
        hops = tuple(map(env.get, self._inductions.get(id(stmt), ())))
        return self._key.get((id(stmt), hops))

    def executions(self, stmt: Stmt) -> List[int]:
        return self.by_stmt.get(id(stmt), [])

    def lanes(self, mask: int) -> Tuple[int, ...]:
        """The lanes of `mask`, the lowest first."""
        out = self._lane_sets.get(mask)
        if out is None:
            out = self._lane_sets[mask] = tuple(
                lane for lane in range(mask.bit_length()) if mask >> lane & 1)
        return out

    # -- running ------------------------------------------------------------- #

    def _new(self, s: Stmt, env: Dict[int, int], mask: int, conds) -> int:
        inductions = self._inductions.get(id(s), ())
        trips = tuple(map(env.get, inductions))
        n = len(self.insts)
        self.insts.append(Inst(s, tuple(zip(inductions, trips)), mask, conds))
        self.by_stmt[id(s)].append(n)
        self._key[(id(s), trips)] = n
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
        lanes = self.lanes(mask)
        found = self._ix.evaluate_lanes(cond, env, lanes, {}, {})
        if not isinstance(found, tuple):
            return mask if found else 0
        for lane, v in zip(lanes, found):
            if v:
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
        # What the run writes, by the executions that touch it: a register's
        # cell, and a block of memory with the elements.  Everything a
        # statement that writes touches, as it may have written it.
        regs: Dict[Tuple, Set[int]] = defaultdict(set)
        mems: Dict[Tuple, List[Tuple[int, int]]] = defaultdict(list)
        for n in range(first, len(self.insts)):
            placed = self._cells.get(n)
            if not placed or not self._writes(self.insts[n].stmt):
                continue
            for _, cell, _ in placed:
                if isinstance(cell, _Cells):
                    if not self._iteration(cell.key, local):
                        for block, bits in cell.blocks().items():
                            mems[(cell.key, cell.lead, block)].append(
                                (n, bits))
                elif not self._iteration(cell[0], local):
                    regs[cell].add(n)
        if not regs and not mems:
            return
        for n in range(first, len(self.insts)):
            reaches = self.reach.get(n)
            if not reaches or all(r.writer is not None and r.writer >= first
                                  and not r.maybe for r in reaches):
                continue
            # Per lane, what the run writes of what the execution touches.
            handed: Dict[int, Set[int]] = defaultdict(set)
            for m, cell, _ in self._cells[n]:
                if isinstance(cell, _Cells):
                    for block, bits in cell.blocks().items():
                        for w, wrote in mems.get((cell.key, cell.lead, block),
                                                 ()):
                            hit = bits & wrote
                            if hit:
                                for lane in self.lanes(
                                        cell.lanes_in(block, hit)):
                                    handed[lane].add(w)
                else:
                    writers = regs.get(cell)
                    if writers:
                        for lane in self.lanes(m):
                            handed[lane].update(writers)
            if not handed:
                continue
            out: List[Reach] = []
            for reach in reaches:
                if (reach.writer is not None and reach.writer >= first
                        and not reach.maybe):
                    out.append(reach)
                    continue
                groups: Dict[FrozenSet[int], int] = {}
                for lane in self.lanes(reach.mask):
                    found = handed.get(lane)
                    key = self._frozen(found) if found else EMPTY
                    groups[key] = groups.get(key, 0) | (1 << lane)
                for part, writers in self._apart(reach, groups):
                    if writers:
                        part.maybe = part.maybe | writers
                        for w in writers:
                            self.readers[w].append((n, part.mask))
                    out.append(part)
            self.reach[n] = out

    def _apart(self, reach: Reach, groups: Dict[FrozenSet[int], int]
               ) -> List[Tuple[Reach, FrozenSet[int]]]:
        """`reach` for each lane group of `groups`, writers -> lanes: itself
        for the first, a reach of its own for each other one (`_parts`)."""
        items = list(groups.items())
        if len(items) == 1:
            return [(reach, items[0][0])]
        root = self._root.get(id(reach), reach)
        entry = self._split.get(id(root))
        if entry is None:
            entry = self._split[id(root)] = (root, [root])
        out = []
        for writers, lanes in items:
            if not out:
                part = reach
                reach.mask = lanes
            else:
                part = Reach(lanes, reach.writer, reach.maybe, reach.own,
                             reach.cell, reach.by)
                entry[1].append(part)
                self._root[id(part)] = root
            out.append((part, writers))
        return out

    @staticmethod
    def _iteration(key: Tuple, local: Set[int]) -> bool:
        """Is a buffer one an iteration has to itself: bound inside the
        region, which every run binds anew?"""
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
        self._cells[n] = placed
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
                if found:
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
                if self._fetches_too:
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
        if (d is not None and d.op == 'select'
                and d.predicate is None and self._static(d.args[0], env)):
            # A select over a condition of the lanes: the lanes it holds for
            # hold the first arm, the others the second -- the rows a store
            # writes and the zeros it pads them with past them, and under the
            # explicit vector a merge, a vector's tail and the zeros past it.
            inside = self._lanes_where(d.args[0], env, mask)
            out = []
            for lanes, arm in ((inside, d.args[1]),
                               (mask & ~inside, d.args[2])):
                if lanes:
                    out += self._arm(arm, env, lanes)
            return out
        return self._loaded(v, d, env, mask)

    def _arm(self, x, env, mask: int) -> List[Tuple[int, Tuple]]:
        """Where an arm `x` of a select came from for the lanes of `mask`:
        a number, or as `_loaded` says."""
        if not isinstance(x, Value):
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return [(mask, ('const', x))]
            return []
        d = self.defs.get(x.id)
        if d is not None and d.op == Op.CONST:
            return [(mask, ('const', d.attr('value')))]
        return self._loaded(x, d, env, mask)

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
        it still -- or, where another lane wrote the element last in a
        store this lane ran as well, what this lane stored, where that is a
        number or a value the same on every lane of the multiplication: the
        other lane stored the same.  With `exchange`, where another lane
        stored a value, or stored it from a register that lane holds still,
        that value or register of that lane (`('lane', lane, origin)`)."""
        out: List[Tuple[int, Tuple, Tuple[Reach, ...]]] = []
        for r in self.reach.get(n, ()):
            if not r.certain or r.writer is None or r.writer < 0:
                continue
            for m, o in self.origin.get(r.writer, ()):
                if o[0] == 'value' and not self._one_trip(o[1], r.writer, n):
                    # The store took the value in another trip of a loop that
                    # computes it anew in each: here it is another.
                    continue
                if (not r.own and self._exchange and o[0] in ('reg', 'value')
                        and r.by >= 0 and not self._same_on_every_lane(o)):
                    # Another lane's store, of a value -- the one it stored,
                    # on every lane -- or from a register that lane holds it
                    # in still: read across.
                    if o[0] == 'value' or self._holds(o[1], o[2], 1 << r.by,
                                                      conds):
                        out.append((r.mask, ('lane', r.by, o), (r,)))
                    continue
                hit = m & r.mask
                if not hit:
                    continue
                if not r.own and not self._same_on_every_lane(o):
                    continue
                if o[0] == 'reg':
                    _, cell, writer = o
                    if self._holds(cell, writer, hit, conds):
                        out.append((hit, ('reg', cell, writer), (r,)))
                else:
                    out.append((hit, o, (r,)))
        return out

    def _one_trip(self, v: Value, writer: int, n: int) -> bool:
        """Is `v` at the load's execution `n` what it was at the store's
        execution `writer`: are both in the same trip of every counted loop
        that computes it anew in each -- those around its definition, and the
        loop itself for an argument of its body?"""
        d = self.defs.get(v.id)
        if d is None:
            return True
        loops = self.loops.get(id(d), ())
        if d.op == Op.FOR and not any(t is v for t in d.target):
            loops = loops + (d,)
        if not loops:
            return True
        here = dict(self.insts[n].hops)
        there = dict(self.insts[writer].hops)
        return all(l.induction.id in here and l.induction.id in there
                   and here[l.induction.id] == there[l.induction.id]
                   for l in loops)

    def _same_on_every_lane(self, origin: Tuple) -> bool:
        """Is what a store took its value from, in `origin`'s form, the same
        on every lane of the multiplication: a number, or a value
        `_Alike` says is?"""
        if origin[0] == 'const':
            return True
        if origin[0] != 'value':
            return False
        return self.alike(origin[1])

    def alike(self, v) -> bool:
        """Is `v` the same on every lane of the multiplication (`_Alike`)?"""
        if self._alike is None:
            self._alike = _Alike(self.defs)
        return self._alike(v)

    def _fetches(self, n: int, conds, taken: int) -> List[Tuple[int, Tuple, Tuple[Reach, ...]]]:
        """What the load's execution `n` of memory reads for certain because
        an earlier load by the same lanes read it: the element unchanged in
        between -- both found the same write there -- and the register that
        earlier load's value went into holding it still.  Not for the lanes
        in `taken`, which read what their own store put there, and take
        that.

        The first such load whose register holds it, not the last: a third
        staging of an element reads the register of the first, and not the
        second's, which the same round takes from the first in turn -- a
        chain one round of the forwarding would leave half done."""
        out: List[Tuple[int, Tuple, Tuple[Reach, ...]]] = []
        for r in self.reach.get(n, ()):
            if not r.certain or not r.cell or isinstance(r.cell, tuple):
                # Of a register, which it reads itself.
                continue
            left = r.mask & ~taken
            if not left:
                continue
            # Lane by lane, and the lanes that found the same together: the
            # lanes may each have loaded their element before in a load of
            # their own.
            found_by: Dict[Tuple, int] = {}
            reaches: Dict[Tuple, Reach] = {}
            for lane in self.lanes(left):
                bit = 1 << lane
                for earlier, found in self._fetched.get(
                        (r.cell_at(lane), lane), ()):
                    if (found.writer != r.writer
                            or not self._part(found, lane).certain):
                        continue
                    holder = next(((cell, store) for m, cell, store
                                   in self._held.get(earlier, ())
                                   if m & bit
                                   and self._holds(cell, store, bit, conds)),
                                  None)
                    if holder is not None:
                        key = (holder, id(found))
                        found_by[key] = found_by.get(key, 0) | bit
                        reaches[key] = found
                        break
            for key, lanes in found_by.items():
                out.append((lanes, ('reg',) + key[0], (r, reaches[key])))
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
                if isinstance(cell, tuple) and cell[0][0] is MemSpace.REGISTER]
        if not held:
            return
        self._held[load].extend(held)
        for r in self.reach.get(load, ()):
            if r.cell:
                for lane in self.lanes(r.mask):
                    self._fetched[(r.cell_at(lane), lane)].append((load, r))

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
        if space is MemSpace.REGISTER:
            for cell in self._cells_of.pop((space, base), ()):
                self._reg.pop(cell, None)
        else:
            for where in self._blocks_of.pop((space, base), ()):
                self._mem.pop(where, None)
        if fresh is not None:
            self._fresh[base] = fresh
        else:
            self._fresh.pop(base, None)
        self._smeared.pop((space, base), None)

    def _place(self, s: Stmt, env, mask):
        """`[(mask, cell, kind)]` for the elements `s` touches, or None where
        they cannot be told."""
        if self.simd:
            return self._place_vector(s, env, mask)
        plan = self._plans.get(id(s))
        if plan is None:
            plan = self._plans[id(s)] = self._plan(s, env)
        if not plan:
            return None
        if len(plan) == 1:
            if plan[0] is None:
                return None
            return self._place_one(*plan[0], env, mask)
        # A copy reads its source and writes its destination; its value is
        # none a pass could name.
        found = [None if part is None else self._place_one(*part, env, mask)
                 for part in plan]
        if any(f is None for f in found):
            return None
        return [p for f in found for p in f]

    def _plan(self, s: Stmt, env) -> Tuple:
        """What placing `s` asks on every execution of it the same:
        `(binding, access, index, width, kind, uniform)` an access, None for
        one placed nowhere -- or nothing, where `s` places none.  The leaves
        of an index every execution binds are the same: the inductions of
        the counted loops around `s`."""
        if s.op == Op.LOAD and len(s.accesses) == 1:
            a = s.accesses[0]
            parts = [(s.args[0], a, s.args[1:],
                      _width(s.target[0] if s.target else None, a,
                             s.args[0]), 'read')]
        elif s.op in (Op.STORE, Op.MARK) and len(s.accesses) == 1:
            # A promise of a zero (`mark zeros`) as the store of that zero.
            a = s.accesses[0]
            parts = [(s.args[0], a, s.args[2:], _width(s.args[1], a, s.args[0]),
                      'rmw' if a.kind & Effect.ATOMIC else 'write')]
        elif s.op == Op.LOAD_ASYNC and len(s.accesses) == 1:
            a = s.accesses[0]
            parts = [(s.load_base, a, s.load_index,
                      _width(s.target[0] if s.target else None, a,
                             s.load_base), 'read')]
        elif s.op == Op.COPY_ASYNC:
            reads = [a for a in s.accesses if a.kind & Effect.READ]
            writes = [a for a in s.accesses if a.kind & Effect.WRITE]
            if len(reads) != 1 or len(writes) != 1:
                return ()
            elems = s.attr('elems', 1) or 1
            parts = [(s.copy_src, reads[0], s.copy_src_index, elems, 'read'),
                     (s.copy_dst, writes[0], s.copy_dst_index, elems,
                      'write')]
        else:
            return ()
        out = []
        for binding, a, index, width, kind in parts:
            if (a.space not in _PLACED or a.base is None
                    or not all(self._static(i, env) for i in index)):
                out.append(None)
                continue
            # An element no lane tells apart from the others' is one for
            # all of them: a register, which each lane holds of its own, and
            # an element of memory they read -- a broadcast, which a read
            # tells the lane that wrote it apart in (`_reaches`).  A write of
            # one element by several lanes is the last lane's: what memory
            # holds.
            uniform = ((a.space is MemSpace.REGISTER or kind == 'read')
                       and not any('x' in self._ix.leaves(i) for i in index
                                   if isinstance(i, Value)))
            out.append((binding, a, tuple(index), width, kind, uniform))
        return tuple(out)

    def _place_one(self, binding, a, index, width, kind, uniform, env, mask):
        """One access of `_plan`'s, for the lanes of `mask` at `env`."""
        key = self._buffer(a, binding, env)
        if key is None:
            return None
        if a.space is not MemSpace.REGISTER:
            out = self._memory(key, index, width, kind, env, mask, uniform)
        elif uniform:
            out = []
            interned = self._interned
            for addr in _span(self._address(index, env), width):
                cell = (key, addr)
                out.append((mask, interned.setdefault(cell, cell), kind))
        else:
            # The index for every lane at once (`_Indices.evaluate_lanes`).
            out = []
            interned = self._interned
            lanes = self.lanes(mask)
            memo: Dict = {}
            fixed: Dict[int, int] = {}
            found = [self._ix.evaluate_lanes(i, env, lanes, memo, fixed)
                     for i in index]
            for k, lane in enumerate(lanes):
                addr = tuple(int(v[k]) if isinstance(v, tuple) else int(v)
                             for v in found)
                for at in _span(addr, width):
                    cell = (key, at)
                    out.append((1 << lane, interned.setdefault(cell, cell),
                                kind))
        self._spend(len(out))
        return out

    def _spend(self, placed: int) -> None:
        self._budget -= placed
        if self._budget < 0:
            raise TooLarge(f'the body places more than {self._limit} '
                           f'accesses')

    def _memory(self, key, index, width: int, kind: str, env, mask: int,
                uniform: bool) -> List[Tuple[int, _Cells, str]]:
        """What an access of memory reaches: the elements of its lanes, one
        `_Cells` a row of the buffer they land in -- the index for every
        lane at once (`_Indices.evaluate_lanes`)."""
        lanes = self.lanes(mask)
        if not lanes:
            return []
        if uniform:
            addr = self._address(index, env)
            if not addr:
                return [(mask, _Cells(key, None, lanes, 0, 1, mask), kind)]
            return [(mask, _Cells(key, addr[:-1], lanes, addr[-1], width,
                                  mask), kind)]
        memo: Dict = {}
        fixed: Dict[int, int] = {}
        found = [self._ix.evaluate_lanes(i, env, lanes, memo, fixed)
                 for i in index]
        if not found:
            return [(mask, _Cells(key, None, lanes, (0,) * len(lanes), 1,
                                  mask), kind)]
        last = found[-1]
        starts = last if isinstance(last, tuple) else (last,) * len(lanes)
        lead = found[:-1]
        if not any(isinstance(v, tuple) for v in lead):
            return [(mask, _Cells(key, tuple(lead), lanes, starts, width,
                                  mask), kind)]
        # The lanes apart by the row they reach, in the order they run.
        rows: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        for k, lane in enumerate(lanes):
            row = tuple(v[k] if isinstance(v, tuple) else v for v in lead)
            rows.setdefault(row, []).append((lane, starts[k]))
        out = []
        for row, members in rows.items():
            m = 0
            for lane, _ in members:
                m |= 1 << lane
            out.append((m, _Cells(key, row, tuple(l for l, _ in members),
                                  tuple(s for _, s in members), width, m),
                        kind))
        return out

    def _place_vector(self, s: Stmt, env, mask):
        """`_place` under the explicit vector: a load, a store and a promise
        of a zero in a register array, of one element a lane.

        A value spread over the lanes is one vector, and an access of it
        reaches an element a lane: lane `l`'s at the index plus `l` -- the
        vector the emitter reads or writes from the index on (`EsimdEmitter`),
        as many lanes as the value spans, the first of the multiplication's.
        In memory that is the element, and every lane may read what another
        wrote.  A register array is the work-item's own, but where every
        vector it is read and written as starts at a multiple of the lanes,
        an element is one lane's in every one of them: the array is placed as
        under SPMD, a register of every lane's own at the vector's index.  A
        value every lane holds whole is one element, every lane's, in memory;
        in a register array it is a scalar part of a vector, and placed
        nowhere.  So is anything of more than one element a lane, of more
        lanes than the multiplication's, or of part of its vector that
        starts past its first lane -- a window's head.  A full-lane tail
        touches its lanes up to `valid`, in memory and in a load of a
        register array, which gives the lanes past it zeros; a store into
        a register array writes every lane of the value whatever `valid`
        says, and is placed as one that does."""
        if s.op == Op.LOAD and len(s.accesses) == 1 and s.target:
            value, index, kind = s.target[0], s.args[1:], 'read'
        elif s.op == Op.STORE and len(s.accesses) == 1:
            value, index = s.args[1], s.args[2:]
            kind = 'rmw' if s.accesses[0].kind & Effect.ATOMIC else 'write'
        elif (s.op == Op.MARK and s.attr('mark') == 'zeros'
              and len(s.accesses) == 1
              and s.accesses[0].space is MemSpace.REGISTER):
            # The zero of a register a lane holds: no value, every lane's.
            value, index, kind = None, s.args[2:], 'write'
        else:
            return None
        a = s.accesses[0]
        if (a.space not in _PLACED or a.base is None
                or s.attr('head') is not None
                or not all(self._static(i, env) for i in index)):
            return None
        if value is not None and (
                not isinstance(value, Value) or value.layout is None
                or getattr(value.type, 'length', None) is not None):
            return None
        spread = value is None or value.layout.is_distributed
        if spread and value is not None:
            span = value.lane_span()
            if span > self.threads:
                return None
            mask &= (1 << span) - 1
        valid = s.attr('valid')
        if valid is not None and not (
                a.space is MemSpace.REGISTER and kind != 'read'):
            if not spread:
                return None
            mask &= (1 << valid) - 1
        key = self._buffer(a, s.args[0], env)
        addr = self._address(index, env)
        if key is None or not addr:
            return None
        out: List[Tuple[int, Any, str]] = []
        if a.space is MemSpace.REGISTER:
            if not spread or addr[-1] % self.threads:
                return None
            if mask:
                cell = (key, addr)
                out.append((mask, self._interned.setdefault(cell, cell),
                            kind))
        else:
            lanes = self.lanes(mask)
            if lanes:
                starts = (tuple(addr[-1] + lane for lane in lanes) if spread
                          else (addr[-1],) * len(lanes))
                out.append((mask, _Cells(key, addr[:-1], lanes, starts, 1,
                                         mask), kind))
        self._spend(len(out))
        return out

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
        layers = self._reg.get(cell)
        if layers is None:
            layers = self._reg[cell] = []
            self._cells_of[cell[0][:2]].add(cell)
        return layers

    def _shared(self, key: Tuple) -> bool:
        """Is the buffer of `key` shared memory lanes of other
        multiplications reach?"""
        return key[0] is MemSpace.SHARED and key[1] not in self._own

    def _read(self, n: int, placed, conds) -> List[Reach]:
        out: List[Reach] = []
        #: Per buffer, what a read of it may find besides what is placed.
        others: Dict[Tuple, FrozenSet[int]] = {}
        for m, cell, _ in placed:
            memory = isinstance(cell, _Cells)
            key = cell.key if memory else cell[0]
            smeared = others.get(key[:2])
            if smeared is None:
                smeared = frozenset(self._smeared.get(key[:2], ())) | \
                    frozenset(self._smeared_space.get(key[0], ()))
                if self._shared(key):
                    smeared = smeared | {EXTERNAL}
                others[key[:2]] = smeared = self._frozen(smeared)
            if memory:
                out.extend(self._read_cells(n, cell, conds, smeared))
            else:
                out.extend(self._read_register(n, m, cell, self._layers(cell),
                                               conds, smeared))
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

    def _read_cells(self, n: int, cells: _Cells, conds,
                    smeared: FrozenSet[int]) -> List[Reach]:
        """What the lanes read of `cells`, elements of memory: of each
        element, the last write they stand under for certain, and those of
        uncertain standing after it, which they may read as well -- the
        elements that found alike together, block by block."""
        found: List[Tuple[int, int, Optional[_Piece], FrozenSet[int]]] = []
        mem = self._mem
        readers = self.readers
        key, lead = cells.key, cells.lead
        for block, bits in cells.blocks().items():
            pieces = mem.get((key, lead, block))
            pending = [(bits, smeared)]
            if pieces:
                for piece in reversed(pieces):
                    later = []
                    for left, maybe in pending:
                        hit = left & piece.bits
                        if not hit:
                            later.append((left, maybe))
                            continue
                        readers[piece.writer].append(
                            (n, cells.lanes_in(block, hit)))
                        if piece.conds <= conds:
                            found.append((block, hit, piece, maybe))
                        else:
                            later.append((hit, self._frozen(
                                maybe | {piece.writer})))
                        if left != hit:
                            later.append((left & ~hit, maybe))
                    pending = later
                    if not pending:
                        break
            for left, maybe in pending:
                found.append((block, left, None, maybe))
        return self._reaches(cells, found)

    @staticmethod
    def _reaches(cells: _Cells, found) -> List[Reach]:
        """`found`, `(block, bits, piece, maybe)` -- what a read of `cells`
        found of the elements of `bits`: the write of `piece` for certain,
        or nothing the body wrote where there is none, and those of `maybe`
        as well -- as the reaches of the lanes: a lane's own store apart
        from another lane's, by the lane that stored it, the lanes that
        read alike together, `k` past their start apart."""
        groups: Dict[Tuple, int] = {}
        for block, bits, piece, maybe in found:
            writer = INITIAL if piece is None else piece.writer
            if not cells.broadcast and (piece is None
                                        or piece.cells.alike(cells)):
                # Every lane reads its own elements: ones nothing wrote, or
                # ones it wrote itself, in a write lane by lane alike.
                if cells.width == 1:
                    key = (writer, maybe, piece is not None, -1, 0)
                    groups[key] = groups.get(key, 0) | cells.lanes_in(block,
                                                                      bits)
                    continue
                if bits == cells.blocks()[block]:
                    for k, lanes in cells.lanes_at(block):
                        key = (writer, maybe, piece is not None, -1, k)
                        groups[key] = groups.get(key, 0) | lanes
                    continue
            readers = cells.readers()
            base = block << 6
            while bits:
                low = bits & -bits
                bits ^= low
                e = base + low.bit_length() - 1
                lane = -1 if piece is None else piece.cells.lane_of(e)
                for lanes, k in readers[e]:
                    own = lanes & (1 << lane) if lane >= 0 else 0
                    if own:
                        key = (writer, maybe, True, -1, k)
                        groups[key] = groups.get(key, 0) | own
                    if lanes & ~own:
                        key = (writer, maybe, False, lane, k)
                        groups[key] = groups.get(key, 0) | (lanes & ~own)
        return [Reach(lanes, writer, maybe, own=own, cell=cells.at(k), by=by)
                for (writer, maybe, own, by, k), lanes in groups.items()]

    def _write(self, n: int, placed, conds) -> None:
        before: Dict[Tuple[int, Tuple], Tuple[Optional[int], bool]] = {}
        for m, cell, _ in placed:
            if isinstance(cell, _Cells):
                self._write_cells(n, cell, conds,
                                  before if self._before else None)
                continue
            layers = self._layers(cell)
            for lane in self.lanes(m) if self._before else ():
                prior, certain = INITIAL, True
                for layer in reversed(layers):
                    if not layer.mask & (1 << lane):
                        continue
                    if layer.conds <= conds:
                        prior = layer.writer
                        break
                    certain = False
                if prior == INITIAL:
                    prior = self._fresh.get(cell[0][1], INITIAL)
                if (self._smeared.get(cell[0][:2])
                        or self._smeared_space.get(cell[0][0])):
                    certain = False
                before[(lane, cell)] = (prior, certain)
            kept: List[_Layer] = []
            for layer in layers:
                if layer.conds >= conds:
                    layer.mask &= ~m
                    if not layer.mask:
                        continue
                kept.append(layer)
            kept.append(_Layer(n, conds, mask=m))
            if len(kept) > _DEPTH:
                kept = kept[-_DEPTH:]
            self._reg[cell] = kept
        if self._before:
            self.before[n] = before

    def _write_cells(self, n: int, cells: _Cells, conds,
                     before: Optional[Dict]) -> None:
        """The write of `cells`, elements of memory, block by block: what it
        leaves of the writes before it, the ones it stands under at least
        the conditions of gone, as it is the last write of its elements
        wherever they are; of each element, `_DEPTH` writes at most."""
        mem = self._mem
        key, lead = cells.key, cells.lead
        for block, bits in cells.blocks().items():
            where = (key, lead, block)
            pieces = mem.get(where)
            if pieces is None:
                pieces = mem[where] = []
                self._blocks_of[key[:2]].add(where)
            if before is not None:
                self._before_cells(n, cells, block, pieces, conds, before)
            kept: List[_Piece] = []
            for piece in pieces:
                if piece.conds >= conds:
                    piece.bits &= ~bits
                    if not piece.bits:
                        continue
                kept.append(piece)
            kept.append(_Piece(n, conds, cells, bits))
            if len(kept) > _DEPTH:
                kept = _deepest(kept, bits)
            mem[where] = kept
        if self._shared(key):
            self.escaped.add(n)

    def _before_cells(self, n: int, cells: _Cells, block: int,
                      pieces: List[_Piece], conds, before: Dict) -> None:
        """`before` for the elements the write of `cells` makes in `block`,
        lane by lane: the write each held right before, and whether for
        certain -- a lane's that another lane of the same write wrote ahead
        of it, that write."""
        key = cells.key
        uncertain = bool(self._smeared.get(key[:2])
                         or self._smeared_space.get(key[0])
                         or self._shared(key))
        done: Set[int] = set()
        for lane, k, e in cells.each():
            if e >> 6 != block:
                continue
            if e in done:
                prior, certain = n, True
            else:
                done.add(e)
                prior, certain = INITIAL, True
                bit = 1 << (e & 63)
                for piece in reversed(pieces):
                    if not piece.bits & bit:
                        continue
                    if piece.conds <= conds:
                        prior = piece.writer
                        break
                    certain = False
            before[(lane, cells.cell(lane, k))] = (prior,
                                                    certain and not uncertain)

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
                    held = (self._cells_of if space is MemSpace.REGISTER
                            else self._blocks_of)
                    keys = ([(space, base)] if base is not None else
                            [k for k in held if k[0] is space])
                    for key in keys:
                        for where in held.get(key, ()):
                            for layer in table.get(where, ()):
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


class _Alike:
    """Whether a value is the same on every lane of the multiplication: as
    its uniformity says (`Value.uniformity`), where nothing it is computed
    from was read out of registers spread over the lanes in a way nobody
    stated.

    A read's uniformity is joined from its index, and the index of a read
    of registers is the same in every lane whatever each lane holds there:
    the answer is the address's, written on the value.  A read whose layout
    is known has the value's answer (`Value.__post_init__`); one whose
    layout is not, has none.  What a loop carries is the same on every lane
    where what it starts from is and what each trip hands the next is,
    given that what the trip started from was; what any other region yields
    is not asked here."""

    def __init__(self, defs: Dict[int, Stmt]):
        self._defs = defs
        self._memo: Dict[int, bool] = {}

    def __call__(self, v) -> bool:
        if not isinstance(v, Value):
            return True
        if v.uniformity < Uniformity.MULT:
            return False
        hit = self._memo.get(v.id)
        if hit is not None:
            return hit
        # A cycle that does not go through what a loop carries proves
        # nothing.
        self._memo[v.id] = False
        d = self._defs.get(v.id)
        out = True
        if d is not None:
            if d.op == Op.FOR:
                out = self._loop(d)
            elif d.regions:
                out = False
            elif d.attr('alike'):
                # The fold of what every lane of the multiplication held,
                # which each of them holds alike (`IRBuilder.all_reduce`).
                out = True
            elif v.layout is None and any(
                    a.space in (MemSpace.REGISTER, MemSpace.UNKNOWN)
                    for a in d.accesses):
                out = False
            else:
                out = all(self(x) for x in d.operands()
                          if not isinstance(x.type, BufferType))
        self._memo[v.id] = out
        return out

    def _loop(self, d: Stmt) -> bool:
        """The induction, what the loop carries and what it leaves: taken to
        be the same on every lane while what a trip hands the next is asked,
        and what was concluded from that dropped where it is not."""
        carried = (d.induction,) + tuple(d.iter_args) + tuple(d.target)
        before = set(self._memo)
        for c in carried:
            self._memo[c.id] = True
        out = (all(self(x) for x in d.operands())
               and all(self(y) for y in d.regions[0].yielded))
        if not out:
            for k in set(self._memo) - before:
                del self._memo[k]
            for c in carried:
                self._memo[c.id] = False
        return out


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


def _deepest(pieces: List[_Piece], bits: int) -> List[_Piece]:
    """`pieces` with each element of `bits` in the `_DEPTH` last of them
    at most."""
    while bits:
        low = bits & -bits
        bits ^= low
        seen = 0
        for piece in reversed(pieces):
            if piece.bits & low:
                seen += 1
                if seen > _DEPTH:
                    piece.bits &= ~low
    return [piece for piece in pieces if piece.bits]


def _span(addr: Tuple[int, ...], width: int) -> List[Tuple[int, ...]]:
    """The elements a `width`-wide access at `addr` covers: the last index
    runs on."""
    if width <= 1 or not addr:
        return [addr]
    return [addr[:-1] + (addr[-1] + k,) for k in range(width)]
