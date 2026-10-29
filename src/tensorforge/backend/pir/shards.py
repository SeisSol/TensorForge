# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: the shards of the batch-constant operands that the block's
shared memory holds.

An operand without a batch offset -- an operator every element multiplies
with -- is read from global memory inside the batch loop, the same numbers
for every element, unless `preload_globals` stages it into shared memory
ahead of the loop.  That is all or nothing per operand, and at order 8 in
double precision one operator alone is past the 64 KiB an AMD block has:
nothing is staged at all.

This keeps what fits, a shard at a time.  A shard is what one load of the
loop reads across the lanes of the block -- a lane block of a column,
typically -- and loads whose reads overlap share one.  The shards that fit
into the budget the pass is handed (`Generator._shard_budget`) are copied in
ahead of the loop by the threads of the block, behind a barrier, and their
loads read them there; the rest stay where they are.  A shard keeps the order
its operand stores its elements in, so a load's index moves by a constant:
where the shard starts in the buffer less where it starts in the operand.
And shards of one length at one distance from each other -- a lane block of
every column -- are copied by one copy (`_copies`).

Which come first: the shards whose loads leave the least of their statement
list between themselves and their first reader (`_Load.cover`).  A load from
global memory is waited for where nothing stands between it and its use, and
that is the latency shared memory takes away; a load with a long cover has it
hidden already.

A load is a candidate where its operand is a read-only buffer bound ahead of
the loop and its index is known for every thread: a function of the thread's
indices, of the counted loops it stands in and of constants (`_Indices`).  A
load whose lanes all read one element is not one: it is a scalar load, which
on AMD comes out of the constant space as cheaply as anything here could, and
it would take a lane block's worth of shared memory for one element.  A load
in a counted loop is one statement for all of its trips, so its shard spans
them and is held whole or not at all -- in a rolled reduction (`k_roll`),
typically all of an operator.  The loads of the elements it spans keep
shards of their own (`_shards`), and what both hold is held twice.
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

from .core import (BOOL, INDEX, Access, BufferType, Effect, MemSpace, Op,
                   Participants, ScalarType, Stmt, Uniformity, Value,
                   accesses_conflict, walk_stmts)


class _NotStatic(Exception):
    """An index, or a condition over one, that is not known per thread."""


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


#: What the integer arithmetic of an index means, as C++ has it.
_BIN = {
    'add': lambda a, b: a + b, 'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _trunc_div, 'rem': lambda a, b: a - b * _trunc_div(a, b),
    'min': min, 'max': max,
    'lt': lambda a, b: a < b, 'le': lambda a, b: a <= b,
    'gt': lambda a, b: a > b, 'ge': lambda a, b: a >= b,
    'eq': lambda a, b: a == b, 'ne': lambda a, b: a != b,
    'and': lambda a, b: bool(a) and bool(b),
    'or': lambda a, b: bool(a) or bool(b),
    'bitand': lambda a, b: a & b, 'bitor': lambda a, b: a | b,
    'bitxor': lambda a, b: a ^ b,
    'shl': lambda a, b: a << b, 'shr': lambda a, b: a >> b,
}

#: The same for the one-operand ops `_BIN` cannot hold.
_UN = {'neg': lambda a: -a, 'popcount': lambda a: int(a).bit_count()}

#: The thread's indices, by the axis they are leaves for.
_THREAD = {'thread_idx_x': 'x', 'thread_idx_y': 'y'}


class _Indices:
    """The elements an index reads, thread by thread.

    The value of an operand is computed from the statements that define it:
    constants, integer arithmetic, the thread's indices and the inductions of
    the counted loops a load stands in.  Those last two are its leaves, and a
    load is evaluated once for every combination of the leaves its index and
    its conditions reach -- the lanes of a multiplication, times the trips of
    the loops around it, times the multiplications of a block where the
    index tells them apart.
    """

    def __init__(self, body: Tuple[Stmt, ...], extents: Mapping[str, int],
                 simd: bool = False):
        self._defs: Dict[int, Stmt] = {}
        for s in walk_stmts(body):
            for t in s.target:
                self._defs[t.id] = s
        #: Per axis, how many threads a block has along it.
        self._extents = dict(extents)
        #: Under the explicit vector, the lanes' own index is a vector of
        #: as many lanes as `x` has (`IRBuilder.lane_index`, `lanes`): the
        #: thread index, lane by lane.
        self._simd = simd
        self._leaves: Dict[int, frozenset] = {}

    def _lanes(self, s: Stmt) -> bool:
        """Is `s` the lanes' own index under the explicit vector -- `0, 1,
        ...` over every lane of `x`?"""
        if not self._simd or s.op != Op.RAWEXPR or s.args:
            return False
        if s.attr('lanes') != self._extents.get('x'):
            return False
        axes = getattr(getattr(s.target[0], 'layout', None), 'axes', ())
        return len(axes) == 1 and axes[0].stride == 1

    def leaves(self, operand) -> frozenset:
        """What `operand` varies over: thread axes, by name, and the ids of
        the values nothing in the body defines -- the inductions."""
        if not isinstance(operand, Value):
            if isinstance(operand, (bool, int)):
                return frozenset()
            raise _NotStatic(repr(operand))
        found = self._leaves.get(operand.id)
        if found is not None:
            return found
        s = self._defs.get(operand.id)
        if s is None:
            found = frozenset((operand.id,))
        elif (s.op == Op.CALL and not s.args
              and s.attr('callee') in _THREAD):
            found = frozenset((_THREAD[s.attr('callee')],))
        elif self._lanes(s):
            found = frozenset(('x',))
        elif s.op == Op.CONST:
            if not isinstance(s.attr('value'), (bool, int)):
                raise _NotStatic(f'const {s.attr("value")!r}')
            found = frozenset()
        elif ((s.op in _BIN or s.op in _UN or s.op == 'select')
              and s.predicate is None):
            found = frozenset().union(*(self.leaves(a) for a in s.args))
        else:
            raise _NotStatic(s.op)
        self._leaves[operand.id] = found
        return found

    def evaluate(self, operand, env: Dict, memo: Dict[int, int]) -> int:
        """`operand` where the leaves are as `env` says; `leaves` has read
        it first."""
        if not isinstance(operand, Value):
            return int(operand)
        if operand.id in env:
            return env[operand.id]
        known = memo.get(operand.id)
        if known is not None:
            return known
        s = self._defs[operand.id]
        if s.op == Op.CALL:
            out = env[_THREAD[s.attr('callee')]]
        elif s.op == Op.RAWEXPR:
            # The lanes' own index (`_lanes`; `leaves` asked first).
            out = env['x']
        elif s.op == Op.CONST:
            out = int(s.attr('value'))
        elif s.op == 'select':
            cond, a, b = s.args
            out = self.evaluate(a if self.evaluate(cond, env, memo) else b,
                                env, memo)
        elif s.op in _UN:
            out = int(_UN[s.op](self.evaluate(s.args[0], env, memo)))
        else:
            a, b = s.args
            out = int(_BIN[s.op](self.evaluate(a, env, memo),
                                 self.evaluate(b, env, memo)))
        memo[operand.id] = out
        return out

    def domain(self, leaves: frozenset,
               loops: Mapping[int, Sequence[int]]) -> Iterator[Dict]:
        """Every combination of `leaves`: the thread indices over the block,
        each induction over its loop's trips."""
        names, ranges = [], []
        for leaf in sorted(leaves, key=str):
            if leaf in self._extents:
                ranges.append(range(self._extents[leaf]))
            elif leaf in loops:
                ranges.append(loops[leaf])
            else:
                raise _NotStatic('a value that is neither a thread index '
                                 'nor the induction of a counted loop')
            names.append(leaf)
        for values in itertools.product(*ranges):
            yield dict(zip(names, values))


@dataclass
class _Load:
    """A load of a batch-constant operand, and the span its lanes read."""
    stmt: Stmt
    buffer: Value
    lo: int
    hi: int
    #: Statements of its list between it and its first reader.
    cover: int
    #: Where it stands in the loop, for an order among equals.
    position: int
    #: Whether its span is over the trips of a counted loop it stands in.
    trips: bool = False
    #: The trip of a merged run whose member it reads, a member a trip
    #: (`_streamed`); None for an operand bound ahead of the batch loop.
    trip: Optional[Stmt] = None


@dataclass
class _Shard:
    """Elements `lo` to `hi` of `buffer`: what overlapping loads read --
    those whose span is over a counted loop's trips apart from the rest."""
    buffer: Value
    lo: int
    hi: int
    loads: List[_Load] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.hi - self.lo + 1

    @property
    def trip(self) -> Optional[Stmt]:
        """The trip it is copied in at the head of, where it is a member's."""
        return self.loads[0].trip

    @property
    def cover(self) -> int:
        return min(load.cover for load in self.loads)

    @property
    def position(self) -> int:
        return min(load.position for load in self.loads)


def shard_loads(body: Tuple[Stmt, ...], scratch: Callable[[], object], *,
                arena: str, align: int, budget: int, mults: int,
                threads: int, fptype: Datatype, static: bool = True,
                stream: bool = False,
                aligns: Optional[Mapping[str, int]] = None,
                extents: Optional[Mapping[str, int]] = None,
                report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with the shards that fit into `budget` elements of shared
    memory read from there.

    `arena` is the block's arena, which the shards go into, each on the
    alignment its first element has in the operand modulo `align`.
    `threads` is how many lanes one multiplication spans and `mults` how
    many multiplications the block holds, whose threads together copy the
    shards in.  `static` takes the operands bound ahead of the batch loop,
    copied in once ahead of it; `stream` the members of a merged run, copied
    in at the head of every trip (`_streamed`) -- which the caller promises
    every multiplication of the block takes together.  Both are held under
    the one budget.  `aligns` is what the caller proves of the operands'
    bases, in bytes, by the name of their binding: how wide the copies may
    read them.  `extents` is how many elements one of them is stored over,
    by the same name, where that is more than its binding's type says
    (`_extents`).  `report` collects a line per operand: `+` and how much of
    it is held, `-` and why nothing is.
    """
    def note(line: str) -> None:
        if report is not None:
            report.append(line)

    loops = [i for i, s in enumerate(body)
             if (s.op is Op.FOR and s.attr('next') is not None)
             or (s.op is Op.RAWBLOCK and s.attr('batch'))]
    if len(loops) != 1:
        note(f'- {len(loops)} batch loops in the body, where one is held')
        return body
    at = loops[0]
    reach = _extents(body, extents or {})
    loads = (_candidates(body, at, threads, mults, fptype, reach, note)
             if static else [])
    if stream:
        loads += _streamed(body, at, threads, mults, fptype, reach, note)
    shards = _shards(loads)
    held = _choose(shards, budget, align)
    if not held:
        note('- no shard fits' if shards else '- no load to hold')
        return body
    return _rewrite(body, at, held, shards, scratch, arena, threads, mults,
                    align, _alignments(body, at, aligns or {}), note)


def _bindings(body: Tuple[Stmt, ...], upto: int, fptype: Datatype
              ) -> Dict[int, Value]:
    """The read-only buffers in global memory bound ahead of the loop, of
    the kernel's own type."""
    out = {}
    for s in body[:upto]:
        for t in s.target:
            ty = t.type
            if (isinstance(ty, BufferType) and ty.space is MemSpace.GLOBAL
                    and ty.readonly and ty.elem == fptype):
                out[t.id] = t
    return out


def _first_reads(stmts: Tuple[Stmt, ...]) -> Dict[int, int]:
    """For every value read in `stmts`, where it is first read."""
    out: Dict[int, int] = {}
    for n, s in enumerate(stmts):
        for x in walk_stmts((s,)):
            for v in x.operands():
                out.setdefault(v.id, n)
    return out


def _extents(body: Tuple[Stmt, ...],
             extents: Mapping[str, int]) -> Dict[int, int]:
    """How many elements the buffers in `body` are stored over, by value,
    where the caller knows better than their types: an operand bound ahead
    of the batch loop by the name of its binding (`extents`), and a pointer
    chosen among such operands -- a merged run's member, through its table --
    by the largest of them.

    A binding is typed when it is bound, from the box, and a multiplication
    may store its operand in an order of its own after that
    (`storage_order`): two lead slots of a lane as a pair, the rows padded to
    the lanes, longer than the box.  A shard clamped to the type would leave
    out what the loads read past it."""
    out: Dict[int, int] = {}
    for s in walk_stmts(body):
        if len(s.target) != 1 or not isinstance(s.target[0].type, BufferType):
            continue
        # A merged run's binding has a name of its own as well, the stand-in's,
        # whose box the members' orders do not reach the stand-in through.
        known = [extents[s.attr('extern')]] if s.attr('extern') in extents \
            else []
        sources = [a for a in s.args if isinstance(a, Value)
                   and isinstance(a.type, BufferType)]
        if (s.op is Op.RAWEXPR and sources
                and all(a.id in out for a in sources)):
            known.append(max(out[a.id] for a in sources))
        if known:
            out[s.target[0].id] = max(known)
    return out


def _within(buffer: Value, span: Tuple[int, int, bool],
            reach: Mapping[int, int]) -> Optional[Tuple[int, int]]:
    """`span` clamped to what `buffer` is stored over -- the elements a load
    reads past it belong to lanes its guard leaves out -- or None where
    nothing of it is left."""
    last = max(buffer.type.volume, reach.get(buffer.id, 0)) - 1
    lo, hi = max(0, span[0]), min(span[1], last)
    return (lo, hi) if lo <= hi else None


def _candidates(body, at: int, threads: int, mults: int, fptype: Datatype,
                reach: Mapping[int, int], note) -> List[_Load]:
    """The loads of the loop at `at` that read a batch-constant operand at
    an index known per thread and varying over the lanes, each with the
    span it reads."""
    loop = body[at]
    bound = _bindings(body, at, fptype)
    if not bound:
        return []
    writes = [a for s in walk_stmts(body) for a in s.accesses
              if a.writes and a.space is MemSpace.GLOBAL]
    indices = _Indices(body, {'x': threads, 'y': mults})
    batch = _batch(loop)
    reads: Dict[int, Dict[int, int]] = {}
    out: List[_Load] = []
    refused: Dict[str, int] = {}
    for position, (stmt, stmts, n, parents) in enumerate(_inside(loop)):
        if stmt.op != Op.LOAD or len(stmt.args) != 2:
            continue
        buffer = stmt.args[0]
        if not isinstance(buffer, Value) or buffer.id not in bound:
            continue
        why = None
        if any(accesses_conflict(w, a) for a in stmt.accesses for w in writes):
            why = 'the kernel writes the operand'
        else:
            try:
                span = _span(stmt, parents, indices, batch)
            except _NotStatic as error:
                why = f'an index not known per thread ({error})'
        if why is not None:
            refused[why] = refused.get(why, 0) + 1
            continue
        if span is None:
            continue
        held = _within(buffer, span, reach)
        if held is None:
            why = 'an index past the end of the operand'
            refused[why] = refused.get(why, 0) + 1
            continue
        first = reads.get(id(stmts))
        if first is None:
            first = reads[id(stmts)] = _first_reads(stmts)
        cover = first.get(stmt.target[0].id, len(stmts)) - n - 1
        out.append(_Load(stmt, buffer, held[0], held[1], cover, position,
                         span[2]))
    for why, count in sorted(refused.items()):
        note(f'- {count} load(s): {why}')
    return out


def _batch(loop: Stmt) -> Optional[int]:
    """The batch loop's induction, or None for a traversal spelled as text
    (`BatchLoop._gen_grouped`), whose element no index reaches but through
    text -- which `_Indices` does not read, and refuses."""
    args = loop.regions[0].args
    return args[0].id if args else None


def _members(trip: Stmt, bound: Mapping[int, Value],
             fptype: Datatype) -> Dict[int, int]:
    """The buffers `trip` binds out of its counter and the operands in
    `bound` alone, at the top of its body -- a merged run's member of the
    trip, its table and its binding as the stand-in (`DeclareOperandTable`),
    which every element reads alike -- with where each is bound."""
    region = trip.regions[0]
    counter = region.args[0]
    known = set(bound) | {counter.id}
    varying = {counter.id}
    out: Dict[int, int] = {}
    for n, s in enumerate(region.body):
        if s.op is not Op.RAWEXPR or len(s.target) != 1 or not s.args:
            continue
        if not all(isinstance(a, Value) and a.id in known for a in s.args):
            continue
        t = s.target[0]
        known.add(t.id)
        if not any(a.id in varying for a in s.args):
            continue
        varying.add(t.id)
        ty = t.type
        if (isinstance(ty, BufferType) and ty.space is MemSpace.GLOBAL
                and ty.readonly and ty.elem == fptype):
            out[t.id] = n
    return out


def _streamed(body, at: int, threads: int, mults: int, fptype: Datatype,
              reach: Mapping[int, int], note) -> List[_Load]:
    """The loads of a merged run's members (`_members`), each with the span
    one trip reads -- the same for every trip, since only the member moves.

    A trip is a counted loop at the top of the batch loop's body, whose
    trips the caller promises every multiplication of the block takes
    together: its head holds the block's copy of the trip's members, between
    two barriers.  A load reads the member of its trip at an index known per
    thread and not moving with the trip, and stands behind every binding the
    head goes after.
    """
    loop = body[at]
    bound = _bindings(body, at, fptype)
    if not bound:
        return []
    indices = _Indices(body, {'x': threads, 'y': mults})
    batch = _batch(loop)
    out: List[_Load] = []
    refused: Dict[str, int] = {}
    for trip in loop.regions[0].body:
        if trip.op is not Op.FOR or not trip.regions:
            continue
        bounds = trip.loop_bounds
        if not all(isinstance(x, int) for x in bounds) or not bounds[2]:
            continue
        members = _members(trip, bound, fptype)
        if not members:
            continue
        top = trip.regions[0].body
        head = max(members.values())
        place = {id(s): n for n, s in enumerate(top)}
        counter = trip.regions[0].args[0].id
        reads: Dict[int, Dict[int, int]] = {}
        for position, (stmt, stmts, n, parents) in enumerate(_inside(trip)):
            if stmt.op != Op.LOAD or len(stmt.args) != 2:
                continue
            buffer = stmt.args[0]
            if not isinstance(buffer, Value) or buffer.id not in members:
                continue
            where = place[id(parents[1])] if len(parents) > 1 else n
            why = None
            if where <= head:
                why = 'a member read ahead of the bindings of its trip'
            else:
                try:
                    if counter in indices.leaves(stmt.args[1]):
                        raise _NotStatic('an index moving with the trip')
                    span = _span(stmt, (loop,) + parents, indices, batch)
                except _NotStatic as error:
                    why = f'an index not known per thread ({error})'
            if why is not None:
                refused[why] = refused.get(why, 0) + 1
                continue
            if span is None:
                continue
            held = _within(buffer, span, reach)
            if held is None:
                why = 'an index past the end of the operand'
                refused[why] = refused.get(why, 0) + 1
                continue
            first = reads.get(id(stmts))
            if first is None:
                first = reads[id(stmts)] = _first_reads(stmts)
            cover = first.get(stmt.target[0].id, len(stmts)) - n - 1
            out.append(_Load(stmt, buffer, held[0], held[1], cover, position,
                             span[2], trip=trip))
    for why, count in sorted(refused.items()):
        note(f'- {count} member load(s): {why}')
    return out


def _inside(loop: Stmt):
    """Every statement inside `loop`: the statement, the list it stands in,
    its place there, and the statements around it."""
    def go(stmts: Tuple[Stmt, ...], parents: Tuple[Stmt, ...]):
        for n, s in enumerate(stmts):
            yield s, stmts, n, parents
            for r in s.regions:
                yield from go(r.body, parents + (s,))
    for r in loop.regions:
        yield from go(r.body, (loop,))


def _span(stmt: Stmt, parents: Tuple[Stmt, ...], indices: _Indices,
          batch: int) -> Optional[Tuple[int, int, bool]]:
    """The first and the last element a load reads, over the threads and
    the trips that reach it, and whether its index goes with the trips; None
    where every lane reads the same element.

    The conditions around it are read where they can be.  One that cannot
    leaves every combination in, and the span is then wider than what the
    load reads, which a shard holds as well.
    """
    index = stmt.args[1]
    leaves = indices.leaves(index)
    if batch in leaves:
        raise _NotStatic('the element')
    if 'x' not in leaves:
        return None
    loops: Dict[int, Sequence[int]] = {}
    conditions: List = [stmt.predicate] if stmt.predicate is not None else []
    for p in parents:
        if p.op is Op.IF:
            conditions.append(p.args[0])
        elif p.op is Op.FOR and p.regions[0].args[0].id != batch:
            lo, hi, step = p.loop_bounds
            if all(isinstance(x, int) for x in (lo, hi, step)) and step:
                loops[p.regions[0].args[0].id] = range(lo, hi, step)
    trips = any(leaf in loops for leaf in leaves)
    readable = []
    for c in conditions:
        try:
            reach = indices.leaves(c)
        except _NotStatic:
            continue
        if batch not in reach and all(
                isinstance(r, str) or r in loops for r in reach):
            readable.append(c)
            leaves = leaves | reach
    width = getattr(stmt.target[0].type, 'length', None) or 1
    lo = hi = None
    for env in indices.domain(leaves, loops):
        memo: Dict[int, int] = {}
        if not all(indices.evaluate(c, env, memo) for c in readable):
            continue
        i = indices.evaluate(index, env, memo)
        lo = i if lo is None else min(lo, i)
        hi = i + width - 1 if hi is None else max(hi, i + width - 1)
    if lo is None:
        return None
    return lo, hi, trips


def _shards(loads: List[_Load]) -> List[_Shard]:
    """Per operand, the spans of its loads, those that overlap merged --
    the loads whose span is over a counted loop's trips apart from the rest:
    such a span is all of an operator, typically, and would make the lane
    blocks the rest read one shard with it, held whole or not at all."""
    out: List[_Shard] = []
    by_buffer: Dict[Tuple[int, bool], List[_Load]] = {}
    for load in loads:
        by_buffer.setdefault((load.buffer.id, load.trips), []).append(load)
    for group in by_buffer.values():
        current: Optional[_Shard] = None
        for load in sorted(group, key=lambda l: (l.lo, l.hi)):
            if current is not None and load.lo <= current.hi:
                current.hi = max(current.hi, load.hi)
                current.loads.append(load)
                continue
            current = _Shard(load.buffer, load.lo, load.hi, [load])
            out.append(current)
    return out


def _choose(shards: List[_Shard], free: int, align: int) -> List[_Shard]:
    """The shards held: least cover first, then in the order the loop reads
    them, each as long as it fits.

    A shard is counted with what it costs where `_layout` puts it: its
    elements, and the gap that keeps its first element on the alignment it
    has in the operand -- which depends on the held shard before it in the
    operand alone, and changes the gap of the one after it.  A buffer after
    the first is counted with what its start's alignment may cost; the first
    starts where the block's own buffers end, aligned."""
    def gap(before: Optional[_Shard], shard: _Shard) -> int:
        end = 0 if before is None else before.hi + 1
        return (shard.lo - end) % align

    held: List[_Shard] = []
    rows: Dict[int, List[_Shard]] = {}
    starts: Dict[int, List[int]] = {}
    used = 0
    for shard in sorted(shards, key=lambda s: (s.cover, s.position)):
        key = shard.buffer.id
        row, lo = rows.get(key, []), starts.get(key, [])
        at = bisect.bisect_left(lo, shard.lo)
        before = row[at - 1] if at else None
        after = row[at] if at < len(row) else None
        need = gap(before, shard) + shard.size
        if after is not None:
            need += gap(shard, after) - gap(before, after)
        if key not in rows and rows:
            need += align - 1
        if used + need > free:
            continue
        row.insert(at, shard)
        lo.insert(at, shard.lo)
        rows[key], starts[key] = row, lo
        held.append(shard)
        used += need
    return held


def _layout(held: List[_Shard], align: int
            ) -> Dict[int, Tuple[int, List[Tuple[_Shard, int]]]]:
    """Per operand, how large its buffer is and where each held shard starts
    in it: in the order of the operand, each on the alignment its first
    element has there, so that a wide load stays as aligned as it was."""
    out: Dict[int, Tuple[int, List[Tuple[_Shard, int]]]] = {}
    by_buffer: Dict[int, List[_Shard]] = {}
    for shard in held:
        by_buffer.setdefault(shard.buffer.id, []).append(shard)
    for key, group in by_buffer.items():
        end = 0
        placed = []
        for shard in sorted(group, key=lambda s: s.lo):
            end += (shard.lo - end) % align
            placed.append((shard, end))
            end += shard.size
        out[key] = (end, placed)
    return out


@dataclass(frozen=True)
class _Copy:
    """`count` elements, `times` times over: from `src + k * src_step` of
    the operand to `dst + k * dst_step` of the buffer, `k` below `times`."""
    src: int
    dst: int
    count: int
    times: int = 1
    src_step: int = 0
    dst_step: int = 0

    @property
    def total(self) -> int:
        return self.count * self.times


def _copies(placed: List[Tuple[_Shard, int]]) -> List[_Copy]:
    """The copies that fill one buffer.  Shards that follow each other in
    the operand and in the buffer are one run, and runs of one length at one
    distance from each other in both -- a lane block of every column, say --
    are one copy."""
    runs: List[List[int]] = []
    for shard, base in placed:
        if runs and runs[-1][0] + runs[-1][2] == shard.lo \
                and runs[-1][1] + runs[-1][2] == base:
            runs[-1][2] += shard.size
            continue
        runs.append([shard.lo, base, shard.size])
    out: List[_Copy] = []
    for src, dst, count in runs:
        last = out[-1] if out else None
        if last is not None and last.count == count:
            src_step = src - (last.src + (last.times - 1) * last.src_step)
            dst_step = dst - (last.dst + (last.times - 1) * last.dst_step)
            if last.times == 1 or (src_step, dst_step) == (last.src_step,
                                                           last.dst_step):
                out[-1] = replace(last, times=last.times + 1,
                                  src_step=src_step, dst_step=dst_step)
                continue
        out.append(_Copy(src, dst, count))
    return out


#: Passes over the block a copy is written out to rather than looped over
#: (`_stage`).  The loop runs by lane -- its trips are `flat`'s -- so the
#: compiler cannot count the loads in flight and waits on each one before its
#: store: a round trip to memory a pass, twelve at the head of every trip of
#: SeisSol's order-8 `localFluxAll` on gfx942.  Written out, the passes'
#: loads are in flight together.  Past this many the loop stays, its code
#: size the bound.
UNROLL = 16


def _stage(b, operand: Value, buf: Value, copy: _Copy, flat: Value,
           block: int, width: int = 1) -> None:
    """`copy` by the threads of the block, `flat` the thread's place among
    them: each thread its share of `block` accesses of `width` elements
    (`_width`), under a guard where the block covers the copy at once, and
    written out pass by pass where it takes up to `UNROLL` passes -- every
    load ahead of the stores, and the lanes past the end of the last pass on
    their access of the first."""
    def plus(a: Value, c, hint: str) -> Value:
        if isinstance(c, int) and c == 0:
            return a
        return b.op('add', INDEX, a, c, hint=hint)

    def at(i: Value, k: Optional[Value], start: int, step: int,
           hint: str) -> Value:
        # Element `i` of the copy is element `i % count` of run `i / count`:
        # `i` itself, moved by where the copy starts and by what the
        # distance between runs adds to the `count` elements of each run
        # ahead of it.
        out = plus(i, start, hint)
        if k is not None and step != copy.count:
            out = plus(out, b.op('mul', INDEX, k, step - copy.count), hint)
        return out

    elem = operand.type.elem
    wide = ScalarType(elem, width) if width > 1 else None

    nbytes = width * elem.size()

    def place(u: Value) -> Tuple[Value, Value]:
        # Access `u` moves elements `width * u` on, which never straddle a
        # run: `width` divides every run's length.
        i = u if width == 1 else b.op('mul', INDEX, u, width, hint='e')
        k = (b.op('div', INDEX, i, copy.count, hint='k')
             if copy.times > 1 else None)
        return (at(i, k, copy.src, copy.src_step, 'g'),
                at(i, k, copy.dst, copy.dst_step, 'l'))

    def fetch(src: Value) -> Value:
        if wide is None:
            return b.load(operand, src, hint='sh')
        return b.load(operand, src, type_=wide, align=nbytes, hint='sh')

    def put(dst: Value, value: Value) -> None:
        if wide is None:
            b.store(buf, value, dst)
        else:
            b.store(buf, value, dst, align=nbytes)

    def one(u: Value) -> None:
        src, dst = place(u)
        put(dst, fetch(src))

    units = copy.total // width
    if units <= block:
        with b.if_(b.op('lt', BOOL, flat, units, hint='s')):
            one(flat)
        return
    passes = -(-units // block)
    if passes > UNROLL:
        with b.for_(flat, units, block, hint='s',
                    uniform=Uniformity.LANE) as f:
            one(f.induction)
        return
    fetched = []
    for p in range(passes):
        u = plus(flat, p * block, 's')
        if (p + 1) * block > units:
            # The last pass, which some lanes are past the end of.  They take
            # their access of the first pass again -- the same value to the
            # same place, by the same lane, so no other lane's write races it
            # -- rather than branch: a guard around the store would draw the
            # load in behind it and wait on it there.
            u = b.op('select', INDEX, b.op('lt', BOOL, u, units, hint='s'),
                     u, flat, hint='s')
        src, dst = place(u)
        fetched.append((dst, fetch(src)))
    for dst, value in fetched:
        put(dst, value)


def _alignments(body: Tuple[Stmt, ...], at: int,
                aligns: Mapping[str, int]) -> Dict[int, int]:
    """What is proven of the bases of the buffers in `body`, in bytes, by
    value: an operand bound ahead of the batch loop by the name of its
    binding (`aligns`), and a pointer chosen among such operands -- a merged
    run's member, through its table -- by the least of them."""
    out: Dict[int, int] = {}
    for s in walk_stmts(body):
        if len(s.target) != 1 or not isinstance(s.target[0].type, BufferType):
            continue
        name = s.attr('extern')
        if name in aligns:
            out[s.target[0].id] = aligns[name]
            continue
        sources = [a for a in s.args if isinstance(a, Value)
                   and isinstance(a.type, BufferType)]
        if (s.op is Op.RAWEXPR and sources
                and all(a.id in out for a in sources)):
            out[s.target[0].id] = min(out[a.id] for a in sources)
    return out


def _width(copy: _Copy, elem: int, proven: int) -> int:
    """How many elements a thread moves at once in `copy`: as many as one
    access of 16 bytes takes, as the operand's base is proven aligned for,
    and as every run's start, length and distance allow -- the buffer keeps
    the operand's alignment (`_layout`), so its side follows."""
    width = max(1, min(16, proven) // elem)
    while width > 1 and any(x % width for x in (
            copy.src, copy.dst, copy.count,
            copy.src_step if copy.times > 1 else 0,
            copy.dst_step if copy.times > 1 else 0)):
        width //= 2
    return width


def _rewrite(body, at: int, held: List[_Shard], shards: List[_Shard],
             scratch, arena: str, threads: int, mults: int, align: int,
             proven: Dict[int, int], note) -> Tuple[Stmt, ...]:
    """`body` with the held shards copied in ahead of the loop at `at`, by
    every thread of the block and behind a barrier, and their loads reading
    them there."""
    layout = _layout(held, align)
    b = scratch()
    flat = b.op('add', INDEX, b.thread_id('x'),
                b.op('mul', INDEX, b.thread_id('y'), threads, hint='row'),
                hint='flat')
    block = threads * mults
    buffers: Dict[int, Value] = {}
    streamed: Dict[int, List] = {}
    copied = False
    for key, (size, placed) in layout.items():
        operand = placed[0][0].buffer
        buf = b.alloc(operand.type.elem, (size,), MemSpace.SHARED,
                      hint='shard', arena=arena)
        buffers[key] = buf
        trip = placed[0][0].trip
        if trip is not None:
            streamed.setdefault(id(trip), []).append((operand, buf, placed))
            continue
        for copy in _copies(placed):
            _stage(b, operand, buf, copy, flat, block,
                   _width(copy, operand.type.elem.size(),
                          proven.get(operand.id, 0)))
            copied = True
    if copied:
        b.barrier(Participants.BLOCK)
    staging = tuple(b.finish())

    # A member's shards are copied in at the head of every trip, behind the
    # bindings of the trip's members: behind a barrier, since the previous
    # trip's readers may still be in the buffer -- or the previous element's
    # last -- and in front of one, since every multiplication of the block
    # reads what the whole block copied.
    heads: Dict[int, Tuple[frozenset, Tuple[Stmt, ...]]] = {}
    for trip, items in streamed.items():
        h = scratch()
        h.barrier(Participants.BLOCK)
        for operand, buf, placed in items:
            for copy in _copies(placed):
                _stage(h, operand, buf, copy, flat, block,
                       _width(copy, operand.type.elem.size(),
                              proven.get(operand.id, 0)))
        h.barrier(Participants.BLOCK)
        heads[trip] = (frozenset(operand.id for operand, _, _ in items),
                       tuple(h.finish()))

    moved: Dict[int, List[Stmt]] = {}
    for key, (_, placed) in layout.items():
        for shard, base in placed:
            for load in shard.loads:
                moved[id(load.stmt)] = _moved(load.stmt, buffers[key],
                                              base - shard.lo, scratch)
    for key in sorted({s.buffer.id for s in shards}):
        mine = [s for s in shards if s.buffer.id == key]
        kept = [s for s in held if s.buffer.id == key]
        operand = mine[0].buffer
        what = f'{operand} a trip' if mine[0].trip is not None else operand
        total = sum(s.size for s in mine)
        if kept:
            note(f'+ {what}: {len(kept)} of {len(mine)} shard(s), '
                 f'{sum(s.size for s in kept)} of {total} element(s)')
        else:
            note(f'- {what}: 0 of {len(mine)} shard(s), 0 of {total} '
                 f'element(s)')

    loop = body[at]
    loop = replace(loop, regions=tuple(
        replace(r, body=_substitute(r.body, moved, heads))
        for r in loop.regions))
    return body[:at] + staging + (loop,) + body[at + 1:]


def _moved(stmt: Stmt, buf: Value, shift: int, scratch) -> List[Stmt]:
    """`stmt` reading `buf`, its index moved by `shift`."""
    index = stmt.args[1]
    out: List[Stmt] = []
    if shift:
        b = scratch()
        index = b.op('add', index.type, index, shift, hint='shard')
        out += list(b.finish())
    out.append(replace(stmt, args=(buf, index),
                       accesses=(Access(Effect.READ, MemSpace.SHARED, buf),),
                       attrs=tuple((k, v) for k, v in stmt.attrs
                                   if k != 'nontemporal')))
    return out


def _substitute(stmts: Tuple[Stmt, ...], moved: Dict[int, List[Stmt]],
                heads: Optional[Dict[int, Tuple[frozenset, Tuple[Stmt, ...]]]]
                = None) -> Tuple[Stmt, ...]:
    """`stmts` with the loads in `moved` replaced, and a trip in `heads`
    with its head behind the last binding of the members it copies."""
    out: List[Stmt] = []
    for s in stmts:
        new = moved.get(id(s))
        if new is not None:
            out.extend(new)
            continue
        if s.regions:
            head = heads.get(id(s)) if heads else None
            regions = []
            for k, r in enumerate(s.regions):
                inner = _substitute(r.body, moved, heads)
                if head is not None and k == 0:
                    operands, copies = head
                    at = max(n for n, x in enumerate(inner)
                             if any(t.id in operands for t in x.target))
                    inner = inner[:at + 1] + copies + inner[at + 1:]
                regions.append(replace(r, body=inner))
            s = replace(s, regions=tuple(regions))
        out.append(s)
    return tuple(out)
