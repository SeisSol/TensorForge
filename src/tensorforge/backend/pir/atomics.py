# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: an accumulation into global memory, added there atomically
where nothing reads the sum.

A body accumulates as it reads: it loads what the destination holds, adds
its terms onto it and stores the sum.  Where that store is the last anybody
in the body sees of the element, the load and the add of what it read are
work the hardware can do instead -- an atomic add of the terms alone, with
nothing read first (`Options.atomic_accumulation`).  Where something reads
the element or the sum afterwards, the sum is what it wants, and the
accumulation stays as it is.

Asked of every store into global memory, execution by execution and lane by
lane (`flow.Flow`):

* its value is the element's own value, as a load of the body read it, plus
  terms: a chain of `add`s (or a `sub` of a term, or an `fma` onto it) back
  to that load, through register arrays as well, every link of which
  nothing but the next one reads;
* nothing wrote the element between that load and the store;
* nothing reads what the store wrote but the load at the head of another
  such chain -- the next trip's, where a loop accumulates through memory;
  those go with it.

Then the load is a zero and the store an atomic add of the terms: memory
ends up holding what it held, plus every term, which is what the chain left
there.  A chain without a term stores back what it loaded, and goes.  The
chains of one store, and the stores whose chains share a statement, change
together or not at all: a load that turns into a zero turns into one for
every execution of it.

A store whose element another lane of the multiplication stores as well is
two chains onto one element, and neither passes the second test.  Elements
another multiplication or another block writes are outside what the body
can know; an accumulation into memory a body owns is what every placement
in the generator assumes, atomic or not.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .core import Access, Effect, MemSpace, Op, Stmt, Value, walk_stmts
from .flow import INITIAL, Flow
from .passes import _zero_like


class _Refused(Exception):
    """Why a chain is not one this pass may change; `depth`, how many
    links down it was found, so that of two operands the one that went
    further says why."""

    def __init__(self, why: str, depth: int = 0):
        super().__init__(why)
        self.depth = depth


@dataclass
class _Link:
    """A register cell on a chain: `store` (an execution) wrote what `load`
    (an execution) reads, for the lanes in `mask`."""

    store: int
    load: int
    mask: int


@dataclass
class _Chain:
    """One execution of a store, for the lanes in `mask`: what it stores is
    what `bias` (an execution of a load of the element) read, plus terms."""

    store: int
    mask: int
    bias: int
    cells: List[_Link] = field(default_factory=list)
    #: The values on the chain, from the stored one down to the bias.
    values: List[Value] = field(default_factory=list)
    #: The statements that add a term, by id.
    adds: List[Stmt] = field(default_factory=list)
    #: The cells between the bias and the first term, which hold the bias
    #: alone.
    plain: List[_Link] = field(default_factory=list)


def accumulate_atomically(body: Tuple[Stmt, ...], *, threads: int,
                          arena: Optional[str] = None,
                          supported: Callable[[Value], bool],
                          report: Optional[List[str]] = None
                          ) -> Tuple[Stmt, ...]:
    """`body` with its accumulations into global memory added atomically
    where nothing reads the sum; `supported(value)` says whether the target
    adds a value of that type atomically."""
    stores = [s for s in walk_stmts(body) if _candidate(s)]
    if not stores:
        return body
    flow = Flow(body, threads, arena)
    chains: Dict[int, List[_Chain]] = {}
    reasons: Dict[int, str] = {}
    for s in stores:
        if not supported(s.args[1]):
            reasons[id(s)] = 'the target adds no such value atomically'
            continue
        try:
            chains[id(s)] = _chains(flow, s)
        except _Refused as why:
            reasons[id(s)] = str(why)
    while True:
        alive = _settle(flow, chains, reasons)
        plan = _Plan(flow, alive, chains)
        if not plan.refused:
            break
        reasons.update(plan.refused)
        for sid in plan.refused:
            chains.pop(sid, None)
    atomic = {sid for sid in alive if sid not in plan.silent}
    silent = set(plan.silent)
    if report is not None:
        names = {id(s): s for s in stores}
        for sid, why in sorted(reasons.items(), key=lambda kv: _where(names[kv[0]])):
            report.append(f'atomics: {_where(names[sid])} stays: {why}')
        for sid in sorted(atomic | silent, key=lambda k: _where(names[k])):
            what = 'goes, it stores what it loaded' if sid in silent \
                else 'adds atomically'
            report.append(f'atomics: {_where(names[sid])} {what}')
    if not atomic and not silent:
        return body
    return _rewrite(body, plan, atomic)


def _candidate(s: Stmt) -> bool:
    if s.op != Op.STORE or len(s.accesses) != 1 or len(s.args) < 3:
        return False
    a = s.accesses[0]
    return (a.space is MemSpace.GLOBAL and a.kind == Effect.WRITE
            and isinstance(s.args[1], Value))


def _where(s: Stmt) -> str:
    base = s.args[0]
    name = getattr(base, 'name', None) or getattr(base, 'hint', None) or '?'
    return f'the store into {name} of %{getattr(s.args[1], "id", "?")}'


# -- the chains of one store ------------------------------------------------- #

def _chains(flow: Flow, s: Stmt) -> List[_Chain]:
    out: List[_Chain] = []
    for n in flow.executions(s):
        inst = flow.insts[n]
        if n in flow.escaped:
            raise _Refused('what it writes is read where nothing can tell')
        for mask, chain in _walk(flow, s.args[1], inst.env(), inst.mask):
            chain.store = n
            chain.mask = mask
            out.append(chain)
    if not out:
        raise _Refused('it never runs')
    return out


def _walk(flow: Flow, v, env: Dict[int, int], mask: int, depth: int = 0
          ) -> List[Tuple[int, _Chain]]:
    """The chains `v` heads, by lane group: back through the adds and the
    register cells to a load of global memory."""
    if not isinstance(v, Value):
        raise _Refused('it stores a constant', depth)
    d = flow.defs.get(v.id)
    if d is None:
        raise _Refused('its value comes from outside the body', depth)
    if d.op == Op.LOAD and len(d.accesses) == 1:
        a = d.accesses[0]
        n = flow.inst(d, env)
        if n is None:
            raise _Refused('a load on the chain runs elsewhere', depth)
        if a.space is MemSpace.GLOBAL:
            return [(mask, _Chain(-1, mask, n, values=[v]))]
        if a.space is not MemSpace.REGISTER:
            raise _Refused('the chain passes through shared memory', depth)
        out = []
        for reach in flow.reach.get(n, ()):
            hit = reach.mask & mask
            if not hit:
                continue
            if not reach.certain or reach.writer == INITIAL:
                raise _Refused('a register on the chain is not certain to '
                               'hold what one store wrote', depth + 1)
            w = flow.insts[reach.writer]
            if w.stmt.op != Op.STORE:
                raise _Refused('a register on the chain holds what the '
                               'allocation put there', depth + 1)
            for sub, chain in _walk(flow, w.stmt.args[1], w.env(), hit,
                                    depth + 1):
                chain.cells.insert(0, _Link(reach.writer, n, sub))
                chain.values.insert(0, v)
                if not chain.adds:
                    chain.plain.insert(0, _Link(reach.writer, n, sub))
                out.append((sub, chain))
        if not out:
            raise _Refused('a register on the chain is not written', depth)
        return out
    if d.op in ('add', 'sub', 'fma') and len(d.target) == 1:
        if d.op == 'add':
            sides = [0, 1]
        elif d.op == 'sub':
            sides = [0]
        else:
            sides = [2]
        deepest: Optional[_Refused] = None
        for side in sides:
            try:
                found = _walk(flow, d.args[side], env, mask, depth + 1)
            except _Refused as why:
                if deepest is None or why.depth > deepest.depth:
                    deepest = why
                continue
            for _, chain in found:
                chain.values.insert(0, v)
                chain.adds.insert(0, d)
            return found
        raise deepest or _Refused('no operand leads to a load', depth)
    raise _Refused(f'the chain passes through `{d.op}`', depth)


# -- what a group of stores needs -------------------------------------------- #

def _settle(flow: Flow, chains: Dict[int, List[_Chain]],
            reasons: Dict[int, str]) -> Set[int]:
    """The stores that change: those whose every chain passes, given which
    others change -- what one of them stores may be read by the head of
    another's chain only where that one changes as well -- until none
    leaves."""
    alive = set(chains)
    while True:
        heads = {c.bias for sid in alive for c in chains[sid]}
        failed: Dict[int, str] = {}
        for sid in alive:
            for c in chains[sid]:
                why = _check_chain(flow, c, heads)
                if why is not None:
                    failed[sid] = why
                    break
        # A load that turns into a zero, or a register store whose value
        # loses the bias, does so for every execution: each has to be on a
        # chain of a store that changes.
        used: Dict[int, Set[int]] = defaultdict(set)
        owners: Dict[int, Set[int]] = defaultdict(set)
        for sid in alive - set(failed):
            for c in chains[sid]:
                at = [c.bias] + [l.store for l in c.cells]
                for n in at:
                    stmt = id(flow.insts[n].stmt)
                    used[stmt].add(n)
                    owners[stmt].add(sid)
        for stmt, seen in used.items():
            if set(flow.by_stmt.get(stmt, ())) - seen:
                for sid in owners[stmt]:
                    failed.setdefault(sid, 'a statement on its chain runs '
                                           'where no chain of it does')
        if not failed:
            return alive
        reasons.update(failed)
        alive -= set(failed)


def _check_chain(flow: Flow, c: _Chain, heads: Set[int]) -> Optional[str]:
    store = flow.insts[c.store]
    bias = flow.insts[c.bias]
    s, b = store.stmt, bias.stmt
    if flow.binding(s.args[0]) is not flow.binding(b.args[0]):
        return 'it loads through another binding than it stores through'
    if not bias.conds <= store.conds:
        return 'it may store where its load did not run'
    if c.mask & ~bias.mask:
        # Those lanes store what the else-value gives plus the terms: an
        # assignment, which an add onto memory is not.
        return 'it stores lanes its load leaves to an else-value'
    loaded = _by_lane(flow.cells.get(c.bias, ()))
    stored = _by_lane(flow.cells.get(c.store, ()))
    before = flow.before.get(c.store, {})
    held: Dict[Tuple[int, Tuple], object] = {}
    for r in flow.reach.get(c.bias, ()):
        for lane in flow.lanes(r.mask & c.mask):
            held[(lane, r.cell)] = r
    # The element holds, right before the store, for certain what the load
    # read: whatever the load may have read besides is something the store
    # would find there as well, or not certain to.
    for lane in flow.lanes(c.mask):
        if loaded.get(lane) != stored.get(lane):
            return 'it stores another element than its load read'
        for cell in stored.get(lane, ()):
            r = held.get((lane, cell))
            if r is None or before.get((lane, cell)) != (r.writer, True):
                return 'the element is written between its load and its store'
    for reader, m in flow.readers.get(c.store, ()):
        if m & c.mask and reader not in heads:
            return 'something reads what it stores'
    for l in c.cells:
        if l.store in flow.escaped:
            return 'a register on its chain is read where nothing can tell'
        for reader, m in flow.readers.get(l.store, ()):
            if m & l.mask and reader != l.load:
                return 'a register on its chain is read elsewhere as well'
    for k, v in enumerate(c.values):
        uses = flow.uses.get(v.id, [])
        if len(uses) != 1:
            return 'a value on its chain has another use'
    if len(flow.uses.get(c.values[-1].id, [])) != 1:
        return 'its load has another use'
    return None


def _by_lane(cells) -> Dict[int, Tuple]:
    """The elements an execution touches, lane by lane, in order."""
    out: Dict[int, List] = defaultdict(list)
    for mask, cell in cells:
        lane = 0
        while mask:
            if mask & 1:
                out[lane].append(cell)
            mask >>= 1
            lane += 1
    return {lane: tuple(cells) for lane, cells in out.items()}


class _Plan:
    """What changes besides the stores.

    The bias goes where it joins the terms: the `add` that takes it is the
    terms alone, where every execution of that add takes the bias and
    nothing else on that side, and the registers that held the bias on the
    way go with it.  Where an add takes the bias on one trip and the sum of
    the trips before on the next -- an image carried over a loop -- the
    image has to start from zero instead: the registers that held the bias
    go where the allocation zeroed them, and the load becomes a zero where
    it did not.  `refused` names the stores that would need a zero no
    constant spells, a vector's.
    """

    def __init__(self, flow: Flow, alive: Set[int], chains):
        self.flow = flow
        #: Loads of the bias that become a zero, by statement id.
        self.zeroed: Set[int] = set()
        #: Register stores that go, by statement id.
        self.dropped: Set[int] = set()
        #: Adds that are their terms, the add's result by id.
        self.mapped: Dict[int, Value] = {}
        self.silent: Set[int] = set()
        self.refused: Dict[int, str] = {}
        plain_loads: Set[int] = set()
        plain_stores: Dict[int, Set[int]] = defaultdict(set)
        biases: Set[int] = set()
        for sid in alive:
            for c in chains[sid]:
                biases.add(c.bias)
                for l in c.plain:
                    plain_loads.add(l.load)
                    plain_stores[id(flow.insts[l.store].stmt)].add(l.store)
        whole = {stmt for stmt, seen in plain_stores.items()
                 if set(flow.by_stmt.get(stmt, ())) == seen}
        joins: Dict[int, bool] = {}
        for sid in alive:
            found = chains[sid]
            if all(not c.adds for c in found):
                self.silent.add(sid)
            for c in found:
                if not c.adds:
                    continue
                join = c.adds[-1]
                feed = c.plain[0].load if c.plain else c.bias
                at = flow.by_stmt.get(id(flow.insts[feed].stmt), ())
                ok = join.op == 'add' and all(
                    n in plain_loads or n in biases for n in at)
                joins[id(join)] = joins.get(id(join), True) and ok
        for sid in alive:
            for c in chains[sid]:
                stmts = {id(flow.insts[l.store].stmt) for l in c.plain}
                if not c.adds:
                    self.dropped.update(stmts & whole)
                    continue
                join = c.adds[-1]
                if joins.get(id(join)):
                    side = 0 if _leads_to(flow, join.args[0], c) else 1
                    self.mapped[join.target[0].id] = join.args[1 - side]
                    self.dropped.update(stmts & whole)
                    continue
                zero = all(stmt in whole and self._zero_before(stmt)
                           for stmt in stmts) and stmts
                if zero:
                    self.dropped.update(stmts)
                    continue
                bias = flow.insts[c.bias].stmt
                if _zero_like(bias.target[0].type) is None:
                    self.refused[sid] = ('its sum is carried over a loop and '
                                         'would have to start from a zero no '
                                         'constant spells')
                    break
                self.zeroed.add(id(bias))

    def _zero_before(self, stmt: int) -> bool:
        """Did the allocation's zero sit in every element right before each
        execution of `stmt`?"""
        flow = self.flow
        for n in flow.by_stmt.get(stmt, ()):
            for prior, certain in flow.before.get(n, {}).values():
                if not certain or prior < 0:
                    return False
                if flow.insts[prior].stmt.op != Op.ALLOC:
                    return False
            if n not in flow.before:
                return False
        return True


def _leads_to(flow: Flow, v, c: _Chain) -> bool:
    """Is `v` on the chain `c`: the bias, or a value of a cell on it?"""
    return isinstance(v, Value) and any(x is v for x in c.values)


# -- the rewrite ------------------------------------------------------------- #

def _rewrite(body, plan: _Plan, atomic: Set[int]) -> Tuple[Stmt, ...]:
    from .passes import substitute

    def visit(stmts: Sequence[Stmt]) -> Tuple[Stmt, ...]:
        out: List[Stmt] = []
        for s in stmts:
            if id(s) in plan.dropped or id(s) in plan.silent:
                continue
            if id(s) in plan.zeroed and s.op == Op.LOAD:
                zero = _zero_like(s.target[0].type)
                out.append(replace(s, op=Op.CONST, args=(), accesses=(),
                                   effect=Effect.NONE, pure=True,
                                   predicate=None, attrs=(('value', zero),)))
                continue
            if id(s) in atomic:
                a = s.accesses[0]
                out.append(replace(s, effect=Effect.ATOMIC, accesses=(
                    Access(Effect.ATOMIC, a.space, a.base),)))
                continue
            if s.regions:
                regions = tuple(replace(r, body=visit(r.body))
                                for r in s.regions)
                s = replace(s, regions=regions)
            out.append(s)
        return tuple(out)
    return substitute(visit(body), plan.mapped)
