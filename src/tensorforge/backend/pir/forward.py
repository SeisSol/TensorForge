# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: a load that reads what its own lane stored takes it from where
the store took it, one that reads what its lanes loaded before from where
that load put it, and a store nothing reads goes.

A body hands values from one operation to the next through memory -- a
result stored to global memory and loaded back by the operation that reads
it, a register image copied into another before it is stored, an operand
each operation loads anew from where it lives.  Asked of every load,
execution by execution and lane by lane (`flow.Flow`, with `values`):
where the store it reads for certain was the lane's own, and its value is

* a number, the load is that number;
* a value of the body that stands where the load does, the load is that
  value;
* what a load read from a register that holds it still where the load
  stands, the load reads that register -- at an index the hops of the loops
  around the load give, of loops that are unrolled, so that the compiler
  names the register (`_fit`).

The last holds as well where the lanes loaded the element before and found
the same write there: the load reads the register that earlier load's value
went into, where it holds it still.

Every execution and every lane the load runs for has to agree: one number,
one value, one register array.  Then the stores: one that nothing reads and
that leaves nothing behind -- no element of global memory the body ends
with, nothing another multiplication reads -- goes.


An element's mask is taken to hold where the body keeps its contract
(`Flow` with `masks`): the store under a batch element's guard is what the
load outside it reads, on AMD, where every store to global memory stands
under one and no load does.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .core import Access, Effect, MemSpace, Op, Stmt, Value, walk_stmts
from .flow import Flow, TooLarge
from .passes import _zero_like, substitute

_SPACES = (MemSpace.GLOBAL, MemSpace.SHARED, MemSpace.REGISTER)


def forward_memory(body: Tuple[Stmt, ...], scratch, *, threads: int,
                   arena: Optional[str] = None, masks: bool = True,
                   rounds: int = 2,
                   report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with its loads taken from what their own lane stored, and its
    stores nothing reads dropped (see the module); `scratch` makes the
    builder new statements are built with.  Until nothing changes, or
    `rounds` times: a load taken from a register may read one that a copy
    filled, which the next round takes from where the copy did.  A body too
    large for the flow (`flow.TooLarge`) is left as it is."""
    for _ in range(rounds):
        # Let the last round's flow go first: two at once would hold twice
        # what the body takes.
        flow = None
        try:
            flow = Flow(body, threads, arena, masks=masks, values=True,
                        before=False)
        except TooLarge as why:
            if report is not None:
                report.append(f'forward: {why}; left as it is')
            break
        out = _Forward(flow, scratch, report).run()
        if out is body:
            break
        body = out
    return body


def drop_dead_stores(body: Tuple[Stmt, ...], *, threads: int,
                     arena: Optional[str] = None, masks: bool = True,
                     report: Optional[List[str]] = None
                     ) -> Tuple[Stmt, ...]:
    """`body` without the stores nothing reads and that leave nothing
    behind: no reader in the body, no element of global memory it ends
    with, nothing another multiplication or a statement the analysis cannot
    place may read.  A body too large for the flow is left as it is."""
    try:
        flow = Flow(body, threads, arena, masks=masks, before=False)
    except TooLarge as why:
        if report is not None:
            report.append(f'forward: {why}; left as it is')
        return body
    dead = _dead_stores(flow, set(), set())
    if not dead:
        return body
    if report is not None:
        report.append(f'forward: {len(dead)} stores nothing reads go')
    body = _strip(body, dead)
    return _strip(body, _unreached(body))


def _dead_stores(flow: Flow, taken: Set[int], kept: Set[int]) -> Set[int]:
    """The stores, by statement id, every execution of which nothing reads
    -- but the loads in `taken`, which read something else from now on --
    and that leave nothing behind; the executions in `kept` are read by
    what those loads read instead."""
    dead: Set[int] = set()
    for s in walk_stmts(flow.body):
        if s.op != Op.STORE or len(s.accesses) != 1:
            continue
        a = s.accesses[0]
        if a.space not in _SPACES or a.kind != Effect.WRITE:
            continue
        execs = flow.executions(s)
        if not execs or id(s) in flow.opaque:
            continue
        if all(n not in flow.escaped and n not in flow.final and n not in kept
               and all(id(flow.insts[r].stmt) in taken
                       for r, _ in flow.readers.get(n, ()))
               for n in execs):
            dead.add(id(s))
    return dead


def _unreached(body: Sequence[Stmt]) -> Set[int]:
    """The shared buffers the body declares that nothing reaches any more
    but a `mark` -- one that says a store defines the buffer, which the
    store went before -- by the ids of the statements that go with them: the
    marks and the declaration.  Left behind, the declaration would name a
    window of an arena in which the allocator, finding nothing to place,
    lays out nothing."""
    buffers: Dict[int, Stmt] = {}
    for s in walk_stmts(body):
        if (s.op == Op.ALLOC and len(s.target) == 1
                and getattr(s.target[0].type, 'space', None)
                is MemSpace.SHARED):
            buffers[s.target[0].id] = s
    if not buffers:
        return set()
    names = {}
    for sid, s in buffers.items():
        for key in (s.attr('identity'), s.attr('extern')):
            if key is not None:
                names[id(key) if not isinstance(key, str) else key] = sid
    marks: Dict[int, List[int]] = {}
    reached: Set[int] = set()
    for s in walk_stmts(body):
        if s.op == Op.ALLOC and s.target and s.target[0].id in buffers:
            continue
        hit = {v.id for v in s.operands() if v.id in buffers}
        for a in s.accesses:
            base = a.base
            if isinstance(base, Value) and base.id in buffers:
                hit.add(base.id)
            elif base is not None:
                key = base if isinstance(base, str) else id(base)
                name = getattr(base, 'name', None)
                for k in (key, name):
                    if k in names:
                        hit.add(names[k])
        if not hit:
            continue
        if s.op == Op.MARK:
            for b in hit:
                marks.setdefault(b, []).append(id(s))
        else:
            reached |= hit
    out: Set[int] = set()
    for b, alloc in buffers.items():
        if b in reached:
            continue
        out.add(id(alloc))
        out.update(marks.get(b, ()))
    return out


def _strip(body: Sequence[Stmt], dead: Set[int]) -> Tuple[Stmt, ...]:
    out: List[Stmt] = []
    for s in body:
        if id(s) in dead:
            continue
        if s.regions:
            s = replace(s, regions=tuple(replace(r, body=_strip(r.body, dead))
                                         for r in s.regions))
        out.append(s)
    return tuple(out)


class _Forward:
    """One round over the loads of a body."""

    def __init__(self, flow: Flow, scratch, report: Optional[List[str]]):
        self.flow = flow
        self.scratch = scratch
        self.report = report
        self.allocs: Dict[int, Stmt] = {
            id(s.target[0]): s for s in walk_stmts(flow.body)
            if s.op == Op.ALLOC and s.target}
        self.where = _Positions(flow.body)
        #: Loads that become another load, by statement id.
        self.rebuilt: Dict[int, List[Stmt]] = {}
        #: Loads whose value is another, by the id of their target.
        self.mapped: Dict[int, Value] = {}
        #: The stores whose registers a load reads from now on.
        self.kept: Set[int] = set()

    def run(self) -> Tuple[Stmt, ...]:
        for s in walk_stmts(self.flow.body):
            if self._candidate(s):
                self._take(s)
        dead = _dead_stores(self.flow, set(self.rebuilt), self.kept)
        if not self.rebuilt and not dead:
            return self.flow.body
        if self.report is not None:
            if self.rebuilt:
                self.report.append(f'forward: {len(self.rebuilt)} loads read '
                                   f'what their lane stored where the store '
                                   f'took it')
            if dead:
                self.report.append(f'forward: {len(dead)} stores nothing '
                                   f'reads go')
        body = _strip(self._rewrite(self.flow.body), dead)
        body = _strip(body, _unreached(body))
        return substitute(body, self.mapped) if self.mapped else body

    @staticmethod
    def _candidate(s: Stmt) -> bool:
        if s.op != Op.LOAD or len(s.accesses) != 1 or len(s.target) != 1:
            return False
        if s.attr('escapes') or s.accesses[0].space not in _SPACES:
            return False
        return getattr(s.target[0].type, 'length', None) is None

    def _take(self, s: Stmt) -> bool:
        flow = self.flow
        kinds: Set[str] = set()
        consts: Set = set()
        values: Set[int] = set()
        value: Optional[Value] = None
        bases: Set[int] = set()
        writers: Set[int] = set()
        points: List[Tuple[Dict[int, int], Tuple]] = []
        #: Per execution, its hops and the lanes it takes from registers.
        lanes: List[Tuple[Dict[int, int], int, int]] = []
        for n in flow.executions(s):
            inst = flow.insts[n]
            covered = held = 0
            cells: Set[Tuple] = set()
            for m, o in flow.forward.get(n, ()):
                covered |= m
                kinds.add(o[0])
                if o[0] == 'const':
                    consts.add(o[1])
                elif o[0] == 'value':
                    values.add(o[1].id)
                    value = o[1]
                else:
                    cells.add(o[1])
                    writers.add(o[2])
                    held |= m
            if covered & inst.mask != inst.mask:
                return False
            if cells:
                if len(cells) != 1:
                    return False
                cell, = cells
                bases.add(cell[0][1])
                points.append((inst.env(), cell[1]))
            lanes.append((inst.env(), inst.mask, held & inst.mask))
        if kinds == {'reg', 'const'} and len(consts) == 1 and len(bases) == 1:
            # Some lanes from a register, the rest a number: where a
            # condition of the lanes tells them apart, a select.
            if s.attr('extern') is None and self._mixed(
                    s, next(iter(bases)), points, next(iter(consts)), lanes):
                self.kept |= writers
                return True
            return False
        if len(kinds) != 1:
            return False
        kind, = kinds
        if kind != 'reg' and s.attr('extern') is not None:
            # Text names the load's value: a register load keeps the name,
            # a number or another value does not.
            return False
        if kind == 'const':
            return len(consts) == 1 and self._const(s, next(iter(consts)))
        if kind == 'value':
            return len(values) == 1 and self._value(s, value)
        if len(bases) == 1 and self._register(s, next(iter(bases)), points):
            self.kept |= writers
            return True
        return False

    # -- what a load becomes --------------------------------------------- #

    def _const(self, s: Stmt, c) -> bool:
        t = s.target[0].type
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            return False
        b = self.scratch()
        number = b.const(c, t)
        self.rebuilt[id(s)] = list(b.finish())
        self.mapped[s.target[0].id] = self._unless(s, number)
        return True

    def _value(self, s: Stmt, v: Value) -> bool:
        if v.type != s.target[0].type or not self._stands(v, s):
            return False
        b = self.scratch()
        out = self._select(b, s, v)
        self.rebuilt[id(s)] = list(b.finish())
        self.mapped[s.target[0].id] = out
        return True

    def _register(self, s: Stmt, base: int, points) -> bool:
        read = self._register_read(s, base, points)
        if read is None:
            return False
        lead, slot, image = read
        attrs = tuple((k, v) for k, v in s.attrs if k in ('other', 'extern'))
        self.rebuilt[id(s)] = lead + [replace(
            s, args=(image, slot),
            accesses=(Access(Effect.READ, MemSpace.REGISTER, image),),
            attrs=attrs)]
        return True

    def _mixed(self, s: Stmt, base: int, points, c, lanes) -> bool:
        """The load as a register where a condition of the lanes holds and
        the number `c` where it does not -- a condition the body tests
        already, that tells the lanes apart at every execution."""
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            return False
        flow = self.flow
        for g, register_lanes in self._lane_conditions():
            if not self._stands(g, s):
                continue
            if all(flow._lanes_where(g, env, mask)
                   == (held if register_lanes else mask & ~held)
                   for env, mask, held in lanes):
                break
        else:
            return False
        read = self._register_read(s, base, points)
        if read is None:
            return False
        lead, slot, image = read
        t = s.target[0].type
        b = self.scratch()
        v = b.load(image, slot, type_=t, hint='fwd')
        number = b.const(c, t)
        picked = (b.op('select', t, g, v, number, hint='fwd') if register_lanes
                  else b.op('select', t, g, number, v, hint='fwd'))
        out = self._select(b, s, picked)
        self.rebuilt[id(s)] = lead + list(b.finish())
        self.mapped[s.target[0].id] = out
        return True

    def _lane_conditions(self):
        """The conditions of the body's branches that a lane decides alone,
        each twice: as holding for the lanes that read a register, and as
        holding for the others."""
        if not hasattr(self, '_conds'):
            found = []
            seen: Set[int] = set()
            for x in walk_stmts(self.flow.body):
                cond = x.cond if x.op == Op.IF else None
                if not isinstance(cond, Value) or cond.id in seen:
                    continue
                seen.add(cond.id)
                try:
                    if self.flow._ix.leaves(cond) != frozenset({'x'}):
                        continue
                except Exception:  # noqa: BLE001 -- not a lane's own
                    continue
                found += [(cond, True), (cond, False)]
            self._conds = found
        return self._conds

    def _register_read(self, s: Stmt, base: int, points):
        """`(statements, index, image)` reading the register of `base` that
        `points` give at the hops of `s`, or None."""
        alloc = self.allocs.get(base)
        if alloc is None or not self.where.dominates(alloc, s):
            return None
        image = alloc.target[0]
        if getattr(s.target[0].type, 'base', None) != getattr(
                image.type, 'elem', None):
            return None
        if any(len(addr) != 1 for _, addr in points):
            return None
        loops = self.flow.loops.get(id(s), ())
        unrolled = {l.induction.id for l in loops if l.attr('unroll')}
        fit = _fit([(env, addr[0]) for env, addr in points])
        if fit is None or set(fit[1]) - unrolled:
            return None
        inductions = {l.induction.id: l.induction for l in loops}
        b = self.scratch()
        slot = _slot(b, fit, inductions)
        return list(b.finish()), slot, image

    def _unless(self, s: Stmt, v):
        """`v` where the load's predicate holds, and what the load gives
        where it does not."""
        if s.predicate is None:
            return v
        b = self.scratch()
        out = self._select(b, s, v)
        self.rebuilt[id(s)] = self.rebuilt.get(id(s), []) + list(b.finish())
        return out

    @staticmethod
    def _select(b, s: Stmt, v):
        if s.predicate is None:
            return v
        t = s.target[0].type
        other = s.attr('other')
        if other is None:
            other = _zero_like(t)
        return b.op('select', t, s.predicate, v, other, hint='fwd')

    def _stands(self, v: Value, s: Stmt) -> bool:
        """Does `v` stand where `s` does: defined ahead of it, in a region
        around it?"""
        d = self.flow.defs.get(v.id)
        if d is None:
            return True
        if any(t is v for t in d.target):
            return self.where.dominates(d, s)
        # An argument of a region: inside it.
        return d in self.flow.parents.get(id(s), ())

    # -- the rewrite ------------------------------------------------------ #

    def _rewrite(self, stmts: Sequence[Stmt]) -> Tuple[Stmt, ...]:
        out: List[Stmt] = []
        for s in stmts:
            new = self.rebuilt.get(id(s))
            if new is not None:
                out.extend(new)
                continue
            if s.regions:
                s = replace(s, regions=tuple(replace(r, body=self._rewrite(
                    r.body)) for r in s.regions))
            out.append(s)
        return tuple(out)


class _Positions:
    """Where every statement of a body stands: the list it is in, and its
    place there, and the same for every statement around it."""

    def __init__(self, body: Sequence[Stmt]):
        self.chain: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        self._index(tuple(body), ())

    def _index(self, stmts: Tuple[Stmt, ...], up: Tuple) -> None:
        for i, s in enumerate(stmts):
            here = up + ((id(stmts), i),)
            self.chain[id(s)] = here
            for r in s.regions:
                self._index(tuple(r.body), here)

    def dominates(self, a: Stmt, b: Stmt) -> bool:
        """Is `a` ahead of `b` in a list `b` stands in, or around?"""
        ca, cb = self.chain.get(id(a)), self.chain.get(id(b))
        if ca is None or cb is None:
            return False
        k = len(ca) - 1
        if len(cb) <= k or cb[:k] != ca[:k]:
            return False
        return cb[k][0] == ca[k][0] and cb[k][1] > ca[k][1]


def _fit(points) -> Optional[Tuple[int, Dict[int, int]]]:
    """`(c, {induction: a})` with `c` plus the sum of `a` times each hop the
    index of every point `(hops, index)`, or None."""
    if not points:
        return None
    keys = sorted(points[0][0])
    env0, r0 = points[0]
    coef: Dict[int, int] = {}
    for k in keys:
        coef[k] = 0
        for env, r in points[1:]:
            if env.get(k) != env0.get(k) and all(
                    env.get(j) == env0.get(j) for j in keys if j != k):
                step = env[k] - env0[k]
                if (r - r0) % step:
                    return None
                coef[k] = (r - r0) // step
                break
    c = r0 - sum(coef[k] * env0[k] for k in keys)
    for env, r in points:
        if c + sum(coef[k] * env.get(k, 0) for k in keys) != r:
            return None
    return c, {k: a for k, a in coef.items() if a}


def _slot(b, fit, inductions: Dict[int, Value]):
    """The index `fit` gives, as an operand: a number, or arithmetic over
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
