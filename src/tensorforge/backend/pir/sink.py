# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: a store of a number nothing reads goes behind the work it
stands in.

A buffer laid out with more rows than an operation writes gets the rows
past them as zeros -- a padding of global memory nothing in the body reads.
It stands where the operation stores its result, and where that is in the
middle of a list -- a branch on the lanes that hold the padding, between a
product and the operation that reads it -- the compiler keeps every value
of the product live across it: `plConvertToNodal` of SeisSol's elastic
order 4, its result stored ahead of the operation that reads it, takes 255
registers and spills on sm_86, where it takes 138 with the padding behind
the last store.

Asked of the statements of every list: a store of a number, or a branch or
a counted loop of nothing but such stores and the arithmetic of their
indices, whose every element nothing reads (`flow.Flow`), goes down its
list, past every statement that touches none of its elements -- to the end
of the list, or to the first that does, a barrier, the end of the list's
run or what may leave it, what the IR does not say the effect of, or what
the flow cannot place.  Several go down together and keep their order.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from dataclasses import replace

from .core import Effect, MemSpace, Op, Stmt, Value, walk_stmts
from .flow import Flow, TooLarge


def sink_numbers(body: Tuple[Stmt, ...], *, threads: int,
                 arena: Optional[str] = None, simd: bool = False,
                 report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with its stores of numbers nothing reads at the end of their
    lists (see the module).  A body too large for the flow is left as it
    is."""
    try:
        flow = Flow(body, threads, arena, masks=True, before=False,
                    simd=simd)
    except TooLarge:
        return body
    sink = _Sink(flow)
    out = sink.list(tuple(body))
    if report is not None and sink.moved:
        report.append(f'sink: {len(sink.moved)} stores of numbers go to the '
                      f'end of their lists')
    return out


class _Sink:

    def __init__(self, flow: Flow):
        self.flow = flow
        self.consts: Set[int] = {
            s.target[0].id for s in walk_stmts(flow.body)
            if s.op == Op.CONST and s.target}
        self.moved: Set[int] = set()
        self._cells: Dict[int, Optional[Set[Tuple]]] = {}

    def list(self, stmts: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        out: List[Stmt] = []
        #: The statements going down, in their order, with their elements.
        going: List[Tuple[Stmt, Set[Tuple]]] = []
        for s in stmts:
            cells = self._numbers(s)
            if cells is not None:
                # Behind the others going down, whatever it writes: they
                # keep their order.
                going.append((s, cells))
                continue
            if going and self._stops(s, going):
                out.extend(x for x, _ in going)
                going = []
            if s.regions:
                s = replace(s, regions=tuple(
                    replace(r, body=self.list(tuple(r.body)))
                    for r in s.regions))
            out.append(s)
            self.moved.update(id(x) for x, _ in going)
        out.extend(x for x, _ in going)
        return tuple(out)

    # -- what goes down ---------------------------------------------------- #

    def _numbers(self, s: Stmt) -> Optional[Set[Tuple]]:
        """The elements `s` writes, where it is a store of a number nothing
        reads, or a branch or a counted loop of nothing but such stores and
        arithmetic; else None."""
        stores: List[Stmt] = []
        for x in walk_stmts((s,)):
            if x.op == Op.STORE:
                if not self._number_store(x):
                    return None
                stores.append(x)
            elif x.op in (Op.IF, Op.FOR):
                if x is not s and not x.regions:
                    return None
            elif x.op in Op.ARITH or x.op == Op.CONST:
                if x.accesses:
                    return None
            else:
                return None
        if not stores or (s.op == Op.FOR and s.attr('rolled')):
            return None
        cells: Set[Tuple] = set()
        for x in stores:
            execs = self.flow.executions(x)
            if not execs:
                return None
            for n in execs:
                if (self.flow.readers.get(n) or n in self.flow.escaped
                        or id(x) in self.flow.opaque):
                    return None
                cells.update(c for _, c in self.flow.cells_of(n))
        return cells

    def _number_store(self, x: Stmt) -> bool:
        if len(x.accesses) != 1:
            return False
        a = x.accesses[0]
        if a.space is not MemSpace.GLOBAL or a.kind != Effect.WRITE:
            return False
        v = x.args[1]
        if isinstance(v, Value):
            return v.id in self.consts
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    # -- what stops it ----------------------------------------------------- #

    def _stops(self, s: Stmt, going) -> bool:
        """Does `s` stand in the way of what goes down (`_touched`), or
        touch an element one of them writes?"""
        mine = self._touched(s)
        if mine is None:
            return True
        return any(mine & cells for _, cells in going)

    def _touched(self, s: Stmt) -> Optional[Set[Tuple]]:
        """The elements `s` and the statements in it touch, or None where it
        is a barrier, ends the list's run or may leave it, does what the IR
        does not say, or touches what the flow did not place."""
        key = id(s)
        if key in self._cells:
            return self._cells[key]
        if s.op == Op.YIELD:
            # The end of the region: nothing goes behind it.
            self._cells[key] = None
            return None
        out: Optional[Set[Tuple]] = set()
        flow = self.flow
        for x in walk_stmts((s,)):
            if x.op == Op.BARRIER or x.effect & Effect.BARRIER:
                out = None
                break
            if x.op == Op.EXIT:
                # It may leave the loop the list runs in, where what goes
                # down would not run.
                out = None
                break
            if (x.effect & Effect.UNKNOWN and not x.accesses
                    and not (x.op == Op.RAWEXPR
                             and x.attr('decl') is not None)):
                # What it does the IR does not say -- but for a binding,
                # which computes where a buffer is.
                out = None
                break
            if id(x) in flow.opaque and any(
                    a.space in (MemSpace.GLOBAL, MemSpace.UNKNOWN)
                    for a in x.accesses):
                out = None
                break
            for n in flow.executions(x):
                out.update(c for _, c in flow.cells_of(n))
        self._cells[key] = out
        return out
