# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: which lanes of a value anything reads.

A value of a body is a number in every lane, and most of what reads it reads
its own lane's: arithmetic, a store into the lane's registers and the load
that takes it back.  A cross-lane primitive does not.  A broadcast gives
every lane of a block what one lane of the block holds, and an FMA under a
DPP control multiplies with what one lane of every row holds
(`tensorforge_device/hip.h`, `cuda.h`).  What a lane no read reaches holds
changes nothing the body keeps: it may hold what a load gives it there, or
what a register holds there instead.

`Demand.lanes` follows the uses of a value (`flow.Flow`): through
arithmetic, which reads the lanes its result is read in; through a store
into registers, to the loads that read what it wrote (`Flow.readers`), lane
by lane; through the primitives of `_READS`, to the lane each takes.
Anything else -- a store to memory, an address, a condition, a call or a text
it does not know, a value text may name -- reads every lane.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from .core import MemSpace, Op, Stmt, Value
from .flow import Flow

#: The primitives whose lanes are known: the callee, the operand the map is
#: about, whether the map is one of the result's lanes (rather than of every
#: lane the call runs for: an FMA in place), and, from the callee's numbers,
#: the block the map repeats over and the lane of the block that lane `p` of
#: it reads.  Each row is run on `harness.wavesim` by `tests/test_demand.py`.
_READS: Tuple[Tuple[re.Pattern, int, bool,
                    Callable[..., Tuple[int, Callable[[int], int]]]], ...] = (
    # `broadcast<Block, Subblock, Lane>(v)`: lane `Lane * Subblock + p %
    # Subblock` of its block.
    (re.compile(r'tensorforge::broadcast<\s*(\d+)\s*,\s*(\d+)\s*,'
                r'\s*(\d+)\s*>$'), 0, True,
     lambda block, sub, lane: (block, lambda p: lane * sub + p % sub)),
    # `fmacdppN<Row>(c, a, b)`, `c += a * b` with `a` from lane `Row` of
    # every row of `N`.
    (re.compile(r'tensorforge::fmacdpp(4|8|16)<\s*(\d+)\s*>$'), 1, False,
     lambda width, row: (width, lambda p: row)),
    # `movdpp16<Row>(v)`: lane `Row` of every row of sixteen.
    (re.compile(r'tensorforge::movdpp16(?:Kept)?<\s*(\d+)\s*>$'), 0, True,
     lambda row: (16, lambda p: row)),
)

#: A primitive of `_READS` as the text of an expression of its one operand.
_SPELLED = re.compile(r'(tensorforge::\w+<[^<>]*>)\(\{0\}\)$')

#: Arithmetic every lane does on its own: what reads a lane of the result
#: reads that lane of every operand.
_LOCAL = Op.ARITH | {Op.MATH}

#: How deep the uses are followed before every lane counts as read.
_DEPTH = 32


def lane_map(callee: str) -> Optional[Tuple[int, bool, int,
                                            Callable[[int], int]]]:
    """`(operand, of_result, block, lane)` for a primitive of `_READS`: lane
    `p` of a block of `block` reads lane `lane(p)` of the block of operand
    `operand` -- `p` a lane of the result where `of_result`, any lane the
    call runs for otherwise.  None for any other callee."""
    for pattern, operand, of_result, make in _READS:
        m = pattern.match(callee or '')
        if m:
            block, lane = make(*(int(g) for g in m.groups()))
            return operand, of_result, block, lane
    return None


class Demand:
    """The lanes of the values of a flow's body that anything reads (see
    the module).  Asked of a value at an execution of what computes it."""

    def __init__(self, flow: Flow):
        self.flow = flow
        self._memo: Dict[Tuple, int] = {}
        self._open: Set[Tuple] = set()

    def lanes(self, v: Value, env: Dict[int, int], depth: int = 0) -> int:
        """The lanes of `v` anything reads, where `env` holds the hops of
        the counted loops around what computes it: a mask over the flow's
        lanes."""
        full = self.flow.full
        key = (v.id, tuple(sorted(env.items())))
        got = self._memo.get(key)
        if got is not None:
            return got
        if key in self._open or depth > _DEPTH:
            return full
        self._open.add(key)
        try:
            out = self._lanes(v, env, depth)
        finally:
            self._open.discard(key)
        self._memo[key] = out
        return out

    def _lanes(self, v: Value, env: Dict[int, int], depth: int) -> int:
        flow = self.flow
        full = flow.full
        d = flow.defs.get(v.id)
        accumulator = d is not None and d.op == Op.DECLARE
        if (d is None or d.attr('extern') is not None
                or (d.attr('escapes') and not accumulator)):
            # An argument of a region, or a value text may name.
            return full
        out = 0
        for u in flow.uses.get(v.id, ()):
            if u.op in _LOCAL and len(u.target) == 1:
                out |= self.lanes(u.target[0], env, depth + 1)
            elif u.op == Op.STORE:
                out |= self._stored(u, v, env, depth)
            elif u.op == Op.CALL and accumulator and self._adds_into(u, v):
                # An FMA in place reads the lanes it writes: what reads the
                # accumulator later reads them.
                continue
            elif u.op == Op.CALL:
                out |= self._called(u, v, env, depth, u.attr('callee'))
            elif u.op == Op.RAWEXPR and u.attr('crosslane'):
                # The same primitives, spelled as text around the operand.
                m = _SPELLED.match(u.text or '')
                if m is None:
                    return full
                out |= self._called(u, v, env, depth, m.group(1))
            else:
                return full
            if out == full:
                return full
        return out

    def _stored(self, u: Stmt, v: Value, env, depth: int) -> int:
        """What reads lanes of `v` through `u`, a store of it."""
        flow = self.flow
        full = flow.full
        if (len(u.accesses) != 1
                or u.accesses[0].space is not MemSpace.REGISTER
                or len(u.args) < 2 or u.args[1] is not v
                or any(a is v for a in u.args[2:]) or u.args[0] is v
                or getattr(v.type, 'length', None) is not None
                or id(u) in flow.opaque):
            return full
        out = 0
        found = False
        for n in self._executions(u, env):
            found = True
            for r, m in flow.readers.get(n, ()):
                x = flow.insts[r].stmt
                if (x.op != Op.LOAD or len(x.target) != 1 or len(x.accesses) != 1
                        or x.accesses[0].space is not MemSpace.REGISTER):
                    # Anything but a load back: an access the flow cannot
                    # place, which reads every element it may reach.
                    return full
                out |= self.lanes(x.target[0], flow.insts[r].env(),
                                  depth + 1) & m
                if out == full:
                    return full
        return out if found else full

    def _called(self, u: Stmt, v: Value, env, depth: int,
                callee: Optional[str]) -> int:
        """What reads lanes of `v` through `u`, a call of `callee` it is an
        operand of."""
        flow = self.flow
        full = flow.full
        known = lane_map(callee)
        if known is None:
            return full
        operand, of_result, block, lane = known
        if (sum(1 for a in u.args if a is v) != 1 or u.predicate is v
                or flow.threads % block):
            return full
        if not of_result and len(u.args) == 3 and u.args[2] is v:
            # The operand an FMA in place multiplies its lane with.
            return self._accumulated(u, env, depth)
        if len(u.args) <= operand or u.args[operand] is not v:
            return full
        if of_result:
            if len(u.target) != 1:
                return full
            wanted = self.lanes(u.target[0], env, depth + 1)
        else:
            wanted = self._accumulated(u, env, depth)
        out = 0
        for p in range(flow.threads):
            if wanted >> p & 1:
                out |= 1 << (p - p % block + lane(p % block))
        return out & full

    def _accumulated(self, u: Stmt, env, depth: int) -> int:
        """The lanes of the accumulator `u`, an FMA in place, adds into that
        anything reads later."""
        acc = u.args[0] if u.args else None
        d = self.flow.defs.get(acc.id) if isinstance(acc, Value) else None
        if d is None or d.op != Op.DECLARE:
            return self.flow.full
        return self.lanes(acc, env, depth + 1)

    @staticmethod
    def _adds_into(u: Stmt, v: Value) -> bool:
        """Is `u` an FMA in place of `_READS` whose accumulator is `v`, and
        nothing else of it?"""
        known = lane_map(u.attr('callee'))
        return (known is not None and not known[1] and len(u.args) == 3
                and u.args[0] is v and u.args[1] is not v
                and u.args[2] is not v and u.predicate is not v)

    def _executions(self, u: Stmt, env: Dict[int, int]) -> Sequence[int]:
        """The executions of `u` at the hops `env` names."""
        flow = self.flow
        return [n for n in flow.executions(u)
                if all(flow.insts[n].env().get(k, x) == x
                       for k, x in env.items())]
