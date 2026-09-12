# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a shared access costs in bank cycles, computed from the IR.

`tools/bank_conflicts.py` answers this by parsing the generated C++, and it
had to: the addresses were `rawexpr` text, so there was nothing in the IR to
evaluate.  They are operations now, and the only leaf that is not a constant
is `thread_idx_x` -- which a pass can read as "the lane".

Two implementations of one question is the arrangement this codebase keeps
finding underneath its bugs, so this exists to replace the text one, not to
sit beside it.  It is checked against it first: `tests/test_pir_banks.py`
compares the two over the corpus, and until they agree everywhere the text
version is the one that decides.

Run it on the *optimised* body.  A freshly finished one still holds the loads
that `dce` and `cse` are about to remove -- twice as many in `chain_three` --
and the addresses the hardware sees are the ones that survive.  That is also
where a pass acting on this would sit: after the passes that change what is
there, before the emitter that fixes it.

What the two count differs by about 200 accesses over the corpus, in one
direction: the text sees every subscript the emitter writes, including those
inside raw statements, and this sees structured loads and stores.  The gap is
therefore a measure of what is still raw -- `addressing_none` writes its
window as text and shows up here as 0 against 65 -- and it closes as that
does, rather than needing to be explained.

What this buys beyond tidiness is the thing the text version cannot do.  It
runs before emission, on a body, which is where a decision could still be
made -- the swizzle width is chosen from the buffer's volume today, three
times by hand, because the access pattern was not available at that point.
It is available here.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .core import (BufferType, Effect, MemSpace, Op, ScalarType, Stmt, Value,
                   XorSwizzle, walk, walk_stmts)

#: Bytes one bank serves per cycle.  The count of banks is a property of the
#: target and comes from `hw_descr.shmem_banks`; the width does not vary
#: across the targets here.
BANK_BYTES = 4

#: The unit that contends for banks.  A CDNA wave64 access to LDS is processed
#: in two halves of 32 and an RDNA wave32 in one, so 32 either way.
LANES = 32

_BIN = {'lt': operator.lt, 'le': operator.le, 'gt': operator.gt,
        'ge': operator.ge, 'eq': operator.eq, 'ne': operator.ne,
        'and': lambda a, b: a and b, 'or': lambda a, b: a or b,
        'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
        'div': operator.floordiv, 'rem': operator.mod,
        'bitand': operator.and_, 'bitor': operator.or_,
        'bitxor': operator.xor, 'shl': operator.lshift,
        'shr': operator.rshift}


class NotStatic(Exception):
    """The address depends on something that is not the lane or a constant."""


@dataclass(frozen=True)
class Access:
    buffer: Value
    kind: str                 # 'load' or 'store'
    ways: int
    lanes: int


def _definitions(body: Sequence[Stmt]) -> Dict[int, object]:
    """Where each value comes from: the statement that produced it, or, for a
    loop's induction variable, its lower bound.

    A loop variable is a region argument, not a statement result, so it has no
    producer to find and 159 addresses came back unresolved.  Substituting the
    bound is sound for this question: these are loops over tensor dimensions,
    every lane in the wave is on the same iteration, so the value shifts every
    lane's address equally and the bank pattern is unchanged.  Anything
    lane-dependent reaches the address through `thread_idx_x`.
    """
    out: Dict[int, object] = {}
    for stmt in walk_stmts(body):
        for t in stmt.target:
            out[id(t)] = stmt
        if (stmt.op in (Op.FOR, Op.WHILE) and stmt.regions
                and stmt.regions[0].args):
            # What the induction starts at.  Later iterations differ from it
            # by an amount that is the same in every lane -- a multiple of the
            # step, or whichever block the queue handed out -- so the pattern
            # across the lanes, which is the only thing counted here, is the
            # one the first iteration has.
            start = stmt.args[0] if stmt.args else 0
            out[id(stmt.regions[0].args[0])] = ('bound', start)
    return out


def evaluate(operand, tid: int, defs: Dict[int, Stmt], depth: int = 0):
    """The value this operand takes on lane `tid`.

    Raises `NotStatic` rather than guessing.  An address the analysis cannot
    read is one it must not count, and saying so is the difference between a
    number and a number-shaped thing.
    """
    try:
        # `operator.index` and not `isinstance(..., int)`: shapes and offsets
        # arrive as `numpy.int64`, which is an integer everywhere except to
        # `isinstance`.  Two dozen addresses were unresolved for that alone.
        return operator.index(operand)
    except TypeError:
        pass
    if depth > 64 or not isinstance(operand, Value):
        raise NotStatic(repr(operand))
    stmt = defs.get(id(operand))
    if stmt is None:
        raise NotStatic(f'no definition for {operand}')
    if isinstance(stmt, tuple):
        return evaluate(stmt[1], tid, defs, depth + 1)   # a loop bound
    name = str(stmt.op)
    if name == 'call':
        callee = stmt.attr('callee')
        if callee == 'thread_idx_x':
            return tid
        if callee in ('thread_idx_y', 'thread_idx_z'):
            return 0          # uniform within a wave
        raise NotStatic(callee or 'call')
    if name == 'const':
        value = stmt.attr('value')
        if isinstance(value, int):
            return value
        raise NotStatic(f'const {value!r}')
    f = _BIN.get(name)
    if f is None or len(stmt.args) != 2:
        raise NotStatic(name)
    return f(evaluate(stmt.args[0], tid, defs, depth + 1),
             evaluate(stmt.args[1], tid, defs, depth + 1))


def ways(addresses: Sequence[int], base_bytes: int, width: int = 1,
         banks: int = 32) -> int:
    """Bank cycles this access costs.

    `base_bytes` is the buffer's element size and `width` how many of them the
    access covers.  The two are separate because the *index* is in base
    elements even when the access is a vector, and a wide access is served in
    phases of `banks * BANK_BYTES // span` lanes -- which is what makes a
    `float4` store conflict-free where a whole-warp model calls it 2-way.
    """
    span = base_bytes * width
    per_phase = max(1, (banks * BANK_BYTES) // span)
    worst = 0
    for start in range(0, len(addresses), per_phase):
        seen: Dict[int, set] = {}
        for a in addresses[start:start + per_phase]:
            byte = a * base_bytes
            for off in range(0, span, BANK_BYTES):
                seen.setdefault(((byte + off) // BANK_BYTES) % banks,
                                set()).add(byte)
        worst = max(worst, max((len(s) for s in seen.values()), default=0))
    return worst


def _active(parents: Sequence[Stmt], defs: Dict[int, Stmt]) -> List[int]:
    """Which lanes reach an access under the guards enclosing it.

    The staging steps here are guarded to a quarter or a half of the wave, and
    counting inactive lanes into a bank is how a diagnostic earns a reputation
    for crying wolf -- 72 accesses in `rectangular` read as 2-way that way and
    are 1-way.

    `walk` yields the enclosing statements, and an `Op.IF` predicate is an
    ordinary value: the same evaluator answers it per lane.  A predicate this
    cannot read leaves the lane set whole, which over-reports rather than
    under-reports.
    """
    lanes = list(range(LANES))
    for parent in parents:
        if parent.op is not Op.IF:
            continue
        try:
            lanes = [t for t in lanes if evaluate(parent.cond, t, defs)]
        except NotStatic:
            continue
    return lanes or list(range(LANES))


def analyse(body: Sequence[Stmt], banks: int = 32) -> Tuple[List[Access], int]:
    """Every shared access in this body, and how many were not static."""
    defs = _definitions(body)
    found: List[Access] = []
    unresolved = 0
    for stmt, parents in walk(body):
        if stmt.op not in (Op.LOAD, Op.STORE):
            continue
        for access in stmt.accesses:
            if access.space != MemSpace.SHARED:
                continue
            buf = access.base
            t = getattr(buf, 'type', None)
            if not isinstance(t, BufferType):
                continue
            index = stmt.args[-1] if stmt.args else None
            lanes = _active(parents, defs)
            try:
                addrs = [evaluate(index, i, defs) for i in lanes]
            except NotStatic:
                unresolved += 1
                continue
            # The width comes from the value, and which operand that is
            # depends on the direction: a load produces it, a store consumes
            # it.  Reading `target` for both made every vector *store* look
            # scalar -- and a scalar model of a `float4` store puts four times
            # the stride between lanes, which is how 72 conflict-free accesses
            # in `rectangular` read as 2-way.
            carrier = (stmt.target[0] if stmt.op == Op.LOAD and stmt.target
                       else stmt.args[1] if len(stmt.args) > 1 else None)
            width = getattr(getattr(carrier, 'type', None), 'length', None) or 1
            found.append(Access(
                buf, 'load' if stmt.op == Op.LOAD else 'store',
                ways(addrs, t.elem.size(), width, banks), len(lanes)))
    return found, unresolved


#: Widths a permutation may take: powers of two up to the bank count.
CANDIDATE_WIDTHS = (1, 2, 4, 8, 16, 32)


def recommend(body: Sequence[Stmt], banks: int = 32):
    """What each shared buffer's permutation costs, and what it could cost.

    The width is chosen at `alloc` today, from the buffer's volume, because
    the access pattern is not known at that point -- which is the ordering
    problem behind three widths picked by hand, each with a measurement beside
    it in a comment.

    This does not fix the ordering.  It answers the question the ordering gets
    in the way of: given the accesses a body actually makes, which width would
    have been right.  The permutation is an involution, so applying the
    buffer's current one to an emitted address recovers the plain index, and
    every candidate can then be tried against the real pattern.

    Returns ``{buffer: (current_width, {width: worst_ways})}``.  A
    recommendation that disagrees with what was chosen is a finding either
    way: the volume rule is wrong there, or the pattern is one this scoring
    does not capture.
    """
    defs = _definitions(body)
    per_buffer = {}
    for stmt, parents in walk(body):
        if stmt.op not in (Op.LOAD, Op.STORE):
            continue
        for access in stmt.accesses:
            if access.space != MemSpace.SHARED:
                continue
            buf = access.base
            t = getattr(buf, 'type', None)
            if not isinstance(t, BufferType):
                continue
            index = stmt.args[-1] if stmt.args else None
            lanes = _active(parents, defs)
            try:
                addrs = [evaluate(index, i, defs) for i in lanes]
            except NotStatic:
                continue
            carrier = (stmt.target[0] if stmt.op == Op.LOAD and stmt.target
                       else stmt.args[1] if len(stmt.args) > 1 else None)
            width = getattr(getattr(carrier, 'type', None), 'length', None) or 1
            per_buffer.setdefault(buf, []).append((addrs, t.elem.size(), width))

    out = {}
    for buf, accesses in per_buffer.items():
        current = getattr(buf.type, 'swizzle', None)
        cw = current.width if current is not None else 1
        scores = {}
        for candidate in CANDIDATE_WIDTHS:
            worst = 0
            for addrs, elem, width in accesses:
                plain = [current.apply(a) for a in addrs] if current else addrs
                if candidate > 1:
                    swz = XorSwizzle(candidate)
                    permuted = [swz.apply(a) for a in plain]
                else:
                    permuted = plain
                worst = max(worst, ways(permuted, elem, width, banks))
            scores[candidate] = worst
        out[buf] = (cw, scores)
    return out
