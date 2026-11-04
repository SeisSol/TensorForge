# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a shared access costs in bank cycles, computed from the IR.

`tools/bank_conflicts.py` answers this by parsing the generated C++.  In the
IR the addresses are operations, and the only leaf that is not a constant is
`thread_idx_x` -- which a pass can read as "the lane".

Two implementations of one question are a standing source of bugs, so this
exists to replace the text one, not to sit beside it.  It is checked against
it first: `tests/test_pir_banks.py` compares the two over the corpus, and
until they agree everywhere the text version is the one that decides.

Run it on the *optimized* body.  A freshly finished one still holds the loads
that `dce` and `cse` are about to remove -- twice as many in `chain_three` --
and the addresses the hardware sees are the ones that survive.  That is also
where a pass acting on this would sit: after the passes that change what is
there, before the emitter that fixes it.

What the two count differs by about 200 accesses over the corpus, in one
direction: the text sees every subscript the emitter writes, including those
inside raw statements, and this sees structured loads and stores.  The gap is
therefore a measure of what is written as raw text -- `addressing_none` writes
its window as text and shows up here as 0 against 65 -- and it shrinks with
that, rather than needing to be explained.

What this buys beyond tidiness is the thing the text version cannot do.  It
runs before emission, on a body, which is where a decision could still be
made -- the swizzle width is chosen from the buffer's volume today, three
times by hand, because the access pattern is not available at that point.
It is available here.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import (BufferType, MemSpace, Op, Operand, Stmt, Value, XorSwizzle,
                   walk, walk_stmts)

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

#: The one-operand ops.  `neg` is deliberately absent: it was never
#: evaluated here, and adding it would change which addresses this
#: pass can read and so which buffers get a swizzle.
_UN = {'popcount': lambda a: int(a).bit_count()}


#: A raw expression that is a read of a named table at one index.
_TABLE_READ = re.compile(r'^(\w+)\[\{0\}\]$')


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
    producer to find, and an address built from one would come back
    unresolved.  Substituting the bound is sound for this question: these are
    loops over tensor dimensions, every lane in the wave is on the same
    iteration, so the value shifts every lane's address equally and the bank
    pattern is unchanged.  Anything lane-dependent reaches the address through
    `thread_idx_x`.
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


def evaluate(operand, tid: int, defs: Dict[int, Stmt], depth: int = 0,
             tables: Optional[Dict[str, Sequence[int]]] = None):
    """The value this operand takes on lane `tid`.

    Raises `NotStatic` rather than guessing.  An address the analysis cannot
    read is one it must not count, and saying so is the difference between a
    number and a number-shaped thing.
    """
    try:
        # `operator.index` and not `isinstance(..., int)`: shapes and offsets
        # arrive as `numpy.int64`, which is an integer everywhere except to
        # `isinstance`.
        return operator.index(operand)
    except TypeError:
        pass
    if depth > 64 or not isinstance(operand, Value):
        raise NotStatic(repr(operand))
    stmt = defs.get(id(operand))
    if stmt is None:
        raise NotStatic(f'no definition for {operand}')
    if isinstance(stmt, tuple):
        return evaluate(stmt[1], tid, defs, depth + 1, tables)   # a loop bound
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
    if name == 'rawexpr' and tables and len(stmt.args) == 1:
        # A read of a table of the generated file (`module_tables`): the
        # numbers are the generator's, so the entry a lane reads is known --
        # the column an ELL gather takes is one (`MultilinearInstruction.
        # _ell_term`).
        m = _TABLE_READ.match(stmt.text or '')
        values = tables.get(m.group(1)) if m else None
        if values is not None:
            at = evaluate(stmt.args[0], tid, defs, depth + 1, tables)
            if 0 <= at < len(values):
                return int(values[at])
            raise NotStatic(f'{m.group(1)}[{at}] past its {len(values)}')
    g = _UN.get(name)
    if g is not None and len(stmt.args) == 1:
        return g(evaluate(stmt.args[0], tid, defs, depth + 1, tables))
    f = _BIN.get(name)
    if f is None or len(stmt.args) != 2:
        raise NotStatic(name)
    return f(evaluate(stmt.args[0], tid, defs, depth + 1, tables),
             evaluate(stmt.args[1], tid, defs, depth + 1, tables))


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


def _active(parents: Sequence[Stmt], defs: Dict[int, Stmt],
            tables: Optional[Dict[str, Sequence[int]]] = None) -> List[int]:
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
            lanes = [t for t in lanes if evaluate(parent.cond, t, defs, 0, tables)]
        except NotStatic:
            continue
    return lanes or list(range(LANES))


def analyze(body: Sequence[Stmt], banks: int = 32) -> Tuple[List[Access], int]:
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
            # it.  Reading `target` for both would make every vector *store*
            # look scalar -- and a scalar model of a `float4` store puts four
            # times the stride between lanes, which would make 72
            # conflict-free accesses in `rectangular` read as 2-way.
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


# --------------------------------------------------------------------------- #
# Deciding the permutation of a staged window
# --------------------------------------------------------------------------- #

#: What a permutation costs an access whose index is not a constant: the
#: shift, the mask and the xor `PirBuilder._swizzled` emits (and one more to
#: put a granule's number back in place).  Counted in issue slots, the unit a
#: bank cycle is counted in too -- close enough for the question it decides,
#: which is whether a permutation pays at all.  Where every index is a
#: constant once the loops are written out, the compiler folds the
#: permutation away and it costs nothing.
PERMUTE_OPS = 3


@dataclass(frozen=True)
class Proposal:
    """What `propose` says about one staged window."""

    #: The permutation, or None for the plain layout.
    swizzle: Optional[XorSwizzle]
    #: What the plain layout costs, in bank cycles weighted by how often each
    #: access runs; and what `swizzle` costs, the permutation's own
    #: arithmetic included.  Equal where nothing was proposed.
    plain: float
    permuted: float
    #: Why the window was not read, where it was not; ''.
    unread: str = ''


def _index_kinds(defs: Dict[int, object], loops: Dict[int, Stmt]):
    """`kind(operand)`: 0 for a constant once the fully unrolled loops are
    written out, 1 for a value that varies but is the same in every lane (a
    rolled loop's counter, a block index), 2 for one that may differ between
    the lanes.

    The middle class is the one the permutation is not free for and cannot
    help: every lane reads the same address, so there is no bank to spread
    the lanes over, and the address changes from trip to trip, so the
    compiler cannot fold the permutation into it.
    """
    memo: Dict[int, int] = {}

    def kind(operand, depth: int = 0) -> int:
        if not isinstance(operand, Value):
            return 0
        key = id(operand)
        got = memo.get(key)
        if got is not None:
            return got
        if depth > 64:
            return 2
        loop = loops.get(key)
        if loop is not None:
            # An induction: the unrolled loop's is a constant in every copy
            # of the body the compiler writes out, the rolled loop's is not.
            out = 0 if loop.attr('unroll') is True else 1
        else:
            stmt = defs.get(key)
            if stmt is None or isinstance(stmt, tuple):
                out = 2
            elif str(stmt.op) == 'call':
                callee = stmt.attr('callee')
                out = 2 if callee in ('thread_idx_x', None) else 1
            elif str(stmt.op) == 'const':
                out = 0
            elif str(stmt.op) in _BIN or str(stmt.op) in _UN:
                out = max((kind(a, depth + 1) for a in stmt.args), default=0)
            else:
                # A load, a raw expression: nothing says it agrees across
                # the lanes.
                out = 2
        memo[key] = out
        return out
    return kind


def _runs(parents: Sequence[Stmt]) -> float:
    """How often a statement under `parents` runs, from the loops whose
    bounds are constants.  A loop with a bound known only at run time -- the
    batch loop -- encloses every access of the body alike and is left out."""
    n = 1.0
    for parent in parents:
        if parent.op != Op.FOR:
            continue
        lo, hi, step = parent.loop_bounds
        try:
            lo, hi, step = (operator.index(lo), operator.index(hi),
                            operator.index(step))
        except TypeError:
            continue
        if step > 0 and hi > lo:
            n *= -(-(hi - lo) // step)
    return n


def _window_access(stmt: Stmt, base) -> Tuple[Optional[Operand], int]:
    """The index of the element `stmt` reaches in the window `base` first,
    and how many adjacent elements it reaches -- or None for an access this
    cannot read."""
    if stmt.op == Op.LOAD:
        index = stmt.args[1:]
        width = getattr(stmt.target[0].type, 'length', None) if stmt.target else None
    elif stmt.op == Op.STORE:
        index = stmt.args[2:]
        width = getattr(getattr(stmt.args[1], 'type', None), 'length', None)
    elif stmt.op == Op.COPY_ASYNC and stmt.copy_dst is base:
        index = stmt.copy_dst_index
        width = stmt.attr('elems', 1)
    elif stmt.op == Op.COPY_ASYNC and stmt.copy_src is base:
        index = stmt.copy_src_index
        width = stmt.attr('elems', 1)
    else:
        return None, 0
    if len(index) != 1:
        return None, 0
    return index[0], width or 1


def propose(body: Sequence[Stmt], lanes: int = LANES, banks: int = 32,
            tables: Optional[Dict[str, Sequence[int]]] = None
            ) -> Dict[Any, Proposal]:
    """A permutation for each staged window of `body` that pays for itself,
    decided from the accesses the body makes -- by window identity.

    The windows are the ones built unpermuted and marked `permutable` with
    the granule their writer needs (`AbstractShrMemWrite._window`).  Every
    access to one is read lane by lane: the lanes of *one* multiplication,
    `lanes` of them up to the 32 that contend for banks together.  Where a
    warp holds several multiplications they reach separate windows, which the
    arena already starts on different banks (`Generator._mult_arena_size`),
    and a permutation of each window's own index changes nothing between
    them.

    The candidates are every power-of-two width up to the bank count, with a
    granule as wide as the widest access and the writer's need, wherever
    the window's volume is a whole number of blocks.  Each is scored as bank
    cycles over the accesses, weighted by how often each runs, plus
    `PERMUTE_OPS` for every run of an access whose index the compiler cannot
    fold.  The plain layout wins ties: a permutation that saves nothing is
    four operations an access in a rolled loop and a vector read the
    hardware cannot make -- SeisSol's `localFluxAll` at order 6 rolled by
    four spent 139 of 254 instructions a trip on it on sm_120.

    A loop's counter is read at its first value (`_definitions`).  For the
    plain layout every trip has the pattern of the first; for a permutation
    the first trip stands for the rest, which holds for the reads along a row
    or down a column that a permutation is for.

    A window with an access this cannot read keeps the plain layout and says
    why (`Proposal.unread`).  `tables` are the numbers of the generated
    file's tables by name (`module_tables`), for an address read from one.
    """
    defs = _definitions(body)
    loops: Dict[int, Stmt] = {}
    windows: Dict[int, Any] = {}
    facts: Dict[Any, dict] = {}
    for stmt in walk_stmts(body):
        if stmt.op == Op.FOR and stmt.regions and stmt.regions[0].args:
            loops[id(stmt.regions[0].args[0])] = stmt
        if stmt.op != Op.ALLOC:
            continue
        identity = stmt.attr('identity')
        need = stmt.attr('permutable')
        if identity is None or need is None:
            continue
        for t in stmt.target:
            tp = t.type
            if not (isinstance(tp, BufferType) and tp.space == MemSpace.SHARED
                    and tp.swizzle is None):
                continue
            windows[id(t)] = identity
            f = facts.setdefault(identity, dict(volume=tp.volume,
                                                elem=tp.elem.size(),
                                                need=need, widest=1,
                                                accesses=[], unread=''))
            f['need'] = max(f['need'], need)
            f['volume'] = min(f['volume'], tp.volume)
    if not facts:
        return {}

    kind = _index_kinds(defs, loops)
    span = max(1, min(lanes, LANES))
    for stmt, parents in walk(body):
        for access in stmt.accesses:
            if access.space != MemSpace.SHARED:
                continue
            identity = windows.get(id(access.base))
            if identity is None:
                continue
            f = facts[identity]
            if f['unread'] or stmt.op in (Op.MARK, Op.ALLOC):
                continue
            index, width = _window_access(stmt, access.base)
            if index is None:
                f['unread'] = f'{stmt.op} reaches it'
                continue
            f['widest'] = max(f['widest'], width)
            runs = _runs(parents)
            k = kind(index)
            if k < 2:
                # The same address in every lane: one bank cycle whatever the
                # layout, and the permutation's arithmetic where it does not
                # fold.
                f['accesses'].append((None, width, runs, k == 1))
                continue
            active = [t for t in _active(parents, defs, tables) if t < span]
            try:
                addrs = [evaluate(index, t, defs, 0, tables) for t in active]
            except NotStatic as e:
                f['unread'] = f'an index this cannot evaluate ({e})'
                continue
            f['accesses'].append((addrs, width, runs, True))

    out: Dict[Any, Proposal] = {}
    for identity, f in facts.items():
        if f['unread'] or not f['accesses']:
            out[identity] = Proposal(None, 0.0, 0.0, f['unread'])
            continue
        elem = f['elem']

        def score(swz: Optional[XorSwizzle]) -> float:
            extra = PERMUTE_OPS + (1 if swz is not None and swz.granule > 1
                                   else 0)
            total = 0.0
            for addrs, width, runs, varies in f['accesses']:
                if addrs is None:
                    cycles = 1
                else:
                    seen = (addrs if swz is None
                            else [swz.apply(a) for a in addrs])
                    cycles = ways(seen, elem, width, banks)
                total += runs * cycles
                if swz is not None and varies:
                    total += runs * extra
            return total

        plain = score(None)
        granule = 1
        while granule < max(f['need'], f['widest']):
            granule *= 2
        best, best_cost = None, plain
        width = 2
        while width <= banks:
            if f['volume'] % (width * granule) == 0:
                swz = XorSwizzle(width, granule)
                cost = score(swz)
                if cost < best_cost:
                    best, best_cost = swz, cost
            width *= 2
        out[identity] = Proposal(best, plain, best_cost)
    return out
