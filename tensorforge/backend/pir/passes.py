# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: verifier and the first three real passes.

All passes are pure functions ``body -> body``.  Nothing mutates in place, so
dumping between passes is a one-liner and a bad pass can never corrupt the
input it was handed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

from .core import (Access, BufferType, Effect, IRError, MemSpace, Op, Operand,
                   Region, ScalarType, Stmt, TokenType, Value,
                   accesses_conflict, collect_accesses, collect_effect,
                   def_use, defined_within, walk)
from .asyncmem import check_tokens, schedule_async


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

_INTEGER_ONLY = frozenset({'bitand', 'bitor', 'bitxor', 'shl', 'shr', 'rem'})
_FLOAT_TYPES = frozenset({Datatype.F16, Datatype.F32, Datatype.F64})


def verify(body: Tuple[Stmt, ...], strict: bool = True) -> List[str]:
    """Structural + SSA + uniformity checks.

    Returns a list of diagnostics.  With ``strict`` it raises on the first
    batch instead, which is what ``gen_ir`` should do in debug builds.
    Legacy ``raw*`` statements are checked more loosely on purpose --- they are
    opaque by construction, and a hard error there would just block migration.
    """
    diag: List[str] = []

    try:
        defs, uses = def_use(body)  # single-assignment / single-binding
        diag.extend(check_tokens(body, defs, uses))
    except IRError as e:
        diag.append(str(e))

    _check_scope(body, set(), diag, divergent=False)
    diag.extend(_check_dangling_names(body))

    if diag and strict:
        raise IRError('pseudo-IR verification failed:\n  ' + '\n  '.join(diag))
    return diag


_NAMED = re.compile(r'\bv(\d+)_\w*\b')


def _check_dangling_names(body: Tuple[Stmt, ...]) -> List[str]:
    """Raw text must not name a value that no statement defines.

    During migration a value's *name* is often interpolated into text the IR
    cannot rewrite.  `escapes` protects such a value from being folded,
    eliminated or inlined --- but a pass that forgets the marker leaves the
    text pointing at nothing, and the result is generated source that does not
    compile.  Cheap to check, and it catches the whole class at once.
    """
    defined = set()
    for s, _ in walk(body):
        for t in s.target:
            defined.add(t.id)
        for r in s.regions:
            for a in r.args:
                defined.add(a.id)
    seen, diag = set(), []
    for s, _ in walk(body):
        if not s.text:
            continue
        for m in _NAMED.finditer(s.text):
            ident = int(m.group(1))
            if ident not in defined and m.group(0) not in seen:
                seen.add(m.group(0))
                diag.append(f'{s.op}: raw text names {m.group(0)}, which no '
                            f'statement defines')
    return diag


def _typeof(x: Operand):
    return x.type if isinstance(x, Value) else None


def _same_types(a: Sequence[Operand], b: Sequence[Value]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        tx = _typeof(x)
        if tx is not None and tx != y.type:
            return False
    return True


def _check_scope(body: Tuple[Stmt, ...], live: set, diag: List[str],
                 divergent: bool) -> None:
    """``live`` is the set of value ids visible from enclosing scopes."""
    live = set(live)

    for i, s in enumerate(body):
        # -- operand dominance ------------------------------------------- #
        for v in s.operands():
            if v.id not in live:
                diag.append(f'{s.op}: operand {v!r} used before definition')

        # -- predicate ----------------------------------------------------- #
        if s.predicate is not None:
            t = s.predicate.type
            if not (isinstance(t, ScalarType) and t.base.name == 'BOOL'):
                diag.append(f'{s.op}: predicate {s.predicate!r} is not bool')

        # -- terminator placement ------------------------------------------ #
        if s.op == Op.YIELD and i != len(body) - 1:
            diag.append('yield must be the last statement of a region')

        # -- barriers must not be reached divergently ---------------------- #
        if (s.effect & Effect.BARRIER) and divergent:
            diag.append(f'{s.op}: barrier inside a thread-divergent region '
                        f'(undefined behaviour on CUDA/HIP)')

        # -- op-specific --------------------------------------------------- #
        if s.op == Op.IF:
            if not 1 <= len(s.regions) <= 2:
                diag.append('if: expects one or two regions')
            if len(s.args) != 1:
                diag.append('if: expects exactly one condition operand')
            for r in s.regions:
                if r.args:
                    diag.append('if: regions take no arguments')
                if s.target and r.terminator is None:
                    diag.append('if: region producing results must end in yield')
                elif s.target and not _same_types(r.yielded, s.target):
                    diag.append('if: yielded arity/type does not match results')
            if s.target and len(s.regions) != 2:
                diag.append('if: producing results requires an else region')

        elif s.op == Op.FOR:
            if len(s.regions) != 1:
                diag.append('for: expects exactly one region')
            elif len(s.args) < 3:
                diag.append('for: expects lo, hi, step and one init per iter_arg')
            else:
                r = s.regions[0]
                if not r.args:
                    diag.append('for: region must bind the induction variable')
                else:
                    n = len(r.args) - 1
                    if n != len(s.loop_inits):
                        diag.append('for: iter_args and inits disagree in arity')
                    if n != len(s.target):
                        diag.append('for: iter_args and results disagree in arity')
                    if n and r.terminator is None:
                        diag.append('for: loop carrying values must end in yield')
                    elif n and not _same_types(r.yielded, r.args[1:]):
                        diag.append('for: yielded types do not match iter_args')

        elif s.op == Op.RAWEXPR:
            if len(s.target) != 1 or s.text is None:
                diag.append('rawexpr: needs exactly one target and text')
            if s.pure and (s.effect & Effect.UNKNOWN):
                diag.append('rawexpr: opaque text must not be marked pure')

        elif s.op == Op.RAWSTMT:
            if s.target or s.text is None:
                diag.append('rawstmt: needs text and no target')

        elif s.op == Op.RAWBLOCK:
            if len(s.regions) != 1 or s.text is None:
                diag.append('rawblock: needs text and exactly one region')

        elif s.op == Op.ALLOC:
            if len(s.target) != 1 or not isinstance(s.target[0].type, BufferType):
                diag.append('alloc: must produce exactly one buffer value')

        elif s.op in (Op.LOAD, Op.STORE):
            if not s.accesses:
                diag.append(f'{s.op}: must declare at least one access')

        elif s.op in _INTEGER_ONLY:
            for x in list(s.args) + list(s.target):
                t = _typeof(x) if not isinstance(x, Value) else x.type
                if isinstance(t, ScalarType) and t.base in _FLOAT_TYPES:
                    diag.append(f'{s.op}: bitwise/shift operand {x!r} is '
                                f'floating point')

        elif s.op == Op.LOAD_ASYNC:
            if len(s.target) != 1 or not isinstance(s.target[0].type, TokenType):
                diag.append('load.async: must produce exactly one token')
            if not s.accesses:
                diag.append('load.async: must declare its read')
            if not (s.effect & Effect.ASYNC):
                diag.append('load.async: must carry Effect.ASYNC')

        elif s.op == Op.COPY_ASYNC:
            if len(s.target) != 1 or not isinstance(s.target[0].type, TokenType):
                diag.append('copy.async: must produce exactly one token')
            if len(s.args) < 2:
                diag.append('copy.async: needs a destination and a source')
            if len(s.accesses) < 2:
                diag.append('copy.async: must declare the read and the write')
            if not (s.effect & Effect.ASYNC):
                diag.append('copy.async: must carry Effect.ASYNC')

        elif s.op == Op.WAIT:
            if len(s.args) > 1:
                diag.append('wait: takes at most one token')
            if s.args and not isinstance(_typeof(s.args[0]), TokenType):
                diag.append('wait: argument must be a completion token')

        # -- recurse ------------------------------------------------------- #
        if s.regions:
            inner_div = divergent or _is_divergent(s)
            for r in s.regions:
                _check_scope(r.body, live | {a.id for a in r.args}, diag, inner_div)

        for t in s.target:
            live.add(t.id)


def _is_divergent(s: Stmt) -> bool:
    """Does entering this statement's regions depend on thread-varying data?"""
    if s.op == Op.IF:
        c = s.cond
        return isinstance(c, Value) and not c.uniform
    if s.op == Op.FOR:
        return any(isinstance(b, Value) and not b.uniform for b in s.loop_bounds)
    if s.op == Op.RAWBLOCK:
        # opaque head text: assume the worst
        return True
    return False


# --------------------------------------------------------------------------- #
# Substitution
# --------------------------------------------------------------------------- #

def substitute(body: Tuple[Stmt, ...], mapping: Dict[int, Value]) -> Tuple[Stmt, ...]:
    """Replace uses (not definitions) of values according to ``mapping``."""
    if not mapping:
        return body

    def sub(x: Operand) -> Operand:
        return mapping.get(x.id, x) if isinstance(x, Value) else x

    out: List[Stmt] = []
    for s in body:
        pred = sub(s.predicate) if s.predicate is not None else None
        out.append(replace(
            s,
            args=tuple(sub(a) for a in s.args),
            predicate=pred,
            regions=tuple(replace(r, body=substitute(r.body, mapping))
                          for r in s.regions),
        ))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Dead code elimination
# --------------------------------------------------------------------------- #

def dce(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Drop pure statements whose results nobody reads.  Runs to fixpoint."""
    while True:
        _, uses = def_use(body)
        new = _dce_body(body, uses)
        if new == body:
            return body
        body = new


def _dce_body(body: Tuple[Stmt, ...], uses) -> Tuple[Stmt, ...]:
    out: List[Stmt] = []
    for s in body:
        s = replace(s, regions=tuple(replace(r, body=_dce_body(r.body, uses))
                                     for r in s.regions))
        if s.op == Op.YIELD or s.has_side_effects:
            out.append(s)
            continue
        if s.attr('escapes'):
            # its name is interpolated into raw text the IR cannot see
            out.append(s)
            continue
        if s.text is not None:
            # A raw statement is output by construction -- a Comment has no
            # target, no region and no effect, and dropping it would silently
            # change the generated source.
            out.append(s)
            continue
        if s.regions and collect_effect(s.regions[0].body) & (
                Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN):
            out.append(s)
            continue
        if s.target and all(not uses.get(t.id) for t in s.target):
            continue                    # dead
        if not s.target and not s.regions:
            continue                    # produces nothing, does nothing
        out.append(s)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Common subexpression elimination
# --------------------------------------------------------------------------- #

def _cse_key(s: Stmt):
    def k(x: Operand):
        return ('v', x.id) if isinstance(x, Value) else ('c', type(x).__name__, x)
    return (s.op, tuple(k(a) for a in s.args), s.text, s.attrs,
            None if s.predicate is None else s.predicate.id)


def cse(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Hash-cons pure, region-free statements.

    Only expressions from *enclosing* scopes are reused, so dominance holds by
    construction --- no dominator tree needed with structured control flow.
    """
    body, mapping = _cse_body(body, {})
    return substitute(body, mapping) if mapping else body


def _cse_body(body: Tuple[Stmt, ...], available: Dict[Any, Tuple[Value, ...]]):
    available = dict(available)
    mapping: Dict[int, Value] = {}
    out: List[Stmt] = []

    for s in body:
        if s.regions:
            regions = []
            for r in s.regions:
                inner, inner_map = _cse_body(r.body, available)
                regions.append(replace(r, body=substitute(inner, inner_map)))
            out.append(replace(s, regions=tuple(regions)))
            continue

        if (s.pure and not s.has_side_effects and s.target
                and s.effect == Effect.NONE and not s.attr('escapes')):
            key = _cse_key(s)
            prev = available.get(key)
            if prev is not None and len(prev) == len(s.target):
                for old, new in zip(s.target, prev):
                    mapping[old.id] = new
                continue
            available[key] = s.target

        out.append(s)

    return tuple(out), mapping


# --------------------------------------------------------------------------- #
# Constant folding and algebraic identities
# --------------------------------------------------------------------------- #

# Arithmetic that can be evaluated when every operand is a known constant.
# Deliberately a table rather than `eval`: an op name pir does not know stays
# untouched, which is what keeps the op set extensible.
_FOLDABLE: Dict[str, Any] = {
    'neg': lambda a: -a,
    'abs': lambda a: abs(a),
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'max': lambda a, b: max(a, b),
    'min': lambda a, b: min(a, b),
    'eq': lambda a, b: a == b,
    'neq': lambda a, b: a != b,
    'lt': lambda a, b: a < b,
    'le': lambda a, b: a <= b,
    'gt': lambda a, b: a > b,
    'ge': lambda a, b: a >= b,
    'and': lambda a, b: a and b,
    'or': lambda a, b: a or b,
}


def _fold_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a / b


def _fold_rem(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a % b


_FOLDABLE['div'] = _fold_div
_FOLDABLE['rem'] = _fold_rem


def fold(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Evaluate constant expressions and apply algebraic identities.

    Two jobs in one walk, because each exposes the other: folding
    ``mul(c1, c2)`` produces a constant that may then make ``add(x, 0)`` an
    identity, and applying an identity may hand a constant to a foldable op.

    The identities are the ones the frontend used to apply inline while building
    the expression tree (``optree.mul`` returned its other operand for a
    multiplication by one, and so on) --- a rewrite that had to happen at
    construction time because there was no pass to do it later.  They belong
    here: the caller writes what it means, and the IR simplifies.

    Conservative by construction: only ``pure`` region-free single-target
    statements are touched, division by a constant zero is left alone rather
    than folded into a trap, and an op name absent from ``_FOLDABLE`` with no
    matching identity passes through untouched.
    """
    body, mapping = _fold_body(body, {})
    return substitute(body, mapping) if mapping else body


def _as_number(x: Operand, consts: Dict[int, Any]):
    """The constant behind an operand, or ``None`` if it is not known."""
    if isinstance(x, Value):
        return consts.get(x.id)
    if isinstance(x, (int, float, bool)):
        return x
    return None


def _is(n, *values) -> bool:
    return n is not None and not isinstance(n, bool) and any(n == v for v in values)


def _identity(s: Stmt, nums: List[Any]) -> Optional[int]:
    """Index of the operand this statement is equal to, if any.

    Returns ``0``/``1`` to mean "the result is just that operand".
    """
    op = s.op
    a, b = (nums + [None, None])[:2]
    if len(s.args) != 2:
        return None
    if op == 'add':
        if _is(a, 0):
            return 1
        if _is(b, 0):
            return 0
    elif op == 'sub':
        if _is(b, 0):
            return 0
    elif op == 'mul':
        if _is(a, 1):
            return 1
        if _is(b, 1):
            return 0
    elif op == 'div':
        if _is(b, 1):
            return 0
    elif op in ('and', 'min'):
        # x and x == x; only safe when both operands are the same value
        if (isinstance(s.args[0], Value) and isinstance(s.args[1], Value)
                and s.args[0].id == s.args[1].id):
            return 0
    elif op in ('or', 'max'):
        if (isinstance(s.args[0], Value) and isinstance(s.args[1], Value)
                and s.args[0].id == s.args[1].id):
            return 0
    return None


def _resolve(mapping: Dict[int, Value], v: Operand) -> Operand:
    """Follow a chain of forwardings to its end.

    Identities compose: ``mul(x, 1)`` forwards to ``x`` and a following
    ``add(that, 0)`` forwards to the *mul's* target, which no longer exists.
    ``substitute`` rewrites each use once, so the mapping has to be transitive
    before it is handed over -- otherwise the second forwarding lands on a
    deleted value and the IR fails to verify with "used before definition".
    """
    seen = set()
    while isinstance(v, Value) and v.id in mapping and v.id not in seen:
        seen.add(v.id)
        v = mapping[v.id]
    return v


def _fold_body(body: Tuple[Stmt, ...], consts: Dict[int, Any]):
    # a copy: constants defined in an enclosing scope are visible here, but
    # ones defined here must not leak out to a sibling region
    consts = dict(consts)
    mapping: Dict[int, Value] = {}
    out: List[Stmt] = []

    for s in body:
        if s.regions:
            regions = []
            for r in s.regions:
                inner, inner_map = _fold_body(r.body, consts)
                regions.append(replace(r, body=substitute(inner, inner_map)))
            out.append(replace(s, regions=tuple(regions)))
            continue

        if s.op == Op.CONST and s.target:
            consts[s.target[0].id] = s.attr('value')
            out.append(s)
            continue

        foldable = (s.pure and not s.has_side_effects
                    and len(s.target) == 1 and s.effect == Effect.NONE
                    and s.predicate is None
                    # an escaping value is named from raw text, so folding it
                    # away would leave that text referring to nothing
                    and not s.attr('escapes'))
        if not foldable:
            out.append(s)
            continue

        nums = [_as_number(a, consts) for a in s.args]

        # 1. every operand known -> evaluate
        fn = _FOLDABLE.get(s.op)
        if fn is not None and s.args and all(n is not None for n in nums):
            try:
                value = fn(*nums)
            except (ZeroDivisionError, ValueError, OverflowError):
                out.append(s)
                continue
            target = s.target[0]
            consts[target.id] = value
            out.append(replace(s, op=Op.CONST, args=(),
                               attrs=(('value', value),)))
            continue

        # 2. algebraic identity -> drop and forward the surviving operand
        keep = _identity(s, nums)
        if keep is not None and isinstance(s.args[keep], Value):
            mapping[s.target[0].id] = _resolve(mapping, s.args[keep])
            continue

        out.append(s)

    return tuple(out), mapping


# --------------------------------------------------------------------------- #
# Loop-invariant code motion
# --------------------------------------------------------------------------- #

def licm(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Hoist loop-invariant statements out of ``for`` loops, innermost first."""
    out: List[Stmt] = []
    for s in body:
        s = replace(s, regions=tuple(replace(r, body=licm(r.body))
                                     for r in s.regions))
        if s.op != Op.FOR:
            out.append(s)
            continue
        hoisted, rest = _split_invariant(s)
        out.extend(hoisted)
        out.append(replace(s, regions=(replace(s.regions[0], body=rest),)))
    return tuple(out)


def _loop_runs_at_least_once(s: Stmt) -> bool:
    lo, hi, _ = s.loop_bounds
    return isinstance(lo, int) and isinstance(hi, int) and lo < hi


def _split_invariant(loop: Stmt) -> Tuple[List[Stmt], Tuple[Stmt, ...]]:
    region = loop.regions[0]
    inner = region.body
    bound = {a.id for a in region.args}

    # everything the loop body may write to
    body_writes = tuple(a for a in collect_accesses(inner) if a.writes)
    body_effect = collect_effect(inner)

    hoisted: List[Stmt] = []
    kept: List[Stmt] = []
    defined_inside = set(bound)

    for s in inner:
        movable = (s.movable
                   and not (s.effect & (Effect.WRITE | Effect.ATOMIC |
                                        Effect.BARRIER | Effect.UNKNOWN))
                   and s.op not in (Op.YIELD,)
                   and not s.regions
                   and s.target)
        invariant = movable and all(v.id not in defined_inside for v in s.operands())

        if invariant and s.accesses:
            # a read may only leave the loop if nothing in it writes the same
            # base *and* the loop is known to execute at least once (otherwise
            # we would introduce a load that never happened before).
            if not _loop_runs_at_least_once(loop):
                invariant = False
            elif body_effect & Effect.UNKNOWN:
                invariant = False
            else:
                for a in s.accesses:
                    if any(accesses_conflict(a, w) for w in body_writes):
                        invariant = False
                        break

        if invariant:
            hoisted.append(s)
        else:
            kept.append(s)
            for t in s.target:
                defined_inside.add(t.id)

    return hoisted, tuple(kept)


# --------------------------------------------------------------------------- #
# Register pressure
# --------------------------------------------------------------------------- #

def _index(body: Tuple[Stmt, ...], start: int = 0):
    """Number every statement in pre-order and record each one's span.

    A statement that owns regions spans everything inside it, which is what
    makes a value used inside a loop live for the whole loop rather than only
    up to the one textual use.
    """
    order: List[Stmt] = []
    span: Dict[int, Tuple[int, int]] = {}

    def walk_(stmts):
        for st in stmts:
            i = len(order)
            order.append(st)
            for r in st.regions:
                walk_(r.body)
            span[i] = (i, len(order) - 1)

    walk_(body)
    return order, span


def pressure(body: Tuple[Stmt, ...]) -> int:
    """Peak number of simultaneously live SSA values.

    The bound every scheduling decision needs: hoisting a load away from its
    use, unrolling further, or deepening a software pipeline all buy latency
    with live values, and on CDNA the occupancy cliff arrives well before the
    latency win does.

    Loop-carried values are handled by extending a value's live range to the
    end of the outermost region that contains a use but not its definition ---
    a value read inside a loop is live across every iteration, not just at the
    one statement that mentions it.
    """
    order, span = _index(body)
    pos: Dict[int, int] = {}          # statement identity -> index
    for i, st in enumerate(order):
        pos[id(st)] = i

    define: Dict[int, int] = {}
    last: Dict[int, int] = {}
    owner: Dict[int, int] = {}        # value id -> index of its defining stmt

    for i, st in enumerate(order):
        for r in st.regions:
            for a in r.args:          # induction variable, iter_args
                define[a.id] = i
                owner[a.id] = i
                last[a.id] = span[i][1]
        for t in st.target:
            define[t.id] = i
            owner[t.id] = i
            last.setdefault(t.id, i)

    # extend each use to the end of the outermost region enclosing the use but
    # not the definition
    enclosing: Dict[int, List[int]] = {}
    def collect(stmts, chain):
        for st in stmts:
            i = pos[id(st)]
            enclosing[i] = chain
            for r in st.regions:
                collect(r.body, chain + [i])
    collect(body, [])

    for i, st in enumerate(order):
        for v in st.operands():
            d = define.get(v.id)
            if d is None:
                continue
            end = i
            for anc in enclosing.get(i, []):
                if anc >= d and not (span[anc][0] <= d <= span[anc][1]):
                    end = max(end, span[anc][1])
                elif span[anc][0] > d:
                    end = max(end, span[anc][1])
            last[v.id] = max(last.get(v.id, d), end)

    peak = 0
    for i in range(len(order)):
        live = sum(1 for vid, d in define.items() if d <= i <= last.get(vid, d))
        peak = max(peak, live)
    return peak


# --------------------------------------------------------------------------- #
# Scope flattening
# --------------------------------------------------------------------------- #

_CDECL = re.compile(r'\b(?:const\s+)?(?:float|double|int32_t|int|auto|bool|'
                    r'__float128|unsigned|char|short|long|\w+_t)\s+(\w+)\s*[=;\[]')


def _declares(body: Tuple[Stmt, ...]) -> bool:
    """Does any *direct* statement declare a C++ name in raw text?

    Structured values do not count: their names come from the shared
    allocator and are unique across the whole generated file.
    """
    for s in body:
        if s.text and _CDECL.search(s.text):
            return True
    return False


def flatten_scopes(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Splice away anonymous `{ }` regions that cannot cause a redeclaration.

    One of these used to wrap every instruction body, so that `value` or
    `data0` from one instruction would not collide with the next.  Once the
    names come from the shared allocator the braces are pure noise --- and
    expensive noise: an opaque block head makes the async scheduler give up its
    state, and nothing can be reordered across one.

    Conservative on purpose: a region that still declares a name in raw text
    keeps its braces.
    """
    out: List[Stmt] = []
    for s in body:
        s = replace(s, regions=tuple(replace(r, body=flatten_scopes(r.body))
                                     for r in s.regions))
        if (s.op == Op.RAWBLOCK and not s.text and not s.target
                and not s.attrs and len(s.regions) == 1
                and not s.regions[0].args
                and not _declares(s.regions[0].body)):
            out.extend(s.regions[0].body)
            continue
        out.append(s)
    return tuple(out)


# --------------------------------------------------------------------------- #
# If-conversion
# --------------------------------------------------------------------------- #

def _convertible(s: Stmt) -> bool:
    """A guard whose body may carry the predicate statement by statement."""
    if s.op != Op.IF or s.target or len(s.regions) != 1:
        return False
    if not isinstance(s.cond, Value):
        return False
    for inner in s.regions[0].body:
        if inner.regions or inner.predicate is not None:
            return False
        # A barrier must not be predicated at all, and an async operation's
        # completion counter would no longer match.
        if inner.effect & (Effect.BARRIER | Effect.ASYNC):
            return False
        # Raw text that declares a C++ name must keep the shared brace: giving
        # it its own `if` would scope the declaration inside it.
        if inner.text and _CDECL.search(inner.text):
            return False
    return True


def if_convert(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Push a guard's condition onto its statements and dissolve the region.

    One rule for everything inside: reads become selects and are free to move,
    pure arithmetic needs no predicate at all, and anything with a side effect
    keeps a real branch.  The guarded store therefore stays conditional --- and
    because it does, the value a masked-out lane loads never reaches memory,
    so the `other` value only has to be defined, not neutral.

    Not in the default pipeline: it trades one shared brace for one per
    side-effecting statement, which is only worth it once something actually
    uses the freedom it buys.
    """
    out: List[Stmt] = []
    for s in body:
        s = replace(s, regions=tuple(replace(r, body=if_convert(r.body))
                                     for r in s.regions))
        if _convertible(s):
            cond = s.cond
            for inner in s.regions[0].body:
                # Pure arithmetic needs no predicate: computing it for a
                # masked-out lane is harmless, and predicating it anyway
                # produces a nested select per operation.
                if inner.pure and not inner.has_side_effects and not inner.accesses:
                    out.append(inner)
                else:
                    out.append(replace(inner, predicate=cond))
            continue
        out.append(s)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Convenience pipeline
# --------------------------------------------------------------------------- #

def optimize(body: Tuple[Stmt, ...], dump_hook=None,
             diagnostics: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """The default pipeline.  ``dump_hook(name, body)`` sees every stage.

    ``fold`` runs first: it turns expressions into constants and removes
    identity operations, which gives ``cse`` more equal keys to merge and
    ``licm`` fewer statements to consider.  It runs a second time after
    ``licm``, because hoisting can bring two constants into the same scope.

    ``schedule_async`` runs last on purpose: the wait counts depend on the
    final issue order, so anything that may still move statements has to have
    happened already.
    """
    stages = (('flatten', flatten_scopes), ('fold', fold), ('cse', cse), ('licm', licm),
              ('fold2', fold), ('cse2', cse), ('dce', dce))
    for name, fn in stages:
        body = fn(body)
        if dump_hook is not None:
            dump_hook(name, body)
    body, diag = schedule_async(body)
    if diagnostics is not None:
        diagnostics.extend(diag)
    if dump_hook is not None:
        dump_hook('async', body)
    return body
