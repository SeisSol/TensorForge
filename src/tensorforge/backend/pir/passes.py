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

from .schedule import can_reorder
from .core import (Access, BufferType, Effect, IRError, MemSpace, Op, Operand,
                   Region, ScalarType, Stmt, TokenType, Value,
                   accesses_conflict, collect_accesses, collect_effect,
                   def_use, defined_within, walk, Uniformity)
from .asyncmem import check_tokens, schedule_async


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

_INTEGER_ONLY = frozenset({'bitand', 'bitor', 'bitxor', 'shl', 'shr', 'rem'})
_FLOAT_TYPES = frozenset({Datatype.F16, Datatype.F32, Datatype.F64})


def verify(body: Tuple[Stmt, ...], strict: bool = True) -> List[str]:
    """Structural, SSA, uniformity and buffer-bounds checks.

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

    _check_scope(body, set(), diag, reachable_at=Uniformity.GRID)
    diag.extend(_check_dangling_names(body))
    diag.extend(_check_buffer_bounds(body))

    if diag and strict:
        raise IRError('pseudo-IR verification failed:\n  ' + '\n  '.join(diag))
    return diag


def _check_buffer_bounds(body: Tuple[Stmt, ...]) -> List[str]:
    """Every access lands inside the buffer it names.

    A register array is a fixed number of registers, and an index one past
    either end is a neighbouring register or a spill slot -- a value that is
    wrong without being absent, so it survives every check that asks whether
    something was computed.  A host interpreter does not see it either: it
    keeps registers in a dict, serves index -1, and answers.

    Only what the IR can resolve is checked.  The index of an access built
    through `Symbol.build_address` is an expression tree over loop counters
    with stated bounds, which is enough to bound it; an index that arrives as
    raw text is opaque and passes silently.  Which accesses those are shifts
    as the migration proceeds, so this reports what it can prove wrong rather
    than claiming coverage it does not have.
    """
    sizes: Dict[str, int] = {}
    counters: Dict[int, Tuple[int, int]] = {}
    defs: Dict[int, Stmt] = {}

    for s, _ in walk(body):
        for t in s.target:
            defs[t.id] = s
        if s.op == Op.ALLOC and isinstance(getattr(s.target[0], 'type', None),
                                           BufferType):
            buffer = s.target[0].type
            if buffer.space == MemSpace.REGISTER and buffer.shape:
                count = 1
                for extent in buffer.shape:
                    count *= extent
                sizes[s.target[0].hint] = count
        if s.op == Op.FOR and s.regions and len(s.args) >= 3:
            start, stop, step = s.args[0], s.args[1], s.args[2]
            if all(isinstance(a, int) for a in (start, stop, step)) and step > 0:
                for counter in s.regions[0].args:
                    counters[counter.id] = (start, stop - 1)

    def bounds(x: Operand, depth: int = 0) -> Optional[Tuple[int, int]]:
        if isinstance(x, int):
            return (x, x)
        if not isinstance(x, Value) or depth > 24:
            return None
        if x.id in counters:
            return counters[x.id]
        s = defs.get(x.id)
        if s is None:
            return None
        if s.op == Op.CONST and s.args and isinstance(s.args[0], int):
            return (s.args[0], s.args[0])
        if s.op not in ('add', 'sub', 'mul') or len(s.args) != 2:
            return None
        left, right = bounds(s.args[0], depth + 1), bounds(s.args[1], depth + 1)
        if left is None or right is None:
            return None
        if s.op == 'add':
            return (left[0] + right[0], left[1] + right[1])
        if s.op == 'sub':
            return (left[0] - right[1], left[1] - right[0])
        corners = [a * b for a in left for b in right]
        return (min(corners), max(corners))

    diag: List[str] = []
    for s, _ in walk(body):
        if s.op == Op.LOAD:
            base, index = (s.args + (None, None))[0], (s.args + (None, None))[1]
        elif s.op == Op.STORE:
            base, index = (s.args + (None,) * 3)[0], (s.args + (None,) * 3)[2]
        else:
            continue
        # A buffer reaches an access either as the symbol it was declared
        # for or as the `alloc` result itself, and both spell the same
        # array.
        name = getattr(base, 'name', None) or getattr(base, 'hint', None)
        if name not in sizes or index is None:
            continue
        span = bounds(index)
        if span is None:
            continue
        low, high = span
        if low < 0 or high >= sizes[name]:
            diag.append(f'{s.op}: {name} holds {sizes[name]} elements and is '
                        f'addressed at {low}..{high}')
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
                 reachable_at: Uniformity) -> None:
    """``live`` is the set of value ids visible from enclosing scopes."""
    live = set(live)

    for i, s in enumerate(body):
        # -- operand dominance ------------------------------------------- #
        for v in s.operands():
            if v.id not in live:
                diag.append(f'{s.op}: operand {v!r} used before definition')

        # -- a wide access carries the alignment it needs -------------------- #
        #
        # `*(T4*)&buf[i]` is undefined unless that address is 16-byte aligned,
        # and nothing in the IR can decide whether it is: the index is an
        # expression, frequently a string, and the base's guarantee lives on
        # the frontend's tensor rather than on the buffer value.  So the claim
        # travels with the access and is checked for *sufficiency* here.
        #
        # Missing is the case worth catching.  A width chosen without anyone
        # asking about alignment is exactly what the parked `for g in [4, 2,
        # 1]` would have reintroduced, and it has no symptom until a tensor
        # happens not to be padded.
        if s.op in (Op.LOAD, Op.STORE):
            carrier = s.target[:1] if s.op == Op.LOAD else s.args[1:2]
            wide = [t for t in carrier
                    if isinstance(getattr(t, 'type', None), ScalarType)
                    and t.type.length is not None]
            if wide:
                need = wide[0].type.length * wide[0].type.base.size()
                claim = s.attr('align')
                if claim == 'relaxed':
                    # Spelled with an element-aligned type, so there is no
                    # alignment to prove: the compiler splits the access
                    # rather than emitting one the hardware needs aligned.
                    # The register side takes this route because a private
                    # address cannot be made contiguous at all.
                    pass
                elif claim is None:
                    diag.append(
                        f'{s.op}: {wide[0].type} access carries no alignment '
                        f'claim; a {need}-byte reinterpret cast needs one and '
                        f'the IR cannot derive it from the address')
                elif claim < need:
                    diag.append(
                        f'{s.op}: {wide[0].type} access needs {need}-byte '
                        f'alignment but only {claim} is claimed')

        # -- distribution agrees with uniformity --------------------------- #
        #
        # The other direction of the rule `Value.__post_init__` enforces.
        # There, a distributed layout narrows an over-broad uniformity, which
        # is always safe.  Here: a value the builder called lane-varying whose
        # layout says every lane holds the same thing.  That cannot be
        # narrowed away -- one of the two is simply wrong, and which one is
        # not decidable from here -- so it is reported rather than repaired.
        #
        # An untracked layout is not checked.  `None` means unknown, and
        # unknown disagrees with nothing.
        for t in s.target:
            if (t.layout is not None and not t.layout.is_distributed
                    and t.uniformity <= Uniformity.LANE):
                diag.append(
                    f'{s.op}: {t!r} is lane-varying but its layout says every '
                    f'lane holds the same element; one of the two is wrong')

        # -- predicate ----------------------------------------------------- #
        if s.predicate is not None:
            t = s.predicate.type
            if not (isinstance(t, ScalarType) and t.base.name == 'BOOL'):
                diag.append(f'{s.op}: predicate {s.predicate!r} is not bool')

        # -- terminator placement ------------------------------------------ #
        if s.op == Op.YIELD and i != len(body) - 1:
            diag.append('yield must be the last statement of a region')

        # -- a barrier may not out-reach the region it sits in -------------- #
        #
        # Legal iff every thread the barrier waits for actually gets here.  The
        # check used to be "is this region divergent at all", which forbade a
        # multiplication-wide barrier inside a multiplication-wide loop -- a
        # rendezvous of exactly the threads that agree on the trip count, and
        # therefore fine.
        if s.effect & Effect.BARRIER:
            scope = s.attr('scope')
            scope = scope if isinstance(scope, Uniformity) else Uniformity.BLOCK
            if reachable_at < scope:
                diag.append(
                    f'{s.op}: {scope.name.lower()}-wide barrier in a region '
                    f'only reached uniformly at {reachable_at.name.lower()} '
                    f'level; the threads that took another path never arrive '
                    f'(deadlock, not a wrong answer)')

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
            if s.attr('fmt') and '{0}' not in (s.text or ''):
                # `fmt` says the emitter fills the operands in; a text that
                # baked a value's *name* in instead survives every pass and
                # then loses its definition to inlining.
                diag.append('rawstmt: marked fmt but the text has no {0} '
                            'placeholder')

        elif s.op == Op.RAWBLOCK:
            if len(s.regions) != 1 or s.text is None:
                diag.append('rawblock: needs text and exactly one region')

        elif s.op == Op.DECLARE:
            if len(s.target) != 1 or s.args or s.regions or s.text is not None:
                diag.append('declare: needs exactly one target and nothing else')
            if s.predicate is not None:
                # There is no initialiser to turn into a select, and wrapping
                # the declaration in a guard would scope the value inside it.
                diag.append('declare: must not be predicated')
            if s.target and isinstance(s.target[0].type, BufferType):
                diag.append('declare: a buffer comes from alloc, not declare')

        elif s.op == Op.PACK:
            if len(s.target) != 1:
                diag.append('pack: produces exactly one value')
            elif isinstance(s.target[0].type, ScalarType):
                n = s.target[0].type.length
                if n is None:
                    diag.append('pack: the result must be a vector type')
                elif n != len(s.args):
                    diag.append(f'pack: {len(s.args)} parts for a {n}-vector')

        elif s.op == Op.EXTRACT:
            if len(s.target) != 1 or len(s.args) != 1:
                diag.append('extract: takes one vector and produces one value')
            elif isinstance(s.args[0], Value) and isinstance(s.args[0].type, ScalarType):
                n = s.args[0].type.length
                lane = s.attr('lane')
                if n is None:
                    diag.append('extract: the operand must be a vector')
                elif not isinstance(lane, int) or not 0 <= lane < n:
                    diag.append(f'extract: lane {lane} out of range for a '
                                f'{n}-vector')

        elif s.op == Op.ACCUM:
            if s.target or len(s.args) != 2:
                diag.append('accum: takes a target operand and a value, '
                            'and produces nothing')
            elif not isinstance(s.args[0], Value):
                diag.append('accum: the accumulated register must be a value')

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
            # Several tokens are allowed, and for a macro copy they are
            # required: `schedule_async` retires the waited token and
            # everything issued before it in its class, so one wait covers a
            # transfer split into hops -- but `check_tokens` runs before the
            # schedule exists and would call the earlier hops unconsumed.
            # Naming them reconciles the two without either guessing, and the
            # emitted code is unchanged because `prior` still comes from one
            # position.
            if any(not isinstance(a.type, TokenType) for a in s.args
                   if isinstance(a, Value)):
                diag.append('wait: every operand must be a completion token')
            if s.args and not isinstance(_typeof(s.args[0]), TokenType):
                diag.append('wait: argument must be a completion token')

        # -- recurse ------------------------------------------------------- #
        if s.regions:
            inner = min(reachable_at, _entry_uniformity(s))
            for r in s.regions:
                _check_scope(r.body, live | {a.id for a in r.args}, diag, inner)

        for t in s.target:
            live.add(t.id)


def _operand_uniformity(x) -> Uniformity:
    return x.uniformity if isinstance(x, Value) else Uniformity.GRID


def _entry_uniformity(s: Stmt) -> Uniformity:
    """Across how many threads is *entering* this statement's regions agreed?

    This replaces a boolean "is it divergent".  Two levels could not express a
    loop whose trip count is the same for every thread of one multiplication and
    different between the multiplications sharing a block -- which is what a
    batch loop is.  Such a loop is not divergent enough to forbid a
    multiplication-wide barrier, and far too divergent to allow a block-wide
    one; the boolean had to pick one answer and was wrong for the other.
    """
    if s.op == Op.IF:
        return _operand_uniformity(s.cond)
    if s.op == Op.FOR:
        return min((_operand_uniformity(b) for b in s.loop_bounds),
                   default=Uniformity.GRID)
    if s.op == Op.RAWBLOCK:
        # opaque head text: assume the worst
        return Uniformity.LANE
    return Uniformity.GRID


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
    # The result layout is part of the key, not a detail of the operands: two
    # statements can agree on op, text and arguments and still land their
    # results in different lanes -- a broadcast is exactly that.  Merging them
    # would silently drop the relayout.  Untracked (`None`) only ever matches
    # untracked, which is what every call site produces today.
    #
    # The result *type* is there for the same reason, and `Op.CONST` is why it
    # had to be: a constant carries its value in `attrs` and takes no
    # arguments, so `const(0, INDEX)` and `const(0, float)` had identical keys
    # and were merged.  Whichever survived decided the spelling for both, and
    # `Datatype.literal` spells them differently -- a float accumulator seeded
    # from an integer zero came out as `0_i32`.  For an operator whose neutral
    # element is 0 that still computes the right answer; the same merge on an
    # infinity would not.
    return (s.op, tuple(k(a) for a in s.args), s.text, s.attrs,
            tuple((t.layout, t.type) for t in s.target),
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
# Redundant load elimination
# --------------------------------------------------------------------------- #

def load_cse(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Reuse a load whose location provably has not been written since.

    `cse` hash-conses only statements with `Effect.NONE`, and rightly so: two
    pure operations on the same operands give the same result *by definition*,
    no analysis needed.  A load is not like that --- reading ``r0[3]`` twice
    yields the same value only if nothing wrote ``r0`` in between --- so it
    needs an availability analysis, which is what this pass adds.

    Until it exists, every `Access` the builder declares is inert: the model
    is there, but the gate in `cse` rejects anything with a memory effect
    before alias information is ever consulted, so declaring an access more
    precisely changes nothing.  This is the pass that consumes the model.

    Availability is killed by any conflicting write (`accesses_conflict`) and,
    for everything that is not thread-private, additionally by a barrier or an
    async wait: those publish *other* threads' writes, which a per-thread
    access model cannot see.  Loads whose name escapes into raw text are left
    alone --- deleting them would leave the text referring to a variable that
    is no longer declared.
    """
    body, mapping = _load_cse_body(body, _Avail())
    return substitute(body, mapping) if mapping else body


def _reusable_load(s: Stmt) -> bool:
    """A pure read with a fully declared location, and nothing else."""
    return (bool(s.target) and not s.regions and not s.attr('escapes')
            and s.effect == Effect.READ and bool(s.accesses)
            and all(a.kind == Effect.READ and a.space is not MemSpace.UNKNOWN
                    for a in s.accesses))


def _load_key(s: Stmt):
    """`_cse_key` plus the buffer identity the text may not distinguish."""
    return (_cse_key(s),
            tuple((a.space, None if a.base is None else id(a.base))
                  for a in s.accesses))


_EMPTY: frozenset = frozenset()
_M_SYNC = int(Effect.BARRIER | Effect.ASYNC)
_M_CLOBBER = int(Effect.WRITE | Effect.ATOMIC | Effect.UNKNOWN)


class _Avail:
    """Availability map for `load_cse`, indexed by what a write can kill.

    The plain dict version re-tested every live entry against every access of
    every statement, so one kill cost O(|available|) and a body cost
    O(n^2) --- on a fully unrolled 56x56x56 GEMM that is ~14k statements
    against ~2.3k live loads, i.e. millions of `accesses_conflict` calls.

    `may_alias` only ever says yes within one address space, and within a
    space only for the same base or an unknown one, so the entries a write can
    reach are known from `(space, base)` alone.  Indexing on that pair turns
    the scan into a couple of set lookups and makes the pass linear in
    practice.  The kill set is exactly the one the scan produced --- this is a
    faster spelling of the same predicate, not a weaker one.
    """

    __slots__ = ('entries', '_by_space', '_by_base', '_nonregister')

    def __init__(self, other: Optional['_Avail'] = None):
        if other is None:
            self.entries: Dict[Any, Any] = {}
            self._by_space: Dict[Any, set] = {}
            self._by_base: Dict[Any, set] = {}
            self._nonregister: set = set()
        else:
            self.entries = dict(other.entries)
            self._by_space = {k: set(v) for k, v in other._by_space.items()}
            self._by_base = {k: set(v) for k, v in other._by_base.items()}
            self._nonregister = set(other._nonregister)

    def get(self, key):
        return self.entries.get(key)

    def add(self, key, target, accesses: Tuple[Access, ...]) -> None:
        # The index is only equivalent to the scan because of what
        # `_reusable_load` admits: a stored access is a pure read with a known
        # space.  Relaxing that gate without revisiting `kill` would make this
        # silently unsound rather than merely imprecise, so pin it here.
        assert all(a.kind == Effect.READ and a.space is not MemSpace.UNKNOWN
                   for a in accesses), \
            'availability entry outside what _reusable_load admits'
        if key in self.entries:
            self._unindex(key)
        self.entries[key] = (target, accesses)
        for a in accesses:
            self._by_space.setdefault(a.space, set()).add(key)
            self._by_base.setdefault(
                (a.space, None if a.base is None else id(a.base)),
                set()).add(key)
            if a.space is not MemSpace.REGISTER:
                self._nonregister.add(key)

    def _unindex(self, key) -> None:
        _, accesses = self.entries[key]
        for a in accesses:
            bucket = self._by_space.get(a.space)
            if bucket is not None:
                bucket.discard(key)
            bucket = self._by_base.get(
                (a.space, None if a.base is None else id(a.base)))
            if bucket is not None:
                bucket.discard(key)
        self._nonregister.discard(key)

    def _drop(self, keys) -> None:
        for key in keys:
            if key in self.entries:
                self._unindex(key)
                del self.entries[key]

    def clear(self) -> None:
        self.entries.clear()
        self._by_space.clear()
        self._by_base.clear()
        self._nonregister.clear()

    def kill(self, accesses: Tuple[Access, ...], effect: Effect) -> None:
        eff = int(effect)
        if eff & _M_SYNC:
            # only thread-private state survives a barrier or a wait
            self._drop(list(self._nonregister))
        if not accesses:
            # an undeclared side effect has to be assumed to hit everything
            if eff & _M_CLOBBER:
                self.clear()
            return
        if not self.entries:
            return
        victims = set()
        for w in accesses:
            if not w.writes:
                continue
            if w.space is MemSpace.UNKNOWN:
                self.clear()
                return
            # a stored entry never has UNKNOWN space and never writes, so
            # `accesses_conflict(w, a)` reduces to "same space, and one of the
            # two bases is unknown or they are the same base".
            if w.base is None:
                victims.update(self._by_space.get(w.space, _EMPTY))
            else:
                victims.update(self._by_base.get((w.space, None), _EMPTY))
                victims.update(
                    self._by_base.get((w.space, id(w.base)), _EMPTY))
        if victims:
            self._drop(victims)


def _load_cse_body(body: Tuple[Stmt, ...], available: '_Avail'):
    available = _Avail(available)
    mapping: Dict[int, Value] = {}
    out: List[Stmt] = []

    for s in body:
        if s.regions:
            # A loop body runs again, so its own writes have to kill *before*
            # the descent as well as after it; doing both with one kill keeps
            # the fixed point trivial.
            inner = tuple(a for r in s.regions
                          for a in collect_accesses(r.body))
            eff = int(s.effect)
            for r in s.regions:
                eff |= int(collect_effect(r.body))
            available.kill(inner + s.accesses, Effect(eff))
            regions = []
            for r in s.regions:
                sub, sub_map = _load_cse_body(r.body, available)
                regions.append(replace(r, body=substitute(sub, sub_map)))
            out.append(replace(s, regions=tuple(regions)))
            continue

        if _reusable_load(s):
            key = _load_key(s)
            prev = available.get(key)
            if prev is not None and len(prev[0]) == len(s.target):
                for old, new in zip(s.target, prev[0]):
                    mapping[old.id] = new
                continue
            available.add(key, s.target, s.accesses)
            out.append(s)
            continue

        available.kill(s.accesses, s.effect)
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


def pressure(body: Tuple[Stmt, ...], in_bytes: bool = False,
             explicit_simd: bool = True) -> int:
    """Peak simultaneously live SSA values, or the bytes they occupy.

    `in_bytes` is what a caller comparing against a register budget has to
    ask.  A count treats `simd<float, 16>` and `float` as one apiece, and it
    cannot see register arrays at all -- which is the larger blind spot: the
    SSA values of `lead_window_spans_two_blocks` peak at 540 bytes while its
    `float r0[4992]` is 19 KB.

    `explicit_simd` says whose registers are being counted, and the byte form
    is wrong on the wrong setting rather than merely imprecise.  See
    `register_bytes`.

    The bound every scheduling decision needs: hoisting a load away from its
    use, unrolling further, or deepening a software pipeline all buy latency
    with live values, and on CDNA the occupancy cliff arrives well before the
    latency win does.

    Loop-carried values are handled by extending a value's live range to the
    end of the outermost region that contains a use but not its definition ---
    a value read inside a loop is live across every iteration, not just at the
    one statement that mentions it.
    """
    values = _value_index(body) if in_bytes else {}
    weight = ((lambda vid: register_bytes(values[vid], explicit_simd)
               if vid in values else 0)
              if in_bytes else (lambda vid: 1))
    # Register arrays are live for the whole body, so they are a constant
    # added to every point rather than something the sweep can see rise and
    # fall.  Added once here instead of extending their live ranges, which
    # would say the same thing less clearly.
    floor = (sum(register_bytes(v, explicit_simd) for v in values.values()
                 if isinstance(v.type, BufferType)) if in_bytes else 0)
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

    v_none = Value(id=-1, type=ScalarType(Datatype.I32))
    peak = 0
    for i in range(len(order)):
        live = sum(weight(vid) for vid, d in define.items()
                   if d <= i <= last.get(vid, d)
                   and not isinstance(values.get(vid, v_none).type, BufferType))
        peak = max(peak, live)
    return peak + floor


def _value_index(body: Tuple[Stmt, ...]) -> Dict[int, Value]:
    """Every value the body defines, by id -- targets and region arguments."""
    out: Dict[int, Value] = {}
    for st, _ in walk(body):
        for r in st.regions:
            for a in r.args:
                out[a.id] = a
        for t in st.target:
            out[t.id] = t
    return out


def register_bytes(v: Value, explicit_simd: bool = True) -> int:
    """How much register file one live value occupies.

    Not the same question as "how many values are live", and the difference is
    the whole reason this exists.  Under SPMD a value is one register per
    thread and counting values is counting registers; under an explicit vector
    the same value is `lanes * slots` elements in *this* work-item's registers,
    so a count understates it by the wave width -- sixteen-fold on PVC, and in
    the direction that matters.

    Both axes multiply.  `ScalarType.length` is the slot axis (how many
    consecutive elements one lane holds) and the layout is the lane axis; they
    are kept apart everywhere else for good reasons and they are still
    different things here, they just both make the register bigger.

    An untracked layout counts as one lane.  That is a floor and not an
    estimate: it is what SPMD would need, so the number never *over*states,
    and a caller comparing against a budget stays on the safe side of a value
    it cannot size.

    `explicit_simd` is which of the two is being asked, and it has to be
    asked. Under SPMD the lane axis is *threads*, so multiplying by it gives
    the wave's total register file rather than one lane's, and the two halves
    of the sum end up in different units -- a register array is per lane on
    that path and a scalar would be per wave.

    That is not academic. Doubling the lane count halves every array, and
    under the wave-total reading it also doubles what each live value costs --
    so that term scales as `lanes * count(lanes)`: flat where the count halves
    with the lanes, doubled where it does not. The only movement left is the
    array's, and the array is the smaller term. Over the cases where the lane
    ceiling binds on gfx90a the median ratio at 64 lanes against 32 is 1.03
    for the wave total and 0.57 per lane -- and 0.57 is what the register
    slots do, which is what the figure is meant to track. A search minimising
    the wave total would keep 32 lanes precisely where 64 is what relieves the
    pressure.
    """
    t = v.type
    if isinstance(t, BufferType):
        # A register-space allocation *is* register file, and on the explicitly
        # vectorised path it is the biggest thing in it by a wide margin: the
        # SSA values of `lead_window_spans_two_blocks` peak at 540 bytes while
        # its `float r0[4992]` is 19 KB.  Counting only the values would have
        # reported that kernel as comfortable.
        #
        # Its whole volume, and for the whole body: an array is live from its
        # declaration to the end of the scope, so there is no live range to
        # narrow.  Every other space is memory and costs no registers.
        return (t.volume * t.elem.size()
                if t.space is MemSpace.REGISTER else 0)
    if not isinstance(t, ScalarType):
        return 0                      # tokens name an event, not storage
    lanes = 1
    if explicit_simd and v.layout is not None:
        for axis in v.layout.axes:
            lanes *= axis.block
    return lanes * (t.length or 1) * t.base.size()


# --------------------------------------------------------------------------- #
# Scope flattening
# --------------------------------------------------------------------------- #

#: A raw statement that introduces a C++ name.  What may follow the name is
#: the whole question: `=` for an initialiser, `;` for a bare declaration, `[`
#: for an array --- and also `{` for brace initialisation and `,` for a second
#: declarator, both of which this missed.  10% of the declarations the corpus
#: emits took one of those two forms, and every one of them was invisible to
#: `flatten_scopes`, which then spliced away braces that were the only thing
#: keeping two declarations of one name apart.
#:
#: Nothing had gone wrong yet because the misses were masked: the accumulator
#: array `float v58[4][2]{};` matches on `[`, so the scope holding it was kept,
#: and the `float v58_0{};` beside it survived by association.  Making the
#: array a structured value removed the match and the braces went with it ---
#: six declarations of one name, at one level, in a kernel that had compiled
#: the day before.
#:
#: Over-matching here costs a pair of braces that could have been removed.
#: Under-matching costs a kernel that does not compile, so the character class
#: errs wide.
_CDECL = re.compile(r'\b(?:const\s+)?(?:float|double|int32_t|int|auto|bool|'
                    r'__float128|unsigned|char|short|long|\w+_t)\s+(\w+)\s*[=;\[{,]')


def _declares(body: Tuple[Stmt, ...]) -> bool:
    """Does any *direct* statement declare a C++ name in raw text?

    Structured values do not count: their names come from the shared
    allocator and are unique across the whole generated file.

    Nor does a *block head*.  A `rawblock`'s text is `for (int32_t i = 0; ...)`
    or `if (...)`, and a name introduced there is scoped to the block it opens
    -- it cannot collide with anything in the enclosing scope, which is the
    only thing this predicate exists to prevent.  Reading the head as a
    declaration kept 38 anonymous scopes alive across the corpus, every one of
    them around an unrolled hop loop whose `i` was never visible outside it.
    """
    for s in body:
        if s.regions:
            continue
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


def _sinkable_loop(s: Stmt) -> Optional[Stmt]:
    """`if (m) { for (...) { ... } }`, where `m` may move inside the loop.

    A mask is not control flow.  Moving a guard into a loop leaves the trip
    count alone and suppresses only what the body *writes*, which is the same
    program -- so the shape `_convertible` refuses (a region inside the guard)
    is convertible after all, one level down.

    Three preconditions, and each is a case where it would not be:

    * The loop carries no values.  A masked-out lane's accumulator would have
      to keep its previous value across the back edge, which is a `merge` and
      not a predicate; predicating the update alone would leave the
      accumulator undefined for that lane.
    * The bounds are lane-uniform.  Sinking says the trip count is the same
      whether or not the guard holds; a bound derived from the mask makes that
      false.
    * Only the loop is in the guard.  With a second statement beside it the
      predicate would have to be applied twice over, in two different ways.

    Returns the loop, or None.
    """
    if s.op != Op.IF or s.target or len(s.regions) != 1:
        return None
    if not isinstance(s.cond, Value):
        return None
    inner = s.regions[0].body
    if len(inner) != 1 or inner[0].op != Op.FOR:
        return None
    loop = inner[0]
    if loop.target or len(loop.regions) != 1:
        return None
    for b in loop.loop_bounds:
        if isinstance(b, Value) and b.layout is not None and b.distributed:
            return None
    if loop.regions[0].args and len(loop.regions[0].args) > 1:
        # induction variable only; anything more is an iteration argument
        return None
    return loop


def if_convert(body: Tuple[Stmt, ...],
               sink_into_loops: bool = False) -> Tuple[Stmt, ...]:
    """Push a guard's condition onto its statements and dissolve the region.

    One rule for everything inside: reads become selects and are free to move,
    pure arithmetic needs no predicate at all, and anything with a side effect
    keeps a real branch.  The guarded store therefore stays conditional --- and
    because it does, the value a masked-out lane loads never reaches memory,
    so the `other` value only has to be defined, not neutral.

    Not in the default pipeline: it trades one shared brace for one per
    side-effecting statement, which is only worth it once something actually
    uses the freedom it buys.

    ``sink_into_loops`` additionally moves a guard *through* a loop; see
    :func:`_sinkable_loop`.  Off by default because for a real branch it is a
    pessimisation -- the loop runs its full trip count instead of being
    skipped.  Where the guard is a lane mask there is no branch to skip with,
    so it is the only lowering rather than a trade.
    """
    out: List[Stmt] = []
    for s in body:
        s = replace(s, regions=tuple(
            replace(r, body=if_convert(r.body, sink_into_loops))
            for r in s.regions))
        if sink_into_loops:
            loop = _sinkable_loop(s)
            if loop is not None:
                region = loop.regions[0]
                # The same transformation, one level down: wrap the loop's
                # body in the guard that was outside the loop and convert
                # that.  Reusing the machinery rather than repeating it is
                # what keeps the two from growing different rules about what
                # may be predicated.
                guarded = Stmt(op=Op.IF, args=(s.cond,),
                               regions=(Region(body=region.body),))
                inner = if_convert((guarded,), sink_into_loops)
                out.append(replace(loop, regions=(replace(region, body=inner),)))
                continue
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
# Load clustering
# --------------------------------------------------------------------------- #

def _reads_only(s: Stmt) -> bool:
    return bool(s.effect & Effect.READ) and not (
        s.effect & (Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN))


def _can_swap(earlier: Stmt, later: Stmt) -> bool:
    """May `later` move in front of `earlier`?

    `schedule.can_reorder` answers the same question, and having two spellings
    of it is how they grow apart: this one tested `later.movable` but not
    `earlier.movable`, and `earlier`'s effect but not `later`'s, so a swap that
    dragged a pinned statement *down* past a movable one was licensed.  Nothing
    in the corpus hit it -- `cluster_loads` hoists reads, and a read that is
    blocked by an allocation is blocked by def-use first -- but it was licensed
    by the predicate rather than by the caller.

    Kept as a name because `cluster_loads` reads better for it, and because a
    later caller may want a directional variant; if one does, it should say so
    here rather than by omitting a check.
    """
    return can_reorder(earlier, later)


def cluster_loads(body: Tuple[Stmt, ...], max_pressure: int = 32,
                  window: int = 64) -> Tuple[Stmt, ...]:
    """Move memory reads earlier, bounded by register pressure.

    The generated bodies put every load immediately in front of its use, so the
    memory pipeline never sees more than one or two requests in flight.  This
    hoists independent reads together, which is the source-level version of the
    prefetch idiom --- and it is the first pass that has to be *stopped* by
    something, because every position gained costs a live value.

    Not in the default pipeline: whether it pays depends on what the vendor
    compiler already does with the same block, and that needs measurement on
    hardware rather than an argument.
    """
    out: List[Stmt] = [replace(s, regions=tuple(
        replace(r, body=cluster_loads(r.body, max_pressure, window))
        for r in s.regions)) for s in body]
    original = list(out)
    moved: List[Tuple[int, int]] = []

    for i in range(len(out)):
        s = out[i]
        if not _reads_only(s) or s.regions:
            continue
        j = i
        while j > 0 and i - j < window and _can_swap(out[j - 1], s):
            j -= 1
        if j == i:
            continue
        moved.append((j, i))
        out = out[:j] + [s] + out[j:i] + out[i + 1:]

    # Pressure is checked once for the whole region rather than per hoist:
    # `pressure` is a full sweep, and calling it inside the loop made the pass
    # quadratic on bodies with thousands of statements.  If the region ends up
    # over budget the hoists are undone wholesale, which is coarse but keeps
    # the pass linear in practice.
    if moved and pressure(tuple(out)) > max_pressure:
        return tuple(original)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Convenience pipeline
# --------------------------------------------------------------------------- #

def optimize(body: Tuple[Stmt, ...], dump_hook=None,
             diagnostics: Optional[List[str]] = None,
             explicit_simd: bool = False) -> Tuple[Stmt, ...]:
    """The default pipeline.  ``dump_hook(name, body)`` sees every stage.

    ``fold`` runs first: it turns expressions into constants and removes
    identity operations, which gives ``cse`` more equal keys to merge and
    ``licm`` fewer statements to consider.  It runs a second time after
    ``licm``, because hoisting can bring two constants into the same scope.

    ``load_cse`` runs after ``cse`` and before ``licm``: it removes the loads
    that would otherwise be hoisting candidates, so ``licm`` sees fewer
    statements.  A second run after ``cse2`` was measured and removes nothing
    on the current corpus, so it is not in the pipeline.

    `schedule.hoist_issues` and `schedule.sink_waits` are deliberately *not*
    here.  Both are correct and both were measured on the corpus: between them
    they move nothing but comments, on 15 of 232 outputs, and the mean
    issue-to-wait distance goes 7.7 to 8.1 statements entirely through comments
    changing places.  The schedule the macro layer produces is already at the
    fixed point of those two greedy moves --- the wait sits immediately before
    the read that needs it, and the issue sits immediately after the pointer
    binding it reads.

    That is worth knowing rather than working around.  The distance that is
    still missing is not reachable by any local swap: more than half the
    transfers have five statements or fewer of cover, and getting more means
    moving an issue across the loop back edge, which is a different
    transformation with a distance parameter and a prologue.  When that pass
    exists it will run here, and `schedule_async` after it, since the wait
    counts describe the final issue order.

    ``schedule_async`` runs last on purpose: the wait counts depend on the
    final issue order, so anything that may still move statements has to have
    happened already.
    """
    stages = (('flatten', flatten_scopes), ('fold', fold), ('cse', cse),
              ('loads', load_cse), ('licm', licm),
              ('fold2', fold), ('cse2', cse), ('dce', dce))
    if explicit_simd:
        # Not an optimisation here.  A guard over a lane-varying condition is
        # a mask in this model and there is no branch to lower it to, so the
        # conversion is the only legal path rather than a trade of one shared
        # brace for several.  It runs before `fold`, so that the predicates it
        # attaches take part in the same simplification as everything else.
        stages = (stages[0],
                  ('if_convert',
                   lambda b: if_convert(b, sink_into_loops=True))) + stages[1:]
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
