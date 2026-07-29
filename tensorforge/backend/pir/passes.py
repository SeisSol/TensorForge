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

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import (Access, BufferType, Effect, IRError, MemSpace, Op, Operand,
                   Region, ScalarType, Stmt, TokenType, Value,
                   accesses_conflict, collect_accesses, collect_effect,
                   def_use, defined_within, walk)
from .asyncmem import check_tokens, schedule_async


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

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

    if diag and strict:
        raise IRError('pseudo-IR verification failed:\n  ' + '\n  '.join(diag))
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

        if s.pure and not s.has_side_effects and s.target and s.effect == Effect.NONE:
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
# Convenience pipeline
# --------------------------------------------------------------------------- #

def optimize(body: Tuple[Stmt, ...], dump_hook=None,
             diagnostics: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """The default pipeline.  ``dump_hook(name, body)`` sees every stage.

    ``schedule_async`` runs last on purpose: the wait counts depend on the
    final issue order, so anything that may still move statements has to have
    happened already.
    """
    stages = (('cse', cse), ('licm', licm), ('cse2', cse), ('dce', dce))
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
