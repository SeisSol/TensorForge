# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: asynchronous memory pipelines.

`copy.async` moves global -> shared, `load.async` moves global -> register;
both produce a completion token that a `wait` consumes.  For `load.async` the
*wait* is what produces the loaded value, so a use before the wait is not
merely diagnosed --- it is unrepresentable.

Counters: AMD tracks global->LDS and global->VGPR in one `vmcnt`, while NVIDIA
has a group counter for `cp.async` and pure hardware scoreboarding for register
loads.  Every wait is therefore annotated twice --- `prior` counts only tokens
of its own class, `prior_unified` counts every class --- and the emitter, which
knows the target, picks one.

Memory operations the compiler emits on its own are not in our list, which can
only make N too small; a smaller N waits longer, so the result is pessimistic
rather than wrong.  Because the token is an ordinary SSA value, the pairing
is a def-use edge, and a double-buffered pipeline is just a `for` loop that
carries a token through its iter_args::

    %t0 = copy.async %lds[0] <- %glb[0]
    %tn = for %k = 0 to N iter(%t = %t0) -> (token) {
            %t1 = copy.async %lds[(%k+1) % 2] <- %glb[%k+1]
            wait %t                       # prior = 1
            ... compute on %lds[%k % 2] ...
            yield %t1
          }
    wait %tn                              # prior = 0

Ordering is expressed entirely through the existing `Access` model: a
`copy.async` and its `wait` carry the *same* accesses, so no read of the
destination can be hoisted above the wait and no write to the source can sink
below it.  Nothing in the reorder machinery needs to know about asynchrony.

What this module does *not* check: whether a read of a copy's destination is
properly separated from the copy by a wait.  Aliasing here is at buffer
granularity, and a double-buffered loop reads ``lds[k % 2]`` while filling
``lds[(k+1) % 2]`` --- the same buffer.  Any check at this granularity would
fire on the one pattern that matters most, so visibility across the two halves
stays the caller's obligation.  What *is* checked: every token is consumed
exactly once, and no wait names a token that is not in flight.

This module derives the hardware wait count.  Both AMD's `s_waitcnt vmcnt(N)`
and NVIDIA's `cp.async.wait_group N` / `__pipeline_wait_prior(N)` mean "wait
until at most N of the outstanding operations are still in flight", counted in
issue order.  So for a wait on the token at position `idx` of the outstanding
list, `N = len(outstanding) - idx - 1`, and positions `0..idx` retire.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .core import Op, Stmt, TokenType, Value, walk


class _State:
    """Tokens in flight as ``(token id, counter class)``, in issue order."""

    __slots__ = ('outstanding', 'known')

    def __init__(self, outstanding=(), known: bool = True):
        self.outstanding: List[Tuple[int, str]] = list(outstanding)
        self.known = known

    def copy(self) -> '_State':
        return _State(self.outstanding, self.known)

    def same(self, other: '_State') -> bool:
        return self.known == other.known and self.outstanding == other.outstanding

    def drain(self):
        self.outstanding.clear()
        self.known = True

    def index_of(self, tid: int) -> int:
        for i, (t, _) in enumerate(self.outstanding):
            if t == tid:
                return i
        return -1

    def __repr__(self):
        return f'_State({self.outstanding}, known={self.known})'


def _remap(state: _State, mapping: Dict[int, int]) -> None:
    """Rename outstanding tokens in place, keeping issue order."""
    state.outstanding = [(mapping.get(t, t), c) for t, c in state.outstanding]


def _is_token(v) -> bool:
    return isinstance(v, Value) and isinstance(v.type, TokenType)


def schedule_async(body: Tuple[Stmt, ...]) -> Tuple[Tuple[Stmt, ...], List[str]]:
    """Annotate every `wait` with a ``prior`` attribute; report hazards.

    Returns ``(body, diagnostics)``.  Whenever the outstanding set cannot be
    determined statically the pass falls back to ``prior = 0`` (drain
    everything), which is always correct and merely slower.
    """
    diag: List[str] = []
    state = _State()
    out = _sched(body, state, diag)
    for tok, _ in state.outstanding:
        diag.append(f'async operation %v{tok} is never waited for')
    return out, diag


def _sched(body: Tuple[Stmt, ...], state: _State,
           diag: List[str]) -> Tuple[Stmt, ...]:
    out: List[Stmt] = []

    for s in body:
        if s.op in Op.ASYNC:
            state.outstanding.append((s.target[0].id, s.counter))
            out.append(s)
            continue

        if s.op == Op.WAIT:
            out.append(_sched_wait(s, state, diag))
            continue

        if s.op == Op.FOR:
            out.append(_sched_for(s, state, diag))
            continue

        if s.op == Op.IF:
            out.append(_sched_if(s, state, diag))
            continue

        if s.regions:                       # rawblock and friends: opaque
            regions = tuple(replace(r, body=_sched(r.body, state.copy(), diag))
                            for r in s.regions)
            if state.outstanding:
                state.known = False
            out.append(replace(s, regions=regions))
            continue

        out.append(s)

    return tuple(out)


def _sched_wait(s: Stmt, state: _State, diag: List[str]) -> Stmt:
    # A wait may name several tokens: the hops of one macro copy are retired
    # together, and naming them is how `check_tokens` sees that.  The count is
    # still derived from a single position -- the *latest* of them, since
    # retiring that one retires everything issued before it in its class.
    tok = None
    if s.args:
        best = -1
        for cand in s.args:
            if not _is_token(cand):
                continue
            i = state.index_of(cand.id) if state.known else -1
            if i > best:
                best, tok = i, cand
        if tok is None:
            tok = s.waited

    if tok is None:                         # explicit full drain
        state.drain()
        return (s.with_attr('prior', 0).with_attr('prior_unified', 0)
                 .with_attr('counter', 'all'))

    idx = state.index_of(tok.id) if state.known else -1
    if idx < 0:
        if state.known:
            diag.append(f'wait: %{tok} is not in flight at this point')
        state.drain()
        return (s.with_attr('prior', 0).with_attr('prior_unified', 0)
                 .with_attr('counter', 'all'))

    cls = state.outstanding[idx][1]
    after = state.outstanding[idx + 1:]
    prior = sum(1 for _, c in after if c == cls)
    unified = len(after)

    # Retire the token and everything issued before it *in its own class*.  A
    # unified counter also retires the other class, but leaving those entries
    # in place is harmless: only entries *after* a token are ever counted, so a
    # stale entry can never inflate a later N.
    state.outstanding = ([e for e in state.outstanding[:idx] if e[1] != cls]
                         + after)

    return (s.with_attr('prior', prior)
             .with_attr('prior_unified', unified)
             .with_attr('counter', cls))


def _sched_for(s: Stmt, state: _State, diag: List[str]) -> Stmt:
    region = s.regions[0]

    # The init tokens *are* the copies the iter_args stand for: rename in
    # place so the issue order carries into the body.
    entry = state.copy()
    rename: Dict[int, int] = {}
    for arg, init in zip(s.iter_args, s.loop_inits):
        if not isinstance(arg.type, TokenType):
            continue
        if _is_token(init) and entry.index_of(init.id) >= 0:
            rename[init.id] = arg.id
        else:
            entry.known = False
    _remap(entry, rename)

    trial = entry.copy()
    inner_diag: List[str] = []
    inner = _sched(region.body, trial, inner_diag)

    balanced = (trial.known and entry.known and
                len(trial.outstanding) == len(entry.outstanding))

    if not balanced:
        # Not a steady state (or opaque): redo conservatively, every wait in
        # the body drains completely.
        inner_diag = []
        trial = _State(known=False)
        inner = _sched(region.body, trial, inner_diag)
        if entry.outstanding:
            diag.append('for: async pipeline is not in steady state '
                        '(issues and waits do not balance per iteration); '
                        'falling back to full drains inside the loop')
    diag.extend(inner_diag)

    # Map the yielded tokens onto the loop results for the enclosing scope.
    exit_state = trial.copy()
    yielded = region.yielded
    back: Dict[int, int] = {}
    for res, y in zip(s.target, yielded):
        if isinstance(res.type, TokenType) and _is_token(y):
            back[y.id] = res.id
    _remap(exit_state, back)

    if not balanced:
        exit_state.known = False
    state.outstanding = exit_state.outstanding
    state.known = exit_state.known

    return replace(s, regions=(replace(region, body=inner),))


def _sched_if(s: Stmt, state: _State, diag: List[str]) -> Stmt:
    entry = state.copy()
    ends: List[_State] = []
    regions = []
    for r in s.regions:
        branch = entry.copy()
        regions.append(replace(r, body=_sched(r.body, branch, diag)))
        ends.append(branch)

    if len(s.regions) == 1:
        ends.append(entry.copy())           # the not-taken path

    merged = ends[0]
    if not all(e.same(merged) for e in ends[1:]):
        merged = _State(merged.outstanding, known=False)
    state.outstanding = merged.outstanding
    state.known = merged.known

    return replace(s, regions=tuple(regions))


# --------------------------------------------------------------------------- #
# Verification helpers used by passes.verify
# --------------------------------------------------------------------------- #

def check_tokens(body: Tuple[Stmt, ...], defs, uses) -> List[str]:
    """Tokens are single-use, and a wait releases what its issue promised."""
    diag: List[str] = []
    for s, _ in walk(body):
        for t in s.target:
            if not isinstance(t.type, TokenType):
                continue
            n = len(uses.get(t.id, ()))
            if n == 0:
                diag.append(f'{s.op}: token %{t} is never consumed')
            elif n > 1:
                diag.append(f'{s.op}: token %{t} is consumed {n} times '
                            f'(a completion token is single-use)')

        if s.op == Op.FOR:
            for arg, init in zip(s.iter_args, s.loop_inits):
                if not isinstance(arg.type, TokenType) or not _is_token(init):
                    continue
                producer = defs.get(init.id)
                if producer is not None and producer.attr('types', ()):
                    diag.append(
                        'for: a load.async token cannot be carried across the '
                        'back edge --- the prefetched value would need two '
                        'registers to ping-pong between iterations.  Unroll by '
                        'two, or use copy.async through shared memory.')
            continue

        if s.op != Op.WAIT or s.waited is None:
            continue
        producer = defs.get(s.waited.id)
        if producer is None or producer.op == Op.FOR:
            continue                        # carried token: checked at the loop
        expected = len(producer.attr('types', ()))
        if len(s.target) != expected:
            diag.append(f'wait: releases {len(s.target)} value(s) but '
                        f'{producer.op} promised {expected}')
    return diag
