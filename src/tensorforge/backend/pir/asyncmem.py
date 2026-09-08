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
loads.  Every wait is therefore annotated twice --- `prior` counts only units
of its own class, `prior_unified` counts every class --- and the emitter, which
knows the target, picks one.

On the copy class the counted unit is not the copy but the *group*, and a
group is closed by a `commit.async`.  That is why the commit is a statement
here rather than a line the emitter appends to each copy: `cp.async.commit_group`
and `__pipeline_commit` are **per thread**, so a lane that skipped a copy also
skipped its commit and has one group fewer in flight than its neighbour, while
the `wait` that counts them is one statement for all of them.  A lane-predicated
tail hop and a hop loop therefore both used to make the count describe one
lane's path and not another's.  Nothing went wrong only because every wait in
the corpus was a full drain, where the difference does not show.

`place_commits` puts the commit where the wait is: tokens accumulate per scope
and rise out of the regions that issued them, and the commit lands after the
last issuing statement *at the wait's own nesting*.  An empty group is legal
and retires at once, so a lane with nothing to commit still commits, and the
copy itself may stay under its predicate.

Memory operations the compiler emits on its own are not in our list, which can
only make N too small; a smaller N waits longer, so the result is pessimistic
rather than wrong.  Because the token is an ordinary SSA value, the pairing
is a def-use edge, and a double-buffered pipeline is just a `for` loop that
carries a token through its iter_args::

    %t0 = copy.async %lds[0] <- %glb[0]
    commit.async {%t0}
    %tn = for %k = 0 to N iter(%t = %t0) -> (token) {
            %t1 = copy.async %lds[(%k+1) % 2] <- %glb[%k+1]
            commit.async {%t1}
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
until at most N of the outstanding units are still in flight", counted in issue
order.  So for a wait on the unit at position `idx` of the outstanding list,
`N = len(outstanding) - idx - 1`, and positions `0..idx` retire.  The unit is
a group on the copy class and a single operation on the load class, which is
what the two counters count.

The direction of that arithmetic is worth stating, because it is the opposite
of the usual one: too *large* an N waits for fewer retirements and may return
before the unit the caller needs.  Under-counting merely waits longer.  So
where a figure is a bound rather than a count -- a group whose issues sit under
a predicate or inside a region holds at most as many operations as it names --
the per-operation total leaves it out instead of guessing high.  The group
count needs no such care: the commit runs whether its copies did or not.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, NamedTuple, Optional, Tuple

from .core import Effect, Op, Stmt, TokenType, Value, walk


class _Unit(NamedTuple):
    """One entry of the outstanding list: what a single wait step retires.

    On the load class that is one operation, so ``tokens`` holds one id.  On
    the copy class it is a group, and ``tokens`` holds every copy the commit
    closed.  ``exact`` says whether that set is also the operation *count*;
    see the module docstring for why a bound may not be counted as one.
    """

    tokens: frozenset
    cls: str
    exact: bool = True

    @property
    def ops(self) -> int:
        """Operations this unit is worth to a per-operation counter."""
        return len(self.tokens) if self.exact else 0


class _State:
    """Units in flight, in issue order."""

    __slots__ = ('outstanding', 'known')

    def __init__(self, outstanding=(), known: bool = True):
        self.outstanding: List[_Unit] = list(outstanding)
        self.known = known

    def copy(self) -> '_State':
        return _State(self.outstanding, self.known)

    def same(self, other: '_State') -> bool:
        return self.known == other.known and self.outstanding == other.outstanding

    def drain(self):
        self.outstanding.clear()
        self.known = True

    def index_of(self, tid: int) -> int:
        for i, u in enumerate(self.outstanding):
            if tid in u.tokens:
                return i
        return -1

    def __repr__(self):
        return f'_State({self.outstanding}, known={self.known})'


def _remap(state: _State, mapping: Dict[int, int]) -> None:
    """Rename outstanding tokens in place, keeping issue order."""
    state.outstanding = [
        u._replace(tokens=frozenset(mapping.get(t, t) for t in u.tokens))
        for u in state.outstanding]


def _is_token(v) -> bool:
    return isinstance(v, Value) and isinstance(v.type, TokenType)


def strip_commits(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Remove every `commit.async`.

    A commit records where a group closed in *this* schedule, so anything that
    moves statements invalidates it as surely as it invalidates `prior`.  Both
    are therefore derived rather than maintained, and both are derived by the
    same pass, which is why this is the first thing it does.
    """
    out: List[Stmt] = []
    for s in body:
        if s.op == Op.COMMIT_ASYNC:
            continue
        if s.regions:
            s = replace(s, regions=tuple(replace(r, body=strip_commits(r.body))
                                         for r in s.regions))
        out.append(s)
    return tuple(out)


def _commit(tokens: Tuple[int, ...], exact: bool) -> Stmt:
    return Stmt(op=Op.COMMIT_ASYNC, pure=False, movable=False,
                effect=Effect.ASYNC,
                attrs=(('tokens', tuple(tokens)), ('exact', exact),
                       ('counter', 'copy')))


_NEVER = 1 << 30


def place_commits(body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
    """Close each transfer into one group, at the wait's own nesting.

    Two rules, and between them they give every placement that matters without
    naming any of them as a case.

    *Which copies share a group*: the ones one wait retires.  Not "everything
    issued before the next wait" --- a body that issues two transfers and then
    waits for them in turn would put both in one group, and a wait cannot
    leave part of a group in flight.  Grouping by the wait is what keeps the
    second transfer available to stay outstanding once the waits are split.

    *Where the group closes*: behind its last issuing statement, in the scope
    that statement belongs to.  Tokens rise out of the regions that issued
    them, so a hop loop and a lane-predicated tail hop both contribute to the
    scope containing them and the commit lands there --- out of the loop, out
    of the predicate, and *into* the flag guard when the wait is in the guard
    too.  Behind the issue rather than in front of the wait, because groups
    have to close in the order they were issued and the waits need not appear
    in that order.
    """
    waits = _wait_index(body)
    out, leftover, exact = _place(body, waits)
    for _, toks in _by_wait(leftover, waits):
        out = _insert(out, toks, exact)
    return out


def _wait_index(body: Tuple[Stmt, ...]) -> Dict[int, int]:
    """Token id -> position of the wait that retires it, in walk order."""
    out: Dict[int, int] = {}
    for i, (s, _) in enumerate(walk(body)):
        if s.op != Op.WAIT:
            continue
        for a in s.args:
            if isinstance(a, Value):
                out.setdefault(a.id, i)

    # A carried token is retired by whatever waits the iteration argument it
    # becomes: the peeled transfer is the first iteration's `%t`, and what the
    # body yields is the next one's.  Neither is named by any wait, and
    # without this they would look alike in that -- which is enough to put
    # them in one group, closed once, after the loop that issues one of them
    # every iteration.
    for s, _ in walk(body):
        if s.op != Op.FOR:
            continue
        yielded = s.regions[0].yielded
        for j, arg in enumerate(s.iter_args):
            if not isinstance(arg.type, TokenType):
                continue
            at = out.get(arg.id)
            if at is None:
                continue
            for other in (s.loop_inits[j] if j < len(s.loop_inits) else None,
                          yielded[j] if j < len(yielded) else None):
                if _is_token(other):
                    out.setdefault(other.id, at)
    return out


def _by_wait(tokens, waits: Dict[int, int]):
    """Split tokens into groups by their wait, in the waits' own order."""
    groups: Dict[int, List[int]] = {}
    for t in tokens or ():
        groups.setdefault(waits.get(t, _NEVER), []).append(t)
    return [(k, groups[k]) for k in sorted(groups)]


def _place(body: Tuple[Stmt, ...], waits: Dict[int, int]):
    """Returns ``(body, uncommitted tokens, exact)`` for one scope.

    ``exact`` is False once anything in the open run was predicated or came
    out of a region, because then the token set bounds the operation count
    instead of being it.
    """
    out: List[Stmt] = []
    run: List[int] = []
    run_wait: Optional[int] = None
    run_at = -1                 # index in `out` of the run's last issue
    exact = True

    def flush():
        nonlocal out, run, run_wait, run_at, exact
        if run:
            at = run_at + 1
            out = out[:at] + [_commit(tuple(run), exact)] + out[at:]
        run, run_wait, run_at, exact = [], None, -1, True

    for s in body:
        issued: Tuple[int, ...] = ()
        from_region = False
        if s.regions:
            regions = []
            for r in s.regions:
                inner, up, up_exact = _place(r.body, waits)
                regions.append(replace(r, body=inner))
                if up:
                    issued += tuple(up)
                    from_region = True
                    exact = exact and up_exact
            s = replace(s, regions=tuple(regions))
        elif s.op == Op.COPY_ASYNC:
            issued = tuple(t.id for t in s.target)

        groups = _by_wait(issued, waits)
        # Both closures happen *before* the statement is appended, so that
        # `_insert` still finds the previous run's last issue behind it.
        if s.op == Op.WAIT:
            flush()
        elif groups and run and groups[0][0] != run_wait:
            flush()

        out.append(s)
        for i, (key, toks) in enumerate(groups):
            if i:
                flush()             # this statement issues for a second wait
            run.extend(toks)
            run_wait = key
            run_at = len(out) - 1
            exact = exact and not from_region and s.predicate is None

    # Whatever is left has no wait in this scope, so it rises to the one that
    # does.  `place_commits` commits the remainder at the top.
    return tuple(out), (run or None), exact


def schedule_async(body: Tuple[Stmt, ...]) -> Tuple[Tuple[Stmt, ...], List[str]]:
    """Place the commits, then annotate every `wait` with ``prior``.

    Returns ``(body, diagnostics)``.  Whenever the outstanding set cannot be
    determined statically the pass falls back to ``prior = 0`` (drain
    everything), which is always correct and merely slower.
    """
    diag: List[str] = []
    body = place_commits(strip_commits(body))
    state = _State()
    out = _sched(body, state, diag)
    for unit in state.outstanding:
        for tok in sorted(unit.tokens):
            diag.append(f'async operation %v{tok} is never waited for')
    return out, diag


def _sched(body: Tuple[Stmt, ...], state: _State,
           diag: List[str]) -> Tuple[Stmt, ...]:
    out: List[Stmt] = []

    for s in body:
        if s.op in Op.ASYNC:
            # A copy is not in flight *as a counted unit* until its group is
            # committed; a register load has no group and is one on its own.
            if s.counter != 'copy':
                state.outstanding.append(
                    _Unit(frozenset({s.target[0].id}), s.counter))
            out.append(s)
            continue

        if s.op == Op.COMMIT_ASYNC:
            state.outstanding.append(
                _Unit(frozenset(s.committed), 'copy', s.exact))
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

    cls = state.outstanding[idx].cls
    after = state.outstanding[idx + 1:]
    prior = sum(1 for u in after if u.cls == cls)
    unified = sum(u.ops for u in after)

    # Retire the unit and everything issued before it *in its own class*.  A
    # unified counter also retires the other class, but leaving those entries
    # in place is harmless: only entries *after* a unit are ever counted, so a
    # stale entry can never inflate a later N.
    state.outstanding = ([u for u in state.outstanding[:idx] if u.cls != cls]
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

def check_commits(body: Tuple[Stmt, ...]) -> List[str]:
    """A commit is never more conditional than the wait that counts it.

    This is the whole property, and it is worth checking rather than trusting,
    because breaking it produces code that runs: the count is derived from the
    statements, so it stays self-consistent while describing a path only some
    lanes take.  The corpus went a long time with a commit inside a hop loop
    and a commit inside a lane predicate, and nothing showed, because every
    wait was a full drain -- where the difference between one group and three
    does not change what the wait does.

    The test is on the nesting chains: the statements enclosing the commit
    have to be a prefix of those enclosing the wait.  A commit further out
    than its wait is fine, and after `wrap_prefetch` is the normal case ---
    the peel commits before the loop whose body waits.
    """
    diag: List[str] = []
    commits: List[Tuple[Stmt, Tuple[Stmt, ...]]] = []
    waits: List[Tuple[Stmt, Tuple[Stmt, ...]]] = []
    for s, parents in walk(body):
        if s.op == Op.COMMIT_ASYNC:
            commits.append((s, parents))
        elif s.op == Op.WAIT:
            waits.append((s, parents))

    for commit, cparents in commits:
        closed = set(commit.committed)
        for wait, wparents in waits:
            named = {a.id for a in wait.args if isinstance(a, Value)}
            if not (closed & named):
                continue
            if len(cparents) > len(wparents) or any(
                    c is not w for c, w in zip(cparents, wparents)):
                inner = (cparents[len(wparents):] or cparents)[0]
                diag.append(
                    f'commit.async: closed inside a `{inner.op}` that the '
                    f'wait counting it is not inside, so the number of groups '
                    f'in flight depends on which lanes took it')
    return diag


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
