# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`_Avail` against the dict scan it replaces.

`load_cse` used to re-test every live entry against every access of every
statement.  `_Avail` indexes on `(space, base)` instead, which is the same
predicate spelled faster --- but only two of its five branches are reachable
from the corpus.  Nothing generated today carries `Effect.BARRIER` or
`Effect.ASYNC` into a PIR body, and the only access with `base is None` also
has `MemSpace.UNKNOWN`, so the barrier/wait kill and the wildcard-base kill
are covered by neither the snapshots nor the generated-source hashes.

So they are covered here, against a verbatim copy of the scan, over random
traces.  The gates that keep those branches unreachable are `vec == 1` in
`Symbol.load_linear` and the commented-out vector widths in
`memory/load.py`; both are on their way out.
"""
import collections
import random

import pytest

from tensorforge.backend.pir.core import Access, Effect, MemSpace, accesses_conflict
from tensorforge.backend.pir.passes import _Avail

SPACES = [MemSpace.REGISTER, MemSpace.SHARED, MemSpace.GLOBAL,
          MemSpace.CONSTANT, MemSpace.SCRATCH, MemSpace.UNKNOWN]
STORED_SPACES = [s for s in SPACES if s is not MemSpace.UNKNOWN]
WRITE_KINDS = [Effect.WRITE, Effect.ATOMIC, Effect.READ | Effect.WRITE,
               Effect.READ | Effect.ATOMIC]
EFFECTS = [Effect.NONE, Effect.READ, Effect.WRITE, Effect.ATOMIC,
           Effect.BARRIER, Effect.ASYNC, Effect.UNKNOWN,
           Effect.WRITE | Effect.BARRIER, Effect.READ | Effect.ASYNC,
           Effect.BARRIER | Effect.ASYNC | Effect.WRITE]


class Base:
    """A stand-in for a Symbol / alloc Value: identity is all that matters."""
    __slots__ = ('n',)
    def __init__(self, n): self.n = n
    def __repr__(self): return f'B{self.n}'


def reference_kill(available, accesses, effect):
    """Verbatim copy of the dict-scan `_kill` from dev3, 891c7b2."""
    if effect & (Effect.BARRIER | Effect.ASYNC):
        available = {k: v for k, v in available.items()
                     if all(a.space is MemSpace.REGISTER for a in v[1])}
    if not accesses:
        if effect & (Effect.WRITE | Effect.ATOMIC | Effect.UNKNOWN):
            return {}
        return available
    if not any(a.writes for a in accesses):
        return available
    return {k: v for k, v in available.items()
            if not any(accesses_conflict(w, a)
                       for w in accesses for a in v[1])}


def run(seed, n_bases=4, n_entries=12, n_steps=25):
    rnd = random.Random(seed)
    bases = [Base(i) for i in range(n_bases)]

    ref = {}
    idx = _Avail()
    for e in range(n_entries):
        # entries obey _reusable_load: pure reads, never UNKNOWN space
        accs = tuple(
            Access(Effect.READ, rnd.choice(STORED_SPACES),
                   rnd.choice(bases + [None]))
            for _ in range(rnd.randint(1, 3)))
        key = f'k{e}'
        ref[key] = ((f't{e}',), accs)
        idx.add(key, (f't{e}',), accs)

    for step in range(n_steps):
        if rnd.random() < 0.15:
            accesses = ()
        else:
            accesses = tuple(
                Access(rnd.choice(WRITE_KINDS + [Effect.READ]),
                       rnd.choice(SPACES),
                       rnd.choice(bases + [None]))
                for _ in range(rnd.randint(1, 3)))
        effect = rnd.choice(EFFECTS)

        ref = reference_kill(ref, accesses, effect)
        idx.kill(accesses, effect)

        if set(ref) != set(idx.entries):
            return (f'seed {seed} step {step}: '
                    f'ref={sorted(ref)} idx={sorted(idx.entries)} '
                    f'effect={effect!r} accesses={accesses}')
        # the index must also stay internally consistent, or a later kill lies
        for key in idx.entries:
            for a in idx.entries[key][1]:
                bkey = (a.space, None if a.base is None else id(a.base))
                if key not in idx._by_space.get(a.space, ()):
                    return f'seed {seed} step {step}: {key} missing from by_space'
                if key not in idx._by_base.get(bkey, ()):
                    return f'seed {seed} step {step}: {key} missing from by_base'
        stale = {k for b in idx._by_space.values() for k in b} - set(idx.entries)
        if stale:
            return f'seed {seed} step {step}: stale index keys {stale}'
        if idx._nonregister - set(idx.entries):
            return f'seed {seed} step {step}: stale _nonregister'
    return None



@pytest.mark.parametrize('lo', range(0, 2000, 200))
def test_index_agrees_with_the_scan(lo):
    for seed in range(lo, lo + 200):
        msg = run(seed)
        assert msg is None, msg


def test_every_branch_is_actually_reached():
    """A trace set that never reaches a branch proves nothing about it."""
    from tensorforge.backend.pir.passes import _Avail
    seen = collections.Counter()
    orig = _Avail.kill

    def kill(self, accesses, effect):
        if int(effect) & int(Effect.BARRIER | Effect.ASYNC):
            seen['sync'] += 1
        for w in accesses:
            if w.writes and w.space is MemSpace.UNKNOWN:
                seen['unknown'] += 1
            elif w.writes and w.base is None:
                seen['wildcard'] += 1
            elif w.writes:
                seen['exact'] += 1
        return orig(self, accesses, effect)

    _Avail.kill = kill
    try:
        for seed in range(200):
            run(seed)
    finally:
        _Avail.kill = orig

    for branch in ('sync', 'unknown', 'wildcard', 'exact'):
        assert seen[branch] > 0, branch + ' branch never reached'
