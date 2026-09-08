# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What would it take for one transfer to still be in flight at the next wait?

Every `__pipeline_wait_prior` in the corpus is `(0)`.  The obvious reading is
that the waits sit too early and want sinking, and it is wrong: a wait is
already immediately before the read that needs it, two to twenty-two lines
after the commit that closed its group.  What is far away is the *next*
transfer, issued fifty-seven to seven-thousand lines later -- after the
compute that consumes this one.  There is no overlap to expose because there
is no overlap: the transfers are issued in sequence, one at a time.

So the question is what stops the next issue from moving up, and this counts
the two candidate answers separately, because they want opposite fixes.

*Order.*  Whether `may_cross` licenses the issue over everything between the
previous wait and itself.  Barriers and def-use edges show up here.

*Space.*  Whether the two destinations occupy the same arena bytes.  They
usually do, and this is the trap: the two are distinct `alloc` values, so
`may_alias` says they never alias and the IR-level check passes.  The
identity is made downstream, by `MemoryRegionAllocation`, out of a liveness
computed from the schedule the hoist is about to change.  Hoisting on the
strength of the IR answer alone writes the next operand into the buffer the
current one is still being read from -- values that are wrong without being
absent, on some launches.

The last column is therefore the price: bytes per work-item that keeping the
pair apart would add to the arena, against what the kernel uses today.

    python3 tools/overlap_census.py             # per-case table and summary
    python3 tools/overlap_census.py --summary
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from collections import Counter
from pathlib import Path

from tensorforge.backend import pir
from tensorforge.backend.pir.core import BufferType, Effect, MemSpace, Op
from tensorforge.backend.pir.schedule import may_cross, touches
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / 'tests' / 'cases'
TARGETS = [('sm_86', 'cuda')]


def _capture():
    """Collect every body that reaches the emitter, after `schedule_async`."""
    bodies = []
    original = pir.emit

    def hooked(body, writer, context=None):
        bodies.append(body)
        return original(body, writer, context)

    pir.emit = hooked
    return bodies, (lambda: setattr(pir, 'emit', original))


def _load(path):
    spec = importlib.util.spec_from_file_location('tf_ov__' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _arena(body):
    """alloc value id -> (arena, offset, bytes) for the shared buffers."""
    out = {}
    for s, _ in pir.walk(body):
        if s.op != Op.ALLOC or not s.target:
            continue
        v = s.target[0]
        t = v.type
        if not isinstance(t, BufferType) or t.space is not MemSpace.SHARED:
            continue
        n = 1
        for d in t.shape:
            n *= d
        out[v.id] = (s.attr('arena'), s.attr('offset', 0), n * t.elem.size())
    return out


def _groups(scope):
    """[(commit index, wait index, destination ids)] for one statement list."""
    out = []
    for i, s in enumerate(scope):
        if s.op != Op.COMMIT_ASYNC:
            continue
        closed = set(s.committed)
        dests = set()
        for j in range(i, -1, -1):
            for x, _ in pir.walk((scope[j],)):
                if x.op is Op.COPY_ASYNC and x.target[0].id in closed:
                    for a in x.accesses:
                        if a.writes and a.base is not None:
                            dests.add(a.base.id)
        w = next((k for k in range(i, len(scope))
                  if scope[k].op == Op.WAIT
                  and closed & {a.id for a in scope[k].args
                                if isinstance(a, pir.Value)}), None)
        out.append((i, w, dests))
    return out


def _first_issue(scope, commit_at):
    for i in range(commit_at, -1, -1):
        if any(x.op is Op.COPY_ASYNC for x, _ in pir.walk((scope[i],))):
            first = i
            while first and any(x.op is Op.COPY_ASYNC
                                for x, _ in pir.walk((scope[first - 1],))):
                first -= 1
            return first
    return commit_at


def _scopes(body):
    """Every statement list that holds a commit, outermost first."""
    yield tuple(body)
    for s, _ in pir.walk(body):
        for r in s.regions:
            if any(x.op == Op.COMMIT_ASYNC for x in r.body):
                yield tuple(r.body)


def _pairs(scope, arena):
    """One row per consecutive pair of groups in this scope."""
    rows = []
    groups = _groups(scope)
    for (ci, wi, di), (cj, wj, dj) in zip(groups, groups[1:]):
        if wi is None:
            continue
        first = _first_issue(scope, cj)
        between = scope[wi + 1:first]
        issue = [scope[k] for k in range(first, cj + 1)]
        blocker = next((s for s in between
                        if not all(may_cross(g, s) for g in issue)), None)
        rooms = {arena[d] for d in di if d in arena}
        roomj = {arena[d] for d in dj if d in arena}
        shared = any(a[0] == b[0] and _overlaps(a, b)
                     for a in rooms for b in roomj)
        cost = max((b[2] for b in roomj), default=0)
        rows.append((first - wi - 1, _name(blocker), shared, cost))
    return rows


def _name(blocker):
    """What stops the hoist, in one word.

    "no" is not a finding.  The barrier that shows up here is not an
    independent obstacle: `SyncThreadsOpt` puts it there because the next
    write lands in the bytes the previous reads are still taking from, which
    is the same fact the space column reports.  Separating the two removes
    the reason for the barrier along with the aliasing.
    """
    if blocker is None:
        return 'free'
    if blocker.effect & Effect.BARRIER:
        return 'barrier'
    if touches(blocker) is None:
        return 'opaque'
    return blocker.op


def _overlaps(a, b):
    _, oa, na = a
    _, ob, nb = b
    return oa < ob + nb and ob < oa + na


def _arena_bytes(body):
    total = 0
    for arena, offset, size in _arena(body).values():
        total = max(total, offset + size)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()

    tally = Counter()
    print(f'{"case":34s} {"grp":>3s} {"cover":>7s} {"blocker":>8s} '
          f'{"space":>6s} {"+B":>7s} {"arena":>7s}')
    for path in sorted(CASES.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        try:
            mod = _load(path)
        except Exception:
            continue
        if not hasattr(mod, 'NAME') or not hasattr(mod, 'descr_list'):
            continue
        for arch, backend in TARGETS:
            bodies, restore = _capture()
            try:
                ctx = Context(arch=arch, backend=backend,
                              fp_type=getattr(mod, 'DTYPE', None))
                with contextlib.redirect_stdout(io.StringIO()):
                    Generator(mod.descr_list(), ctx).generate()
            except Exception:
                continue
            finally:
                restore()
            for body in bodies:
                arena = _arena(body)
                rows = [r for scope in _scopes(body) for r in _pairs(scope, arena)]
                if not rows:
                    continue
                have = _arena_bytes(body)
                # One alternate region, not one per buffer: a transfer has to
                # miss the buffer the *previous* one is being read from, and
                # ping-ponging between two regions does that for a chain of
                # any length.  Summing every participant would price a copy
                # per transfer, which is not what overlap needs.
                extra = max((c for _, _, s, c in rows if s), default=0)
                tally['pairs'] += len(rows)
                tally['blocked by a barrier'] += sum(
                    1 for _, w, _, _ in rows if w == 'barrier')
                tally['space shared'] += sum(1 for _, _, s, _ in rows if s)
                tally['extra bytes'] += extra
                tally['arena bytes'] += have
                if not args.summary:
                    for cover, w, shared, cost in rows:
                        print(f'{mod.NAME[:34]:34s} {len(rows) + 1:3d} '
                              f'{cover:7d} {w:>8s} '
                              f'{"same" if shared else "apart":>6s} '
                              f'{cost:7d} {have:7d}')

    print()
    print(f'{tally["pairs"]} consecutive group pairs')
    print(f'{tally["blocked by a barrier"]} first blocked by a barrier')
    print(f'{tally["space shared"]} writing bytes the previous group is read from')
    if tally['arena bytes']:
        pct = 100.0 * tally['extra bytes'] / tally['arena bytes']
        print(f'{tally["extra bytes"]} B of {tally["arena bytes"]} B to '
              f'separate them  (+{pct:.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
