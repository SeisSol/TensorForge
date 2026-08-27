# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How much of the wave does each descr actually use?

`Generator._deduce_num_threads` takes the *maximum* over the descrs of a
kernel and gives that thread count to all of them.  For a single GEMM that is
exact.  For a fused chain whose leading dimension shrinks -- an ADER-DG
space-time predictor is the case that matters, 56 rows down to 1 over its
derivative recursion -- every descr after the first is paying for lanes it
does not use.

This counts that.  It reports, per case:

    used   sum of the leading dimensions, i.e. the elements there are
    slots  sum of `T * ceil(L / T)`, the register slots the kernel reserves:
           T lanes each holding `ceil(L / T)` entries of the dimension

and the same figure under a hypothetical per-descr thread count, so the
headroom is a number rather than an intuition.

Read the second column with care.  A smaller `T` for a narrow descr does not
by itself free anything: the launch geometry is

    block = (num_threads, mults_per_block, 1)

with the element index bound to `threadIdx.y`, so lanes given up on the x
axis idle rather than moving to another element -- the block owns
`mults_per_block` elements and no more.  Turning the headroom into occupancy
needs the block to own more elements than one pass processes, which is a
shared-memory decision, not a thread-count one.  The number here is the size
of the prize, not a patch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

CAP = 32  # Generator._deduce_num_threads caps here when no ElementwiseDescr


def slots(lead: int, threads: int) -> int:
    """Register slots reserved: `threads` lanes, `ceil(lead/threads)` each."""
    return threads * -(-lead // threads)


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def survey(arch: str = 'pvc', backend: str = 'acpp'):
    rows = []
    for path in sorted(Path('tests/cases').rglob('*.py')):
        try:
            case = load(path)
        except Exception:
            continue
        if not hasattr(case, 'descr_list'):
            continue
        try:
            ctx = Context(arch=arch, backend=backend,
                          fp_type=getattr(case, 'DTYPE', Datatype.F32))
            descrs = case.descr_list()
            want = [d.get_num_threads(ctx) for d in descrs]
        except Exception:
            continue
        if not want:
            continue

        leads = [lead for _, lead in want]
        kernel_t = min(CAP, max(t for t, _ in want))
        per_descr = [min(CAP, t) for t, _ in want]

        used = sum(leads)
        have = sum(slots(l, kernel_t) for l in leads)
        ideal = sum(slots(l, t) for l, t in zip(leads, per_descr))
        rows.append((getattr(case, 'NAME', path.stem), len(descrs), leads,
                     kernel_t, used, have, ideal))
    return rows


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else 'pvc'
    backend = sys.argv[2] if len(sys.argv) > 2 else 'acpp'
    rows = survey(arch, backend)

    print(f'{arch}/{backend}: lane utilisation per case\n')
    print(f'{"case":34s} {"n":>2s} {"T":>3s} {"used":>6s} {"slots":>6s} '
          f'{"util":>6s} {"per-descr":>10s}')
    tot_used = tot_have = tot_ideal = 0
    for name, n, leads, t, used, have, ideal in rows:
        tot_used += used
        tot_have += have
        tot_ideal += ideal
        flag = ' *' if ideal < have else ''
        print(f'{name[:34]:34s} {n:2d} {t:3d} {used:6d} {have:6d} '
              f'{100*used/have:5.0f}% {100*used/ideal:9.0f}%{flag}')
    print(f'\n{"corpus":34s} {"":2s} {"":3s} {tot_used:6d} {tot_have:6d} '
          f'{100*tot_used/tot_have:5.0f}% {100*tot_used/tot_ideal:9.0f}%')
    print('\n* = a per-descr thread count would reserve fewer slots here.')

    # The shapes the corpus does not have.  SeisSol's STP is the reason this
    # question is being asked at all, and no case in `tests/cases` has its
    # shrinking-chain shape -- so it is stated rather than measured, and
    # marked as such.
    print('\nSeisSol space-time predictor (modelled, not in the corpus):')
    print(f'{"order":>6s} {"leading dims":34s} {"util":>6s} {"per-descr":>10s}')
    for order in (4, 5, 6, 7):
        nb = lambda o: o * (o + 1) * (o + 2) // 6
        leads = [nb(order - d) for d in range(order)]
        t = min(CAP, max(_pow2(l) for l in leads))
        have = sum(slots(l, t) for l in leads)
        ideal = sum(slots(l, min(CAP, _pow2(l))) for l in leads)
        used = sum(leads)
        print(f'{order:6d} {str(leads)[:34]:34s} {100*used/have:5.0f}% '
              f'{100*used/ideal:9.0f}%')


def _pow2(lead: int) -> int:
    """`AbstractDescr.get_num_threads`, without needing a Context."""
    for t in (1, 2, 4, 8, 16, 32):
        if lead <= t:
            return t
    return 64


if __name__ == '__main__':
    main()
