# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which regions does a kernel read that nothing wrote first?

Run it from SeisSol's `codegen/` with `generate.py`'s arguments.  Reads before
a write are not always defects --- a global output may be filled by the caller,
and yateto's eqspp contract makes anything outside a window zero --- but every
one is worth an explanation.
"""
import sys, os, runpy
sys.path.insert(0, os.getcwd())
from tensorforge.backend.instructions.builders import multilinear_builder as MB
from tensorforge.common.matrix.boundingbox import BoundingBox
import itertools

B = MB.MultilinearBuilder
findings = []
orig_plan = B.plan

def uncovered(read, boxes):
    """First sub-box of `read` no box in `boxes` covers, by coordinate compression."""
    rank = read.rank()
    if rank == 0 or not boxes or any(b.rank() != rank for b in boxes):
        return None
    cuts = []
    for j in range(rank):
        lo, hi = read.lower()[j], read.upper()[j]
        if lo >= hi:
            return None
        pts = {lo, hi}
        for b in boxes:
            for v in (b.lower()[j], b.upper()[j]):
                if lo < v < hi:
                    pts.add(v)
        cuts.append(sorted(pts))
    for corner in itertools.product(*[range(len(c) - 1) for c in cuts]):
        lo = [cuts[j][corner[j]] for j in range(rank)]
        hi = [cuts[j][corner[j] + 1] for j in range(rank)]
        if any(all(b.lower()[j] <= lo[j] and hi[j] <= b.upper()[j]
                   for j in range(rank)) for b in boxes):
            continue
        return BoundingBox(lo, hi)
    return None

def plan(self, descr_list):
    orig_plan(self, descr_list)
    written = {}          # tensor -> list of boxes written so far
    for n, descr in enumerate(descr_list):
        eff = self._effective_boxes(descr)
        if eff is None:
            continue
        reads, write = eff
        for t, box in reads.items():
            if t not in written:
                continue          # never written here: a genuine input
            gap = uncovered(box, written[t])
            if gap is not None:
                findings.append((getattr(t, 'alias', None) or str(t), n,
                                 str(box), str(gap),
                                 [str(b) for b in written[t]]))
        dest = getattr(descr, 'dest', None)
        t = getattr(dest, 'tensor', None) if dest is not None else None
        if t is not None:
            written.setdefault(t, []).append(write)
B.plan = plan

sys.argv = ['generate.py'] + sys.argv[1:]
try:
    runpy.run_path('generate.py', run_name='__main__')
except SystemExit:
    pass
print("\n==== read before anything wrote it ====")
seen = set()
for name, n, box, gap, w in findings:
    k = (name, box, gap)
    if k in seen: continue
    seen.add(k)
    print(f"  {name:20} descr #{n} reads {box}")
    print(f"  {'':20} uncovered {gap}")
    print(f"  {'':20} written so far {w[:8]}{' …' if len(w) > 8 else ''}")
print(f"total distinct: {len(seen)}")
