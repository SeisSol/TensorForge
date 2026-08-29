# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which constant operands get staged, and in what layout.

The first piece of the cost model that decides rather than measures, and the
corpus is why it has to exist.  At order 8 the three stiffness matrices of the
volume kernel come to 63.4 KiB with every structural zero already squeezed
out, against 64 KiB of LDS per CU.  There is no encoding that makes all three
fit with room to spare, so the question is not how to compress each one but
which of them earns the space.

Two resources, kept apart because they are spent from different budgets:

``lds_bytes``
    What staging the operand costs in shared memory, which is the resource
    that sets occupancy.

``global_bytes``
    What the operand costs in traffic from global memory per block.  Staged,
    it is read once and then served from LDS.  Unstaged, it is read once per
    traversal, and ``traversals`` is a property of the kernel's loop
    structure, not a constant anyone has to guess.

Minimising traffic alone would answer every question with the tightest
encoding there is, because a wide access buys nothing it can see.  So the
objective is cycles, with traffic converted at a stated bandwidth and the
access count carried alongside it.  A window only pays if it holds more than
one non-zero, and whether it does is exactly what the corpus disagrees about
between one instance of a family and the next.

A window is not one instruction either: the widest single load on both
ISAs moves sixteen bytes, so a four-wide FP64 window is two accesses and an
eight-wide one is four.  Counting windows rather than accesses would make
wide windows look free and pick them everywhere.

Index metadata is reported but not optimised over.  It is lane-uniform and
goes to constant space, so it competes for a budget these operators come
nowhere near filling.

The choice is a multiple-choice knapsack and is solved exactly.  The operand
count is small and the budget quantises to a few hundred buckets, so there is
no reason to be greedy about it and then wonder whether the answer was the
answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from spp_metrics import PatternMetrics

#: LDS is quantised to this many bytes when planning.  Rounding *up* keeps a
#: plan feasible; the residue is at most this much per operand.
QUANTUM = 256

#: The widest single load either ISA issues, in bytes.  Two FP64 elements,
#: four FP32.
WIDEST_LOAD = 16


@dataclass(frozen=True)
class Placement:
    """One way to hold one operand."""

    operand: str
    label: str
    staged: bool
    lds_bytes: int
    global_bytes: int
    meta_bytes: int
    accesses: int
    traversals: int
    fill: float

    def cycles(self, bytes_per_cycle: float) -> float:
        """Traffic and issue in one currency.

        One memory instruction per cycle is the issue side; the traffic side
        is whatever the caller says the path from global memory delivers.
        Both are per block.
        """
        return (self.global_bytes / bytes_per_cycle
                + self.accesses * self.traversals)

    def __str__(self) -> str:
        where = 'lds' if self.staged else 'global'
        return (f'{self.operand}: {self.label} @{where} '
                f'{self.lds_bytes / 1024:.1f}K lds, '
                f'{self.global_bytes / 1024:.1f}K glb, '
                f'{self.accesses} acc, fill {self.fill:.2f}')


def placements(metrics: PatternMetrics, fp_bytes: int, traversals: int = 1,
               axis: int = 0) -> List[Placement]:
    """Every layout `metrics` was measured for, priced in both resources.

    The element-wise floor is always among them: it stores exactly the
    non-zeros and gives up the wide access, which for an operand whose
    non-zeros are isolated along the lead axis costs nothing it had.

    ``axis`` is the contiguous one, so a window along it can be widened and a
    tile's extent along any other axis cannot.
    """
    out: List[Placement] = []
    name = metrics.name or 'operand'
    per_load = max(1, WIDEST_LOAD // fp_bytes)

    def add(label: str, stored: int, meta: int, accesses: int, fill: float):
        size = stored * fp_bytes
        out.append(Placement(name, label, True, size, size, meta,
                             accesses, traversals, fill))
        out.append(Placement(name, label, False, 0, size * traversals, meta,
                             accesses, traversals, fill))

    lines = metrics.volume // metrics.shape[axis]
    add('dense', metrics.volume, 0,
        lines * math.ceil(metrics.shape[axis] / per_load), metrics.density)
    # 4 bytes of index per non-zero: the floor on storage, the ceiling on
    # index cost, and no access wider than one element.
    add('element', metrics.nnz, 4 * metrics.nnz, metrics.nnz, 1.0)
    for (a, width, mode), cover in metrics.covers.items():
        if a != axis:
            continue
        add(f'w{width}/{mode}', cover.stored, cover.metadata,
            cover.windows * math.ceil(width / per_load), cover.fill)
    for shape, block in metrics.blocks.items():
        # The tile is contiguous only along ``axis``; every other extent is a
        # separate address however small it is.
        strided = math.prod(e for d, e in enumerate(shape) if d != axis)
        per_tile = strided * math.ceil(shape[axis] / per_load)
        add('tile' + '×'.join(str(s) for s in shape), block.stored,
            4 * block.occupied, block.occupied * per_tile, block.fill)
    return out


@dataclass
class Plan:
    chosen: List[Placement]
    lds_bytes: int
    global_bytes: int
    meta_bytes: int
    cycles: float

    @property
    def evicted(self) -> List[Placement]:
        return [p for p in self.chosen if not p.staged]

    def report(self) -> str:
        lines = [f'lds {self.lds_bytes / 1024:.1f} KiB   '
                 f'global {self.global_bytes / 1024:.1f} KiB/Block   '
                 f'meta {self.meta_bytes} B   '
                 f'{self.cycles:.0f} cyc']
        for p in sorted(self.chosen, key=lambda q: -q.lds_bytes):
            lines.append('  ' + str(p))
        return '\n'.join(lines)


def plan(groups: Sequence[Sequence[Placement]], lds_budget: int,
         bytes_per_cycle: float = 64.0,
         quantum: int = QUANTUM) -> Optional[Plan]:
    """Fewest cycles among the layouts that fit ``lds_budget``.

    ``groups`` is one list of candidates per operand; exactly one is taken
    from each.  Returns ``None`` when no combination fits, which for a budget
    that cannot even hold the unstaged choice cannot happen -- an operand
    always has a zero-LDS candidate -- but a caller that filters the
    candidates can still get there.
    """
    buckets = lds_budget // quantum
    # cost[b] is the least cycle count reachable using at most b buckets.
    inf = float('inf')
    cost: List[float] = [inf] * (buckets + 1)
    back: List[List[Optional[Tuple[int, Placement]]]] = []
    cost[0] = 0.0
    for group in groups:
        if not group:
            return None
        nxt: List[float] = [inf] * (buckets + 1)
        step: List[Optional[Tuple[int, Placement]]] = [None] * (buckets + 1)
        for cand in group:
            need = math.ceil(cand.lds_bytes / quantum)
            for b in range(need, buckets + 1):
                if cost[b - need] == inf:
                    continue
                total = cost[b - need] + cand.cycles(bytes_per_cycle)
                if total < nxt[b]:
                    nxt[b] = total
                    step[b] = (b - need, cand)
        cost = nxt
        back.append(step)

    best = min(range(buckets + 1), key=lambda b: cost[b])
    if cost[best] == inf:
        return None

    chosen: List[Placement] = []
    b = best
    for step in reversed(back):
        prev, cand = step[b]
        chosen.append(cand)
        b = prev
    chosen.reverse()
    return Plan(chosen=chosen,
                lds_bytes=sum(p.lds_bytes for p in chosen),
                global_bytes=sum(p.global_bytes for p in chosen),
                meta_bytes=sum(p.meta_bytes for p in chosen),
                cycles=cost[best])


def frontier(groups: Sequence[Sequence[Placement]], lds_budget: int,
             bytes_per_cycle: float = 64.0,
             quantum: int = QUANTUM) -> List[Tuple[int, float]]:
    """``(lds_bytes, cycles)`` for every budget up to ``lds_budget``.

    The whole trade curve rather than one point on it: where it is flat,
    shared memory is being spent on nothing, and where it is steep is where
    the occupancy step actually has to be weighed.
    """
    out = []
    seen = None
    for b in range(0, lds_budget + 1, quantum):
        p = plan(groups, b, bytes_per_cycle, quantum)
        if p is None:
            continue
        if seen is None or p.cycles < seen:
            out.append((b, p.cycles))
            seen = p.cycles
    return out
