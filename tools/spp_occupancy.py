# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What staging an operand costs in residency.

`spp_plan` answers "given this much shared memory, what is the cheapest
layout".  The budget in that question is an input, and choosing it is the
decision it cannot make: shared memory is not free at the margin, it is free
until it crosses an occupancy step and then costs a whole block's worth of
residency at once.

So the budget becomes endogenous here.  Rather than pick a number and defend
it, every block count the hardware can hold is enumerated, each one implying
the shared memory per block that permits it, and the planner is run against
each.  The output is cost against residency, which is the axis the decision is
actually made on, and it needs no constant that is not already in
`hw_descr_db.yml`.

Deliberately only the shared-memory side.  Registers also bound residency, but
the pressure model calibrates to a ranking and not to a threshold -- it can say
which of two lane configurations is tighter, not whether either fits -- and a
residency figure computed from a number that does not convert to registers
would look exact while being a guess.  `lanes.search` owns that side and owns
it honestly; this owns the side where bytes are bytes.

One consequence of that split is worth stating, because it is not obvious: the
lane count does not appear here at all.  `RegmaxBlockPolicy` sizes the mults so
that lanes times mults stays at 256, so doubling the lanes halves the mults and
leaves the block the same size and the same shared memory.  What moves is the
per-element share of anything staged block-wide, which is why `traversals` and
`mults` are inputs rather than something this infers.

Turning a residency into a time takes one number this cannot avoid.  What
residency buys is latency hiding, and latency is a quantity: below the point
where the resident waves cover it, throughput scales with them, and above it
throughput is bound by whatever shared resource saturates first.  So the model
is the usual three-way maximum -- issue, bandwidth, exposed latency -- and
``latency`` is an argument with a stated default rather than a constant buried
in the arithmetic.  It is also the one input here worth sweeping, which is why
the frontier is cheap enough to run several times.

The known gap, stated rather than hidden: there is no cache here.  An operand
left in global memory and read once per traversal is charged a miss every
time, which is the pessimistic bound and not the truth -- a constant matrix
read forty-eight times a block is in L1 after the first pass, and for one small
enough to stay there, staging it in shared memory buys little beyond the
difference between two hit latencies.  ``cache_hit`` is the crude handle on
that, and it is crude on purpose: the honest version needs cache sizes per
target, which `hw_descr_db.yml` does not yet carry.  Until it does, read the
unstaged branch as a lower bound on its own merit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from spp_plan import Placement, Plan, plan


@dataclass(frozen=True)
class Machine:
    """The residency-relevant half of a hardware description."""

    #: Shared memory one CU or SM can hand out, in bytes.
    #:
    #: Taken from ``max_local_mem_size_per_block``, which is a per-*block*
    #: ceiling and so a lower bound on the per-CU pool.  On every target in
    #: the database the two coincide or nearly so -- 64 kB against 64 on
    #: CDNA, 163 against 164 on sm_80 -- because the per-block ceiling is set
    #: by the pool.  A target where they diverge would need its own field
    #: rather than this reading.
    lds_capacity: int
    max_blocks: int
    max_threads: int
    wave: int

    @classmethod
    def from_hw(cls, hw) -> 'Machine':
        return cls(lds_capacity=hw.max_local_mem_size_per_block,
                   max_blocks=hw.max_block_per_sm,
                   max_threads=hw.max_threads_per_sm,
                   wave=hw.vec_unit_length)


@dataclass(frozen=True)
class Residency:
    blocks: int
    waves: int
    limit: str

    def __str__(self) -> str:
        return f'{self.blocks} Blöcke, {self.waves} Wellen ({self.limit})'


def residency(machine: Machine, lds_per_block: int,
              threads_per_block: int) -> Residency:
    """Blocks and waves resident per CU, and which ceiling decided it.

    Naming the binding limit matters more than the figure: shared memory that
    is spent below the limit is spent on nothing, and the whole point of the
    frontier is to see where that stops being true.
    """
    per_block_waves = max(1, math.ceil(threads_per_block / machine.wave))
    limits = [(machine.max_blocks, 'blocks'),
              (machine.max_threads // max(1, threads_per_block), 'threads')]
    if lds_per_block > 0:
        limits.append((machine.lds_capacity // lds_per_block, 'lds'))
    blocks, which = min(limits, key=lambda p: p[0])
    blocks = max(0, blocks)
    return Residency(blocks=blocks, waves=blocks * per_block_waves,
                     limit=which)


@dataclass
class Level:
    """One point of the frontier: a residency and the best plan that holds it."""

    residency: Residency
    lds_per_block: int
    plan: Plan
    mults: int

    @property
    def cycles_per_element(self) -> float:
        """The comparable figure.

        Cycles per *block* are not: two configurations that hold a different
        number of elements per block would be compared on how much work they
        do rather than on how fast they do it.
        """
        return self.plan.cycles / max(1, self.mults)

    #: Memory instructions per element that go to global memory and so carry
    #: its latency.  A staged operand is fetched once per block however often
    #: it is then read, so only its staging counts.
    global_ops: float = 0.0
    #: Bytes per element from global memory, and instructions per element.
    traffic: float = 0.0
    issue: float = 0.0

    def seconds_per_element(self, bytes_per_cycle: float,
                            latency: float, cache_hit: float = 0.0) -> float:
        """Cycles one element costs the CU, as the binding one of three.

        Issue and bandwidth are shared and do not improve with residency;
        exposed latency is what residency actually buys, and it divides by the
        waves that are there to cover it.  ``cache_hit`` is the share of
        unstaged reads that never reach memory, and it scales both the traffic
        and the latency they would have carried.
        """
        waves = max(1, self.residency.waves)
        miss = 1.0 - min(1.0, max(0.0, cache_hit))
        return max(self.issue,
                   self.traffic * miss / bytes_per_cycle,
                   self.global_ops * miss * latency / waves)


#: Cycles a global-memory access takes to come back on a cold cache.  Both
#: vendors land in the same few hundred; the figure decides how many waves are
#: enough and nothing else, so it is worth sweeping rather than trusting.
DEFAULT_LATENCY = 600.0


def frontier(groups: Sequence[Sequence[Placement]], machine: Machine,
             threads_per_block: int, mults: int, fixed_lds: int = 0,
             bytes_per_cycle: float = 64.0,
             latency: float = DEFAULT_LATENCY,
             cache_hit: float = 0.0,
             quantum: int = 256) -> List[Level]:
    """The best plan at every residency the hardware can reach.

    ``fixed_lds`` is what the block needs regardless of the constants -- the
    staging windows and working set, ``get_size_per_mult()`` times ``mults`` --
    and it is charged before any operand is offered a place.

    Returns one entry per *achieved* residency, best first by throughput.  A
    target block count that the resulting plan overshoots is reported at the
    residency it actually reaches, not the one it aimed at, and duplicates are
    resolved in favour of the cheaper plan.
    """
    best: dict = {}
    for target in range(1, machine.max_blocks + 1):
        budget = machine.lds_capacity // target - fixed_lds
        if budget < 0:
            continue
        got = plan(groups, budget, bytes_per_cycle, quantum)
        if got is None:
            continue
        lds = fixed_lds + got.lds_bytes
        res = residency(machine, lds, threads_per_block)
        level = Level(
            residency=res, lds_per_block=lds, plan=got, mults=mults,
            global_ops=sum(p.accesses * (1 if p.staged else p.traversals)
                           for p in got.chosen) / max(1, mults),
            traffic=got.global_bytes / max(1, mults),
            issue=sum(p.accesses * p.traversals
                      for p in got.chosen) / max(1, mults))
        key = res.blocks
        prev = best.get(key)
        score = level.seconds_per_element(bytes_per_cycle, latency, cache_hit)
        if prev is None or score < prev.seconds_per_element(
                bytes_per_cycle, latency, cache_hit):
            best[key] = level
    return sorted(best.values(),
                  key=lambda l: l.seconds_per_element(bytes_per_cycle,
                                                      latency, cache_hit))


def report(levels: Sequence[Level], bytes_per_cycle: float = 64.0,
           latency: float = DEFAULT_LATENCY, cache_hit: float = 0.0) -> str:
    head = (f'{"Blocks":>7} {"Waves":>6} {"limit":<8} {"lds/blk":>9} '
            f'{"issue":>8} {"bandw":>8} {"latency":>8} {"cyc/elem":>9}')
    lines = [head, '-' * len(head)]
    miss = 1.0 - min(1.0, max(0.0, cache_hit))
    for lv in levels:
        waves = max(1, lv.residency.waves)
        lines.append(
            f'{lv.residency.blocks:>7} {lv.residency.waves:>6} '
            f'{lv.residency.limit:<8} {lv.lds_per_block / 1024:>8.1f}K '
            f'{lv.issue:>8.0f} {lv.traffic * miss / bytes_per_cycle:>8.0f} '
            f'{lv.global_ops * miss * latency / waves:>8.0f} '
            f'{lv.seconds_per_element(bytes_per_cycle, latency, cache_hit):>9.0f}')
    return '\n'.join(lines)
