# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A protocol model of the folded star's ring of A in tensor memory
(`compute.multi_element._Fold`), for the tests: element warps in bands of
four filling it -- band `j % bands` block `j` of a trip, into slot `j %
slots` -- and one issuing thread draining it in the blocks' order, each
warp an agent of its own and every step of theirs taken in an order a seeded
random scheduler picks.  Its mbarriers keep their phases as the hardware
does: a wait names the parity of the phase it waits for and returns where
the barrier's current phase has the other one, as `mbarrier.try_wait.
parity` does -- so that a waiter two phases ahead of a barrier passes it.

What the sequential host oracle cannot show (`reference.kernel_eval`): it
runs a role's warps together, so that no band ever gets ahead of another.
`violations` counts the schedules in which the issuing thread reads a slot
some band wrote for another block, or in which every agent waits."""

from __future__ import annotations

import random
from typing import List

#: How the bands meet between trips: `trip` -- the star role's warps at
#: each trip's head, as the three roles stage their inputs
#: (`_tcgen05_roles`); `none` -- never, a band a trip ahead of another, as
#: the high-order tile's star bands run on an input ring of two slots.
SYNCS = ('trip', 'none')


class _Barrier:
    def __init__(self, count: int):
        self.count = self.pending = count
        self.phase = 0

    def ready(self, parity: int) -> bool:
        # The phase of that parity complete: the phase before the first is.
        return self.phase % 2 != parity

    def arrive(self, n: int = 1) -> None:
        self.pending -= n
        if self.pending == 0:
            self.phase += 1
            self.pending = self.count


def _schedule(seed: int, bands: int, slots: int, blocks: int, trips: int,
              sync: str) -> List[str]:
    rnd = random.Random(seed)
    full = [_Barrier(128) for _ in range(slots)]
    empty = [_Barrier(1) for _ in range(slots)]
    data = [dict() for _ in range(slots)]
    commits: List[_Barrier] = []
    errors: List[str] = []
    met = [0, 0]   # the star role's barrier: arrivals, generation
    warps = 4 * bands

    def where(k: int, j: int):
        s = j % slots
        uses = -(-(blocks - s) // slots)
        return s, (k * uses + j // slots) % 2

    def meet():
        gen = met[1]
        met[0] += 1
        if met[0] == warps:
            met[0], met[1] = 0, met[1] + 1
        return lambda: met[1] > gen

    def band(w: int):
        for k in range(trips):
            if sync == 'trip':
                yield meet()
            for j in range(w // 4, blocks, bands):
                s, p = where(k, j)
                yield lambda s=s, p=p: empty[s].ready(1 - p)
                data[s][w % 4] = (k, j)
                yield None
                full[s].arrive(32)

    def issuer():
        for k in range(trips):
            for j in range(blocks):
                s, p = where(k, j)
                yield lambda s=s, p=p: full[s].ready(p)
                if any(data[s].get(q) != (k, j) for q in range(4)):
                    errors.append(f'trip {k} block {j}: slot {s} holds '
                                  f'{sorted(data[s].items())}')
                # The products' commit arrives once they complete.
                commits.append(empty[s])
                yield None

    agents = {f'w{w}': band(w) for w in range(warps)}
    agents['mma'] = issuer()
    waits = {name: None for name in agents}
    finished = set()
    while len(finished) < len(agents):
        movers = [n for n in agents if n not in finished
                  and (waits[n] is None or waits[n]())]
        if not movers and not commits:
            errors.append('every agent waits')
            break
        if commits and (not movers or rnd.random() < 0.3):
            commits.pop(0).arrive()
            continue
        name = rnd.choice(movers)
        try:
            waits[name] = next(agents[name])
        except StopIteration:
            finished.add(name)
            waits[name] = None
    return errors


def violations(bands: int, slots: int, blocks: int, trips: int = 4,
               sync: str = 'trip', seeds: int = 100) -> int:
    """The schedules of `seeds` in which the ring of `slots` slots, filled
    by `bands` bands with `blocks` blocks a trip and drained by one thread,
    hands the issuing thread a slot of another block, or hangs."""
    if sync not in SYNCS:
        raise ValueError(f'sync is one of {SYNCS}, not {sync!r}')
    return sum(bool(_schedule(seed, bands, slots, blocks, trips, sync))
               for seed in range(seeds))


#: How B's ring numbers its slots (`compute.multi_element._OperatorRing`,
#: `Options.matrix_slots`): `round` -- chunk `j` of trip `k` the ring's
#: round `k blocks + j`, its slot that modulo the slots; `trip` -- slot `j %
#: slots` every trip, as A's ring is, the slot's rounds counted over the
#: trips, `ceil((blocks - slot) / slots)` a trip.
NUMBERINGS = ('round', 'trip')


def _where(numbering: str, slots: int, blocks: int, k: int, j: int):
    if numbering == 'round':
        g = k * blocks + j
        return g % slots, (g // slots) % 2
    s = j % slots
    uses = -(-(blocks - s) // slots)
    return s, (k * uses + j // slots) % 2


def _operator_schedule(seed: int, slots: int, blocks: int, trips: int,
                       numbering) -> List[str]:
    """B's ring between the operator warp's first lane, which fills every
    chunk of a trip in turn -- its slot back from the products of its last
    round, then the copy and the arrival with its bytes -- and the issuing
    thread, which waits for each chunk's bytes and commits its products
    onto the slot's empty side, the commit arriving once they complete.
    `numbering` the producer's and the consumer's, a pair where they
    differ."""
    rnd = random.Random(seed)
    made, used = ((numbering, numbering) if isinstance(numbering, str)
                  else numbering)
    full = [_Barrier(1) for _ in range(slots)]
    empty = [_Barrier(1) for _ in range(slots)]
    data = [None] * slots
    commits: List[_Barrier] = []
    errors: List[str] = []

    def operator():
        for k in range(trips):
            for j in range(blocks):
                s, p = _where(made, slots, blocks, k, j)
                yield lambda s=s, p=p: empty[s].ready(1 - p)
                data[s] = (k, j)
                yield None
                full[s].arrive()

    def issuer():
        for k in range(trips):
            for j in range(blocks):
                s, p = _where(used, slots, blocks, k, j)
                yield lambda s=s, p=p: full[s].ready(p)
                if data[s] != (k, j):
                    errors.append(f'trip {k} block {j}: slot {s} holds '
                                  f'{data[s]}')
                commits.append(empty[s])
                yield None

    agents = {'operator': operator(), 'mma': issuer()}
    waits = {name: None for name in agents}
    finished = set()
    while len(finished) < len(agents):
        movers = [n for n in agents if n not in finished
                  and (waits[n] is None or waits[n]())]
        if not movers and not commits:
            errors.append('every agent waits')
            break
        if commits and (not movers or rnd.random() < 0.3):
            commits.pop(0).arrive()
            continue
        name = rnd.choice(movers)
        try:
            waits[name] = next(agents[name])
        except StopIteration:
            finished.add(name)
            waits[name] = None
    return errors


def operator_violations(slots: int, blocks: int, trips: int = 4,
                        numbering='trip', seeds: int = 100) -> int:
    """The schedules of `seeds` in which B's ring of `slots` slots, filled by
    the operator warp with `blocks` chunks a trip and drained by the issuing
    thread, both numbering it as `numbering` says -- one of `NUMBERINGS`,
    or a pair, the producer's and the consumer's -- hands the issuing thread
    another chunk, or hangs."""
    for n in ((numbering,) if isinstance(numbering, str) else numbering):
        if n not in NUMBERINGS:
            raise ValueError(f'numbering is one of {NUMBERINGS}, not {n!r}')
    return sum(bool(_operator_schedule(seed, slots, blocks, trips, numbering))
               for seed in range(seeds))


def _shared_schedule(seed: int, bands: int, slots: int, blocks: int,
                     trips: int, sync: str) -> List[str]:
    """A's ring and B's as one, both numbered by the trip (`Options.
    matrix_slots` `trip`): slot `s` of block `j` one full barrier -- a
    phase the band's 128 threads and the operator warp's arrival with B's
    bytes -- and one empty barrier, the products' one commit, which both
    producers wait for before they fill the slot again: the star bands A
    into its columns of tensor memory, the operator warp B into its slot of
    shared memory.  The issuing thread waits once a block, for both."""
    rnd = random.Random(seed)
    full = [_Barrier(128 + 1) for _ in range(slots)]
    empty = [_Barrier(1) for _ in range(slots)]
    a_data = [dict() for _ in range(slots)]
    b_data = [None] * slots
    commits: List[_Barrier] = []
    errors: List[str] = []
    met = [0, 0]
    warps = 4 * bands

    def meet():
        gen = met[1]
        met[0] += 1
        if met[0] == warps:
            met[0], met[1] = 0, met[1] + 1
        return lambda: met[1] > gen

    def band(w: int):
        for k in range(trips):
            if sync == 'trip':
                yield meet()
            for j in range(w // 4, blocks, bands):
                s, p = _where('trip', slots, blocks, k, j)
                yield lambda s=s, p=p: empty[s].ready(1 - p)
                a_data[s][w % 4] = (k, j)
                yield None
                full[s].arrive(32)

    def operator():
        for k in range(trips):
            for j in range(blocks):
                s, p = _where('trip', slots, blocks, k, j)
                yield lambda s=s, p=p: empty[s].ready(1 - p)
                b_data[s] = (k, j)
                yield None
                full[s].arrive()

    def issuer():
        for k in range(trips):
            for j in range(blocks):
                s, p = _where('trip', slots, blocks, k, j)
                yield lambda s=s, p=p: full[s].ready(p)
                if any(a_data[s].get(q) != (k, j) for q in range(4)) \
                        or b_data[s] != (k, j):
                    errors.append(f'trip {k} block {j}: slot {s} holds A '
                                  f'{sorted(a_data[s].items())}, B '
                                  f'{b_data[s]}')
                commits.append(empty[s])
                yield None

    agents = {f'w{w}': band(w) for w in range(warps)}
    agents['operator'] = operator()
    agents['mma'] = issuer()
    waits = {name: None for name in agents}
    finished = set()
    while len(finished) < len(agents):
        movers = [n for n in agents if n not in finished
                  and (waits[n] is None or waits[n]())]
        if not movers and not commits:
            errors.append('every agent waits')
            break
        if commits and (not movers or rnd.random() < 0.3):
            commits.pop(0).arrive()
            continue
        name = rnd.choice(movers)
        try:
            waits[name] = next(agents[name])
        except StopIteration:
            finished.add(name)
            waits[name] = None
    return errors


def shared_violations(bands: int, slots: int, blocks: int, trips: int = 4,
                      sync: str = 'none', seeds: int = 100) -> int:
    """`violations` for A's ring and B's as one (`_shared_schedule`): the
    schedules of `seeds` in which the issuing thread reads a slot either
    producer filled for another block, or in which every agent waits."""
    if sync not in SYNCS:
        raise ValueError(f'sync is one of {SYNCS}, not {sync!r}')
    return sum(bool(_shared_schedule(seed, bands, slots, blocks, trips, sync))
               for seed in range(seeds))
