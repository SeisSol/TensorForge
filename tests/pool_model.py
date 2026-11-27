# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An exhaustive model of the high-order tile's mixer pool on tcgen05, a pool
a trip (`compute.multi_element`, `Options.matrix_pool`), for the tests:
every interleaving of its agents at a small size, each state once.

The agents: in each quarter of tensor memory's lanes, band warps and one
epilogue warp, all of them mixers.  In each trip's body a mixer takes
tickets of its quarter's counter of the trip's parity -- ticket `b` below
the trip's blocks the trip's block `b`: it waits for the block's slot of A's
ring back from the products of its last round, fills its quarter of it and
arrives on the slot's full barrier, whose phase completes with every
quarter's arrival -- and on a ticket past the blocks it arrives at the
mixers' trip barrier, an mbarrier counting every mixer, and waits for the
trip's phase by its parity.  Behind the barrier one thread resets the
counters of the trip's parity, for the trip two on.  The issuing thread
takes the blocks in order -- a trip's first one once D's slot is back from
the epilogue -- and commits each onto its A slot's empty barrier, the trip's
last onto D's full one; the epilogue warps drain D(t), a quarter each,
behind trip t's barrier, as the epilogue drains it without the pool, and
give D's slot back once every quarter has.

No wait on the trip barrier can alias: a warp's next arrival there needs the
phase it waits for.  A wait for an A slot can, where A's ring has fewer
slots than a quarter has mixers (`element_group.pool_refusal`) -- and, with
A's slots numbered per trip as the kernel numbers them (`numbering`
`trip`: a block's slot its number in the trip modulo the slots), where a
trip has 1 mod 4 blocks on four slots: slot 0 takes a trip's last block and
the next trip's first back to back, and a quarter's other mixers reach the
next trip's fifth block, slot 0 again, two uses ahead -- what P3 found
wrong at o6 and o10 (`element_group.POOL_P3`).  Numbered as a ring over
the trips (`ring`, the default), no such wait.

`violations` reports every state from which no agent can go on before all
are done (a deadlock); a wait that passes on the parity of a phase two short
of the one it means (aliasing); a draw from a counter not reset since its
parity's last trip, and a reset while a draw of its parity is still to come.
The mutants' switches: `reset` -- `none`, or `early`: ahead of the barrier;
`drain` -- `none`, or `ahead` of the barrier, which is safe but holds the
other mixers at trip t's barrier where they could mix trip t + 1 (so the
drain sits behind it, a performance choice and not a safety one).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class Model:
    """The pool at `quarters` quarters of `bands` band warps and one
    epilogue warp each, `trips` trips of `blocks` blocks, A's ring of
    `aslots` slots, D's of `dslots`."""

    def __init__(self, quarters: int = 2, bands: int = 1, trips: int = 3,
                 blocks: int = 2, aslots: int = 2, dslots: int = 2,
                 reset: str = 'after', drain: str = 'behind',
                 numbering: str = 'ring'):
        #: How A's slots are numbered: `ring` over the trips (an item's slot
        #: its number over every trip modulo the slots), or `trip` as the
        #: kernel numbers them (`compute.multi_element._Fold.round`): from 0
        #: again each trip, a slot's use the trip's uses of it before and the
        #: block's own (`slot_use`).
        self.numbering = numbering
        self.quarters, self.bands, self.trips = quarters, bands, trips
        self.blocks, self.aslots, self.dslots = blocks, aslots, dslots
        self.reset = reset
        #: Where an epilogue warp drains D(t): `behind` trip t's barrier
        #: (option B), `ahead` of it (behind its last item), `none`.
        self.drain = drain
        self.items = trips * blocks
        self.agents = [(q, 'band') for q in range(quarters)
                       for _ in range(bands)] + \
            [(q, 'epi') for q in range(quarters)]
        self.mixers = len(self.agents)

    # The state: (ctr, afull, aempty, dfull, dempty, mma, bar, agents)
    # ctr: per quarter and parity, (value, the trip it was last reset for)
    # bar: (arrivals in the open phase, completed phases) of the trip barrier
    # agents: per agent (pc, trip, item, drained): pc `take`, `wait`
    #   (holding an item), `bar` (arrived, waiting for the barrier's phase
    #   `trip`), `reset` (behind the barrier, to reset), `end` (behind the
    #   last trip, draining), `done`.

    def slot_use(self, item: int) -> Tuple[int, int]:
        """Item `item`'s slot of A's ring and its use of it (`numbering`)."""
        if self.numbering == 'ring':
            return item % self.aslots, item // self.aslots
        trip, block = divmod(item, self.blocks)
        slot = block % self.aslots
        per = (self.blocks - 1 - slot) // self.aslots + 1
        return slot, trip * per + block // self.aslots

    def start(self):
        ctr = tuple((0, p) for q in range(self.quarters) for p in range(2))
        return (ctr, tuple([(0, 0)] * self.aslots), tuple([0] * self.aslots),
                tuple([0] * self.dslots), tuple([(0, 0)] * self.dslots), 0,
                (0, 0), tuple(('take', 0, -1, 0) for _ in self.agents))

    @staticmethod
    def _passes(completed: int, phase: int) -> Tuple[bool, bool]:
        returns = completed % 2 == (phase + 1) % 2
        return returns, returns and completed != phase + 1

    def _with(self, state, i=None, agent=None, **parts):
        names = ('ctr', 'afull', 'aempty', 'dfull', 'dempty', 'mma', 'bar',
                 'agents')
        values = dict(zip(names, state))
        values.update(parts)
        if i is not None:
            agents = list(values['agents'])
            agents[i] = agent
            for q in range(self.quarters):
                # Interchangeable: a quarter's band warps, but the first
                # agent, which resets the counters.
                at = [k for k, (aq, kind) in enumerate(self.agents)
                      if aq == q and kind == 'band' and k != 0]
                for k, v in zip(at, sorted(agents[k] for k in at)):
                    agents[k] = v
            values['agents'] = tuple(agents)
        return tuple(values[n] for n in names)

    def _drain(self, state, i):
        """Epilogue warp `i` drains its quarter of D's next trip, its part
        of D's slot back."""
        dempty = list(state[4])
        pc, trip, item, drained = state[7][i]
        d = drained % self.dslots
        count, phases = dempty[d]
        count += 1
        if count == self.quarters:
            count, phases = 0, phases + 1
        dempty[d] = (count, phases)
        return self._with(state, i, (pc, trip, item, drained + 1),
                          dempty=tuple(dempty))

    def _d_ready(self, state, drained):
        d = drained % self.dslots
        return self._passes(state[3][d], drained // self.dslots)

    def steps(self, state) -> List[Tuple[str, object, List[str]]]:
        ctr, afull, aempty, dfull, dempty, mma, bar, agents = state
        out = []
        for i, (q, kind) in enumerate(self.agents):
            pc, trip, item, drained = agents[i]
            if pc == 'done':
                continue
            if pc == 'take':
                k = q * 2 + trip % 2
                value, reset_for = ctr[k]
                errors = []
                if reset_for != trip:
                    errors.append(f'draw before reset (trip {trip})')
                nc = list(ctr)
                nc[k] = (value + 1, reset_for)
                if value >= self.blocks and kind == 'epi' and \
                        self.drain == 'ahead':
                    out.append((f'{i} leaves trip {trip} to drain', self._with(
                        state, i, ('drainA', trip, -1, drained),
                        ctr=tuple(nc)), errors))
                    continue
                if value >= self.blocks:
                    # Past the trip's blocks: to the trip barrier.
                    count, phases = bar
                    count += 1
                    if count == self.mixers:
                        count, phases = 0, phases + 1
                    nxt = self._with(state, i, ('bar', trip, -1, drained),
                                     ctr=tuple(nc), bar=(count, phases))
                    if self.reset == 'early' and i == 0:
                        nxt = self._reset(nxt, trip, errors)
                    out.append((f'{i} leaves trip {trip}', nxt, errors))
                else:
                    out.append((f'{i} takes {trip}:{value}', self._with(
                        state, i, ('wait', trip, trip * self.blocks + value,
                                   drained), ctr=tuple(nc)), errors))
            elif pc == 'wait':
                s, use = self.slot_use(item)
                alias = False
                if use > 0:
                    ready, alias = self._passes(aempty[s], use - 1)
                    if not ready:
                        continue
                count, phases = afull[s]
                count += 1
                if count == self.quarters:
                    count, phases = 0, phases + 1
                na = list(afull)
                na[s] = (count, phases)
                out.append((f'{i} fills {item}', self._with(
                    state, i, ('take', trip, -1, drained), afull=tuple(na)),
                    ['A empty alias'] if alias else []))
            elif pc == 'bar':
                ready, alias = self._passes(bar[1], trip)
                if ready:
                    nxt_pc = 'reset' if (i == 0 and self.reset == 'after') \
                        else ('take' if trip + 1 < self.trips else 'end')
                    if kind == 'epi' and self.drain == 'behind':
                        nxt_pc = 'drainD'
                    out.append((f'{i} passes the barrier of {trip}',
                                self._with(state, i, (nxt_pc, trip + 1
                                                      if nxt_pc != 'reset'
                                                      else trip, -1,
                                                      drained)),
                                ['barrier alias'] if alias else []))
            elif pc == 'drainD':
                # Behind trip t's barrier: D(t), as the epilogue drains it
                # today -- a wait for its full barrier, then the drain; the
                # warp's trip the next one already.
                ready, alias = self._d_ready(state, drained)
                if ready:
                    nxt = self._drain(state, i)
                    nxt_pc = 'take' if trip < self.trips else 'end'
                    out.append((f'{i} drains {drained} behind the barrier',
                                self._with(nxt, i, (nxt_pc, trip, -1,
                                                    drained + 1)),
                                ['D full alias'] if alias else []))
            elif pc == 'drainA':
                # Ahead of the barrier (a mutant): D(t) behind the warp's
                # last item, then the barrier.
                ready, alias = self._d_ready(state, drained)
                if ready:
                    nxt = self._drain(state, i)
                    count, phases = nxt[6]
                    count += 1
                    if count == self.mixers:
                        count, phases = 0, phases + 1
                    out.append((f'{i} drains {drained} ahead of the barrier',
                                self._with(nxt, i, ('bar', trip, -1,
                                                    drained + 1),
                                           bar=(count, phases)),
                                ['D full alias'] if alias else []))
            elif pc == 'reset':
                errors = []
                nxt = self._reset(state, trip, errors)
                nxt_pc = 'take' if trip + 1 < self.trips else 'end'
                out.append((f'{i} resets parity {trip % 2}', self._with(
                    nxt, i, (nxt_pc, trip + 1, -1, drained)), errors))
            elif pc == 'end':
                if kind == 'epi' and drained < self.trips:
                    ready, alias = self._d_ready(state, drained)
                    if ready:
                        out.append((f'{i} drains {drained} last',
                                    self._drain(state, i),
                                    ['D full alias'] if alias else []))
                    continue
                out.append((f'{i} done', self._with(
                    state, i, ('done', trip, -1, drained)), []))
        if mma < self.items:
            trip, block = divmod(mma, self.blocks)
            s, use = self.slot_use(mma)
            d, duse = trip % self.dslots, trip // self.dslots
            ok, errors = True, []
            if block == 0 and duse > 0:
                ready, alias = self._passes(dempty[d][1], duse - 1)
                ok = ready
                if alias:
                    errors.append('D empty alias')
            if ok:
                ready, alias = self._passes(afull[s][1], use)
                if ready:
                    if alias:
                        errors.append('A full alias')
                    ne = list(aempty)
                    ne[s] += 1
                    nd = list(dfull)
                    if block == self.blocks - 1:
                        nd[d] += 1
                    out.append((f'mma {mma}', self._with(
                        state, aempty=tuple(ne), dfull=tuple(nd),
                        mma=mma + 1), errors))
        return out

    def _reset(self, state, trip, errors):
        """The counters of `trip`'s parity reset for the trip two on: a draw
        of that parity still to come for `trip` an error."""
        if self.reset == 'none':
            return state
        nc = list(state[0])
        agents = state[7]
        for q in range(self.quarters):
            k = q * 2 + trip % 2
            if any(a[0] in ('take', 'wait') and a[1] == trip
                   for a, (aq, _) in zip(agents, self.agents) if aq == q):
                errors.append(f'reset races a draw of trip {trip}')
            nc[k] = (0, trip + 2)
        return self._with(state, ctr=tuple(nc))

    def done(self, state) -> bool:
        return state[5] == self.items and all(
            a[0] == 'done' for a in state[7])


def violations(model: Model, limit: int = 4_000_000) -> Dict[str, object]:
    """Every state `model` reaches, each once: the deadlocks, the errors by
    kind, and how many states there were."""
    seen = set()
    stack = [model.start()]
    deadlocks: List[object] = []
    errors: Dict[str, int] = {}
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        if len(seen) > limit:
            raise RuntimeError(f'more than {limit} states')
        steps = model.steps(state)
        if not steps and not model.done(state):
            deadlocks.append(state)
        for _, nxt, errs in steps:
            for e in errs:
                kind = e.split(' (')[0].split(' of trip')[0]
                errors[kind] = errors.get(kind, 0) + 1
            if nxt not in seen:
                stack.append(nxt)
    return {'states': len(seen), 'deadlocks': len(deadlocks),
            'errors': errors, 'example': deadlocks[:1]}
