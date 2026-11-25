# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: the hand-offs of a body's rings.

A ring (`IRBuilder.mbarrier_ring`, `engines.RingSpec`) hands its slots
between a producer and its consumers -- roles of the block, or the threads
of one role -- through a pair of barriers a slot.  What a body does on a
ring is six kinds of statement, the acts of the protocol:

* `setup`: the ring set up ahead of everything (`SYNC_INIT`, the ring its
  target);
* `acquire`: the producer's wait for its round of a slot on its empty side
  (`IRBuilder.ring_acquire`);
* `copy`: a copy into the slot that completes its bytes on the full side
  (`copy_bulk`, `copy_tma` with a `RingSlot`);
* `publish`: the producer's arrival on the full side, with the bytes it
  announces (`ring_publish`);
* `take`: a consumer's wait for the slot on its full side (`ring_take`);
* `release`: a consumer's arrival on the empty side, or an engine's commit
  of its products onto it (`ring_release`).

`handoffs` lists them by ring, each with the role of the block it stands in
(`pir.roles`): what a pass that weighs the hand-offs -- a cost model, a
scheduler -- reads of a body.  `refusal` is the verifier's rule over one
statement: a ring that names its roles is acquired and published by its
producer alone, taken and released by its consumers alone, and a round's
bytes, where its description fixes them, are what each publish
announces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

from .core import IRError, Op, Stmt, Value
from . import engines
from . import roles as role_split

#: The acts of the protocol a ring's statements are, in the order a round
#: takes them.
ACTS = ('setup', 'acquire', 'copy', 'publish', 'take', 'release')


class Handoff(NamedTuple):
    """One statement on a ring: the act it is (`ACTS`), the side of a slot
    it names, the role of the block it stands in (None outside the roles
    of a block), and the bytes it announces -- a publish, the round's -- or
    copies: a copy, one issuing thread's, which as many threads as issue
    it move each."""
    stmt: Stmt
    act: str
    side: Optional[str]
    role: Optional[str]
    bytes: int = 0


@dataclass
class RingUse:
    """What a body does on one ring: its description, its setup and its
    hand-offs in the order they stand."""
    ring: Value
    spec: Optional[engines.RingSpec]
    setup: Optional[Stmt] = None
    handoffs: List[Handoff] = field(default_factory=list)

    def acts(self, act: str) -> List[Handoff]:
        """The hand-offs that are `act`."""
        return [h for h in self.handoffs if h.act == act]

    def roles(self, act: str) -> Tuple[Optional[str], ...]:
        """The roles the hand-offs that are `act` stand in, each once, in
        the order they first do."""
        out: List[Optional[str]] = []
        for h in self.acts(act):
            if h.role not in out:
                out.append(h.role)
        return tuple(out)


def act_of(s: Stmt) -> Optional[Tuple[str, Optional[str]]]:
    """`(act, side)` of a statement on a ring, or None for any other."""
    if not engines.on_ring(s):
        return None
    side = s.attr('side')
    if s.op == Op.SYNC_INIT:
        return 'setup', None
    if s.op == Op.WAIT_PHASE:
        return ('acquire' if side == 'empty' else 'take'), side
    if s.op == Op.ARRIVE:
        return ('release' if side == 'empty' else 'publish'), side
    if s.op == Op.COMMIT_GROUP:
        if side == 'full':
            # The products' arrival once complete onto the slot's full
            # side: the producer's publish of what they wrote.
            return 'publish', 'full'
        return 'release', 'empty'
    if s.op == Op.COPY_ASYNC:
        return 'copy', 'full'
    return None


def _bytes(s: Stmt, act: str) -> int:
    if act == 'publish':
        return int(s.attr('tx') or 0)
    if act == 'copy':
        return int(s.attr('width') or 0)
    return 0


def handoffs(body) -> Dict[int, RingUse]:
    """Every ring the statements of `body` act on, by its value's id: the
    ring's description (`engines.RingSpec`), its setup, and each statement
    on it (`Handoff`) with the role of the block it stands in -- the roles'
    regions read through their chain (`pir.roles.regions`)."""
    uses: Dict[int, RingUse] = {}

    def record(s: Stmt, role: Optional[str]) -> None:
        found = act_of(s)
        if found is None:
            return
        act, side = found
        ring = engines.object_value(s)
        if ring is None:
            return
        use = uses.get(ring.id)
        if use is None:
            use = uses[ring.id] = RingUse(ring, engines.ring_spec(ring))
        if act == 'setup':
            use.setup = s
            return
        use.handoffs.append(Handoff(s, act, side, role, _bytes(s, act)))

    def visit(stmts, role):
        for s in stmts:
            if role_split.is_split(s):
                try:
                    regions = role_split.regions(s)
                except IRError:
                    regions = None
                if regions is not None:
                    for r, region in regions:
                        visit(region.body, r.name)
                    continue
            if s.regions:
                for region in s.regions:
                    visit(region.body, role)
                continue
            record(s, role)

    visit(tuple(body), None)
    return uses


def refusal(s: Stmt, role: Optional[str]) -> Optional[str]:
    """What is wrong with the ring statement `s` standing in the role
    `role` (None outside the roles of a block), or None: a ring that names
    its roles is acquired, copied into and published by its producer, and
    taken and released by its consumers (`RingSpec.role_refusal`); a
    publish announces the bytes of a round where the description fixes
    them."""
    found = act_of(s)
    if found is None or found[0] == 'setup':
        return None
    act, side = found
    spec = engines.ring_spec(engines.object_value(s))
    if spec is None:
        return None
    why = spec.role_refusal(role, side, act in ('acquire', 'take'))
    if why is not None:
        return f'{s.op}: {why}'
    if act == 'publish' and spec.tx and int(s.attr('tx') or 0) != spec.tx:
        return (f'{s.op}: a publish of {int(s.attr("tx") or 0)} bytes on a '
                f'ring whose rounds are {spec.tx} bytes each')
    return None


__all__ = ['ACTS', 'Handoff', 'RingUse', 'act_of', 'handoffs', 'refusal']
