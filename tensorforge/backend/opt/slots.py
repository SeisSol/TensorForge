# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""The schedule geometry of a batch-loop body, in slots rather than iterations.

``Pipeline`` measures prefetch distance in *iterations*: depth ``d`` fills the
buffer for element ``k + d - 1`` while computing on element ``k``, so every
staged buffer needs ``d`` copies.  That is the only distance the pass can
express, and at ``d = 2`` it costs twice the staging space for one body's worth
of latency cover.

The finer unit is the *slot*: one compute instruction and whatever transfers
are issued before it.  With ``n`` slots in the body and a transfer issued ``d``
slots ahead of its consumer, the value's lifetime is ``d + 1`` slots against an
initiation interval of ``n`` slots, so modulo variable expansion needs

    copies = ceil((d + 1) / n)

copies of the buffer.  ``d <= n - 1`` therefore needs *one* -- no rotation, no
stage index, static offsets -- and the transfer simply wraps across the back
edge: the loads that run off the front of the body are issued for element
``k + 1`` at the end of the previous iteration.  ``d = n`` is the first value
that needs two, and is what today's ``depth = 2`` already does.

So the useful distance at no extra space is bounded by the body's slot count,
which is why this is measured before anything is transformed: whether the idea
pays depends on how many slots the corpus's bodies actually have, and that is a
number, not an opinion.

This module answers only that.  It transforms nothing.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from tensorforge.backend.instructions.abstract_instruction import (
    AbstractInstruction, BarrierScope)
from tensorforge.backend.instructions.allocate import RegisterAlloc, ShrMemAlloc
from tensorforge.backend.instructions.batch_loop import BatchLoop
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite
from tensorforge.backend.instructions.memory.load import (GlbToRegLoader,
                                                          GlbToShrLoader,
                                                          LoadWait)
from tensorforge.backend.symbol import SymbolType


class Transfer(NamedTuple):
    """One global-to-local transfer, placed in the body's slot grid."""

    load: AbstractInstruction
    index: int              # position in the body
    slot: int               # how many compute slots precede it
    first_use_slot: Optional[int]   # slot of the first consumer, if any
    shared: bool            # destination is shared memory, not a register

    @property
    def distance(self) -> Optional[int]:
        """Slots between issue and consumption, as currently scheduled."""
        if self.first_use_slot is None:
            return None
        return self.first_use_slot - self.slot

    def wraps_at(self, d: int, n: int) -> bool:
        """Would a prefetch distance of ``d`` push this transfer past the top?"""
        if self.first_use_slot is None or n == 0:
            return False
        return self.first_use_slot - d < 0

    @staticmethod
    def copies(d: int, n: int) -> int:
        """Buffer copies needed at distance ``d`` in a body of ``n`` slots."""
        if n <= 0:
            return 1
        return -(-(d + 1) // n)


class SlotModel:
    """Slot structure of one loop body.

    ``n`` counts compute instructions, because those are the units a transfer
    can hide behind.  A body with one compute slot cannot hide anything without
    a second copy of the buffer, which is exactly the case ``Pipeline`` was
    written for and exactly the case the wrap-around does not improve.
    """

    def __init__(self, body: Sequence[AbstractInstruction]):
        self._body = list(body)
        self.compute_at: List[int] = []
        self.transfers: List[Transfer] = []
        self.barriers: int = 0
        self.rejected: List[Tuple[AbstractInstruction, str]] = []

    # ------------------------------------------------------------------ #

    def run(self) -> 'SlotModel':
        slot_of: Dict[int, int] = {}
        slot = 0
        for index, instr in enumerate(self._body):
            slot_of[index] = slot
            if isinstance(instr, ComputeInstruction):
                self.compute_at.append(index)
                slot += 1
            if instr.barrier_scope() is not BarrierScope.NONE:
                self.barriers += 1

        for index, instr in enumerate(self._body):
            if not isinstance(instr, (GlbToShrLoader, GlbToRegLoader)):
                continue
            dest = instr.defs()[0] if instr.defs() else None
            if dest is None:
                self.rejected.append((instr, 'no destination symbol'))
                continue
            use = self._first_use(index, dest)
            self.transfers.append(Transfer(
                load=instr,
                index=index,
                slot=slot_of[index],
                first_use_slot=None if use is None else slot_of[use],
                shared=isinstance(instr, AbstractShrMemWrite)))
        return self

    def _first_use(self, after: int, sym) -> Optional[int]:
        for index in range(after + 1, len(self._body)):
            instr = self._body[index]
            if isinstance(instr, LoadWait):
                # the wait delegates uses() to the transfer it awaits; it marks
                # completion, not consumption
                continue
            if any(u is sym for u in instr.uses()):
                return index
        return None

    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        return len(self.compute_at)

    @property
    def free_distance(self) -> int:
        """Largest ``d`` that still needs one copy per buffer."""
        return max(self.n - 1, 0)

    def cost(self, d: int) -> Dict[str, int]:
        """What distance ``d`` costs this body, in copies and wrapped loads."""
        c = Transfer.copies(d, self.n)
        return {
            'copies': c,
            'wrapped': sum(1 for t in self.transfers if t.wraps_at(d, self.n)),
            'shared_wrapped': sum(1 for t in self.transfers
                                  if t.shared and t.wraps_at(d, self.n)),
        }

    def report(self) -> str:
        lines = [f'slots: n={self.n} transfers={len(self.transfers)} '
                 f'barriers={self.barriers} free_distance={self.free_distance}']
        for t in self.transfers:
            dest = t.load.defs()[0] if t.load.defs() else None
            name = getattr(dest, 'name', '?')
            where = 'shr' if t.shared else 'reg'
            lines.append(f'  {name} [{where}] slot {t.slot} -> '
                         f'{t.first_use_slot} (distance {t.distance})')
        return '\n'.join(lines)


def models_for(stream: Sequence[AbstractInstruction]) -> List[SlotModel]:
    """One model per ``BatchLoop`` in the stream."""
    out = []
    for instr in stream:
        if isinstance(instr, BatchLoop):
            out.append(SlotModel(instr.region).run())
    return out
