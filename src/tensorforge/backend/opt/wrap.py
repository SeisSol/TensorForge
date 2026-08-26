# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Prefetch at slot distance, wrapping across the back edge.

``Pipeline`` advances a transfer by whole iterations, so depth ``d`` needs
``d`` copies of every staged buffer.  This advances it by *slots* -- one
compute instruction each -- and the accounting (see ``slots.py``) is

    copies = ceil((d + 1) / n)

for a body of ``n`` compute slots.  Every ``d <= n - 1`` therefore needs one
copy: no rotation, no stage index, no second view into the allocation.  The
transfers that run off the front of the body are issued at the end of the
*previous* iteration, for element ``k + 1``::

    before:  [ l1 c1 l2 c2 l3 c3 ]
    after:   l1 | [ l2 c1 l3 c2 l1' c3 ]

with ``l1'`` reading element ``k + 1`` and the peeled ``l1`` reading the
thread's first element.  The buffer is written at the tail of iteration ``k``
and read at the head of ``k + 1``, and the read precedes the next write, so one
buffer is enough -- which is the whole point, and why this is worth having
where ``Pipeline`` is not.

Restricted to register destinations, for two reasons that are the same reason.
A wrapped write to a *shared* buffer overtakes the read of the value the
previous iteration put there, so it needs a barrier against that read -- one
per slot, where double buffering needs two per body.  A register buffer is
thread-private and needs none.  The corpus says this is not a hypothetical
preference: on HIP every per-element transfer already lands in registers and
the bodies carry 6 barriers in total, while the CUDA path stages 54 transfers
through shared memory and carries 93.

Known hole, deliberately left: a wrapped transfer sits mid-body and so stays
inside the per-element flag guard, where a masked element skips it and the
next iteration reads what the one before that loaded.  ``BatchLoop`` can
currently only lift a *prefix* of the region out of the guard, and the correct
placement is not a prefix.  ``verify`` reports it rather than the pass
pretending otherwise.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.instructions.memory.load import (GlbToRegLoader,
                                                          LoadWait)
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.symbol import Symbol

from .abstract import AbstractTransformer, Context
from .slots import SlotModel, Transfer


class Wrapped(NamedTuple):
    transfer: GlbToRegLoader
    producer: GetElementPtr
    alloc: Optional[RegisterAlloc]
    target_index: int          # body index to insert before
    first_use_slot: int
    distance: int              # slots, after clamping to this buffer's span


class WrapLoads(AbstractTransformer):
    """Move register transfers ``distance`` slots ahead of their consumers."""

    def __init__(self,
                 context: Context,
                 instructions: List[AbstractInstruction],
                 distance: int = 1):
        super(WrapLoads, self).__init__(context, instructions)
        if distance < 1:
            raise ValueError(f'wrap distance must be >= 1, got {distance}')
        self._distance = distance
        self.rejected: List[Tuple[object, str]] = []
        self.wrapped: List[str] = []

    # ------------------------------------------------------------------ #

    def apply(self) -> None:
        out: List[AbstractInstruction] = []
        for instr in self._instrs:
            if not isinstance(instr, BatchLoop):
                out.append(instr)
                continue
            prologue, body = self._wrap_loop(instr)
            out.extend(prologue)
            instr.replace_region(0, body)
            out.append(instr)
        self._instrs = out

    # ------------------------------------------------------------------ #

    def _wrap_loop(self, loop: BatchLoop):
        body = list(loop.region)
        if loop._mode is LoopMode.SINGLE:
            # one iteration, so there is no next element to prefetch and no
            # back edge to wrap across
            return [], body

        model = SlotModel(body).run()
        n = model.n
        if n < 2:
            # free distance is n - 1 = 0: nothing to win that Pipeline's second
            # copy does not already do
            return [], body

        plans = [p for p in (self._plan(body, model, t, self._distance, n)
                             for t in model.transfers) if p is not None]
        if not plans:
            return [], body

        prologue: List[AbstractInstruction] = []
        # Insert from the back so earlier target indices stay valid.
        for plan in sorted(plans, key=lambda p: p.target_index, reverse=True):
            prologue = self._apply_one(loop, body, plan) + prologue
        return prologue, body

    # ------------------------------------------------------------------ #

    def _plan(self, body, model: SlotModel, t: Transfer,
              d: int, n: int) -> Optional[Wrapped]:
        dest = t.load.defs()[0] if t.load.defs() else None
        name = getattr(dest, 'name', '?')

        if not isinstance(t.load, GlbToRegLoader):
            return self._reject(name, 'destination is not a register buffer; a '
                                      'wrapped shared write needs a barrier '
                                      'against the read it overtakes')
        if t.first_use_slot is None:
            return self._reject(name, 'loaded value is never read in this body')
        # Clamp to what this buffer can take at one copy.  The distance is not
        # uniform over the body: a value read in slots 1 and 3 stays live for
        # three slots, so the wrapped write has to land after slot 3, and the
        # slots it may be moved back by are `n - 1 - span`, not `n - 1`.
        # Wrapping to the first consumer while a later one still holds the
        # buffer is a write-after-read on the value the next iteration wants --
        # silent wrong numbers, and the reason this is clamped per transfer
        # rather than per body.
        d = min(d, t.free_distance(n))
        if d < 1:
            return self._reject(
                name, f'consumers span {t.span} of {n} slots, leaving no '
                      f'distance at one copy')
        if not t.wraps_at(d, n):
            # it fits inside the body; MoveLoads already had the chance and the
            # placement is not this pass's business
            return None

        src = t.load.uses()[0] if t.load.uses() else None
        if src is None:
            return self._reject(name, 'transfer reports no source symbol')
        producer = self._producer(body, src)
        if producer is None:
            return self._reject(
                name, 'source pointer is not computed by a GetElementPtr in '
                      'this body, so there is no element index to advance')
        if isinstance(producer._batch_offset, str):
            return self._reject(name, 'source pointer already names an index '
                                      'verbatim; already pipelined')

        writers = [i for i in body
                   if any(o is dest for o in i.defs())
                   and not isinstance(i, (RegisterAlloc, LoadWait))]
        if len(writers) > 1:
            kinds = ', '.join(sorted({type(i).__name__ for i in writers}))
            return self._reject(
                name, f'buffer is written {len(writers)} times per iteration '
                      f'({kinds}); a wrapped buffer must hold one element for '
                      f'the whole iteration')
        if self._read_before(body, body.index(t.load), dest):
            return self._reject(name, 'buffer is read before the transfer, so '
                                      'moving the transfer changes what that '
                                      'read observes')

        target_slot = t.first_use_slot - d + n
        if not 0 <= target_slot < n:
            return self._reject(name, f'wrapped slot {target_slot} is outside '
                                      f'the body')
        target_index = model.compute_at[target_slot]
        if target_index <= body.index(t.load):
            return self._reject(name, 'wrapped position is not later in the '
                                      'body than the transfer already is')

        alloc = next((i for i in body if isinstance(i, RegisterAlloc)
                      and i._dest is dest), None)
        if alloc is None:
            return self._reject(
                name, 'no RegisterAlloc for the destination; the declaration '
                      'has to leave the loop or the value does not survive '
                      'the back edge')
        if alloc._init_value not in (None, 0):
            return self._reject(name, 'buffer is declared with a non-zero '
                                      'initialiser, which a hoisted '
                                      'declaration would apply once')

        return Wrapped(transfer=t.load, producer=producer, alloc=alloc,
                       target_index=target_index,
                       first_use_slot=t.first_use_slot,
                       distance=d)

    def _reject(self, name, reason) -> None:
        self.rejected.append((name, reason))
        return None

    @staticmethod
    def _producer(body, sym) -> Optional[GetElementPtr]:
        for instr in body:
            if isinstance(instr, GetElementPtr) and any(d is sym
                                                        for d in instr.defs()):
                return instr
        return None

    @staticmethod
    def _read_before(body, before: int, sym) -> bool:
        for index in range(before):
            if any(u is sym for u in body[index].uses()):
                return True
        return False

    # ------------------------------------------------------------------ #

    def _apply_one(self, loop: BatchLoop, body: List[AbstractInstruction],
                   plan: Wrapped) -> List[AbstractInstruction]:
        """Rewrite one transfer in place; return what it adds to the prologue."""
        transfer = plan.transfer
        old_src = transfer.uses()[0]
        dest = transfer.defs()[0]

        # A pointer to element k + 1.  `index_name(1)` is the loop's own
        # lookahead binding, already clamped to the last element, so the final
        # iteration prefetches a valid address it will never read rather than
        # running off the end.
        ahead = Symbol(f'wrap_{old_src.name}', old_src.stype, old_src.obj)
        ahead.data_view = old_src.data_view
        ahead_ptr = GetElementPtr(self._context,
                                  src=plan.producer._src,
                                  dest=ahead,
                                  include_extra_offset=plan.producer._include_extra_offset,
                                  batch_offset=1)

        # ... and one to the thread's first element, for the peeled copy.
        peeled = Symbol(f'peel_{old_src.name}', old_src.stype, old_src.obj)
        peeled.data_view = old_src.data_view
        peeled_ptr = GetElementPtr(self._context,
                                   src=plan.producer._src,
                                   dest=peeled,
                                   include_extra_offset=plan.producer._include_extra_offset,
                                   batch_offset=loop.prologue_index())

        peeled_load = GlbToRegLoader(context=self._context,
                                     src=peeled,
                                     dest=dest,
                                     num_threads=transfer._num_threads,
                                     linearize=transfer._linearize,
                                     src_bbox=transfer._bbox,
                                     src_offset=transfer._offset)

        # Retarget the body transfer.  Mutating `_src` rather than rebuilding
        # keeps the LoadWait that MoveLoads left at the consumer pointing at
        # this same object, which is what orders the consumer after it.
        self._drop_user(old_src, transfer)
        transfer._src = ahead
        ahead.add_user(transfer)

        # Move the declaration out of the loop: a register declared inside the
        # body is a fresh object every iteration, so a value written at the
        # tail of iteration k would not be there at the head of k + 1.
        body.remove(plan.alloc)
        body.remove(transfer)
        insert_at = self._target_after_removals(body, plan)
        body[insert_at:insert_at] = [ahead_ptr, transfer]

        # The original GetElementPtr may now feed nothing.
        if not any(any(u is old_src for u in i.uses()) for i in body):
            body.remove(plan.producer)
            self._drop_user(old_src, plan.producer)

        self.wrapped.append(getattr(dest, 'name', '?'))
        return [plan.alloc, peeled_ptr, peeled_load]

    def _target_after_removals(self, body, plan: Wrapped) -> int:
        """Where the transfer goes, recomputed after alloc/transfer removal."""
        model = SlotModel(body).run()
        n = model.n
        slot = plan.first_use_slot - plan.distance + n
        slot = min(max(slot, 0), n - 1)
        return model.compute_at[slot]

    @staticmethod
    def _drop_user(sym, instr) -> None:
        users = sym.get_user_list()
        while instr in users:
            users.remove(instr)

    # ------------------------------------------------------------------ #

    def report(self) -> str:
        lines = [f'wrap: {len(self.wrapped)} transfer(s) wrapped, '
                 f'{len(self.rejected)} rejected']
        for name in self.wrapped:
            lines.append(f'  + {name}')
        for name, reason in self.rejected:
            lines.append(f'  - {name}: {reason}')
        return '\n'.join(lines)
