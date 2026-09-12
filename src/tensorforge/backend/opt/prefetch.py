# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Hint for the next element's pointer, at the top of the body that hides it.

`WrapLoads` and `Pipeline` move the transfer: they buy latency and pay for it
in registers, in buffer copies, and in what a masked element does to a value
carried across the back edge. This buys less and pays nothing. Nothing is
loaded, no value is produced, no buffer is needed, and a target that cannot
spell a prefetch drops the statement -- so the failure mode of getting this
wrong is a wasted memory request, not a wrong number.

What it goes after is the one address in a batched kernel that costs a
dependent load. Under `Addressing.PTR_BASED` an iteration cannot form a single
address until `m[batchId0]` has arrived, so the whole body waits on a load
that nothing was scheduled ahead of. Issued one element early, that load has a
full iteration of arithmetic in front of it.

The hint goes at the head of the region, which is as far from its use as this
pass can put it and still be in the same body: the use is in the next
iteration, so every statement of this one is cover.

The hint is issued outside the per-element flag guard: a masked element
still asks for its successor's pointer, which is in range whatever the
mask (see `PrefetchBatch.apply`).
"""

from typing import List

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.instructions.prefetch import (PrefetchBatchPointer,
                                                      PrefetchData as _Hint)
from tensorforge.backend.instructions.memory.load import (GlbToRegLoader,
                                                          GlbToShrLoader)
from tensorforge.backend.symbol import Symbol
from tensorforge.common.basic_types import Addressing
from tensorforge.backend.instructions.ptr_manip import GetElementPtr

from .abstract import AbstractTransformer, Context


class PrefetchBatch(AbstractTransformer):
    """Insert one pointer hint per pointer-based operand, per batch loop."""

    def __init__(self,
                 context: Context,
                 instructions: List[AbstractInstruction],
                 level: str = 'l2'):
        super(PrefetchBatch, self).__init__(context, instructions)
        self._level = level
        self.hinted: List[str] = []
        self.rejected: List[tuple] = []

    def apply(self) -> None:
        for instr in self._instrs:
            if not isinstance(instr, BatchLoop):
                continue
            hints = self._hints_for(instr)
            if hints:
                instr.replace_region(0, hints + list(instr.region))
                # Outside the flag guard: the address is arithmetic on a
                # kernel argument and a clamped index, and nothing here
                # dereferences the pointer it asks for.  Inside, a masked
                # element issued no hint -- and next to `WrapLoads`, whose
                # unguarded prefix it preceded, the guard could not be one
                # block and the kernel did not generate.
                instr.mark_unguarded(hints)

    # ------------------------------------------------------------------ #

    def _hints_for(self, loop: BatchLoop) -> List[AbstractInstruction]:
        if loop._mode is LoopMode.SINGLE:
            # one iteration, so there is no next element to ask for
            return []
        if loop._mode is LoopMode.LAUNCHCTRL:
            # The next element exists and this pass cannot name it. `batchId1`
            # is `batchId0 + stride`, which is what the strided traversal will
            # reach; the queue hands out whatever CTA the launcher cancels, so
            # the name points at some other block's element. For a transfer
            # that is wrong numbers, which is why `WrapLoads` refuses it too;
            # here it is only a request for a line nobody wants, and asking
            # for one is worse than asking for nothing.
            self.rejected.append((loop, 'the queue decides the next element'))
            return []

        out: List[AbstractInstruction] = []
        seen = set()
        for instr in loop.region:
            if not isinstance(instr, GetElementPtr):
                continue
            if not instr.reads_the_pointer_array():
                # Strided and batch-invariant addressing reach no memory to
                # compute an address, so there is no dependent load to shadow;
                # a table-fed or lookahead binding does not read one by this
                # loop's index.
                continue
            src = instr.uses()[0]
            if id(src) in seen:
                continue
            seen.add(id(src))
            out.append(PrefetchBatchPointer(self._context, src, loop,
                                            level=self._level))
            self.hinted.append(src.name)
        return out


class PrefetchData(AbstractTransformer):
    """Hint the next element's operands where `WrapLoads` would fetch them.

    `WrapLoads` issues the transfer for element ``k + 1`` at the tail of
    iteration ``k`` and pays for it: the destination has to survive the back
    edge -- a register image carried, or a buffer the next iteration reads --
    and a peel and a drain.  This leaves every transfer where it is and puts
    at that same tail a *hint* for the data the transfer will read, one span
    per `Lexic.prefetch_line_bytes`, through a pointer to ``k + 1`` bound at
    the head of the body.  Nothing waits on it and nothing is produced, so it
    changes no result; the transfer then finds its lines on their way in.

    Which transfers: global-to-shared and global-to-register ones whose
    source is bound per element in this body (`PTR_BASED` or `STRIDED`),
    each source once.  A batch-invariant operand is the same data for every
    element and already cached; a transfer `WrapLoads` moved already fetches
    ``k + 1``.  The pointer out of an array is followed only for an element
    the caller did not mask, as the wrapped transfer is (`_guard_by_own_flag`).
    """

    def __init__(self,
                 context: Context,
                 instructions: List[AbstractInstruction],
                 level: str = 'l2'):
        super(PrefetchData, self).__init__(context, instructions)
        self._level = level
        self._names = set()
        self.hinted: List[str] = []
        self.rejected: List[tuple] = []

    def apply(self) -> None:
        for instr in self._instrs:
            if isinstance(instr, BatchLoop):
                self._hint(instr)

    @staticmethod
    def _producer(body, sym):
        for instr in body:
            if isinstance(instr, GetElementPtr) and any(d is sym for d in instr.defs()):
                return instr
        return None

    def _hint(self, loop: BatchLoop) -> None:
        if loop._mode in (LoopMode.SINGLE, LoopMode.LAUNCHCTRL):
            self.rejected.append((loop, 'no next element to name'))
            return
        body = list(loop.region)
        heads, tails, seen = [], [], set()
        for instr in body:
            if not isinstance(instr, (GlbToShrLoader, GlbToRegLoader)):
                continue
            if getattr(instr, '_wrapped', False):
                continue
            src = instr._src
            if id(src) in seen:
                continue
            addressing = getattr(src.obj, 'addressing', None)
            if addressing not in (Addressing.PTR_BASED, Addressing.STRIDED):
                continue
            producer = self._producer(body, src)
            if producer is None or isinstance(producer._batch_offset, str):
                self.rejected.append((src.name, 'not bound per element here'))
                continue
            seen.add(id(src))
            name = f'pf_{src.name}'
            while name in self._names:
                name = f'pf_{src.name}_{len(self._names)}'
            self._names.add(name)
            ahead = Symbol(name, src.stype, src.obj)
            ahead.data_view = src.data_view
            ptr = GetElementPtr(self._context, src=producer._src, dest=ahead,
                                include_extra_offset=producer._include_extra_offset,
                                batch_offset=1)
            hint = _Hint(self._context, ahead, 0, src.obj.storage_volume(),
                         level=self._level)
            ahead.add_user(hint)
            if producer.dereferences_the_batch():
                hint._guard_by_own_flag = True
            heads.append(ptr)
            tails.append(hint)
            self.hinted.append(src.name)
        if not heads:
            return
        loop.replace_region(0, heads + body + tails)
        loop.mark_unguarded(heads)
        loop.mark_unguarded_tail(tails)
