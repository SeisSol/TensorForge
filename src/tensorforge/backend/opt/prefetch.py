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

Known limitation, deliberately left: the head of the region is *inside* the
per-element flag guard, so a masked element issues no hint and the element
after it pays the dependent load in full. Lifting it out is safe -- the
address is arithmetic on a kernel argument and a clamped index, and nothing
here dereferences the pointer it asks for -- but `BatchLoop` lifts only a
prefix of the region and does so under `enable_wrap_loads`, which is a
different switch answering a different question. A hint that is skipped is a
hint that was not taken; the guard costs nothing else.
"""

from typing import List

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.instructions.prefetch import PrefetchBatchPointer
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
