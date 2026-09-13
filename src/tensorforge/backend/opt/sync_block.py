# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import List
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.backend.symbol import SymbolType
from .abstract import AbstractTransformer, Context, AbstractInstruction
from .mem_region_allocation import Region


def _one_lane_stores(symbol) -> bool:
  """Whether a computation's store of `symbol` is one lane's.

  A destination without axes has no lane axis to spread over: every lane holds
  the value and the owner alone writes it to memory.  In a register there is
  no store at all, and nothing to fence.
  """
  obj = getattr(symbol, 'obj', None)
  rank = len(getattr(obj, 'shape', ()) or ())
  return rank == 0 and symbol.stype in (SymbolType.Global, SymbolType.Batch,
                                        SymbolType.SharedMem)


class SyncThreadsOpt(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction],
               regions: List[Region],
               num_threads: int,
               loop_body: bool = False):

    super(SyncThreadsOpt, self).__init__(context, instructions)
    self._regions = regions
    self._num_threads = num_threads
    # Whether this block runs again after its last instruction.  A loop body
    # does, so a shared write near its tail is read near its head one
    # iteration later -- see `_insert_sync_before_use`.
    self._loop_body = loop_body

  def apply(self) -> None:
    self._remove_previous_sync_instructions()
    self._insert_sync_before_use()
    self._insert_sync_after_use()

  def _insert_sync_before_use(self):
    # Around the back edge, for a loop body: the writes still unfenced when
    # the body ends are the ones its head reads next iteration.  Nothing put
    # them there before `WrapLoads` learnt shared memory -- a transfer for
    # element k + 1 issued after the last read of its buffer, and read at the
    # top of the next iteration, behind its wait.  Scanning once to find what
    # is carried and once more starting from it puts the barrier where a
    # straight-line scan of the unrolled loop would.
    #
    # One direction only.  The other -- a read at the tail against a write at
    # the head -- is what the barrier the generator appends to every
    # persistent loop is for, and seeding it here too would add a second one
    # per iteration to every kernel.
    carried = self._scan_before_use([])[1] if self._loop_body else []
    selected, _ = self._scan_before_use(carried)
    for instr, handoff in selected:
      index = self._instrs.index(instr)
      self._instrs.insert(index, SyncThreads(self._context, self._num_threads,
                                             handoff=handoff))

  def _scan_before_use(self, writes):
    """`(computes needing a barrier before them, writes unfenced at the end)`.

    Two kinds of write are fenced before whatever reads them next: a transfer
    into shared memory, and a value without axes a computation stores in
    memory -- one number, stored by one lane (the owner) and read back by all
    of them.  The second went unfenced, and with it the guard over a condition
    the kernel had just reduced (`X1 = all(B >= C)`, then `if (X1)`): the
    lanes that read before the owner's store took the other branch, and one
    element mixed both.  A guard reads its condition through `uses`, not as an
    operand, so it is asked that way.

    Both lists pair each entry with whether a one-lane store is what it
    fences: the barrier then also owes the compiler a fence where the
    rendezvous costs no instruction (`Lexic.handoff_fence`).
    """
    selected = []
    writes = list(writes)
    for instr in self._instrs:
      if isinstance(instr, AbstractShrMemWrite):
        writes.append((instr.get_dest(), False))

      if isinstance(instr, ComputeInstruction):
        reads = instr.get_operands()
      elif hasattr(instr, 'region') and hasattr(instr, 'uses'):
        reads = instr.uses()
      else:
        reads = ()
      hits = [handoff for sym, handoff in writes if sym in reads]
      if hits:
        selected.append((instr, any(hits)))
        writes = []

      if isinstance(instr, ComputeInstruction):
        writes.extend((sym, True) for sym in instr.defs()
                      if _one_lane_stores(sym))
    return selected, writes

  def _insert_sync_after_use(self):
    selected = []
    flags = [False] * len(self._regions)
    for index, instr in enumerate(self._instrs):
      if isinstance(instr, ComputeInstruction):
        for src in instr.get_operands():
          if src.stype == SymbolType.SharedMem and self._get_region_id(src) is not None:
            flags[self._get_region_id(src)] = True

      if isinstance(instr, SyncThreads):
        flags = [False] * len(self._regions)

      if isinstance(instr, AbstractShrMemWrite):
        dest = instr.get_dest()
        if self._get_region_id(dest) is not None:
          if flags[self._get_region_id(dest)]:
            selected.append(instr)
            flags = [False] * len(self._regions)

    self._insert_sync_instrs(selected)

  def _insert_sync_instrs(self, selected):
    for instr in selected:
      index = self._instrs.index(instr)
      self._instrs.insert(index, SyncThreads(self._context, self._num_threads))

  def _get_region_id(self, symbol):
    for region_id, region in enumerate(self._regions):
      if symbol in region:
        return region_id

  def _remove_previous_sync_instructions(self):
    self._instrs = [item for item in self._instrs if not isinstance(item, SyncThreads)]
