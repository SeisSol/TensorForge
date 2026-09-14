# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import List
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite
from tensorforge.backend.instructions.memory.load import LoadWait
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
               loop_body: bool = False,
               wraps_reads: bool = False):

    super(SyncThreadsOpt, self).__init__(context, instructions)
    self._regions = regions
    # A buffer placed on its own spans every region it covers (`Region`), so
    # a symbol maps to a list.  With colors the list has one entry.
    self._region_ids = {}
    for region_id, region in enumerate(regions):
      for symbol in region:
        self._region_ids.setdefault(id(symbol), []).append(region_id)
    self._num_threads = num_threads
    # Whether this block runs again after its last instruction.  A loop body
    # does, so a shared write near its tail is read near its head one
    # iteration later -- see `_insert_sync_before_use`.
    self._loop_body = loop_body
    # Whether the other direction around the back edge is this pass's too: a
    # read near the tail against a write near the head.  The generator appends
    # a barrier to every batch loop for it; a merged run's `VariantLoop` has
    # none, and its body reused one shared window for the staged operand at
    # the head and the product read at the tail.
    self._wraps_reads = wraps_reads

  def apply(self) -> None:
    self._remove_previous_sync_instructions()
    self._insert_sync_before_use()
    self._insert_sync_after_use()

  def _insert_sync_before_use(self):
    # Around the back edge, for a loop body: the writes still unfenced when
    # the body ends are the ones its head reads next iteration.  Nothing put
    # them there before `WrapLoads` learned shared memory -- a transfer for
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
      # An asynchronous transfer lands at its wait, and the wait makes it
      # visible to the lane that issued it and no other.  A barrier between
      # issue and wait -- the one some other buffer's consumer needed --
      # fences nothing of it, so the wait arms the write again.  Left to the
      # issue, every staged operator after the first was read across lanes
      # with no barrier behind its wait: racecheck on the poroelastic time
      # derivative, and 8 % off once the merged run shifted the timing.
      if isinstance(instr, LoadWait):
        awaited = instr.awaited()
        if (isinstance(awaited, AbstractShrMemWrite)
            and getattr(awaited, 'lands_at_wait', lambda: False)()):
          writes.append((awaited.get_dest(), False))

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
    # Around the back edge where nobody else fences it: the regions still read
    # and unfenced when the body ends are the ones a write at its head
    # overwrites next iteration.  Scanned once for that, then again from it.
    carried = self._scan_after_use(None)[1] if self._wraps_reads else None
    selected, _ = self._scan_after_use(carried)
    self._insert_sync_instrs(selected)

  def _scan_after_use(self, flags):
    """`(shared writes needing a barrier before them, regions read and not yet
    fenced at the end)`."""
    selected = []
    flags = list(flags) if flags is not None else [False] * len(self._regions)
    for index, instr in enumerate(self._instrs):
      if isinstance(instr, ComputeInstruction):
        for src in instr.get_operands():
          if src.stype == SymbolType.SharedMem:
            for region_id in self._get_region_ids(src):
              flags[region_id] = True

      if isinstance(instr, SyncThreads):
        flags = [False] * len(self._regions)

      if isinstance(instr, AbstractShrMemWrite):
        dest = instr.get_dest()
        if any(flags[region_id] for region_id in self._get_region_ids(dest)):
          selected.append(instr)
          flags = [False] * len(self._regions)

    return selected, flags

  def _insert_sync_instrs(self, selected):
    for instr in selected:
      index = self._instrs.index(instr)
      self._instrs.insert(index, SyncThreads(self._context, self._num_threads))

  def _get_region_ids(self, symbol):
    return self._region_ids.get(id(symbol), ())

  def _remove_previous_sync_instructions(self):
    self._instrs = [item for item in self._instrs if not isinstance(item, SyncThreads)]
