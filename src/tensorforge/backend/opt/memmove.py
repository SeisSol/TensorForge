# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import List
from .abstract import AbstractTransformer, Context, AbstractInstruction
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.instructions.memory.load import LoadInstruction, LoadWait
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.abstract_instruction import BarrierScope
from tensorforge.backend.symbol import SymbolType

class MoveLoads(AbstractTransformer):
  """Issue loads early and wait where the value is needed.

  A load is split into the transfer and a `LoadWait`: the transfer walks up
  the stream to hide its latency, the wait stays where the consumer is.  That
  is only sound while nothing between the two positions writes what the load
  reads --- otherwise the transfer picks the value up from *before* that write
  and the consumer silently gets a stale one.

  The pass used to move every load as far as it could go, stopping only at
  another load or a `GetElementPtr`.  In a chain that accumulates into a
  tensor and then reads it back --- `d += ...` several times, then
  `out += d * c`, which is the shape of every ADER derivative kernel --- it
  hoisted the read of `d` above the last accumulation's store, so the final
  term was missing from the result and nothing else about the kernel looked
  wrong.

  `defs()`/`uses()` already say what each instruction touches, so the test is
  the ordinary dependence one; a load stops at the first instruction it
  conflicts with.
  """

  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(MoveLoads, self).__init__(context, instructions)

  @staticmethod
  def _symbols(syms):
    return {id(s) for s in syms if s is not None}

  @staticmethod
  def _touches_shared(load) -> bool:
    return any(getattr(s, 'stype', None) is SymbolType.SharedMem
               for s in tuple(load.defs()) + tuple(load.uses()) if s is not None)

  @classmethod
  def _conflicts(cls, load, instr) -> bool:
    """May `load` not be hoisted above `instr`?

    Conservative on purpose: an instruction that does not describe its
    dataflow conflicts with everything, and all three dependence kinds are
    barriers.  This is a latency optimisation --- giving up on one load costs
    a few cycles, getting it wrong costs the answer.
    """
    if instr.barrier_scope() is not BarrierScope.NONE:
      # A barrier is not a memory operation --- `accesses()` is empty --- it
      # orders what *other* threads did.  Only a load that reads or writes
      # shared memory can see that, so a global-to-register transfer crosses
      # it freely, which is what the pass is for.
      return cls._touches_shared(load)
    if not instr.describes_dataflow() or not load.describes_dataflow():
      return True
    reads, writes = cls._symbols(load.uses()), cls._symbols(load.defs())
    idefs, iuses = cls._symbols(instr.defs()), cls._symbols(instr.uses())
    return bool(reads & idefs           # read-after-write: the point of this
                or writes & idefs       # write-after-write
                or writes & iuses)      # write-after-read

  def apply(self) -> None:
    instrsOut = []
    stored = []
    def clear_stored(instrsOut):
        while len(stored) > 0:
            delayed = stored.pop(0)
            instrsOut += [delayed]
    for instr in reversed(self._instrs):
        if isinstance(instr, LoadInstruction):
            instrsOut += [LoadWait(instr)]
            clear_stored(instrsOut)
            stored.append(instr)
        elif isinstance(instr, RegisterAlloc):
            for st in stored:
                if st._dest is instr._dest:
                    stored.append(instr)
                    break
            else:
                instrsOut += [instr]
        else:
            # `stored` holds loads whose transfer is still travelling up the
            # stream.  Any of them that this instruction feeds --- or whose
            # destination it touches --- has gone as far as it may.  Releasing
            # them together keeps their relative order without a second sort.
            if isinstance(instr, GetElementPtr) or any(
                    self._conflicts(st, instr)
                    for st in stored if isinstance(st, LoadInstruction)):
                clear_stored(instrsOut)
            instrsOut += [instr]
    clear_stored(instrsOut)

    self._instrs = instrsOut[::-1]
