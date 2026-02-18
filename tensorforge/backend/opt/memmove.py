from typing import List
from .abstract import AbstractTransformer, Context, AbstractInstruction
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.instructions.memory.load import LoadInstruction, LoadWait
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.symbol import SymbolType

class MoveLoads(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(MoveLoads, self).__init__(context, instructions)

  def apply(self) -> None:
    instrsOut = []
    stored = []
    for instr in reversed(self._instrs):
        if isinstance(instr, LoadInstruction):
            instrsOut += [LoadWait(instr)]
            while len(stored) > 0:
                delayed = stored.pop()
                instrsOut += [delayed]
            stored.append(instr)
        else:
            if not isinstance(instr, ComputeInstruction):
                while len(stored) > 0:
                    delayed = stored.pop()
                    instrsOut += [delayed]
            instrsOut += [instr]
    instrsOut += stored[::-1]

    self._instrs = instrsOut[::-1]
