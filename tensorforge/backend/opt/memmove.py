from typing import List
from .abstract import AbstractTransformer, Context, AbstractInstruction
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.instructions.memory.load import LoadInstruction, LoadWait
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.symbol import SymbolType

class MoveLoads(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(MoveLoads, self).__init__(context, instructions)

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
            if isinstance(instr, GetElementPtr):
                clear_stored(instrsOut)
            instrsOut += [instr]
    clear_stored(instrsOut)

    self._instrs = instrsOut[::-1]
