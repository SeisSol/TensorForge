from typing import List
from .abstract import AbstractTransformer, Context, AbstractInstruction
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.instructions.memory.load import LoadInstruction, LoadWait, GlbToRegLoader
from tensorforge.backend.instructions.memory.store import StoreRegToReg
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.symbol import SymbolType, Symbol
from copy import deepcopy

class PtrPipe(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(PtrPipe, self).__init__(context, instructions)
    self._global_instrs = []

  def apply(self) -> None:
    globalinstrs = []
    newinstrs = []

    for i, instr in enumerate(self._instrs):
      if isinstance(instr, GetElementPtr):
        newdest = Symbol(f'preload{instr._batch_offset}_{instr._src.name}', instr._src.stype, instr._src.obj)
        newgep = GetElementPtr(self._context, instr._src, newdest, batch_offset=instr._batch_offset + 1, update_dest=instr._dest)
        newinstrs += [newgep]
        newgepstart = GetElementPtr(self._context, instr._src, newdest, batch_offset=instr._batch_offset + 1, pipeline=True)
        globalinstrs += [newgepstart]
      else:
        newinstrs += [instr]

    self._instrs = newinstrs
    self._global_instrs = globalinstrs
