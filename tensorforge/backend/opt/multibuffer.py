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

class MultiBuffer(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(MultiBuffer, self).__init__(context, instructions)
    self._global_instrs = []

  def apply(self) -> None:
    globalinstrs = []
    newinstrs = []

    epmap = {}

    for i, instr in enumerate(self._instrs):
        if isinstance(instr, LoadInstruction) and not isinstance(instr, LoadWait):
            newregs = deepcopy(instr._dest.obj)
            newregs.name = f'preload_{newregs.name}'
            newregsym = Symbol(newregs.name, SymbolType.Register, newregs)
            newregsym.data_view = instr._dest.data_view
            newregsym.num_threads = instr._dest.num_threads
            newregsym.datatype = instr._dest.datatype
            newsym = Symbol(f'next_{instr._src.name}', instr._src.stype, instr._src.obj)
            newsym.data_view = instr._src.data_view
            newsym.num_threads = instr._src.num_threads
            newsym.datatype = instr._src.datatype
            newload1 = GlbToRegLoader(self._context, newsym, newregsym, instr._num_threads, instr._linearize)
            newload2 = GlbToRegLoader(self._context, newsym, newregsym, instr._num_threads, instr._linearize)
            globalinstrs += [GetElementPtr(self._context, epmap[instr._src.name], newsym, batch_offset=1)]
            globalinstrs += [RegisterAlloc(self._context, newregsym, 0, 0.0)]
            globalinstrs += [newload1]
            newinstrs += [GetElementPtr(self._context, epmap[instr._src.name], newsym, batch_offset=1)]
            newinstrs += [LoadWait(newload1)]
            newinstrs += [StoreRegToReg(self._context, newregsym, instr._dest, instr._num_threads)]
            newinstrs += [newload2]
        elif isinstance(instr, GetElementPtr) or isinstance(instr, RegisterAlloc):
            newinstrs += [instr]

            # hack
            if isinstance(instr, GetElementPtr):
                epmap[instr._dest.name] = instr._src
        else:
            self._global_instrs += globalinstrs
            self._instrs = newinstrs + self._instrs[i:]
            break
