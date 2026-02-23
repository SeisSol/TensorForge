from typing import List
from .abstract import AbstractTransformer, Context, AbstractInstruction
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.instructions.memory.load import LoadInstruction, LoadWait, GlbToRegLoader, GlbToShrLoader
from tensorforge.backend.instructions.memory.store import StoreRegToReg, StoreShrMemToGlb
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.symbol import SymbolType, Symbol
from copy import deepcopy

class MultiBuffer(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction],
               shm, scopes):
    super(MultiBuffer, self).__init__(context, instructions)
    self._global_instrs = []
    self._shm = shm
    self._shm_symbol = scopes.get_symbol(self._shm)

  def apply(self) -> None:
    earlystop = False

    globalinstrs = []
    newinstrs = []

    epmap = {}

    for i, instr in enumerate(self._instrs):
        if isinstance(instr, GlbToRegLoader):
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
        elif isinstance(instr, GlbToShrLoader):
            newshrsym = Symbol(f'preload_{instr._dest.name}', SymbolType.SharedMem, instr._dest.obj)
            newshrsym.data_view = instr._dest.data_view
            newshrsym.num_threads = instr._dest.num_threads
            newshrsym.datatype = instr._dest.datatype
            newsym = Symbol(f'next_{instr._src.name}', instr._src.stype, instr._src.obj)
            newsym.data_view = instr._src.data_view
            newsym.num_threads = instr._src.num_threads
            newsym.datatype = instr._src.datatype
            newload1 = GlbToShrLoader(context=self._context, src=newsym, dest=newshrsym, shr_mem=self._shm_symbol, num_threads=instr._num_threads, permute=None)
            newload2 = GlbToShrLoader(context=self._context, src=newsym, dest=newshrsym, shr_mem=self._shm_symbol, num_threads=instr._num_threads, permute=None)
            globalinstrs += [GetElementPtr(self._context, epmap[instr._src.name], newsym, batch_offset=1)]
            globalinstrs += [newload1]
            newinstrs += [GetElementPtr(self._context, epmap[instr._src.name], newsym, batch_offset=1)]
            newinstrs += [LoadWait(newload1)]
            newinstrs += [GlbToShrLoader(context=self._context, src=newshrsym, dest=instr._dest, shr_mem=self._shm_symbol, num_threads=instr._num_threads, permute=None, no_memcpy=True)]
            newinstrs += [newload2]
        elif isinstance(instr, GetElementPtr) or isinstance(instr, RegisterAlloc) or isinstance(instr, LoadWait):
            newinstrs += [instr]

            # hack
            if isinstance(instr, GetElementPtr):
                epmap[instr._dest.name] = instr._src
        else:
            if earlystop:
                newinstrs += self._instrs[i:]
                break
            else:
                newinstrs += [instr]

    self._instrs = newinstrs
    self._global_instrs += globalinstrs
