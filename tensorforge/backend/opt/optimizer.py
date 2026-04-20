from typing import List, Dict, Set
from tensorforge.common.context import Context
from tensorforge.backend.symbol import Symbol
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.data_types import ShrMemObject
from .liveness import LivenessAnalysis
from .mem_region_allocation import MemoryRegionAllocation, Region
from .shr_mem_analyzer import ShrMemOpt
from .sync_block import SyncThreadsOpt
from .remove_redundancy import RemoveRedundancyOpt
from .memmove import MoveLoads
from .multibuffer import MultiBuffer
from .ptrpipe import PtrPipe

class OptimizationStage:
  def __init__(self,
               context: Context,
               shr_mem: ShrMemObject,
               instructions: List[AbstractInstruction],
               num_threads: int,
               scopes):
    self._context = context
    self._shr_mem: ShrMemObject = shr_mem
    self._instrs: List[AbstractInstruction] = instructions
    self._global_instrs: List[AbstractInstruction] = []
    self._num_instrs: int = len(instructions)
    self._user_options = context.get_user_options()
    self._num_threads = num_threads
    self._scopes = scopes

  def optimize(self):
    opt = MoveLoads(self._context, self._instrs)
    opt.apply()
    self._instrs = opt.get_instructions()

    # opt = MultiBuffer(self._context, self._instrs, self._shr_mem, self._scopes)
    # opt.apply()
    # self._instrs = opt.get_instructions()
    # self._global_instrs = opt._global_instrs

    # opt = PtrPipe(self._context, self._instrs)
    # opt.apply()
    # self._instrs = opt.get_instructions()
    # self._global_instrs += opt._global_instrs

    opt = LivenessAnalysis(self._context, self._global_instrs + self._instrs)
    opt.apply()
    live_map: Dict[int, Set[Symbol]] = opt.get_live_map()

    opt = MemoryRegionAllocation(self._context, live_map)
    opt.apply()
    regions: List[Region] = opt.get_regions()

    overhead = self._num_threads % self._context.get_vm().get_hw_descr().shmem_banks
    overhead //= 4
    overhead *= 4

    tmp_overhead = 0
    for instr in self._instrs:
      tmp_overhead = max(tmp_overhead, instr.temp_shmem())

    opt = ShrMemOpt(context=self._context,
                    shr_mem_obj=self._shr_mem,
                    regions=regions,
                    live_map=live_map,
                    thread_overhead=overhead,
                    tmp_overhead=tmp_overhead)
    opt.apply()

    if self._user_options.enable_sync_block_opt:
      opt = SyncThreadsOpt(self._context, self._instrs, regions, self._num_threads)
      opt.apply()
      self._instrs = opt.get_instructions()

    opt = RemoveRedundancyOpt(self._context, self._instrs)
    # opt.apply()
    self._instrs = opt.get_instructions()

  def get_instructions(self):
    return self._instrs

  def get_global_instructions(self):
    return self._global_instrs
