import os
from typing import List, Dict, Set
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
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
from .inspect import dump, verify, format_diagnostics

class OptimizationStage:
  def __init__(self,
               context: Context,
               shr_mem: ShrMemObject,
               instructions: List[AbstractInstruction],
               num_threads: int,
               scopes,
               global_ir: List[AbstractInstruction] = None):
    self._context = context
    self._shr_mem: ShrMemObject = shr_mem
    self._instrs: List[AbstractInstruction] = instructions
    self._global_instrs: List[AbstractInstruction] = []
    self._num_instrs: int = len(instructions)
    self._user_options = context.get_user_options()
    self._num_threads = num_threads
    self._scopes = scopes
    # Section.global_ir is built *before* this stage and never passed through
    # it -- that is exactly why preloaded shared-memory buffers never enter
    # LivenessAnalysis.  Until that split is removed, at least make their
    # definitions visible to verify().
    self._section_global_ir: List[AbstractInstruction] = list(global_ir or [])
    # TF_IR_DEBUG=verify -> run verify() after every pass
    # TF_IR_DEBUG=dump   -> also print the stream after every pass
    self._debug = os.environ.get('TF_IR_DEBUG', '')

  def _check(self, stage: str, allocated: bool = True) -> None:
    """Verify (and optionally print) the stream after a pass.

    Running this *between* passes is the point: a diagnostic reported here
    names the pass that introduced it, whereas the emitter's is_ready()
    check only fires at the very end, after the evidence is gone.
    """
    if not self._debug:
      return
    instrs = self._global_instrs + self._instrs
    if 'dump' in self._debug:
      print(dump(instrs, title=f'after {stage}'))
    predefined = list(self._scopes.get_global_scope().values())
    for instr in self._section_global_ir:
      predefined += list(instr.defs())
    diags = verify(instrs,
                   inside_batch_loop=False,
                   predefined=predefined,
                   allocated=allocated,
                   backend=self._context.get_vm().get_lexic()._backend)
    errors = [d for d in diags if d.severity == 'error']
    if errors:
      raise GenerationError(f'macro-ir invalid after {stage}:\n'
                            + format_diagnostics(diags))

  def optimize(self):
    self._check('build', allocated=False)

    opt = MoveLoads(self._context, self._instrs)
    opt.apply()
    self._instrs = opt.get_instructions()
    self._check('MoveLoads', allocated=False)

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

    alignment = 16 // self._context.fp_type.size()
    overhead = self._num_threads % self._context.get_vm().get_hw_descr().shmem_banks
    overhead //= alignment
    overhead *= alignment

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

    self._check('ShrMemOpt')

    if self._user_options.enable_sync_block_opt:
      opt = SyncThreadsOpt(self._context, self._instrs, regions, self._num_threads)
      opt.apply()
      self._instrs = opt.get_instructions()
      self._check('SyncThreadsOpt')

    opt = RemoveRedundancyOpt(self._context, self._instrs)
    # opt.apply()
    self._instrs = opt.get_instructions()

  def get_instructions(self):
    return self._instrs

  def get_global_instructions(self):
    return self._global_instrs
