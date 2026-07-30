# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Macro-level optimisation stage, driven by the pass manager.

The pipeline used to be a hardcoded sequence with the order implicit, three
passes commented out, one constructed but never applied, and analysis results
handed on as bare dictionaries.  It is now a registered pass list: each entry
names what it consumes and produces, the manager schedules and verifies, and
a disabled pass is a flag rather than a comment.
"""

from typing import List

from tensorforge.common.context import Context
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.data_types import ShrMemObject

from .liveness import LivenessAnalysis
from .manager import LegacyAnalysis, LegacyTransform, PassContext, PassManager
from .mem_region_allocation import MemoryRegionAllocation
from .memmove import MoveLoads
from .multibuffer import MultiBuffer
from .ptrpipe import PtrPipe
from .shr_mem_analyzer import ShrMemOpt
from .sync_block import SyncThreadsOpt


class OptimizationStage:
  def __init__(self,
               context: Context,
               shr_mem: ShrMemObject,
               instructions: List[AbstractInstruction],
               num_threads: int,
               scopes,
               global_ir: List[AbstractInstruction] = None):
    self._context = context
    self._instrs: List[AbstractInstruction] = list(instructions)
    self._user_options = context.get_user_options()
    self._pc = PassContext(context,
                           self._instrs,
                           shr_mem=shr_mem,
                           num_threads=num_threads,
                           scopes=scopes,
                           global_ir=global_ir)
    self._manager = self._build_pipeline()

  # ------------------------------------------------------------------ #

  def _build_pipeline(self) -> PassManager:
    opts = self._user_options
    pm = PassManager()

    # Hoist loads away from their uses.  Must run before liveness, since it
    # changes the distance between a definition and its consumers.
    pm.add(LegacyTransform(
        'MoveLoads',
        lambda pc: MoveLoads(pc.context, pc.instrs)))

    # Software pipelining.  Both are disabled: they publish a prologue
    # through `_global_instrs`, i.e. a second list the rest of the pipeline
    # does not index, so definition and use end up in different streams.
    # Reviving them needs a real loop op first -- see multibuffer.py.
    pm.add(LegacyTransform(
        'MultiBuffer',
        lambda pc: MultiBuffer(pc.context, pc.instrs, pc.shr_mem, pc.scopes),
        collects_global=True,
        enabled=lambda pc: getattr(opts, 'enable_multibuffer', False)))
    pm.add(LegacyTransform(
        'PtrPipe',
        lambda pc: PtrPipe(pc.context, pc.instrs),
        collects_global=True,
        enabled=lambda pc: getattr(opts, 'enable_ptrpipe', False)))

    # NOTE on `predefined`: Section.global_ir also defines shared-memory
    # symbols (the preloaded global operators), but those are allocated by
    # ShrMemObject.alloc_global -- a separate bump allocator in a separate
    # arena.  Feeding them in here would hand them to the region allocator
    # as well and they would get two offsets.  Unifying the two allocators
    # is a deliberate change, not a side effect of this one.
    pm.add(LegacyAnalysis(
        'LivenessAnalysis',
        lambda pc: LivenessAnalysis(pc.context, pc.local_stream),
        lambda opt: opt.get_live_map(),
        provides='live_map'))

    pm.add(LegacyAnalysis(
        'MemoryRegionAllocation',
        lambda pc: MemoryRegionAllocation(pc.context, pc.get('live_map')),
        lambda opt: opt.get_regions(),
        provides='regions',
        requires=('live_map',)))

    pm.add(_AssignShrMemOffsets())

    # Barrier insertion keys on region membership, so it must follow the
    # allocation it depends on.
    pm.add(LegacyTransform(
        'SyncThreadsOpt',
        lambda pc: SyncThreadsOpt(pc.context, pc.instrs, pc.get('regions'),
                                  pc.num_threads),
        preserves=('live_map', 'regions'),
        enabled=lambda pc: opts.enable_sync_block_opt))

    # RemoveRedundancyOpt is deliberately absent: its _remove_bottom_instrs
    # pops every instruction but one when the stream contains no
    # StoreRegToGlb, which is why the call was commented out.  Dead-code
    # elimination over defs()/uses() subsumes it.

    return pm

  # ------------------------------------------------------------------ #

  def optimize(self):
    self._manager.run(self._pc)

  def get_instructions(self):
    return self._pc.instrs

  def get_global_instructions(self):
    return self._pc.produced

  def get_timings(self):
    return self._manager.timings


class _AssignShrMemOffsets(LegacyTransform):
  """ShrMemOpt: turn regions into byte offsets and size the arena.

  A transform rather than an analysis -- it mutates the instructions' offsets
  and flips `is_ready` -- but it neither adds nor removes instructions, so the
  index-keyed `live_map` and the `regions` survive it.
  """

  name = 'ShrMemOpt'
  requires = ('live_map', 'regions')
  preserves = ('live_map', 'regions')
  is_transform = True

  def __init__(self):
    pass

  def enabled(self, pc: PassContext) -> bool:
    return True

  def run(self, pc: PassContext) -> None:
    fp_size = pc.context.fp_type.size()
    alignment = 16 // fp_size
    overhead = pc.num_threads % pc.context.get_vm().get_hw_descr().shmem_banks
    overhead //= alignment
    overhead *= alignment

    tmp_overhead = 0
    for instr in pc.instrs:
      tmp_overhead = max(tmp_overhead, instr.temp_shmem())

    opt = ShrMemOpt(context=pc.context,
                    shr_mem_obj=pc.shr_mem,
                    regions=pc.get('regions'),
                    live_map=pc.get('live_map'),
                    thread_overhead=overhead,
                    tmp_overhead=tmp_overhead)
    opt.apply()
    # from here on every shared-memory writer has an offset, so verify() can
    # check buffer aliasing (readiness still needs the thread-block policy)
    pc.extra['offsets_assigned'] = True
