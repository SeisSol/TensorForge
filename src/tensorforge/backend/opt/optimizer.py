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
from .manager import (LegacyAnalysis, LegacyTransform, PassContext, PassManager,
                      PassScope)
from .mem_region_allocation import MemoryRegionAllocation
from .memmove import MoveLoads
from .pipeline import Pipeline
from .shr_mem_analyzer import ShrMemOpt
from .wrap import WrapLoads
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
    # Scheduling within a straight-line block: per region, or it would hoist a
    # load across a loop boundary.
    pm.add(LegacyTransform(
        'MoveLoads',
        lambda pc, instrs: MoveLoads(pc.context, instrs),
        scope=PassScope.PER_REGION))

    # Slot-granular prefetch.  Whole nest, like Pipeline: the peeled transfer
    # lands outside the loop.  Runs after MoveLoads, which splits the transfer
    # from its wait -- this pass moves the transfer and leaves the wait where
    # the consumer is -- and before Pipeline, so a body it has already wrapped
    # is not also rotated.
    #
    # Off by default.
    pm.add(LegacyTransform(
        'WrapLoads',
        lambda pc, instrs: WrapLoads(
            pc.context, instrs,
            distance=getattr(opts, 'wrap_distance', 1)),
        enabled=lambda pc: getattr(opts, 'enable_wrap_loads', False)))

    # Software pipelining, one pass where there used to be two (MultiBuffer
    # and PtrPipe were the same transform at two granularities).  Whole nest:
    # the peeled iteration has to land outside the loop.
    #
    # Off by default. The pointer-advance half is implemented; buffer rotation
    # raises with the precise reason (see pipeline.py) rather than emitting
    # something plausible.
    pm.add(LegacyTransform(
        'Pipeline',
        lambda pc, instrs: Pipeline(
            pc.context, instrs,
            depth=getattr(opts, 'pipeline_depth', 2),
            rotate_buffers=getattr(opts, 'enable_multibuffer', False)),
        enabled=lambda pc: getattr(opts, 'enable_pipeline', False)))

    # Whole nest: a value carried across the loop's back edge is only visible
    # to a fixed point over the region structure.
    #
    # NOTE on the preloaded globals: the shared-memory symbols defined by the
    # section prologue are allocated by ShrMemObject.alloc_global -- a separate
    # bump allocator in a separate arena -- so they are deliberately *not* fed
    # to the region allocator, which would give them a second offset.  Unifying
    # the two allocators is its own change.
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
    # allocation it depends on.  Per region: "the previous write to this buffer"
    # must not be read across a loop boundary, where the previous write is the
    # previous *iteration*.
    pm.add(LegacyTransform(
        'SyncThreadsOpt',
        lambda pc, instrs: SyncThreadsOpt(pc.context, instrs,
                                          pc.get('regions'), pc.num_threads),
        preserves=('live_map', 'regions'),
        scope=PassScope.PER_REGION,
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

    # whole nest: temp_shmem() of a BatchLoop is the max over its body
    tmp_overhead = 0
    for instr in pc.stream:
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
