# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Macro-level optimization stage, driven by the pass manager.

A registered pass list: each entry names what it consumes and produces, the
manager schedules and verifies, and a disabled pass is a switch.
"""

from typing import List

from tensorforge.common.context import Context
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.data_types import ShrMemObject

from tensorforge.backend.passmanager import PassManager, PassScope

from .manager import StreamContext, Transform
from .memmove import MoveLoads
from .pipeline import Pipeline
from .prefetch import PrefetchBatch, PrefetchData
from .wrap import WrapLoads


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
    self._pc = StreamContext(context,
                             self._instrs,
                             shr_mem=shr_mem,
                             num_threads=num_threads,
                             scopes=scopes,
                             global_ir=global_ir)
    self._manager = self._build_pipeline()

  # ------------------------------------------------------------------ #

  def _build_pipeline(self) -> PassManager:
    opts = self._user_options
    pm = PassManager(debug=opts.ir_debug)

    # Hoist loads away from their uses.  Scheduling within a straight-line
    # block: per region, or it would hoist a load across a loop boundary.
    pm.add(Transform(
        'MoveLoads',
        lambda pc, instrs: MoveLoads(pc.context, instrs,
                                     distance=opts.move_distance),
        scope=PassScope.PER_REGION,
        enabled=lambda pc: opts.enable_move_loads))

    # Prefetch across the back edge, placed the way MoveLoads would place a
    # transfer in the loop unrolled once.  Whole nest, like Pipeline: it
    # rewrites the loop's body and hands the loop its peel.  Runs after
    # MoveLoads, which splits the transfer from its wait -- this pass moves
    # the transfer and leaves the wait where the consumer is -- and before
    # Pipeline, so a body it has already wrapped is not also rotated.
    #
    # Off by default.
    pm.add(Transform(
        'WrapLoads',
        lambda pc, instrs: WrapLoads(pc.context, instrs,
                                     distance=opts.move_distance),
        enabled=lambda pc: opts.enable_wrap_loads))

    # Software pipelining: the address of the next element's transfer ahead of
    # the iteration that consumes it, and with `enable_multibuffer` the
    # transfer itself into a rotating buffer.  Rotation is implemented for a
    # depth of two; any other depth raises with the reason (see pipeline.py).
    # Whole nest: the peeled iteration has to land outside the loop.
    #
    # Off by default.
    pm.add(Transform(
        'Pipeline',
        lambda pc, instrs: Pipeline(
            pc.context, instrs,
            depth=opts.pipeline_depth,
            rotate_buffers=opts.enable_multibuffer),
        enabled=lambda pc: opts.enable_pipeline))

    # The cache hint for the next element's pointer.  After the two passes
    # above and not before: both rewrite the head of the region, and the head
    # is where this inserts.  It moves nothing itself, so nothing downstream
    # has to be told it ran.
    #
    # Off by default.
    pm.add(Transform(
        'PrefetchBatch',
        lambda pc, instrs: PrefetchBatch(
            pc.context, instrs,
            level=opts.prefetch_level),
        enabled=lambda pc: opts.enable_prefetch))

    # The next element's data, hinted at the tail of the body where
    # `WrapLoads` would issue its transfer -- the transfer stays.  After
    # `WrapLoads`, whose wrapped transfers need no hint, and after the pointer
    # hints, which it shares the head with.  Off by default.
    pm.add(Transform(
        'PrefetchData',
        lambda pc, instrs: PrefetchData(
            pc.context, instrs,
            level=opts.prefetch_level),
        enabled=lambda pc: opts.prefetch_data))

    return pm

  # ------------------------------------------------------------------ #

  def optimize(self):
    self._manager.run(self._pc)

  def get_instructions(self):
    return self._pc.instrs
