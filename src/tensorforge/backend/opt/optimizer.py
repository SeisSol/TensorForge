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
from .prefetch import PrefetchBatch, PrefetchData


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

    # The cache hint for the next element's pointer, at the head of the
    # region.  It moves nothing itself, so nothing downstream has to be told
    # it ran.
    #
    # Off by default.
    pm.add(Transform(
        'PrefetchBatch',
        lambda pc, instrs: PrefetchBatch(
            pc.context, instrs,
            level=opts.prefetch_level),
        enabled=lambda pc: opts.enable_prefetch))

    # The next element's data, hinted at the tail of the body, where
    # `enable_wrap_loads` would issue its transfer -- the transfer stays.
    # After the pointer hints, which it shares the head with.  Off by
    # default.
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
