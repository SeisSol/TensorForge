# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.context import Context
from .abstract_instruction import AbstractInstruction, BarrierScope
from tensorforge.backend.pir.core import Uniformity

class SyncThreads(AbstractInstruction):
  def __init__(self, context: Context, num_threads_per_mult):
    super().__init__(context)
    self._num_threads = num_threads_per_mult
    self._is_ready = True

  def barrier_scope(self) -> BarrierScope:
    # same predicate __str__ uses to pick sync_block vs sync_simd -- resolved
    # once, here, so that passes and verify() can see the scope
    #
    # Under an explicit vector there is nothing to synchronise.  The wave is
    # not a hardware sub-group whose width the multiplication has to fit
    # inside; it *is* the work-item, and `num_threads` is the length of its
    # registers.  A 32-thread multiplication is a 32-wide vector held by one
    # work-item, executed in order, with no second party to wait for.
    #
    # This is the structural difference the whole path was chosen for.  On
    # PVC's 16-wide sub-group the SPMD lowering turns any wider multiplication
    # into a GROUP barrier, and a `BatchLoop` is only MULT-uniform, so
    # `verify()` rejects it -- correctly, it would deadlock.  16 of the 54
    # cases in the corpus fail there and none of them for a reason that has
    # anything to do with the operator.
    lex = self._vm.get_lexic()
    if getattr(lex, 'simd_mode', False):
      return BarrierScope.SIMD
    if self._num_threads > self._vm.get_hw_descr().vec_unit_length:
      return BarrierScope.GROUP
    return BarrierScope.SIMD

  def accesses(self):
    return ()

  def gen_ir(self, writer):
    """Emit the scope `barrier_scope` decided, not a second opinion.

    These two used to disagree.  `barrier_scope` weighs the thread count
    against the wave -- which is the whole reason it takes one -- and answers
    `GROUP` for a multiplication that does not fit in a wave; `gen_ir` asked
    for `MULT` regardless, and the emitter turned that into `__syncwarp()`.
    So `verify` would refuse the construct while the emitter, had it run,
    would have synchronised a warp where a block was needed: a barrier that
    reaches part of the threads it was asked to reach.

    Nothing in the corpus is wide enough to have shown it -- `lanes.py` clamps
    to `vec_unit_length` for every descriptor but `ElementwiseDescr`, and no
    elementwise case reaches the cap.  One answer, taken from the resolver
    that has the numbers, is what keeps it from mattering later.
    """
    if self.barrier_scope() is BarrierScope.SIMD:
      # A wave, and it is a wave whatever the multiplication is: these threads
      # are in lockstep, so the barrier is about ordering memory, not about
      # arrival.
      writer.barrier(Uniformity.MULT)
      return
    # Wider than a wave.  `BLOCK` is what may legally be claimed -- the width
    # rides along so a vendor with a sub-block rendezvous can narrow it.
    writer.barrier(Uniformity.BLOCK, threads=self._num_threads)

  def __str__(self) -> str:
    return self.barrier_scope()

  def gen_mask_threads(self, num_threads) -> str:
    return ''

class SyncBlock(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._is_ready = True

  def barrier_scope(self) -> BarrierScope:
    return BarrierScope.GROUP

  def accesses(self):
    return ()

  def gen_ir(self, writer):
    writer.barrier(Uniformity.BLOCK)

  def __str__(self) -> str:
    return self.barrier_scope()

  def gen_mask_threads(self, num_threads) -> str:
    return ''

class SyncGrid(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._is_ready = True

  def barrier_scope(self) -> BarrierScope:
    return BarrierScope.GRID

  def accesses(self):
    return ()

  def gen_ir(self, writer):
    writer.barrier(Uniformity.GRID)

  def __str__(self) -> str:
    return self.barrier_scope()

  def gen_mask_threads(self, num_threads) -> str:
    return ''
