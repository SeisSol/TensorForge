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
    writer.barrier(Uniformity.MULT)

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
