# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.context import Context
from .abstract_instruction import AbstractInstruction
from tensorforge.backend.pir.core import Participants, Uniformity

class SyncThreads(AbstractInstruction):
  """A rendezvous of the threads of one multiplication."""

  def __init__(self, context: Context, num_threads_per_mult):
    super().__init__(context)
    self._num_threads = num_threads_per_mult
    self._is_ready = True

  def _wave(self) -> int:
    return self._vm.get_hw_descr().vec_unit_length

  def participants(self) -> Participants:
    """The narrowest set of threads this barrier can be spelled over.

    Narrowest, because a barrier that reaches further than it has to is a
    barrier the block cannot pack multiplications around: whatever it covers
    has to arrive, and rows that have to arrive together cannot run the body a
    different number of times.  So the width chosen here is what
    `AbstractThreadBlockPolicy` sizes a block from, and asking for less is
    occupancy.

    Under an explicit vector there is nothing to synchronise.  The wave is not
    a hardware sub-group whose width the multiplication has to fit inside; it
    *is* the work-item, and `num_threads` is the length of its registers.  A
    32-thread multiplication is a 32-wide vector held by one work-item,
    executed in order, with no second party to wait for.
    """
    lex = self._vm.get_lexic()
    if getattr(lex, 'simd_mode', False):
      return Participants.MULT
    n, wave = self._num_threads, self._wave()
    if n == wave:
      # The multiplication *is* the wave, so the wave barrier meets exactly
      # the threads that have to meet and nothing narrower exists.
      return Participants.WAVE
    if lex.has_sync_mult(n, self._vm.get_hw_descr()):
      return Participants.MULT
    # No sub-block rendezvous.  What is left is the smallest set of whole
    # waves that holds this multiplication, which is its group -- and the
    # block is sized to one group, so the block barrier is the group's.
    return Participants.MULTGROUP

  def barrier_scope(self) -> Uniformity:
    """Who has to arrive, derived from what the barrier covers.

    One answer, taken from the participant set, so that the claim `verify`
    checks and the instruction the emitter picks cannot disagree.  A barrier
    that claims more than it covers is refused where it would have been legal;
    one that claims less reaches part of the threads it was asked to reach.
    """
    return self.participants().arrival(self._num_threads, self._wave())

  def accesses(self):
    return ()

  def gen_ir(self, writer):
    writer.barrier(self.participants(), threads=self._num_threads)

  def __str__(self) -> str:
    return f'{self.participants().value}({self._num_threads})'

  def gen_mask_threads(self, num_threads) -> str:
    return ''


class SyncBlock(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._is_ready = True

  def barrier_scope(self) -> Uniformity:
    return Uniformity.BLOCK

  def accesses(self):
    return ()

  def gen_ir(self, writer):
    writer.barrier(Participants.BLOCK)

  def __str__(self) -> str:
    return self.barrier_scope()

  def gen_mask_threads(self, num_threads) -> str:
    return ''

class SyncGrid(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._is_ready = True

  def barrier_scope(self) -> Uniformity:
    return Uniformity.GRID

  def accesses(self):
    return ()

  def gen_ir(self, writer):
    writer.barrier(Participants.GRID)

  def __str__(self) -> str:
    return self.barrier_scope()

  def gen_mask_threads(self, num_threads) -> str:
    return ''
