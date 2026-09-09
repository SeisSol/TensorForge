# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A cache hint for the element the loop will reach next."""

from tensorforge.common.context import Context

from .abstract_instruction import AbstractInstruction


class PrefetchBatchPointer(AbstractInstruction):
  """Ask for `m[batchId1]`, the pointer the next iteration has to load first.

  Under `Addressing.PTR_BASED` the operand is an array of pointers, so every
  address in an iteration hangs off one dependent load: `m[batchId0]` has to
  arrive before `&m[batchId0][off]` exists, and nothing in the body can be
  started ahead of it. That is the load this shadows, one element early, out
  of the iteration whose own latency there is something to hide it behind.

  Only the pointer, and only `PTR_BASED`. The *data* of the next element sits
  at `batchId1 * volume + offset`, and hinting for that would mean a second
  copy of the address formula `GetElementPtr` already carries -- for the
  smaller of the two wins, since a strided address needs no load to compute
  and the existing wrap and pipeline passes are what move the transfer that
  reads it. A formula written twice drifts; this way there is nothing to
  drift.

  The index is `batchId1`, clamped into the batch by the loop that binds it.
  That clamp is what makes this safe rather than merely likely: an unclamped
  lookahead runs past the end of the pointer array for every thread whose
  start index is beyond the element count, which is the common case and not
  the edge.
  """

  def __init__(self, context: Context, src, loop, level: str = 'l2'):
    super().__init__(context)
    self._src = src
    self._loop = loop
    self._level = level
    self._is_ready = True

  def gen_ir(self, writer):
    """Nothing where the loop bound no lookahead value.

    Two paths reach here with none. The legacy writer emits the bindings as
    text, so there is no operand to name the next element with -- and a hint
    whose address is a string would be the thing `Op.PREFETCH` was given a
    base and an index to avoid. A body that is not a loop has no next element
    at all.

    Emitting nothing in both cases rather than raising: this instruction is
    the one thing in the stream whose absence changes no result.
    """
    if not hasattr(writer, 'prefetch'):
      return
    index = self._loop.lookahead_value()
    if index is None:
      return
    writer.prefetch(self._src, index, level=self._level)

  def __str__(self) -> str:
    return f'prefetch {self._src.name}[{self._loop.index_name(1)}];'
