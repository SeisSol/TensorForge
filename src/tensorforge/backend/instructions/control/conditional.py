# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A region of instructions that runs only where a conjunction holds."""

from typing import List, Tuple

from tensorforge.common.context import Context
from tensorforge.backend.pir.core import BOOL, Access, Effect, MemSpace
from ..abstract_instruction import AbstractInstruction
from tensorforge.backend.pir.core import Uniformity


class GuardedRegion(AbstractInstruction):
  """`if (c0 && !c1 && ...) { ... }` over a stretch of the section.

  A region rather than a flag on each instruction: the operations under one
  guard are built as they always were, and the guard is the thing that
  contains them. Everything that walks a stream -- the passes, `verify` --
  already recurses through `regions()`, so a body inside one stays visible
  without any of them learning what a guard is.

  The condition is read once for the whole region, not once per operation.
  Two reads of the same tensor may be two different values: yateto states a
  version alongside each literal for exactly that reason, and a region only
  ever holds literals of one version.
  """

  def __init__(self, context: Context, literals, region):
    super().__init__(context)
    #: `(symbol, negated)` per literal, in the order yateto stated them.
    self._literals = list(literals)
    self._region = list(region)
    self._is_ready = True

  # -- structure --------------------------------------------------------- #

  def region(self) -> List[AbstractInstruction]:
    return self._region

  def regions(self) -> Tuple[Tuple[AbstractInstruction, ...], ...]:
    return (tuple(self._region),)

  def replace_region(self, index: int, instrs) -> None:
    assert index == 0, f'GuardedRegion has one region, not {index + 1}'
    self._region = list(instrs)

  def uniform_scope(self) -> Uniformity:
    """How far the guard's decision is uniform.

    The condition is a tensor addressed per batch element, so two elements
    may decide differently -- and a block holds several. That makes the
    region `SIMD`-uniform, the same answer `BatchLoop` gives for the same
    reason: a block-wide barrier inside it is reached by some threads and
    not others, which deadlocks rather than computing the wrong thing.

    `verify` tightens its limit through this on the way into the region, so
    a body that needs a wider barrier is reported there rather than here.
    """
    return Uniformity.MULT

  # -- data flow --------------------------------------------------------- #

  def uses(self) -> Tuple:
    out, seen, defined = [], set(), set()
    for symbol, _ in self._literals:
      if id(symbol) not in seen:
        seen.add(id(symbol))
        out.append(symbol)
    for instr in self._region:
      for sym in instr.uses():
        if id(sym) not in defined and id(sym) not in seen:
          seen.add(id(sym))
          out.append(sym)
      for sym in instr.defs():
        defined.add(id(sym))
    return tuple(out)

  def defs(self) -> Tuple:
    """What the region writes.

    Reported as definitions even though the guard may skip them. A pass that
    did not see them would move a later read of the destination across this
    region, and the value it then reads is whichever one the guard left --
    the failure that leaves no trace.
    """
    out, seen = [], set()
    for instr in self._region:
      for sym in instr.defs():
        if id(sym) not in seen:
          seen.add(id(sym))
          out.append(sym)
    return tuple(out)

  def accesses(self) -> Tuple:
    out = []
    for symbol, _ in self._literals:
      space = MemSpace.from_symbol_type(getattr(symbol, 'stype', None))
      if space is not MemSpace.NONE:
        out.append(Access(Effect.READ, space, symbol))
    for instr in self._region:
      out.extend(instr.accesses())
    return tuple(out)

  def barrier_scope(self) -> Uniformity:
    """A region containing a barrier synchronises, seen from outside."""
    inner = [instr.barrier_scope() for instr in self._region]
    return max((s for s in inner if s is not None), default=None)

  def temp_shmem(self) -> int:
    return max((instr.temp_shmem() for instr in self._region), default=0)

  # -- emission ---------------------------------------------------------- #

  def _condition(self, builder):
    """The conjunction, as one value.

    A negated literal is spelled as a comparison against zero rather than a
    logical not: the operator table has an infix form for `eq` and none for
    `not`, and a name with no infix form falls through to an unqualified
    call.
    """
    value = None
    for symbol, negated in self._literals:
      literal = symbol.load(builder, self._context, None, [], False)
      if negated:
        literal = builder.op('eq', BOOL, literal, 0, hint='g')
      value = literal if value is None else builder.op('and', BOOL, value,
                                                       literal, hint='g')
    return value

  def _emit(self, builder) -> None:
    with builder.if_(self._condition(builder)):
      for instr in self._region:
        instr.gen_code(builder)

  def gen_code(self, writer) -> None:
    """Emit the guard and its body.

    Like `BatchLoop`, this drives child instructions that route themselves,
    so it overrides `gen_code` rather than `gen_ir`. When the caller handed
    over a plain `Writer` -- one body per instruction rather than one for the
    region -- a body is opened here, because the condition is a value and
    only the builder can make one.
    """
    if hasattr(writer, 'if_') and hasattr(writer, 'op'):
      self._emit(writer)
      return
    with AbstractInstruction.shared_body(self._context, writer,
                                         scratch=self.temp_shmem()) as builder:
      self._emit(builder)

  def __str__(self) -> str:
    terms = ' && '.join(f'{"!" if negated else ""}{symbol.name}'
                        for symbol, negated in self._literals)
    return f'if ({terms}) {{ {len(self._region)} instr }}'
