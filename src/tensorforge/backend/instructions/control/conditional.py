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
  guard are built exactly as they are without one, and the guard is the thing
  that contains them. Everything that walks a stream -- the passes, `verify` --
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
    """A region containing a barrier synchronizes, seen from outside."""
    inner = [instr.barrier_scope() for instr in self._region]
    return max((s for s in inner if s is not None), default=None)

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

  def _declare_windows_early(self, builder, cleared) -> None:
    """Declare the shared windows this body fills ahead of the guard.

    `s0 = &localShrMem0[64]` is where a transfer writes, and its offset comes
    from the shared-memory layout rather than from the condition -- the window
    is the same whether the guard holds or not. It is declared by whichever
    instruction fills it first, though, so a first writer inside the region
    scopes the name to the region, and every later reader of that window names
    something that was never declared where it stands. Both halves of a
    hoisted `where` write one tensor: the first half declares its home and the
    second refers to it.

    The same hoist `BatchLoop` does out of its flag guard, for the same reason
    and on the same grounds: nothing in the declaration depends on the guard,
    so moving it out cannot observe anything the guarded body would not have.
    """
    for instr in self._region:
      declare = getattr(instr, 'gen_code_declare', None)
      if declare is None or not getattr(instr, '_declare', False):
        continue
      declare(builder)
      instr._declare = False
      cleared.append(instr)

  def _emit(self, builder) -> None:
    # Restored afterwards: clearing the flag is a statement about this build,
    # and a body built twice would otherwise declare nothing at all.
    cleared = []
    try:
      self._declare_windows_early(builder, cleared)
      with builder.if_(self._condition(builder)):
        for instr in self._region:
          instr.gen_code(builder)
    finally:
      for instr in cleared:
        instr._declare = True

  def gen_code(self, writer) -> None:
    """Emit the guard and its body.

    Like `BatchLoop`, this drives child instructions that route themselves,
    so it overrides `gen_code` rather than `gen_ir`. When the caller hands
    over a plain `Writer` -- no body is open yet -- a body is opened here,
    because the condition is a value and only the builder can make one.
    """
    if hasattr(writer, 'if_') and hasattr(writer, 'op'):
      self._emit(writer)
      return
    AbstractInstruction.build_shared_body(self._context, writer, self._emit)

  def __str__(self) -> str:
    terms = ' && '.join(f'{"!" if negated else ""}{symbol.name}'
                        for symbol, negated in self._literals)
    return f'if ({terms}) {{ {len(self._region)} instr }}'
