# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Liveness over the macro instruction stream, including loop back edges.

Liveness is a least fixed point over the region structure.  For a loop
with body ``B`` and live-out ``X``::

    live_in(loop) = LFP(S -> X union live_in(B, S))

starting from ``S = X``.  The union with ``X`` rather than just ``live_in(B, S)``
is because the loop may execute zero times.  The lattice is finite (subsets of
the tracked symbols) and the transfer function monotone, so the iteration
terminates; the bound is asserted rather than assumed.

On a straight-line stream this reduces to the previous single pass, which is the
test criterion: the emitted code must not change until a region actually
carries a live range.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from collections import OrderedDict

from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.exceptions import InternalError
from tensorforge.common.ordered import OrderedSet

from .abstract import AbstractOptStage, AbstractInstruction, Context


# Which spaces to track.  A parameter rather than a hardcoded check, so the same
# analysis can serve register pressure later.
SHARED = (SymbolType.SharedMem,)


class LivenessAnalysis(AbstractOptStage):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction],
               stypes: Sequence = SHARED,
               predefined: Sequence[Symbol] = ()):
    super(LivenessAnalysis, self).__init__(context)
    self._instrs: List[AbstractInstruction] = list(instructions)
    self._stypes = tuple(stypes)
    # Symbols already live on entry (kernel parameters, anything defined by
    # Section.global_ir).  Without these, a buffer defined outside this stream
    # looks like it is never defined, and so never live.
    self._predefined = OrderedSet(
        s for s in predefined if getattr(s, 'stype', None) in self._stypes)
    self._live_map: Optional[Dict[int, OrderedSet]] = None
    self._ranges: Optional[List[Tuple[Symbol, int, int]]] = None
    self._fixpoint_iterations = 0

  # ------------------------------------------------------------------ #

  def _tracked(self, sym) -> bool:
    return getattr(sym, 'stype', None) in self._stypes

  def _all_symbols(self) -> OrderedSet:
    out = OrderedSet(self._predefined)
    for instr in _flatten(self._instrs):
      for sym in list(instr.defs()) + list(instr.uses()):
        if self._tracked(sym):
          out.add(sym)
    return out

  # -- backward: liveness with a fixed point over back edges ----------- #

  def _transfer(self, instr, live_out: OrderedSet) -> OrderedSet:
    """Live-in of a single region-free instruction."""
    killed = {id(s) for s in instr.defs() if self._tracked(s)}
    out = OrderedSet(s for s in live_out if id(s) not in killed)
    for sym in instr.uses():
      if self._tracked(sym):
        out.add(sym)
    return out

  def _backward(self, body: Sequence[AbstractInstruction],
                live_out: OrderedSet,
                live_out_map: Dict[int, OrderedSet]) -> OrderedSet:
    live = live_out
    for instr in reversed(list(body)):
      # record the live-out, which is what the forward pass needs
      live_out_map[id(instr)] = live
      if instr.regions():
        live = self._region_transfer(instr, live, live_out_map)
      else:
        live = self._transfer(instr, live)
    return live

  def _region_transfer(self, instr, live_out: OrderedSet,
                       live_out_map: Dict[int, OrderedSet]) -> OrderedSet:
    """``LFP(S) = live_out union (union over regions of live_in(region, S))``.

    The union with ``live_out`` accounts for zero iterations.  Each round
    rewrites ``live_out_map`` for the region's instructions; the round that
    reaches the fixed point is the last one, so the map ends up consistent.
    """
    bound = len(self._all_symbols()) + 2
    state = live_out
    for iteration in range(bound):
      nxt = live_out.copy()
      for region in instr.regions():
        nxt = nxt.union(self._backward(region, state, live_out_map))
      self._fixpoint_iterations += 1
      if nxt == state:
        return state
      state = nxt
    raise InternalError(
        f'liveness fixed point did not converge within {bound} iterations at '
        f'{type(instr).__name__}; the transfer function is not monotone')

  # -- forward: definedness, which is what splits the ranges ----------- #

  def _forward(self, body: Sequence[AbstractInstruction],
               defined: OrderedSet,
               live_out_map: Dict[int, OrderedSet],
               records: List[OrderedSet]) -> None:
    for instr in body:
      for sym in instr.defs():
        if self._tracked(sym):
          defined.add(sym)

      still_needed = live_out_map.get(id(instr), OrderedSet())
      here = OrderedSet(s for s in defined if s in still_needed)
      for sym in instr.uses():
        if self._tracked(sym) and sym in defined:
          here.add(sym)
      # A construct with a region is live wherever its body is, so record the
      # union rather than just the boundary: the allocator asks "may these two
      # share an offset", and a buffer live anywhere inside the loop conflicts
      # with one live at the loop.
      records.append(here)

      if instr.regions():
        # Definitions inside are available from the top of the body, because a
        # later iteration sees what an earlier one wrote.  Seeding `defined`
        # with them is the conservative choice: over-estimating liveness costs
        # memory, under-estimating aliases two live buffers.
        inner = defined.copy()
        for region in instr.regions():
          for nested in _flatten(region):
            for sym in nested.defs():
              if self._tracked(sym):
                inner.add(sym)
        for region in instr.regions():
          self._forward(region, inner, live_out_map, records)

      for sym in list(defined):
        if sym not in still_needed and sym not in instr.defs():
          defined.discard(sym)

  # ------------------------------------------------------------------ #

  def apply(self) -> None:
    live_out_map: Dict[int, OrderedSet] = {}
    self._fixpoint_iterations = 0
    self._backward(self._instrs, OrderedSet(), live_out_map)

    records: List[OrderedSet] = []
    self._forward(self._instrs, OrderedSet(self._predefined),
                  live_out_map, records)

    live = {index: value for index, value in enumerate(records)}
    # Preserve the reversed iteration order the previous implementation handed
    # to MemoryRegionAllocation, so downstream vertex numbering is unchanged.
    # NOTE: neither consumer reads the keys -- MemoryRegionAllocation iterates
    # values() and ShrMemOpt never touches the map -- so the key space is free
    # and is simply the depth-first program order.
    self._live_map = OrderedDict(reversed(list(live.items())))
    self._ranges = self._compute_ranges(live)

  # ------------------------------------------------------------------ #

  @staticmethod
  def _compute_ranges(live: Dict[int, OrderedSet]
                      ) -> List[Tuple[Symbol, int, int]]:
    """Maximal ``[start, end]`` intervals per symbol, split at holes.

    A symbol with two disjoint live ranges yields two entries.  This is the form
    a linear-scan allocator wants; the region allocator currently re-derives an
    interference graph from ``live_map`` instead.
    """
    open_at: Dict[int, Tuple[Symbol, int]] = {}
    out: List[Tuple[Symbol, int, int]] = []
    prev: OrderedSet = OrderedSet()
    for index in sorted(live):
      current = live[index]
      for sym in current:
        if id(sym) not in open_at:
          open_at[id(sym)] = (sym, index)
      for sym in prev:
        if sym not in current:
          start_sym, start = open_at.pop(id(sym), (sym, index))
          out.append((start_sym, start, index - 1))
      prev = current
    last = max(live) if live else -1
    for sym, start in open_at.values():
      out.append((sym, start, last))
    out.sort(key=lambda r: (r[1], r[2]))
    return out

  # ------------------------------------------------------------------ #

  def get_live_map(self) -> Dict[int, OrderedSet]:
    return self._live_map

  def get_ranges(self) -> List[Tuple[Symbol, int, int]]:
    """One entry per live range, so a reused buffer appears more than once."""
    return self._ranges

  def max_pressure(self) -> int:
    return max((len(v) for v in self._live_map.values()), default=0)

  def fixpoint_iterations(self) -> int:
    """Rounds spent on back edges; zero for a straight-line stream."""
    return self._fixpoint_iterations


def _flatten(instrs: Sequence[AbstractInstruction]
             ) -> List[AbstractInstruction]:
  out: List[AbstractInstruction] = []
  for instr in instrs:
    out.append(instr)
    for region in instr.regions():
      out.extend(_flatten(region))
  return out
