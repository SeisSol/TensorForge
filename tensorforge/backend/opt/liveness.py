# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Liveness over the macro instruction stream.

This version reads ``defs()``/``uses()``, so it knows no instruction class, and
walks backwards computing genuine live-in sets with kills.  A symbol defined
twice therefore produces two disjoint ranges and the allocator can put
something else in the hole between them.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from collections import OrderedDict

from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.ordered import OrderedSet

from .abstract import AbstractOptStage, AbstractInstruction, Context


# Which spaces to track.  A parameter rather than a hardcoded check, so the
# same analysis can serve register pressure later.
SHARED = (SymbolType.SharedMem,)


class LivenessAnalysis(AbstractOptStage):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction],
               stypes: Sequence = SHARED,
               predefined: Sequence[Symbol] = ()):
    super(LivenessAnalysis, self).__init__(context)
    self._instrs: List[AbstractInstruction] = instructions
    self._stypes = tuple(stypes)
    # Symbols already live on entry (kernel parameters, anything defined by
    # Section.global_ir).  Without these, a buffer defined outside this stream
    # looks like it is never defined, and so never live.
    self._predefined = OrderedSet(
        s for s in predefined if getattr(s, 'stype', None) in self._stypes)
    self._live_map: Optional[Dict[int, OrderedSet]] = None
    self._ranges: Optional[List[Tuple[Symbol, int, int]]] = None

  # ------------------------------------------------------------------ #

  def _tracked(self, sym) -> bool:
    return getattr(sym, 'stype', None) in self._stypes

  def apply(self) -> None:
    n = len(self._instrs)

    # Backwards: live_in[i] = uses(i) ∪ (live_in[i+1] \ defs(i))
    #
    # The set difference is the kill that was missing.  It applies to
    # live_in[i+1], not live_in[i], so an instruction that both reads and
    # writes one symbol (a shared-memory accumulator) keeps it live.
    live_in: List[OrderedSet] = [OrderedSet() for _ in range(n + 1)]
    for index in range(n - 1, -1, -1):
      instr = self._instrs[index]
      killed = {id(s) for s in instr.defs() if self._tracked(s)}
      nxt = OrderedSet(s for s in live_in[index + 1] if id(s) not in killed)
      for sym in instr.uses():
        if self._tracked(sym):
          nxt.add(sym)
      live_in[index] = nxt

    # Forwards: a symbol is live *at* i if some later instruction still reads
    # it and it has been defined at or before i.  Tracking definedness
    # forwards is what splits the ranges: after the last use the symbol leaves
    # the set and, if written again later, re-enters at the new definition.
    defined = OrderedSet(self._predefined)
    live: Dict[int, OrderedSet] = {}
    for index in range(n):
      instr = self._instrs[index]
      for sym in instr.defs():
        if self._tracked(sym):
          defined.add(sym)
      still_needed = live_in[index + 1]
      here = OrderedSet(s for s in defined if s in still_needed)
      for sym in instr.uses():
        if self._tracked(sym) and sym in defined:
          here.add(sym)
      live[index] = here
      for sym in list(defined):
        if sym not in still_needed:
          defined.discard(sym)

    # Preserve the reversed iteration order the previous implementation handed
    # to MemoryRegionAllocation, so downstream vertex numbering is unchanged.
    self._live_map = OrderedDict(reversed(list(live.items())))
    self._ranges = self._compute_ranges(live)

  # ------------------------------------------------------------------ #

  @staticmethod
  def _compute_ranges(live: Dict[int, OrderedSet]
                      ) -> List[Tuple[Symbol, int, int]]:
    """Maximal ``[start, end]`` intervals per symbol, split at holes.

    A symbol with two disjoint live ranges yields two entries.  This is the
    form a linear-scan allocator wants; the region allocator currently
    re-derives an interference graph from ``live_map`` instead.
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
