# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Observability for the macro instruction stream: ``dump`` and ``verify``.

The macro level had neither.  The only check was ``is_ready()``, consulted
by the emitter one instruction at a time, so the first unprepared
instruction aborted code generation and hid every other problem behind it.
``verify`` collects *all* diagnostics instead, and ``dump`` prints the
stream in a form that survives a diff (no heap addresses, stable ordering).

Both work purely through ``AbstractInstruction.defs/uses/accesses/
barrier_scope``, so neither knows any concrete instruction class.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tensorforge.backend.instructions.abstract_instruction import (
    AbstractInstruction, BarrierScope)
from tensorforge.backend.pir.core import Effect, MemSpace, accesses_conflict
from tensorforge.backend.symbol import SymbolType
from tensorforge.common.ordered import OrderedSet


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #

def _sym(s) -> str:
    return getattr(s, 'name', None) or f'<{type(s).__name__}>'


def _effect_str(instr: AbstractInstruction) -> str:
    eff = instr.effect()
    if eff is Effect.NONE:
        return 'pure'
    flags = [f.name.lower() for f in Effect if f and (eff & f)]
    scope = instr.barrier_scope()
    if scope is not BarrierScope.NONE:
        flags = [f for f in flags if f != 'barrier'] + [f'barrier:{scope.name.lower()}']
    return '+'.join(flags)


def dump(instrs: Sequence[AbstractInstruction],
         title: str = 'macro-ir',
         show_dataflow: bool = True) -> str:
    """Render the stream.  Deliberately diffable: no addresses, no ids."""
    lines = [f'--- {title} ({len(instrs)} instructions) ---']
    width = max((len(str(i)) for i in range(len(instrs))), default=1)
    for index, instr in enumerate(instrs):
        text = str(instr).replace('\n', ' ')
        lines.append(f'{index:>{width}}  {text}')
        if not show_dataflow:
            continue
        defs = ', '.join(_sym(s) for s in instr.defs())
        uses = ', '.join(_sym(s) for s in instr.uses())
        note = f'{" " * (width + 2)}   [{_effect_str(instr)}]'
        if defs:
            note += f' def={{{defs}}}'
        if uses:
            note += f' use={{{uses}}}'
        if not instr.describes_dataflow() and instr.barrier_scope() is BarrierScope.NONE:
            note += '  OPAQUE'
        if not instr.is_ready():
            note += '  NOT-READY'
        lines.append(note)
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

class Diagnostic:
    __slots__ = ('severity', 'index', 'message')

    def __init__(self, severity: str, index: Optional[int], message: str):
        self.severity = severity
        self.index = index
        self.message = message

    def __str__(self) -> str:
        where = '' if self.index is None else f'@{self.index}: '
        return f'[{self.severity}] {where}{self.message}'

    __repr__ = __str__


# Symbol kinds that are live on kernel entry and therefore need no
# preceding definition in the stream.
_PREDEFINED = (SymbolType.Batch, SymbolType.Global, SymbolType.Scalar,
               SymbolType.Data)


def verify(instrs: Sequence[AbstractInstruction],
           *,
           inside_batch_loop: bool = False,
           predefined: Iterable[Any] = (),
           backend: Optional[str] = None,
           allocated: bool = True) -> List[Diagnostic]:
    """Structural checks over one instruction stream.

    ``allocated`` says whether shared-memory offsets have been assigned yet.
    Before ``ShrMemOpt`` runs, ``is_ready()`` is legitimately False on every
    shared-memory writer and every offset is still the constructor default 0,
    so the readiness and aliasing checks would report the *absence* of a pass
    rather than a defect.  They are therefore skipped pre-allocation.

    ``inside_batch_loop`` marks the per-element body (``Section.ir``), which
    the generator wraps in the persistent-threading ``for``.  That matters
    for grid barriers -- see below.

    ``predefined`` are symbols already live on entry (kernel parameters,
    the shared-memory arena, anything defined by ``Section.global_ir``).
    """
    diags: List[Diagnostic] = []
    defined = OrderedSet(predefined)

    for index, instr in enumerate(instrs):
        # -- 1. readiness: collect them all instead of aborting on the first
        if allocated and not instr.is_ready():
            diags.append(Diagnostic(
                'error', index,
                f'not ready to emit ({type(instr).__name__}); an offset or '
                f'thread configuration was never assigned'))

        # -- 2. use before def
        for sym in instr.uses():
            if sym in defined:
                continue
            if getattr(sym, 'stype', None) in _PREDEFINED:
                continue
            diags.append(Diagnostic(
                'error', index,
                f'reads {_sym(sym)} ({getattr(sym, "stype", "?")}) with no '
                f'preceding definition'))

        # -- 3. grid barriers must not sit inside the batch loop
        scope = instr.barrier_scope()
        if scope is BarrierScope.GRID and inside_batch_loop:
            diags.append(Diagnostic(
                'error', index,
                'grid barrier inside the per-element loop. The trip count is '
                'ceil((numElements - start)/stride) and `start` depends on the '
                'block id, so it is not grid-uniform: blocks with fewer '
                'iterations exit without arriving and the kernel deadlocks. '
                'Grid barriers belong between sections.'))

        # -- 4. backend actually supports the requested scope
        if scope is BarrierScope.GRID and backend == 'sycl':
            diags.append(Diagnostic(
                'error', index,
                'grid barrier requested but the SYCL lexic raises '
                'NotImplementedError for sync_grid()'))

        # -- 5. opaque instructions: the migration worklist
        if (not instr.describes_dataflow()
                and scope is BarrierScope.NONE
                and Effect.UNKNOWN & instr.effect()):
            diags.append(Diagnostic(
                'info', index,
                f'{type(instr).__name__} does not describe its data flow; '
                f'passes must treat it as opaque'))

        for sym in instr.defs():
            defined.add(sym)

    if allocated:
        diags.extend(_check_shared_aliasing(instrs))
    return diags


def _check_shared_aliasing(instrs: Sequence[AbstractInstruction]
                           ) -> List[Diagnostic]:
    """Two simultaneously-live shared-memory buffers must not overlap.

    This is the check that would have caught a mis-colouring: the region
    allocator assigns byte offsets, and nothing downstream ever validated
    that co-live buffers landed in disjoint ranges.
    """
    diags: List[Diagnostic] = []
    # (symbol -> (offset, size, global_arena)) as far as it is observable
    extents: Dict[int, Tuple[Any, int, int, bool]] = {}
    for instr in instrs:
        offset = getattr(instr, '_shr_mem_offset', None)
        size_fn = getattr(instr, 'compute_shared_mem_size', None)
        if offset is None or not callable(size_fn):
            continue
        for sym in instr.defs():
            if getattr(sym, 'stype', None) is not SymbolType.SharedMem:
                continue
            try:
                size = size_fn()
            except Exception:
                continue
            extents[id(sym)] = (sym, offset, size,
                                bool(getattr(instr, '_global_offset', False)))

    live = _live_shared(instrs)
    for index, live_set in live.items():
        seen: List[Tuple[Any, int, int, bool]] = []
        for sym in live_set:
            rec = extents.get(id(sym))
            if rec is None:
                continue
            _, off, size, arena = rec
            for other, ooff, osize, oarena in seen:
                if arena != oarena:
                    continue        # different arenas cannot overlap
                if off < ooff + osize and ooff < off + size:
                    diags.append(Diagnostic(
                        'error', index,
                        f'shared-memory buffers {_sym(sym)} '
                        f'[{off}, {off + size}) and {_sym(other)} '
                        f'[{ooff}, {ooff + osize}) overlap while both live'))
            seen.append((sym, off, size, arena))
    return diags


def _live_shared(instrs: Sequence[AbstractInstruction]
                 ) -> Dict[int, OrderedSet]:
    """Live shared-memory symbols per program point, from defs/uses only.

    Unlike ``LivenessAnalysis`` this needs no ``isinstance`` and it *kills*
    at the last use, so a buffer written twice yields two ranges rather than
    one merged one.
    """
    last_use: Dict[int, int] = {}
    for index, instr in enumerate(instrs):
        for sym in list(instr.uses()) + list(instr.defs()):
            if getattr(sym, 'stype', None) is SymbolType.SharedMem:
                last_use[id(sym)] = index

    live: Dict[int, OrderedSet] = {}
    current: OrderedSet = OrderedSet()
    for index, instr in enumerate(instrs):
        for sym in instr.defs():
            if getattr(sym, 'stype', None) is SymbolType.SharedMem:
                current.add(sym)
        live[index] = current.copy()
        for sym in list(current):
            if last_use.get(id(sym), -1) <= index:
                current.discard(sym)
    return live


def format_diagnostics(diags: Sequence[Diagnostic]) -> str:
    if not diags:
        return 'verify: ok'
    errors = sum(1 for d in diags if d.severity == 'error')
    head = f'verify: {errors} error(s), {len(diags) - errors} note(s)'
    return '\n'.join([head] + [f'  {d}' for d in diags])
