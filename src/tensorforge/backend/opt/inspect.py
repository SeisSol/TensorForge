# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Observability for the macro instruction stream: ``dump`` and ``verify``.

``is_ready()`` alone, consulted by the emitter one instruction at a time,
would let the first unprepared instruction abort code generation and hide
every other problem behind it.  ``verify`` collects *all* diagnostics
instead, and ``dump`` prints the stream in a form that survives a diff (no
heap addresses, stable ordering).

Both work purely through ``AbstractInstruction.defs/uses/accesses/
barrier_scope``, so neither knows any concrete instruction class.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from tensorforge.backend.instructions.abstract_instruction import (
    AbstractInstruction, Uniformity)
from tensorforge.backend.pir.core import Effect
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
    if scope is not None:
        flags = [f for f in flags if f != 'barrier'] + [f'barrier:{scope.name.lower()}']
    return '+'.join(flags)


def dump(instrs: Sequence[AbstractInstruction],
         title: str = 'macro-ir',
         show_dataflow: bool = True) -> str:
    """Render the stream.  Deliberately diffable: no addresses, no ids."""
    lines = [f'--- {title} ({len(instrs)} instructions) ---']
    _dump_into(lines, instrs, show_dataflow, indent=0)
    return '\n'.join(lines)


def _dump_into(lines: List[str], instrs: Sequence[AbstractInstruction],
               show_dataflow: bool, indent: int) -> None:
    pad = '  ' * indent
    width = max((len(str(i)) for i in range(len(instrs))), default=1)
    for index, instr in enumerate(instrs):
        text = str(instr).replace('\n', ' ')
        lines.append(f'{pad}{index:>{width}}  {text}')
        if show_dataflow:
            defs = ', '.join(_sym(s) for s in instr.defs())
            uses = ', '.join(_sym(s) for s in instr.uses())
            note = f'{pad}{" " * (width + 2)}   [{_effect_str(instr)}]'
            if defs:
                note += f' def={{{defs}}}'
            if uses:
                note += f' use={{{uses}}}'
            if (not instr.describes_dataflow()
                    and instr.barrier_scope() is None
                    and not instr.regions()):
                note += '  OPAQUE'
            if not instr.is_ready():
                note += '  NOT-READY'
            lines.append(note)
        for region in instr.regions():
            lines.append(f'{pad}    {{ uniform='
                         f'{instr.uniform_scope().name.lower()}')
            _dump_into(lines, region, show_dataflow, indent + 3)
            lines.append(f'{pad}    }}')


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


def entering(region: Sequence[AbstractInstruction],
             predefined: Iterable[Any] = ()) -> tuple:
    """`(carried, missing)` — what a region needs that it does not start with.

    A region a loop repeats reads some things before it writes them.  Two
    kinds, and they want opposite treatments, which is why one answer would
    not do:

    *carried*  is also written later in the region.  The value the first
               iteration reads is the one the *previous* iteration wrote, so it
               is a loop-carried argument: initialized before the header,
               updated across the back edge.  An accumulator is this.

    *missing*  is never written in the region at all.  Nothing carries it and
               nothing produces it, so the read is simply unbound and whoever
               assembled the region owes it a definition ahead of the loop.

    Same notion of a read and a write as `verify` uses, on purpose: a region
    that reports nothing here is a region `verify` will not complain about,
    and the two drifting apart would make one of them a lie.
    """
    defined = OrderedSet(predefined)
    written = OrderedSet()
    for instr in region:
        for sym in instr.defs():
            written.add(sym)
    carried, missing = [], []
    for instr in region:
        for sym in instr.uses():
            if sym in defined or getattr(sym, 'stype', None) in _PREDEFINED:
                continue
            (carried if sym in written else missing).append(sym)
            defined.add(sym)
        for sym in instr.defs():
            defined.add(sym)
    return tuple(carried), tuple(missing)


def verify(instrs: Sequence[AbstractInstruction],
           *,
           max_barrier_scope: Uniformity = Uniformity.GRID,
           predefined: Iterable[Any] = (),
           grid_barrier: bool = True,
           check_ready: bool = True) -> List[Diagnostic]:
    """Structural checks over one instruction stream.

    ``check_ready`` is phase-gated: it needs the windows declared, which
    happens once the section's loop is assembled
    (`Generator._declare_buffers`), so it is an emit-time check -- reported
    earlier, it would describe the absence of a later step rather than a
    defect.  Where a buffer sits is not asked here: the allocator decides it
    in the body (`pir.allocate`), and `pir.layout_check` checks it there.

    ``max_barrier_scope`` is the strongest barrier legal at this level.  The
    loop is an instruction with a region, so recursion derives it from
    ``uniform_scope`` and no caller has to set it by hand.

    ``predefined`` are symbols already live on entry (kernel parameters,
    the shared-memory arena, anything defined by ``Section.global_ir``).

    ``grid_barrier`` is whether the target can spell a barrier across the
    whole grid (`Target.grid_barrier`).
    """
    diags: List[Diagnostic] = []
    defined = OrderedSet(predefined)

    for index, instr in enumerate(instrs):
        # -- 1. readiness: collect them all instead of aborting on the first
        if check_ready and not instr.is_ready():
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

        # -- 3. a barrier may not exceed the enclosing constructs' uniformity
        scope = instr.barrier_scope()
        if scope is not None and scope > max_barrier_scope:
            diags.append(Diagnostic(
                'error', index,
                f'{scope.name.lower()} barrier inside a construct whose trip '
                f'count is only {max_barrier_scope.name.lower()}-uniform. '
                f'Threads that execute a different number of iterations never '
                f'arrive at the barrier, so the kernel deadlocks rather than '
                f'producing a wrong answer.'))

        # -- 3b. a wave-collective instruction asks the same of its region
        conv = instr.convergence_scope()
        if conv is not None and conv > max_barrier_scope:
            diags.append(Diagnostic(
                'error', index,
                f'{type(instr).__name__} is issued by the whole wave together '
                f'and needs its region {conv.name.lower()}-uniform, but the '
                f'construct around it is only '
                f'{max_barrier_scope.name.lower()}-uniform: the '
                f'multiplications sharing a wave would take different trips '
                f'and reach it apart.'))

        # -- 4. the target can spell the requested scope
        if scope is Uniformity.GRID and not grid_barrier:
            diags.append(Diagnostic(
                'error', index,
                'a grid barrier was requested, and this target has none '
                '(`Target.grid_barrier`)'))

        # -- 5. opaque instructions: the migration worklist
        if (not instr.describes_dataflow()
                and scope is None
                and Effect.UNKNOWN & instr.effect()):
            diags.append(Diagnostic(
                'info', index,
                f'{type(instr).__name__} does not describe its data flow; '
                f'passes must treat it as opaque'))

        # -- 6. recurse into regions, tightening the barrier limit
        inner_limit = min(max_barrier_scope, instr.uniform_scope())
        for region in instr.regions():
            diags.extend(verify(region,
                                max_barrier_scope=inner_limit,
                                predefined=list(defined),
                                grid_barrier=grid_barrier,
                                check_ready=check_ready))

        for sym in instr.defs():
            defined.add(sym)

    return diags


def format_diagnostics(diags: Sequence[Diagnostic]) -> str:
    if not diags:
        return 'verify: ok'
    errors = sum(1 for d in diags if d.severity == 'error')
    head = f'verify: {errors} error(s), {len(diags) - errors} note(s)'
    return '\n'.join([head] + [f'  {d}' for d in diags])
