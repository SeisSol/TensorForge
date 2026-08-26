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
                    and instr.barrier_scope() is BarrierScope.NONE
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


def verify(instrs: Sequence[AbstractInstruction],
           *,
           max_barrier_scope: BarrierScope = BarrierScope.GRID,
           predefined: Iterable[Any] = (),
           backend: Optional[str] = None,
           check_offsets: bool = True,
           check_ready: bool = True) -> List[Diagnostic]:
    """Structural checks over one instruction stream.

    Two checks are phase-gated, because "ready to emit" is reached in two
    steps and reporting either one early describes the *absence* of a later
    pass rather than a defect:

    ``check_offsets``  needs ``ShrMemOpt``: before it, every offset is still
                       the constructor default 0 and every pair of buffers
                       looks like it overlaps.
    ``check_ready``    needs the thread-block policy, which runs *after* the
                       optimisation stage -- ``ShrMemAlloc.is_ready()`` asks
                       for the arena size.  So this is an emit-time check,
                       not a between-passes one.

    ``max_barrier_scope`` is the strongest barrier legal at this level.  It
    used to be a boolean ``inside_batch_loop`` that callers had to set by hand;
    now the loop is an instruction with a region, so recursion derives it from
    ``uniform_scope`` and no caller has to know.

    ``predefined`` are symbols already live on entry (kernel parameters,
    the shared-memory arena, anything defined by ``Section.global_ir``).
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
        if scope is not BarrierScope.NONE and scope > max_barrier_scope:
            diags.append(Diagnostic(
                'error', index,
                f'{scope.name.lower()} barrier inside a construct whose trip '
                f'count is only {max_barrier_scope.name.lower()}-uniform. '
                f'Threads that execute a different number of iterations never '
                f'arrive at the barrier, so the kernel deadlocks rather than '
                f'producing a wrong answer.'))

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

        # -- 5b. a prefetch for the next element, left under the flag guard
        diags.extend(_check_guarded_prefetch(instr, index))

        # -- 6. recurse into regions, tightening the barrier limit
        inner_limit = min(max_barrier_scope, instr.uniform_scope())
        for region in instr.regions():
            diags.extend(verify(region,
                                max_barrier_scope=inner_limit,
                                predefined=list(defined),
                                backend=backend,
                                check_offsets=False,
                                check_ready=check_ready))

        for sym in instr.defs():
            defined.add(sym)

    if check_offsets:
        diags.extend(_check_shared_aliasing(instrs))
    diags.extend(_check_async_depth(instrs))
    return diags


def _check_guarded_prefetch(instr: AbstractInstruction,
                            index: Optional[int]) -> List['Diagnostic']:
    """A transfer for element ``k + 1`` must not sit under the element guard.

    The batch loop wraps its body in ``if (flags[batchId0])``, a runtime mask
    over individual elements.  A transfer addressed from a *lookahead* index is
    issued in iteration ``k`` for element ``k + 1``, so skipping element ``k``
    skips the prefetch, and iteration ``k + 1`` reads whatever the iteration
    before that left in the buffer.  Nothing crashes; the numbers are wrong for
    every element after the first masked one.

    ``BatchLoop.mark_unguarded`` lifts instructions out of the guard, but only
    a *prefix* of the region -- the guard is one contiguous block -- and the
    slot-granular prefetch belongs mid-body, which is not a prefix.  So this is
    reported rather than fixed: closing it needs either a second unguarded
    region after the guard or a predicated transfer, and until then a caller
    enabling wrap-around on a section that passes a flags array should know.
    """
    from tensorforge.backend.instructions.batch_loop import BatchLoop
    from tensorforge.backend.instructions.ptr_manip import GetElementPtr

    if not isinstance(instr, BatchLoop):
        return []
    unguarded = getattr(instr, '_unguarded', set())
    ahead = set()
    diags: List['Diagnostic'] = []
    for inner in instr.region:
        if isinstance(inner, GetElementPtr):
            offset = inner._batch_offset
            if isinstance(offset, int) and offset > 0:
                ahead.update(id(d) for d in inner.defs())
            continue
        if id(inner) in unguarded:
            continue
        if any(id(u) in ahead for u in inner.uses()):
            diags.append(Diagnostic(
                'warning', index,
                f'{type(inner).__name__} prefetches the next element from '
                f'inside the per-element flag guard; a masked element skips '
                f'it and the next iteration reads a stale buffer'))
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
    for instr in _flatten(instrs):
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
    """Live shared-memory symbols per program point.

    Delegates to ``LivenessAnalysis`` rather than approximating.  An earlier
    version of this function held a symbol live from its first definition to
    its last appearance, i.e. without a kill -- the same over-approximation
    that ``LivenessAnalysis`` used to make.  Once the analysis learned to
    split ranges, the two disagreed and this check reported overlaps at
    program points where the real liveness had a hole.  A verifier that
    reimplements the analysis it is checking will always drift from it.
    """
    # local import: liveness imports .abstract, which does not import this
    # module, so there is no cycle
    from .liveness import LivenessAnalysis

    analysis = LivenessAnalysis(None, list(_flatten(instrs)))
    analysis.apply()
    return dict(analysis.get_live_map())


def async_depth(instrs: Sequence[AbstractInstruction]) -> int:
    """Peak number of asynchronous transfers outstanding at once.

    This is the stage count the section's ``cuda::pipeline`` must provide.
    """
    from tensorforge.backend.instructions.memory.load import (GlbToShrLoader,
                                                              LoadWait)

    def is_async(instr) -> bool:
        return (isinstance(instr, GlbToShrLoader)
                and getattr(instr, '_use_cuda_memcpy', False))

    level = sum(1 for i in instrs if is_async(i))
    peak = level
    for instr in instrs:
        for region in instr.regions():
            # two laps: the first reaches the steady state, the second measures
            # it, so a transfer carried across the back edge is counted
            for _ in range(2):
                for inner in region:
                    if isinstance(inner, LoadWait) and is_async(inner.awaited()):
                        level -= 1
                    elif is_async(inner):
                        level += 1
                    peak = max(peak, level)
    return peak


def _check_async_depth(instrs: Sequence[AbstractInstruction]
                       ) -> List[Diagnostic]:
    """How many asynchronous transfers may be outstanding at once.

    There is one ``cuda::pipeline`` per section and it is a FIFO with a fixed
    number of stages.  ``cuda::make_pipeline()`` gives a thread-scope pipeline
    with one, so a second ``producer_acquire()`` before the matching
    ``consumer_wait()`` blocks on a slot that never frees -- a hang, not a wrong
    answer, and one that only appears once two transfers are in flight.

    That happens for two independent reasons: software pipelining commits a
    peeled transfer before the loop, and a loop body containing two
    shared-memory loads commits twice before waiting.  The second predates
    pipelining entirely.
    """
    from tensorforge.backend.instructions.memory.load import (GlbToShrLoader,
                                                              LoadWait)

    def is_async(instr) -> bool:
        return (isinstance(instr, GlbToShrLoader)
                and getattr(instr, '_use_cuda_memcpy', False))

    level = 0
    for instr in instrs:
        if is_async(instr):
            level += 1

    diags: List[Diagnostic] = []
    for index, instr in enumerate(instrs):
        for region in instr.regions():
            peak, level = level, level
            # two laps: the first reaches the steady state, the second measures
            # it, so a value carried across the back edge is counted
            for _ in range(2):
                for inner in region:
                    if isinstance(inner, LoadWait) and is_async(inner.awaited()):
                        level -= 1
                    elif is_async(inner):
                        level += 1
                    peak = max(peak, level)
            if peak > 1:
                diags.append(Diagnostic(
                    'info', index,
                    f'{peak} asynchronous transfers may be outstanding at once; '
                    f'the section pipeline must provide that many stages'))
    return diags


def _flatten(instrs: Sequence[AbstractInstruction]
             ) -> List[AbstractInstruction]:
    """Depth-first linearisation, for checks that only need an ordering.

    Correct for the shared-memory aliasing check because a region executes
    where it sits; it is *not* a substitute for a real analysis over the
    region structure, which loop-carried live ranges will need.
    """
    out: List[AbstractInstruction] = []
    for instr in instrs:
        out.append(instr)
        for region in instr.regions():
            out.extend(_flatten(region))
    return out


def format_diagnostics(diags: Sequence[Diagnostic]) -> str:
    if not diags:
        return 'verify: ok'
    errors = sum(1 for d in diags if d.severity == 'error')
    head = f'verify: {errors} error(s), {len(diags) - errors} note(s)'
    return '\n'.join([head] + [f'  {d}' for d in diags])
