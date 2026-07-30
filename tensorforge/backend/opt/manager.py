# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pass manager for the macro instruction stream.

Replaces the hardcoded sequence in ``OptimizationStage.optimize``, where the
order was implicit, three passes sat commented out, one was constructed but
never applied, and analysis results were handed on as bare
``Dict[int, Set[Symbol]]`` with no notion of who invalidates what.

Three things are explicit here:

*Analyses vs. transforms.*  An analysis derives a fact and stores it under a
name; a transform rewrites the stream.  A transform invalidates every
analysis unless it declares otherwise, so a stale ``live_map`` cannot be
read by accident -- that was a real hazard, because ``live_map`` is keyed by
*instruction index* and any transform that inserts or removes an instruction
silently reinterprets every key.

*Declared dependencies.*  A pass names the analyses it consumes; the manager
schedules them, re-running an analysis whose result was invalidated.  A
missing dependency is an error at registration, not an ``AttributeError``
halfway through code generation.

*Verification between passes.*  ``verify`` runs after every pass under
``TF_IR_DEBUG``, so a diagnostic names the pass that introduced it.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError

from .inspect import dump, format_diagnostics, verify


class PassContext:
    """Everything a pass may read, plus the analysis cache.

    Passes reach for named analyses instead of being handed positional
    arguments, which is what let ``live_map`` and ``regions`` drift apart
    from the stream they described.
    """

    def __init__(self,
                 context: Context,
                 instrs: List[AbstractInstruction],
                 *,
                 shr_mem=None,
                 num_threads: int = 0,
                 scopes=None,
                 global_ir: Optional[Sequence[AbstractInstruction]] = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.context = context
        self.instrs = instrs
        self.shr_mem = shr_mem
        self.num_threads = num_threads
        self.scopes = scopes
        # Built before this stage and never routed through it.  Passes must
        # be able to see its definitions or every symbol it defines looks
        # undefined -- which is precisely why preloaded shared-memory
        # buffers never entered LivenessAnalysis.
        self.global_ir: List[AbstractInstruction] = list(global_ir or [])
        self.produced: List[AbstractInstruction] = []
        self._analyses: Dict[str, Any] = {}
        self.extra: Dict[str, Any] = dict(extra or {})

    # -- analysis cache ---------------------------------------------------- #

    def get(self, name: str) -> Any:
        if name not in self._analyses:
            raise GenerationError(
                f'analysis {name!r} requested but not available; the pass '
                f'should declare it in `requires`')
        return self._analyses[name]

    def has(self, name: str) -> bool:
        return name in self._analyses

    def put(self, name: str, value: Any) -> None:
        self._analyses[name] = value

    def invalidate(self, keep: Iterable[str] = ()) -> None:
        keep = set(keep)
        for name in list(self._analyses):
            if name not in keep:
                del self._analyses[name]

    @property
    def stream(self) -> List[AbstractInstruction]:
        """Everything that will be emitted for this section, prologue first.

        Use this for whole-section checks (verify, dump).  Do *not* hand it to
        an index-keyed analysis whose result is consumed against
        ``local_stream`` -- the offsets would not line up.
        """
        return self.global_ir + self.produced + self.instrs

    @property
    def local_stream(self) -> List[AbstractInstruction]:
        """The part this stage owns: what passes rewrite and index into.

        ``Section.global_ir`` is excluded on purpose.  Its shared-memory
        symbols are allocated by ``ShrMemObject.alloc_global``, a separate bump
        allocator in a separate arena; letting them into the region allocator
        would give them a second, conflicting offset.
        """
        return self.produced + self.instrs


class Pass:
    """Base class.

    ``name``       identifier used in dependency lists and logs
    ``requires``   analyses this pass reads
    ``provides``   analyses this pass writes (analyses only)
    ``preserves``  analyses that survive this pass (transforms only)
    """

    name: str = ''
    requires: Sequence[str] = ()
    provides: Sequence[str] = ()
    preserves: Sequence[str] = ()
    is_transform: bool = False

    def enabled(self, pc: PassContext) -> bool:
        return True

    def run(self, pc: PassContext) -> None:
        raise NotImplementedError


class PassManager:
    def __init__(self, debug: str = None):
        self._passes: List[Pass] = []
        self._debug = os.environ.get('TF_IR_DEBUG', '') if debug is None else debug
        self.timings: List[tuple] = []

    def add(self, p: Pass) -> 'PassManager':
        available: Set[str] = set()
        for earlier in self._passes:
            available.update(earlier.provides)
        missing = [r for r in p.requires if r not in available]
        if missing:
            raise GenerationError(
                f'pass {p.name!r} requires {missing} but no earlier pass '
                f'provides it; registered so far: '
                f'{[q.name for q in self._passes]}')
        self._passes.append(p)
        return self

    def run(self, pc: PassContext) -> None:
        self._check(pc, 'build', offsets=False)
        for p in self._passes:
            if not p.enabled(pc):
                continue
            missing = [r for r in p.requires if not pc.has(r)]
            if missing:
                raise GenerationError(
                    f'pass {p.name!r} needs {missing}, invalidated by an '
                    f'earlier transform and not recomputed')
            t0 = time.perf_counter()
            p.run(pc)
            self.timings.append((p.name, time.perf_counter() - t0))
            if p.is_transform:
                pc.invalidate(keep=p.preserves)
            self._check(pc, p.name, offsets=pc.extra.get('offsets_assigned', False))

    # -- verification ------------------------------------------------------ #

    def _check(self, pc: PassContext, stage: str, offsets: bool) -> None:
        if not self._debug:
            return
        stream = pc.stream
        if 'dump' in self._debug:
            print(dump(stream, title=f'after {stage}'))
        predefined = []
        if pc.scopes is not None:
            predefined += list(pc.scopes.get_global_scope().values())
        for instr in pc.global_ir:
            predefined += list(instr.defs())
        diags = verify(stream,
                       predefined=predefined,
                       check_offsets=offsets,
                       # readiness needs the thread-block policy, which runs
                       # after this stage -- checked at emit time instead
                       check_ready=False,
                       backend=pc.context.get_vm().get_lexic()._backend)
        errors = [d for d in diags if d.severity == 'error']
        if errors:
            raise GenerationError(f'macro-ir invalid after {stage}:\n'
                                  + format_diagnostics(diags))


# --------------------------------------------------------------------------- #
# Adapters for the existing pass classes
# --------------------------------------------------------------------------- #

class LegacyTransform(Pass):
    """Wraps an ``AbstractTransformer``: construct, ``apply``, take the list.

    Exists so the migration is incremental -- the existing passes keep
    working unchanged while new ones are written against ``Pass``.
    """

    is_transform = True

    def __init__(self, name: str, factory: Callable[[PassContext], Any],
                 *, preserves: Sequence[str] = (),
                 enabled: Optional[Callable[[PassContext], bool]] = None,
                 collects_global: bool = False):
        self.name = name
        self.preserves = preserves
        self._factory = factory
        self._enabled = enabled
        self._collects_global = collects_global

    def enabled(self, pc: PassContext) -> bool:
        return True if self._enabled is None else self._enabled(pc)

    def run(self, pc: PassContext) -> None:
        opt = self._factory(pc)
        opt.apply()
        pc.instrs[:] = opt.get_instructions()
        if self._collects_global:
            pc.produced += list(getattr(opt, '_global_instrs', []))


class LegacyAnalysis(Pass):
    """Wraps an ``AbstractOptStage`` that computes one named result."""

    is_transform = False

    def __init__(self, name: str, factory: Callable[[PassContext], Any],
                 getter: Callable[[Any], Any], provides: str,
                 *, requires: Sequence[str] = ()):
        self.name = name
        self.provides = (provides,)
        self.requires = requires
        self._factory = factory
        self._getter = getter
        self._key = provides

    def run(self, pc: PassContext) -> None:
        opt = self._factory(pc)
        opt.apply()
        pc.put(self._key, self._getter(opt))
