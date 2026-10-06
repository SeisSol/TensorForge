# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""The macro instruction stream as something `PassManager` runs passes over.

The manager itself is IR-agnostic (`backend.passmanager`).  What is specific
to this level lives here: the stream and the prologue it may read but not
rewrite, the shared-memory object the allocation passes fill in, the
per-region dispatch over instruction regions, the macro verifier between
passes, and the adapters that turn an optimization stage into a pass.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.passmanager import Pass, PassContext, PassScope
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError

from .inspect import dump, format_diagnostics, verify


class StreamContext(PassContext):
    """Everything a pass over the macro stream may read, plus the analysis
    cache."""

    def __init__(self,
                 context: Context,
                 instrs: List[AbstractInstruction],
                 *,
                 shr_mem=None,
                 num_threads: int = 0,
                 scopes=None,
                 global_ir: Optional[Sequence[AbstractInstruction]] = None,
                 extra: Optional[Dict[str, Any]] = None):
        super().__init__(extra)
        self.context = context
        self.instrs = instrs
        self.shr_mem = shr_mem
        self.num_threads = num_threads
        self.scopes = scopes
        # Built before this stage and never routed through it.  Passes must
        # be able to see its definitions or every symbol it defines looks
        # undefined -- a preloaded shared-memory buffer, for one, would never
        # enter LivenessAnalysis.
        self.global_ir: List[AbstractInstruction] = list(global_ir or [])

    @property
    def stream(self) -> List[AbstractInstruction]:
        """Everything that will be emitted for this section, prologue first.

        Use this for whole-section checks (verify, dump).  Do *not* hand it to
        an index-keyed analysis whose result is consumed against
        ``local_stream`` -- the offsets would not line up.
        """
        return self.global_ir + self.instrs

    @property
    def local_stream(self) -> List[AbstractInstruction]:
        """The part this stage owns: what passes rewrite and index into.

        ``Section.global_ir`` is excluded on purpose.  Its shared-memory
        symbols are allocated by ``ShrMemObject.alloc_global``, a separate bump
        allocator in a separate arena; letting them into the region allocator
        would give them a second, conflicting offset.
        """
        return list(self.instrs)

    # -- per-region dispatch ---------------------------------------------- #

    def run_per_region(self, p: Pass) -> None:
        """Invoke ``p`` on every region, innermost first, then the top level.

        Innermost first so that a pass which inspects an instruction's regions
        sees them already rewritten -- and so that the top-level invocation acts
        on a settled nest.
        """

        # What each region belongs to, innermost last, for a pass that has to
        # know what kind of block it was handed.  A straight-line block and a
        # loop body are the same list to `run_region`, and for most passes
        # that is the point -- a scheduler must not move anything across the
        # back edge.  Barrier insertion is the exception: a write at the tail
        # of a loop body is read at its head, one iteration later, and a pass
        # that cannot tell a body from the top level cannot put the barrier
        # between the two.
        enclosing = self.extra.setdefault('enclosing', [])

        def visit(instrs: List[AbstractInstruction]) -> List[AbstractInstruction]:
            for instr in instrs:
                for index, region in enumerate(instr.regions()):
                    enclosing.append(instr)
                    try:
                        instr.replace_region(index, visit(list(region)))
                    finally:
                        enclosing.pop()
            return list(p.run_region(instrs, self))

        self.instrs[:] = visit(self.instrs)

    # -- verification ------------------------------------------------------ #

    def check(self, stage: str, debug: str) -> None:
        stream = self.stream
        if 'dump' in debug:
            print(dump(stream, title=f'after {stage}'))
        predefined = []
        if self.scopes is not None:
            predefined += list(self.scopes.get_global_scope().values())
        for instr in self.global_ir:
            predefined += list(instr.defs())
        diags = verify(stream,
                       predefined=predefined,
                       # from `ShrMemOpt` on, every shared-memory writer has
                       # an offset, so buffer aliasing can be checked
                       check_offsets=self.extra.get('offsets_assigned', False),
                       # readiness needs the thread-block policy, which runs
                       # after this stage -- checked at emit time instead
                       check_ready=False,
                       backend=self.context.get_vm().get_lexic()._backend)
        errors = [d for d in diags if d.severity == 'error']
        if errors:
            raise GenerationError(f'macro-ir invalid after {stage}:\n'
                                  + format_diagnostics(diags))


# --------------------------------------------------------------------------- #
# Passes written as optimization stages
# --------------------------------------------------------------------------- #

class Transform(Pass):
    """An ``AbstractTransformer`` as a pass: construct, ``apply``, take the list.

    The stage keeps its own constructor arguments; the factory binds them from
    the pass context.  It takes the instruction list explicitly rather than
    reading ``pc.instrs``, so the same wrapper serves both scopes.
    """

    is_transform = True

    def __init__(self, name: str,
                 factory: Callable[[StreamContext, List[AbstractInstruction]], Any],
                 *, preserves: Sequence[str] = (),
                 enabled: Optional[Callable[[StreamContext], bool]] = None,
                 scope: PassScope = PassScope.WHOLE_NEST):
        self.name = name
        self.preserves = preserves
        self.scope = scope
        self._factory = factory
        self._enabled = enabled

    def enabled(self, pc: StreamContext) -> bool:
        return True if self._enabled is None else self._enabled(pc)

    def run(self, pc: StreamContext) -> None:
        pc.instrs[:] = self.run_region(pc.instrs, pc)

    def run_region(self, instrs: List[AbstractInstruction],
                   pc: StreamContext) -> List[AbstractInstruction]:
        opt = self._factory(pc, instrs)
        opt.apply()
        return list(opt.get_instructions())


class Analysis(Pass):
    """An ``AbstractOptStage`` that computes one named result, as a pass.

    Always ``WHOLE_NEST``: an analysis whose result is consumed against the
    whole stream must be computed over the whole stream.
    """

    is_transform = False
    scope = PassScope.WHOLE_NEST

    def __init__(self, name: str, factory: Callable[[StreamContext], Any],
                 getter: Callable[[Any], Any], provides: str,
                 *, requires: Sequence[str] = ()):
        self.name = name
        self.provides = (provides,)
        self.requires = requires
        self._factory = factory
        self._getter = getter
        self._key = provides

    def run(self, pc: StreamContext) -> None:
        opt = self._factory(pc)
        opt.apply()
        pc.put(self._key, self._getter(opt))
