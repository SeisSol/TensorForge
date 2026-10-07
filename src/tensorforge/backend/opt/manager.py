# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""The macro instruction stream as something `PassManager` runs passes over.

The manager itself is IR-agnostic (`backend.passmanager`).  What is specific
to this level lives here: the stream and the prologue it may read but not
rewrite, the macro verifier between passes, and the adapter that turns an
optimization stage into a pass.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.passmanager import Pass, PassContext
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
        # undefined -- a preloaded shared-memory buffer, for one.
        self.global_ir: List[AbstractInstruction] = list(global_ir or [])

    @property
    def stream(self) -> List[AbstractInstruction]:
        """Everything that will be emitted for this section, prologue first.

        For whole-section checks (verify, dump).  What the passes rewrite is
        `instrs` alone: the prologue is built before this stage and never
        routed through it.
        """
        return self.global_ir + self.instrs

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
                       # readiness needs the windows declared, which happens
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
    the pass context and the stream.
    """

    is_transform = True

    def __init__(self, name: str,
                 factory: Callable[[StreamContext, List[AbstractInstruction]], Any],
                 *, preserves: Sequence[str] = (),
                 enabled: Optional[Callable[[StreamContext], bool]] = None):
        self.name = name
        self.preserves = preserves
        self._factory = factory
        self._enabled = enabled

    def enabled(self, pc: StreamContext) -> bool:
        return True if self._enabled is None else self._enabled(pc)

    def run(self, pc: StreamContext) -> None:
        opt = self._factory(pc, pc.instrs)
        opt.apply()
        pc.instrs[:] = list(opt.get_instructions())
