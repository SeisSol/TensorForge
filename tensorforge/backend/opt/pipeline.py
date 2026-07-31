# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Software pipelining over the batch loop.

Depth ``d`` means: iteration ``k`` issues the transfer for element ``k + d - 1``
while computing on element ``k``.  ``d = 2`` is plain double buffering.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.batch_loop import BatchLoop
from tensorforge.backend.instructions.allocate import RegisterAlloc, ShrMemAlloc
from tensorforge.backend.instructions.memory import AbstractShrMemWrite
from tensorforge.backend.instructions.memory.load import (GlbToRegLoader,
                                                          GlbToShrLoader,
                                                          LoadInstruction,
                                                          LoadWait)
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.pir.core import accesses_conflict
from tensorforge.backend.symbol import Symbol
from tensorforge.common.exceptions import InternalError

from .abstract import AbstractTransformer, Context


class Candidate(NamedTuple):
    """A transfer that could be moved one or more iterations earlier."""

    load: LoadInstruction
    producer: GetElementPtr      # the GEP that computes the source pointer
    load_index: int              # position of the load within the region
    producer_index: int
    first_use_index: int         # first consumer of the loaded buffer

    @property
    def slack(self) -> int:
        """Instructions between the load and its first consumer.

        Zero slack is the case pipelining exists for: the consumer waits on the
        transfer with nothing to overlap it with.
        """
        return self.first_use_index - self.load_index - 1


class PipelineAnalysis:
    """Which transfers in a loop body may be pipelined, and why not.

    Kept separate from the transform so that the decision is inspectable: the
    old passes made it implicitly, by transforming everything they recognised.
    """

    def __init__(self, body: Sequence[AbstractInstruction]):
        self._body = list(body)
        self.candidates: List[Candidate] = []
        self.rejected: List[tuple] = []

    def run(self) -> 'PipelineAnalysis':
        # producer map: symbol id -> (index, instruction) that defines it
        producers: Dict[int, tuple] = {}
        for index, instr in enumerate(self._body):
            for sym in instr.defs():
                producers[id(sym)] = (index, instr)

        # Any global->local transfer, not just the shared-memory one.  Measured
        # over the corpus, the two backends produce structurally different loop
        # bodies: the CUDA path emits GlbToShrLoader, the AMD path
        # GlbToRegLoader, because preloading the Addressing.NONE operators into
        # LDS up front leaves only register loads per element.  A pass that
        # matched one class would silently do nothing on the other backend --
        # which is what the old MultiBuffer did.
        for index, instr in enumerate(self._body):
            if not isinstance(instr, (GlbToShrLoader, GlbToRegLoader)):
                continue
            reason = self._reject(index, instr, producers)
            if reason is not None:
                self.rejected.append((instr, reason))
                continue
            src = self._src_of(instr)
            producer_index, producer = producers[id(src)]
            self.candidates.append(Candidate(
                load=instr,
                producer=producer,
                load_index=index,
                producer_index=producer_index,
                first_use_index=self._first_use(index, self._dest_of(instr))))
        return self

    # NOTE: defs()/uses(), never get_src()/get_dest().  GlbToShrLoader has both
    # accessors, GlbToRegLoader has neither -- the asymmetry the data-flow
    # interface exists to hide.  A pass that reaches for an accessor name works
    # on one loader class and raises AttributeError on the other.
    @staticmethod
    def _src_of(instr):
        uses = instr.uses()
        return uses[0] if uses else None

    @staticmethod
    def _dest_of(instr):
        defs = instr.defs()
        return defs[0] if defs else None

    def _reject(self, index, instr, producers) -> Optional[str]:
        src = self._src_of(instr)
        if src is None:
            return 'load does not report a source symbol'
        if id(src) not in producers:
            return ('source pointer is not produced in this region, so there '
                    'is nothing to advance')
        producer_index, producer = producers[id(src)]
        if not isinstance(producer, GetElementPtr):
            return (f'source pointer comes from {type(producer).__name__}, '
                    f'not a GetElementPtr')
        if producer_index > index:
            # this is the implicit ordering MultiBuffer assumed and would have
            # died on with a KeyError
            return 'source pointer is produced after the load'
        if isinstance(producer._batch_offset, str):
            return 'source pointer already uses a named index; already pipelined'
        dest = self._dest_of(instr)
        if dest is None:
            return 'load does not report a destination symbol'
        if self._first_use(index, dest) is None:
            return 'loaded buffer is never read in this region'
        if self._read_before(index, dest):
            return ('buffer is read before the load, so advancing the load '
                    'would change what that read observes')
        # A buffer written more than once per iteration cannot be rotated by
        # index: rotation assumes stage k holds element k's data for the whole
        # iteration.
        #
        # An allocation is not a write of data, so it does not count -- with it
        # counted, every register buffer looked unrotatable, since RegisterAlloc,
        # the load and the accumulating compute all appear in defs().
        # LoadWait is excluded too: it delegates defs() to the transfer it
        # awaits, which is right for ordering -- consumers must come after the
        # wait, not merely after the issue -- but it completes that write rather
        # than starting a second one.
        writers = [i for i in self._body
                   if any(d is dest for d in i.defs())
                   and not isinstance(i, (RegisterAlloc, ShrMemAlloc, LoadWait))]
        if len(writers) > 1:
            kinds = ', '.join(sorted({type(i).__name__ for i in writers}))
            return (f'buffer is written {len(writers)} times per iteration '
                    f'({kinds}); rotation assumes one write per stage')
        return None

    def _first_use(self, after: int, sym) -> Optional[int]:
        for index in range(after + 1, len(self._body)):
            instr = self._body[index]
            if isinstance(instr, LoadWait):
                continue
            if any(u is sym for u in instr.uses()):
                return index
        return None

    def _read_before(self, before: int, sym) -> bool:
        for index in range(before):
            if any(u is sym for u in self._body[index].uses()):
                return True
        return False

    def report(self) -> str:
        lines = [f'pipeline: {len(self.candidates)} candidate(s), '
                 f'{len(self.rejected)} rejected']
        for c in self.candidates:
            lines.append(f'  + {self._dest_of(c.load).name} '
                         f'(slack {c.slack}, load@{c.load_index}, '
                         f'first use@{c.first_use_index})')
        for instr, reason in self.rejected:
            dest = self._dest_of(instr)
            lines.append(f'  - {getattr(dest, "name", "?")}: {reason}')
        return '\n'.join(lines)


class Pipeline(AbstractTransformer):
    """Advance transfers by ``depth - 1`` iterations.

    Runs on the top level, where the ``BatchLoop`` sits, because the peeled
    iteration has to land *outside* the loop.
    """

    def __init__(self,
                 context: Context,
                 instructions: List[AbstractInstruction],
                 depth: int = 2,
                 rotate_buffers: bool = False):
        super(Pipeline, self).__init__(context, instructions)
        if depth < 2:
            raise ValueError(f'pipelining needs depth >= 2, got {depth}')
        self._depth = depth
        self._rotate = rotate_buffers
        self.analyses: List[PipelineAnalysis] = []

    def apply(self) -> None:
        out: List[AbstractInstruction] = []
        for instr in self._instrs:
            if not isinstance(instr, BatchLoop):
                out.append(instr)
                continue
            prologue, body = self._pipeline_loop(instr)
            out.extend(prologue)
            instr.replace_region(0, body)
            out.append(instr)
        self._instrs = out

    def _pipeline_loop(self, loop: BatchLoop):
        analysis = PipelineAnalysis(loop.region).run()
        self.analyses.append(analysis)
        if not analysis.candidates:
            return [], list(loop.region)

        if self._rotate:
            # See the module docstring for what is in place and the note below
            # for what is not.
            raise InternalError(
                'buffer rotation is not wired up yet. The allocation side is '
                'done -- AbstractShrMemWrite.set_stages reserves every stage '
                'and the declaration selects one at run time -- but the '
                'transform needs two *pointers* into that allocation: the '
                'consumer must read stage k % d while the advanced transfer '
                'writes stage (k + d - 1) % d, and a GlbToShrLoader writes '
                'through the single pointer its own declaration emits. That '
                'needs the declaration split from the write address, which is '
                'a change to AbstractShrMemWrite, not to this pass.')

        return self._advance_pointers(loop, analysis)

    def _advance_pointers(self, loop: BatchLoop, analysis: PipelineAnalysis):
        """Hoist the address computation, leaving the transfers in place.

        This is what ``PtrPipe`` attempted, minus transforming every
        ``GetElementPtr`` whether or not it fed anything: only producers of a
        pipelineable transfer are advanced, and the peeled copy uses the loop's
        prologue index rather than the loop variable, which does not exist
        outside the loop.

        No buffer rotation, so no aliasing question: the transfer still writes
        the same buffer in the same iteration, only its source pointer is
        computed one iteration earlier.
        """
        prologue: List[AbstractInstruction] = []
        body = list(loop.region)
        advanced = {id(c.producer) for c in analysis.candidates}

        for index, instr in enumerate(body):
            if id(instr) not in advanced:
                continue
            original = instr._dest
            # A *distinct* symbol for the rolling pointer.  Reusing `original`
            # for both `dest` and `update_dest` emits
            #     const auto glb_m0 = glb_m0;
            #     glb_m0 = &m0[...];
            # -- a self-referential declaration of a name that was also declared
            # const in the prologue.  The rolling pointer and the value this
            # iteration reads are two different things and need two names.
            rolling = Symbol(f'pipe_{original.name}',
                             original.stype,
                             original.obj)
            rolling.data_view = original.data_view

            # Peeled iteration: compute the pointer for the first element, and
            # declare it mutable (`pipeline=True` drops the const) so the body
            # can advance it.
            prologue.append(GetElementPtr(
                self._context,
                src=instr._src,
                dest=rolling,
                include_extra_offset=instr._include_extra_offset,
                batch_offset=loop.prologue_index(),
                pipeline=True))

            # In the loop: hand this iteration the pointer computed last time,
            # then advance to the element `depth - 1` ahead.
            body[index] = GetElementPtr(
                self._context,
                src=instr._src,
                dest=rolling,
                include_extra_offset=instr._include_extra_offset,
                batch_offset=loop.index_name(self._depth - 1),
                update_dest=original,
                pipeline=True)
        return prologue, body
