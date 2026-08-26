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

        prologue, body = self._advance_pointers(loop, analysis)
        if self._rotate:
            prologue, body = self._rotate_buffers(loop, analysis, prologue, body)
        body = self._hoist_out_of_guard(loop, body)
        return prologue, body

    def _hoist_out_of_guard(self, loop: BatchLoop, body):
        """Move everything carried across the back edge ahead of the flag guard.

        Order within the prefix is the order it was registered in: the pointer
        advances first, because the transfer reads the advanced pointer, then
        the wait, then the transfer.
        """
        if not self._hoist:
            return body
        marked = {id(i) for i in self._hoist}
        rest = [i for i in body if id(i) not in marked]
        head = [i for i in self._hoist if any(o is i for o in body)]
        loop.mark_unguarded(head)
        return head + rest

    def _rotate_buffers(self, loop: BatchLoop, analysis: PipelineAnalysis,
                        prologue, body):
        """Give each pipelined shared-memory buffer ``depth`` stages, and move
        the transfer along with the pointer.

        Advancing the pointer alone is only address pipelining: the load still
        happens in iteration k for iteration k, so nothing overlaps.  Rotation
        moves the transfer too --- iteration k fills stage ``(k + d - 1) % d``
        with element ``k + d - 1`` while consuming stage ``k % d``, filled
        ``d - 1`` iterations ago --- which is what needs the two views into one
        allocation.

        The first ``d - 1`` elements have no earlier iteration to fill them, so
        the prologue peels that many transfers.  With ``d = 2`` that is one:
        element 0 into stage 0, using the pointer the peeled ``GetElementPtr``
        already computed.

        The prefetch is hoisted out of the per-element flag guard (see
        ``_hoist_out_of_guard``), because it is issued in iteration ``k`` for
        element ``k + 1``: left under the guard, a skipped element would leave
        the stage it should have filled untouched while iteration ``k + 1``
        read it anyway.  No placement of the stage counter fixes that -- inside
        the guard the counter and the element sequence drift apart, outside it
        the stage is simply unfilled -- so the transfer has to move instead.

        The stage is indexed by the loop's *iteration counter*, not by
        ``batchId0``.  Those differ: the batch loop is grid-strided, so one
        thread's consecutive elements are ``stride`` apart, and ``stride`` is
        ``gridDim.x * blockDim.y`` -- almost always even.  With ``d = 2`` that
        makes ``batchId0 % d`` invariant over a thread's iterations, so the
        consumer stays pinned to one stage while the producer fills the other,
        and every iteration after the first re-reads the prologue's element.
        The prologue compounds it: it can only name a literal stage, so it
        writes stage 0, while iteration 0 reads ``batchId0 % d`` -- unfilled
        for every thread group with an odd start index.  Neither shows up as a
        crash, only as wrong numbers.
        """
        d = self._depth
        if d != 2:
            # Stages 0..d-2 would all have to be filled by the prologue, each
            # from a *different* element, but the peeled GetElementPtr leaves
            # exactly one pointer behind (element 0).  Filling them all from it
            # would put element 0 in every stage.  Raising beats emitting that.
            raise InternalError(
                f'buffer rotation is implemented for depth 2, got {d}: '
                f'the prologue would have to peel {d - 1} transfers from '
                f'{d - 1} distinct element pointers, and only the first is '
                f'computed')
        k = loop.request_stage_counter(d)
        read = f'{k}'
        write = f'({k} + {d - 1}) % {d}'

        peeled: List[AbstractInstruction] = []
        for c in analysis.candidates:
            load = c.load
            if not isinstance(load, AbstractShrMemWrite):
                # a register destination is private to the thread; there is no
                # allocation to stage
                continue
            rolling = self._advanced_ptr.get(id(load.get_src()))
            if rolling is None:
                # the producer was not advanced, so there is no pointer to the
                # element this iteration should be fetching
                continue

            # The body transfer now reads through the advanced pointer, i.e.
            # element k + d - 1, and fills the stage that element belongs to.
            replacement = load.clone(src=rolling)
            replacement.set_stages(d, read, write)
            body[body.index(load)] = replacement
            # clone() registered the replacement as a *new* user; the symbol
            # must forget the object that is no longer in the stream, or
            # ShrMemOpt sizes the region from it and the extra stages are never
            # reserved.  Order matters: drop the appended duplicate first, then
            # substitute, because list.remove() takes the earliest match and
            # would otherwise delete the entry at the replaced position.
            for sym in (replacement.get_dest(), replacement.get_src()):
                users = sym.get_user_list()
                while replacement in users:
                    users.remove(replacement)
                if not sym.replace_user(load, replacement):
                    users.append(replacement)

            waits = [j for j, o in enumerate(body)
                     if isinstance(o, LoadWait) and o.awaited() is load]
            for j in waits:
                body[j] = LoadWait(replacement)
            # The wait goes ahead of the transfer, and both go ahead of the
            # flag guard.
            #
            # Ahead of the transfer: there is one cuda::pipeline per section and
            # it is a FIFO.  With the wait after the commit, the batch this
            # iteration issues and the one the previous iteration issued are
            # both outstanding at producer_acquire(), which needs two stages --
            # more than the thread-scope pipeline from cuda::make_pipeline() has,
            # so iteration 0 blocks on a slot that only frees further down.  That
            # is why the NVIDIA path hung while AMD, which emits no pipeline
            # object at all, was fine.  Waiting first keeps only one batch in
            # flight without giving up memcpy_async: FIFO order means
            # consumer_wait() retires the batch issued in iteration k-1 --
            # exactly the stage this iteration reads.
            #
            # This is what the token in the async pir instructions states
            # explicitly: the wait consumes the token of the *previous*
            # iteration.  Here the FIFO supplies that implicitly.
            self._hoist.extend(body[j] for j in sorted(waits))
            self._hoist.append(replacement)
            load = replacement

            # ... and the prologue fills the stages nobody else will.  The
            # peeled GetElementPtr left `rolling` pointing at element 0.
            for stage in range(d - 1):
                fill = load.clone(src=rolling)
                # The peeled fill writes stage `stage` through its own view.
                # It must *not* declare the buffer: the declaration addresses
                # `k % d` and so mentions the loop variable, which does not
                # exist in the prologue.  Cloning registers the copy as a later
                # user of the symbol, so ShrMemOpt leaves declaring to the body
                # transfer, which is still the first user.
                fill.set_stages(d, read, str(stage))
                peeled.append(fill)

        if not peeled:
            return prologue, body
        return list(prologue) + peeled, body

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
        advanced_ptr = self._advanced_ptr = {}
        hoist = self._hoist = []

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
            advanced_ptr[id(original)] = rolling
            hoist.append(body[index])
        return prologue, body
