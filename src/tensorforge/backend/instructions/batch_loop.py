# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""The per-element loop, as an instruction with a region.

It used to be raw text in ``Generator._generate_kernel``, written three times
over -- once per traversal strategy -- with the body handed in as a closure.
Two consequences followed from it not being in the IR:

*No prologue could be expressed.*  A software-pipelining pass needs to peel an
iteration, and with no loop to peel from it had to publish the peeled copy
through a second list (``OptimizationStage._global_instrs``) that the rest of
the pipeline neither indexed nor verified.  Definition and use ended up in
different streams, which is why both ``MultiBuffer`` and ``PtrPipe`` are
disabled.  With a region, a prologue is a peeled iteration in the same stream
and ``def_use`` sees it.

*Barrier legality was unrepresentable.*  Whether a barrier inside the loop is
legal depends on the trip count being uniform across the barrier's scope, and
the trip count is a property of the traversal strategy.  ``uniform_scope``
states it, so ``verify`` can check it instead of the invariant living in a
comment.
"""

import os
from enum import Enum
from typing import List, Optional, Tuple

from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError

from .abstract_instruction import AbstractInstruction, BarrierScope


class LoopMode(Enum):
    """How the next element is obtained."""

    SINGLE = 'single'          # one element per block, no loop
    PERSISTENT = 'persistent'  # grid-stride loop
    LAUNCHCTRL = 'launchctrl'  # hardware work queue (Blackwell cluster launch)


class BatchLoop(AbstractInstruction):
    def __init__(self,
                 context: Context,
                 section_index: int,
                 mode: LoopMode,
                 start: str,
                 stride: str,
                 region: List[AbstractInstruction],
                 lookahead: int = 2):
        super().__init__(context)
        self._section_index = section_index
        self._mode = mode
        self._start = start
        self._stride = stride
        self._region = list(region)
        # how many elements ahead are bound as batchid1, batchid2, ... for
        # prefetching; only the strided loop rebinds them per iteration
        self._lookahead = lookahead
        # Set by the pipelining pass; emits a rolling iteration counter.  Not
        # emitted otherwise, so a disabled pass leaves the text unchanged.
        self._stage_depth: Optional[int] = None
        # ids of leading region instructions emitted outside the flag guard
        self._unguarded: set = set()
        self._is_ready = True

    # -- structure ------------------------------------------------------- #

    @property
    def region(self) -> List[AbstractInstruction]:
        return self._region

    def regions(self) -> Tuple[Tuple[AbstractInstruction, ...], ...]:
        return (tuple(self._region),)

    def replace_region(self, index: int,
                       instrs: List[AbstractInstruction]) -> None:
        assert index == 0, f'BatchLoop has one region, not {index + 1}'
        self._region = list(instrs)

    def append(self, instr: AbstractInstruction) -> None:
        self._region.append(instr)

    def uniform_scope(self) -> BarrierScope:
        """How far the body's execution count is uniform, i.e. the strongest
        barrier that may legally appear inside.

        The answer is ``SIMD`` for every mode, because the element index is

            batchId0 = threadIdx.y + blockDim.y * blockIdx.x

        so it varies *within* a block, not just between blocks.

        ``PERSISTENT``: the trip count is
        ``ceil((numElements - batchId0) / stride)``.  Two thread groups in one
        block start at indices differing by their ``threadIdx.y``, so their trip
        counts differ by one whenever ``numElements`` is not a multiple of
        ``gridDim.x * blockDim.y`` -- and ``gridDim.x`` is occupancy-derived
        (``min(gridsize, numElements0)``), not ``ceil(numElements/blockDim.y)``,
        so alignment is a coincidence rather than a guarantee.  A block-wide
        barrier in the body is then reached a different number of times by
        different thread groups.

        ``SINGLE``: the body sits under ``if (batchId0 < numElements)``, the
        same non-uniform predicate.  This previously claimed ``GRID`` on the
        grounds that "exactly one iteration everywhere" -- which ignored the
        guard.  The tail block skips the body entirely, so a grid barrier there
        deadlocks.

        ``LAUNCHCTRL``: the queue hands out work per block, and the size guard
        is the same.
        """
        return BarrierScope.SIMD

    # -- data flow ------------------------------------------------------- #
    #
    # The union over the region, so the loop is not opaque to any pass.  A
    # symbol defined inside is *not* reported as a definition of the loop,
    # because it does not survive the iteration -- only the region's reads of
    # symbols defined outside are uses of the loop.

    def uses(self) -> Tuple:
        defined = set()
        out, seen = [], set()
        for instr in self._region:
            for sym in instr.uses():
                if id(sym) not in defined and id(sym) not in seen:
                    seen.add(id(sym))
                    out.append(sym)
            for sym in instr.defs():
                defined.add(id(sym))
        return tuple(out)

    def defs(self) -> Tuple:
        out, seen = [], set()
        for instr in self._region:
            for sym in instr.defs():
                if id(sym) not in seen:
                    seen.add(id(sym))
                    out.append(sym)
        return tuple(out)

    def accesses(self) -> Tuple:
        out = []
        for instr in self._region:
            out.extend(instr.accesses())
        return tuple(out)

    def barrier_scope(self) -> BarrierScope:
        """A loop containing a barrier synchronises, seen from outside."""
        inner = [i.barrier_scope() for i in self._region]
        return max(inner) if inner else BarrierScope.NONE

    def temp_shmem(self) -> int:
        return max((i.temp_shmem() for i in self._region), default=0)

    # -- emission -------------------------------------------------------- #

    def _batch(self, n: int = 0) -> str:
        return f'{GeneralLexicon.BATCH_ID_NAME}{n}'

    def index_name(self, lookahead: int = 0) -> str:
        """The variable holding the element index ``lookahead`` iterations ahead.

        Valid *inside* the region only.  ``index_name(0)`` is the loop variable.
        """
        if lookahead > self._lookahead:
            raise InternalError(
                f'loop binds {self._lookahead} lookahead indices, '
                f'{lookahead} requested')
        return self._batch(lookahead)

    def stage_counter_name(self) -> str:
        return f'pipeStage{self._section_index}'

    def request_stage_counter(self, depth: int) -> str:
        """Bind a counter that runs ``0, 1, ... depth-1, 0, ...`` per iteration.

        A rotating buffer has to be indexed by the *iteration*, not by the
        element.  Those are the same thing only in a unit-stride loop; here the
        strided loop advances the element index by ``gridDim.x * blockDim.y``
        per iteration, and the queue-driven loop takes whatever the hardware
        hands it.  Indexing stages by ``batchId0 % depth`` therefore fails
        twice over: the stage never alternates when ``depth`` divides the
        stride -- the common case, since the stride is a product of two block
        counts -- and the first iteration reads ``batchId0 % depth`` while a
        peeled prologue can only name a literal, so every thread group with an
        odd start index reads a stage nobody filled.

        ``SINGLE`` has one iteration and so nothing to rotate; it raises rather
        than returning a counter that would always read zero.
        """
        if self._mode is LoopMode.SINGLE:
            raise InternalError(
                'a single-iteration loop has no iteration to pipeline against')
        if depth < 2:
            raise InternalError(f'stage counter needs depth >= 2, got {depth}')
        if self._stage_depth not in (None, depth):
            raise InternalError(
                f'loop already pipelined at depth {self._stage_depth}, '
                f'cannot also serve depth {depth}')
        self._stage_depth = depth
        return self.stage_counter_name()

    def mark_unguarded(self, instrs) -> None:
        """Emit these region instructions *outside* the per-element flag guard.

        The guard is ``if (flags[batchId0])``: a runtime mask that skips
        individual elements.  Anything the loop carries across the back edge
        has to sit outside it, or a skipped element desynchronises it from the
        element sequence for the rest of the loop.  Two things in a pipelined
        loop do:

        * the rolling pointer's advance --- skip it once and the pointer
          trails the loop variable permanently, so every later iteration
          computes on the wrong element;
        * the prefetch itself, issued in iteration ``k`` for element
          ``k + 1`` --- skip it and iteration ``k + 1`` reads a stage nobody
          filled.

        Neither is specific to rotation: the address-advance half has the same
        hole.  The marked instructions must form a prefix of the region,
        because the guard is one contiguous block; the pass moves them there.
        """
        self._unguarded = {id(i) for i in instrs}

    def _split_guard(self):
        """``(unguarded prefix, guarded remainder)``."""
        if not self._unguarded:
            return [], list(self._region)
        cut = 0
        while cut < len(self._region) and id(self._region[cut]) in self._unguarded:
            cut += 1
        stray = [i for i in self._region[cut:] if id(i) in self._unguarded]
        if stray:
            raise InternalError(
                f'{len(stray)} instruction(s) marked unguarded do not form a '
                f'prefix of the region, first is {type(stray[0]).__name__}; '
                f'the flag guard is one block and cannot be reopened')
        return self._region[:cut], self._region[cut:]

    def prologue_index(self) -> str:
        """The index a peeled iteration should use.

        The loop variable does not exist before the loop, so a peeled iteration
        cannot name it.  The generator binds ``batchId_start`` ahead of the
        loop for exactly this.
        """
        return f'{GeneralLexicon.BATCH_ID_NAME}_start'

    def _num_elements(self) -> str:
        return f'{GeneralLexicon.NUM_ELEMENTS}{self._section_index}'

    def _block_id(self, block: Optional[str] = None) -> str:
        lexic = self._vm.get_lexic()
        if block is None:
            block = lexic.block_idx_x
        return f'{lexic.thread_idx_y} + {lexic.block_dim_y} * ({block})'

    def _size_guard(self) -> str:
        return f'{self._batch(0)} < {self._num_elements()}'

    def _flag_guard(self, writer) -> str:
        flags = f'{GeneralLexicon.FLAGS_NAME}{self._section_index}'
        writer(f'bool allowed = true;')
        with writer.If(f'{flags} != nullptr'):
            writer(f'allowed = static_cast<bool>({flags}[{self._batch(0)}]);')
        return 'allowed'

    def _lookahead_bindings(self, writer) -> None:
        """Bind batchid1..N as clamped element indices, for prefetching."""
        for n in range(1, self._lookahead + 1):
            prev = self._batch(n - 1)
            writer(f'const auto {self._batch(n)} = '
                   f'{prev} + {self._stride} < {self._num_elements()} ? '
                   f'{prev} + {self._stride} : {prev};')

    def _declare_stage_counter(self, writer) -> None:
        if self._stage_depth is None:
            return
        writer(f'uint32_t {self.stage_counter_name()} = 0;')

    def _advance_stage_counter(self, writer) -> None:
        """Advance the counter *outside* the flag guard.

        Inside would tie the counter to the compute, and the transfer it indexes
        is issued for the *next* element -- so a skipped element desynchronises
        them either way.  Outside is the placement that stays correct once the
        pipelined transfer is hoisted out of the guard, which is what closing
        that hole needs; see the note in opt/pipeline.py.
        """
        if self._stage_depth is None:
            return
        d = self._stage_depth
        name = self.stage_counter_name()
        if d & (d - 1) == 0:
            writer(f'{name} = ({name} + 1) & {d - 1};')
        else:
            writer(f'{name} = ({name} + 1) % {d};')

    def _emit_body(self, writer) -> None:
        head, guarded = self._split_guard()
        for instr in head:
            instr.gen_code(writer)
        with writer.If(self._flag_guard(writer)):
            if os.environ.get('TF_IR_WIDE'):
                # one body for every instruction of the region
                with AbstractInstruction.shared_body(self._context, writer):
                    for instr in guarded:
                        instr.gen_code(writer)
            else:
                for instr in guarded:
                    instr.gen_code(writer)

    def gen_code(self, writer) -> None:
        # Deliberately no writer.Scope() and no comment: the loop used to be
        # emitted inline by the generator, and adding either would change the
        # generated text.
        self.gen_code_inner(writer)

    def gen_code_inner(self, writer) -> None:
        if self._mode is LoopMode.PERSISTENT:
            # TODO: OMP target
            # TODO: maybe iterate over adjacent elements? (for indirect pointers)
            self._declare_stage_counter(writer)
            with writer.For(f'size_t {self._batch(0)} = {self._start}; '
                            f'{self._batch(0)} < {self._num_elements()}; '
                            f'{self._batch(0)} += {self._stride}'):
                self._lookahead_bindings(writer)
                self._emit_body(writer)
                self._advance_stage_counter(writer)
        elif self._mode is LoopMode.LAUNCHCTRL:
            self._declare_stage_counter(writer)
            writer(f'__shared__ tensorforge::ClusterLaunchCtrl launchctrl;')
            writer(f'int phase = 0;')
            writer(f'launchctrl.init();')
            writer(f'size_t {self._batch(0)} = {self._block_id()};')
            with writer.While(f'true'):
                writer('launchctrl.setupNext();')
                with writer.If(self._size_guard()):
                    self._emit_body(writer)
                self._advance_stage_counter(writer)
                writer('const auto nextBlock = launchctrl.queryNext(phase);')
                with writer.If('!nextBlock.has_value()'):
                    writer('break;')
                writer(f'{self._batch(0)} = '
                       f'{self._block_id("nextBlock.value()")};')
        else:
            writer(f'const size_t {self._batch(0)} = {self._block_id()};')
            with writer.If(self._size_guard()):
                self._emit_body(writer)

    def __str__(self) -> str:
        return (f'batchloop.{self._mode.value} '
                f'[{self._num_elements()}] '
                f'uniform={self.uniform_scope().name.lower()} '
                f'({len(self._region)} instructions)')
