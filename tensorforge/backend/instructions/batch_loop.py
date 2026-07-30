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

from enum import Enum
from typing import List, Optional, Tuple

from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.common.context import Context

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
        self._is_ready = True

    # -- structure ------------------------------------------------------- #

    @property
    def region(self) -> List[AbstractInstruction]:
        return self._region

    def regions(self) -> Tuple[Tuple[AbstractInstruction, ...], ...]:
        return (tuple(self._region),)

    def uniform_scope(self) -> BarrierScope:
        """How far the trip count is uniform, i.e. the strongest barrier that
        may legally appear inside.

        ``PERSISTENT``: the count is ``ceil((numElements - start) / stride)``
        and ``start`` depends on the block id, so blocks disagree.  Threads
        within a block agree, hence ``GROUP``.  A ``GRID`` barrier here
        deadlocks: blocks with fewer iterations exit without arriving.

        ``LAUNCHCTRL``: the queue hands out a different number of elements per
        block, so likewise ``GROUP``.

        ``SINGLE``: exactly one iteration everywhere, so anything is fine.
        """
        if self._mode is LoopMode.SINGLE:
            return BarrierScope.GRID
        return BarrierScope.GROUP

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

    def _emit_body(self, writer) -> None:
        with writer.If(self._flag_guard(writer)):
            for instr in self._region:
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
            with writer.For(f'size_t {self._batch(0)} = {self._start}; '
                            f'{self._batch(0)} < {self._num_elements()}; '
                            f'{self._batch(0)} += {self._stride}'):
                self._lookahead_bindings(writer)
                self._emit_body(writer)
        elif self._mode is LoopMode.LAUNCHCTRL:
            writer(f'__shared__ tensorforge::ClusterLaunchCtrl launchctrl;')
            writer(f'int phase = 0;')
            writer(f'launchctrl.init();')
            writer(f'size_t {self._batch(0)} = {self._block_id()};')
            with writer.While(f'true'):
                writer('launchctrl.setupNext();')
                with writer.If(self._size_guard()):
                    self._emit_body(writer)
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
