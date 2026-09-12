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

from contextlib import contextmanager
from enum import Enum
from typing import List, Optional, Tuple

from tensorforge.common.basic_types import FlagMode, GeneralLexicon
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError

from tensorforge.backend import elementmask
from tensorforge.backend.pir.core import Participants, Uniformity
from tensorforge.common.threads import mults_per_group
from .abstract_instruction import AbstractInstruction


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
                 lookahead: int = 2,
                 flags: FlagMode = FlagMode.OPTIONAL,
                 queue_depth: int = 1,
                 group_size: int = 1,
                 narrow_group: bool = False):
        super().__init__(context)
        self._section_index = section_index
        self._mode = mode
        self._start = start
        self._stride = stride
        self._region = list(region)
        # whether this section has a `flags{i}` parameter, and whether it may
        # be null; `ABSENT` leaves the body unguarded
        self._flags = flags
        # how many elements ahead are bound as batchid1, batchid2, ... for
        # prefetching; only the strided loop rebinds them per iteration
        self._lookahead = lookahead
        # Set by the pipelining pass; emits a rolling iteration counter.  Not
        # emitted otherwise, so a disabled pass leaves the text unchanged.
        self._induction = None
        self._first_lookahead = None
        self._loop_handle = None
        self._stage_depth: Optional[int] = None
        # ids of leading region instructions emitted outside the flag guard
        self._unguarded: set = set()
        # Emitted inside the loop's own body, around the `for`: the peeled
        # transfers of `WrapLoads` ahead of it and their drains after it.  See
        # `add_wrap_prologue`.
        self._wrap_prologue: List[AbstractInstruction] = []
        self._wrap_epilogue: List[AbstractInstruction] = []
        # `LAUNCHCTRL` only: how many cancel requests are kept in flight.  One
        # is enough to hide the queue's own latency behind the body, because
        # the request for the next element is posted before the current one is
        # computed.  It is *not* enough to prefetch the next element's data:
        # at depth one the next index arrives at the bottom of the iteration,
        # with no body left to overlap a transfer with.  Two is the first
        # depth at which the index is known at the top.
        if mode is LoopMode.LAUNCHCTRL and queue_depth < 1:
            raise InternalError(
                f'the work queue needs at least one request in flight, '
                f'got {queue_depth}')
        self._queue_depth = queue_depth
        # How many multiplications share this loop's block.  Set once the
        # thread-block policy has decided; None until then, and `uniform_scope`
        # answers conservatively while it is.
        self._mults_per_block: Optional[int] = None
        # How many multiplications share a whole number of waves, and so the
        # smallest set a hardware barrier can separate.  `1` disables the group
        # traversal entirely and leaves the emission per row, which is what a
        # rotated section asks for: its start is taken modulo the stride, and a
        # leader's start plus a lane offset is then not the row's own element.
        self._group_size: int = max(1, int(group_size))
        # Whether the group may be narrower than the block.  Asked for by a
        # wave-collective instruction (`convergence_scope`), which needs the
        # rows of one wave in step and nothing of the rest of the block.
        self._narrow_group = bool(narrow_group)
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

    def set_mults_per_block(self, mults: int) -> None:
        self._mults_per_block = int(mults)

    def _grouped(self) -> bool:
        """Whether the traversal is driven by a group of rows rather than one.

        Only where the group is the whole block.  A group narrower than the
        block leaves the rows outside it free to run the body a different
        number of times, and the barrier this buys reaches the block -- so it
        would be licensing exactly what it is meant to make safe.  Sizing the
        block to the group is `AbstractThreadBlockPolicy._barrier_cap`.

        `LAUNCHCTRL` is excluded: its hand-off carries a block barrier of its
        own outside the size guard, and a second traversal on top of that is a
        second answer to a question already answered.

        Narrower than the block where that was asked for: a wave-collective
        instruction needs the group in step and nothing more, so the block
        may hold several groups -- and the loop is then uniform per group,
        which `uniform_scope` says, rather than per block.
        """
        return (self._group_size > 1
                and (self._narrow_group
                     or self._mults_per_block == self._group_size)
                and self._mode is not LoopMode.LAUNCHCTRL)

    def _group_batch(self) -> str:
        return f'{GeneralLexicon.BATCH_ID_NAME}Group{self._section_index}'

    def _lane(self) -> str:
        return f'{GeneralLexicon.BATCH_ID_NAME}Lane{self._section_index}'

    def _active(self) -> str:
        return f'{GeneralLexicon.BATCH_ID_NAME}Active{self._section_index}'

    def _group_start(self) -> str:
        """The leader's element index: the row rounded down to its group."""
        lexic = self._vm.get_lexic()
        row = f'({lexic.thread_idx_y} - {self._lane()})'
        return f'{row} + {lexic.block_dim_y} * ({lexic.block_idx_x})'

    def _declare_lane(self, writer) -> None:
        lexic = self._vm.get_lexic()
        writer(f'const auto {self._lane()} = '
               f'{lexic.thread_idx_y} % {self._group_size};')

    def _declare_row_element(self, writer) -> str:
        """Bind this row's element and whether it has one, and return the mask.

        The index is clamped to the leader's, which the loop condition has just
        established is in range, so a row with no element of its own reads a
        duplicate rather than past the end.  What keeps the duplicate out of
        the result is the mask, which every global write runs under.
        """
        raw = f'{self._group_batch()} + {self._lane()}'
        cond = f'{raw} < {self._num_elements()}'
        if self._flags is not FlagMode.ABSENT:
            flags = f'{GeneralLexicon.FLAGS_NAME}{self._section_index}'
            read = f'static_cast<bool>({flags}[{raw}])'
            if self._flags is FlagMode.OPTIONAL:
                read = f'({flags} == nullptr || {read})'
            # Folded into the mask instead of wrapping the body in a block.
            # A flags array is read per element, so the block would be entered
            # by some rows of the group and not others -- and the barrier the
            # group exists for sits inside it.
            cond = f'{cond} && {read}'
        writer(f'const bool {self._active()} = {cond};')
        writer(f'const size_t {self._batch(0)} = '
               f'{self._active()} ? {raw} : {self._group_batch()};')
        return self._active()

    def uniform_scope(self) -> Uniformity:
        """How far the body's execution count is uniform, i.e. the strongest
        barrier that may legally appear inside.

        Two things narrow it, and the answer is the weaker of them: how often
        the loop runs, and what the body sits under.  Both turn on the element
        index

            batchId0 = threadIdx.y + blockDim.y * blockIdx.x

        which is the same for every thread of one multiplication and different
        between the multiplications packed into a block.

        *The trip count.*  ``PERSISTENT`` runs
        ``ceil((numElements - batchId0) / stride)`` times.  Two rows of one
        block start at indices differing by their ``threadIdx.y``, so their
        trip counts differ by one whenever ``numElements`` is not a multiple of
        ``gridDim.x * blockDim.y`` -- and ``gridDim.x`` is occupancy-derived
        (``min(gridsize, numElements0)``), not ``ceil(numElements/blockDim.y)``,
        so alignment is a coincidence rather than a guarantee.  ``SINGLE`` runs
        once or not at all.  ``LAUNCHCTRL`` is the exception: every thread
        reads the same cancel response out of shared memory behind a barrier,
        so all of them leave on the same iteration.

        *The guards.*  The body sits under the size guard ``batchId0 <
        numElements``, and under ``flags[batchId0]`` where the section has a
        flags array.  Both are per element, so both are decided per row.

        At one multiplication per block the two coincide: ``blockDim.y == 1``
        makes ``batchId0`` equal to ``blockIdx.x``, so the trip count, the size
        guard and the flag guard are alike for every thread of the block, and a
        block barrier in the body meets all of them.  That is the whole reason
        `Lexic.has_sync_mult` reaches the thread-block policy: a target with no
        sub-block rendezvous can still run a multiplication wider than a wave,
        by being given a block that holds nothing else.

        Above one, a row that is masked off does not arrive, so the answer is
        the multiplication.  Widening it to a group of rows that share a wave
        needs the mask off the body and onto the accesses it guards, which is
        what an unguarded body would have to mean.
        """
        if self._mults_per_block == 1 or (
                self._grouped() and self._mults_per_block == self._group_size):
            return Uniformity.BLOCK
        if self._grouped():
            return Uniformity.MULTGROUP
        return Uniformity.MULT

    # -- data flow ------------------------------------------------------- #
    #
    # The union over the region, so the loop is not opaque to any pass.  A
    # symbol defined inside is *not* reported as a definition of the loop,
    # because it does not survive the iteration -- only the region's reads of
    # symbols defined outside are uses of the loop.

    def _emitted(self) -> List[AbstractInstruction]:
        """Everything this instruction emits, in the order it emits it.

        The region, and around it what `WrapLoads` handed over to be emitted
        in the loop's own body: the peels ahead of the `for`, the drains after
        it.  Data flow is read off this and not off the region alone, because
        the peel is what defines a wrapped buffer before the region reads it.
        """
        return self._wrap_prologue + self._region + self._wrap_epilogue

    def entry_defs(self) -> Tuple:
        """What is defined on the way into the region: the peeled transfers.

        A wrapped buffer is read at the head of the body and written at its
        tail, so `entering` calls it carried -- initialised ahead of the loop,
        updated across the back edge.  The initialisation is emitted by this
        instruction, not by one before it, so a check that walks the region
        with only the stream's definitions in hand has to be told.
        """
        out, seen = [], set()
        for instr in self._wrap_prologue:
            for sym in instr.defs():
                if id(sym) not in seen:
                    seen.add(id(sym))
                    out.append(sym)
        return tuple(out)

    def uses(self) -> Tuple:
        defined = set()
        out, seen = [], set()
        for instr in self._emitted():
            for sym in instr.uses():
                if id(sym) not in defined and id(sym) not in seen:
                    seen.add(id(sym))
                    out.append(sym)
            for sym in instr.defs():
                defined.add(id(sym))
        return tuple(out)

    def defs(self) -> Tuple:
        out, seen = [], set()
        for instr in self._emitted():
            for sym in instr.defs():
                if id(sym) not in seen:
                    seen.add(id(sym))
                    out.append(sym)
        return tuple(out)

    def accesses(self) -> Tuple:
        out = []
        for instr in self._emitted():
            out.extend(instr.accesses())
        return tuple(out)

    def barrier_scope(self) -> Uniformity:
        """A loop containing a barrier synchronises, seen from outside."""
        inner = [i.barrier_scope() for i in self._region]
        return max((s for s in inner if s is not None), default=None)

    def temp_shmem(self) -> int:
        return max((i.temp_shmem() for i in self._emitted()), default=0)

    # -- emission -------------------------------------------------------- #

    def _batch(self, n: int = 0) -> str:
        return f'{GeneralLexicon.BATCH_ID_NAME}{n}'

    # -- the indices this loop binds, for the body it binds them in -------- #

    #: One entry per open loop: the builder it is being emitted into, and what
    #: it has bound so far.
    _indices: List[Tuple] = []

    #: One entry per open loop that carries tokens: the builder, and each
    #: wrapped transfer's carried tokens -- see `carried_tokens`.
    _carried: List[Tuple] = []

    #: While the body of a loop that carries the mask is being emitted: the
    #: flag words of this element and the next, as the loop hands them in --
    #: see `_carries_flags`.
    _carried_flags: Optional[Tuple] = None

    @classmethod
    @contextmanager
    def batch_indices(cls, builder):
        """Publish the element indices this loop binds, while its body is open.

        A loop binds more than one -- ``batchId0`` is the element the body is
        on, ``batchId1..N`` are the ones it will be on -- and an address is
        entitled to name any of them, so this is a mapping and not a single
        value.  Keyed by the name the macro layer gives each index, which is
        the thing an address would otherwise have spelled.  Taking the
        innermost loop's induction instead answers a question about
        ``batchId0`` with whatever loop was entered last, and a ``VariantLoop``
        inside this one binds a counter that is not an element index at all.

        Filled as the bindings are emitted rather than up front, so that an
        instruction built before them sees exactly what exists at that point.

        Carried with the builder it belongs to, and that half is not
        bookkeeping.  A value belongs to one body and means nothing in another,
        and the same *name* means two different things on either side of the
        loop header: ``batchId1`` is the next element inside the body and the
        clamped first element ahead of it.  Handing one out across that
        boundary would address a different element and compile perfectly well.
        """
        frame = (builder, {})
        cls._indices.append(frame)
        try:
            yield frame[1]
        finally:
            cls._indices.pop()

    @classmethod
    def carried_tokens(cls, builder, transfer) -> list:
        """The tokens a loop in `builder`'s body carries for `transfer`.

        The iteration arguments while the body is being built, the loop's
        results once it is closed -- whichever the loop has published when
        asked.  Empty where no loop in this body carries the transfer, which
        is the caller's cue to drain.
        """
        for owner, table in reversed(cls._carried):
            if owner is builder:
                return list(table.get(id(transfer), ()))
        return []

    @classmethod
    def _push_carried(cls, builder) -> dict:
        table: dict = {}
        cls._carried.append((builder, table))
        return table

    @classmethod
    def _pop_carried(cls) -> None:
        cls._carried.pop()

    def _carried_transfers(self) -> List[AbstractInstruction]:
        """The wrapped shared transfers, in the order the tail issues them."""
        return [i for i in self._region
                if getattr(i, '_wrapped', False)
                and getattr(i, '_peel', None) is not None]

    @classmethod
    def indices_in(cls, builder) -> Optional[dict]:
        """What the loop in ``builder``'s body binds, or ``None`` if it holds none.

        The distinction is what a caller needs: ``None`` says the indices this
        body names are bound somewhere else, so there is no value here to reach
        for -- not that the loop binds nothing.
        """
        for owner, bound in reversed(cls._indices):
            if owner is builder:
                return bound
        return None

    def index_name(self, lookahead: int = 0) -> str:
        """The variable holding the element index ``lookahead`` iterations ahead.

        Valid *inside* the region only.  ``index_name(0)`` is the loop variable.
        """
        if lookahead > self._lookahead:
            raise InternalError(
                f'loop binds {self._lookahead} lookahead indices, '
                f'{lookahead} requested')
        return self._batch(lookahead)

    def lookahead_value(self):
        """``batchId1`` as an operand, once the loop has bound it.

        The name is available from construction; the value is not, and the
        difference matters to anything inside the region that means the *next*
        element. An index spelled into text is a computation the IR reads as
        having no inputs, so it is loop-invariant as far as any pass can tell
        -- which is how it ends up hoisted out of the loop that defines what
        it reads.

        ``None`` before the bindings are emitted, and on the legacy writer,
        which has no operands to hand out.
        """
        return self._first_lookahead

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
        `mark_unguarded_tail` is the other shape the guard can leave, a
        suffix, for what is issued once the body is done with it.

        Adds to what is marked rather than replacing it, so a head and a tail
        marked by different passes both survive.
        """
        self._unguarded = set(self._unguarded) | {id(i) for i in instrs}

    def mark_unguarded_tail(self, instrs) -> None:
        """Emit these region instructions *after* the flag guard, unguarded.

        The other half of what `mark_unguarded` says it cannot do.  A transfer
        `WrapLoads` moves across the back edge is issued for element k + 1 once
        iteration k has finished with its buffer, which is the end of the body
        and not its beginning -- so it can only leave the guard as a suffix.
        The instructions must end the region, barriers aside: see
        `_split_guard`.
        """
        self._unguarded = set(self._unguarded) | {id(i) for i in instrs}

    def add_wrap_prologue(self, instrs) -> None:
        """Emit these ahead of the loop, but inside the body that holds it.

        For the peel of a shared transfer `WrapLoads` moved across the back
        edge.  The top-level stream is the obvious place for a peel, and the
        register path puts its own there -- but a shared transfer is only a
        `copy.async` where its window and pointer are values of the body it is
        emitted in, and the window is declared by `_declare_windows_early`,
        in this body, just before the `for`.  A peel ahead of the loop sits in
        a body of its own, names a window that body never bound, and renders
        as text through a pipeline object nothing declares.

        Not part of the region, so no analysis walks it: the buffer is live
        into the loop through the wait at its head anyway, and `ShrMemOpt`
        reaches the peel through the buffer's user list.
        """
        self._wrap_prologue.extend(instrs)

    def add_wrap_epilogue(self, instrs) -> None:
        """Emit these after the loop, still inside its body.

        The drain for a wrapped transfer: the last iteration prefetches a
        clamped element nobody reads, and that copy is in flight into shared
        memory the next section may reuse.  Inside the body for the same
        reason as the prologue -- `asyncmem` retires what it can see issued.
        """
        self._wrap_epilogue.extend(instrs)

    def _split_guard(self):
        """``(unguarded prefix, guarded middle, unguarded suffix)``.

        The suffix is a trailing run of marked instructions, and barriers may
        sit inside and after it.  Two land there without being marked: the one
        `SyncThreadsOpt` puts in front of a wrapped shared transfer, and the
        one the generator appends to every persistent loop after optimisation
        -- which would otherwise break the run it is appended to.  A barrier
        outside the guard is reached by every lane of the multiplication, a
        masked element's included, so moving one out is never the unsafe
        direction.  The run is trimmed to start at a marked instruction, so a
        body with nothing marked at its end keeps its closing barrier inside
        the guard, as before.
        """
        unguarded = set(self._unguarded) | self._address_prefix()
        if not unguarded:
            return [], list(self._region), []
        self._unguarded = unguarded
        region = self._region
        cut = 0
        while cut < len(region) and id(region[cut]) in unguarded:
            cut += 1
        start = len(region)
        while start > cut and (id(region[start - 1]) in unguarded
                               or region[start - 1].barrier_scope() is not None):
            start -= 1
        while start < len(region) and id(region[start]) not in unguarded:
            start += 1
        stray = [i for i in region[cut:start] if id(i) in unguarded]
        if stray:
            raise InternalError(
                f'{len(stray)} instruction(s) marked unguarded form neither a '
                f'prefix nor a suffix of the region, first is '
                f'{type(stray[0]).__name__}; the flag guard is one block and '
                f'cannot be reopened')
        return region[:cut], region[cut:start], region[start:]

    def _declare_windows_early(self, writer, guarded) -> None:
        """Declare the shared windows ahead of the flag guard.

        `s0 = &localShrMem0[512]` is where a transfer writes, and the offset
        comes from `ShrMemOpt` rather than from `batchId0` -- the window is the
        same for every element.  It was declared by whichever instruction fills
        it, so it landed inside the guard, and a transfer cannot be issued
        outside a guard that defines the buffer it fills.

        That was the last thing keeping the moved transfers in: the address
        bindings came out one commit ago, and `wrap_prefetch` then refused 13
        loops for reading their own destination.

        Same rule as the addresses, and it holds more easily here: nothing
        about this declaration depends on the element, so hoisting it cannot
        observe anything a masked element would not have.

        Unconditional, where this used to run only with `enable_wrap_loads`.
        That gate made the switch decide something it has no business
        deciding: whether the window a *consumer* reads is in scope where it
        reads it.  The declaration is emitted by whichever instruction fills
        the window first, so when that instruction sits inside the flag guard
        the name is scoped to the guard -- and every later reader of the same
        window is then referring to something that was never declared where it
        stands.  With prefetch on it was hoisted and the kernel compiled; with
        prefetch off, which is the default, it did not compile at all.

        The corpus never showed it because no case here has a first writer
        inside the guard and a reader outside it.  Nothing about the hoist
        depends on prefetching: the offset comes from `ShrMemOpt` and is the
        same for every element, so there was never a reason for the two to be
        tied together.
        """
        for instr in guarded:
            declare = getattr(instr, 'gen_code_declare', None)
            if declare is None or not getattr(instr, '_declare', False):
                continue
            declare(writer)
            instr._declare = False

    def _address_prefix(self) -> set:
        """The leading address bindings, when a prefetch will need them outside.

        A transfer moved to the previous iteration has to be issued outside the
        flag guard -- the mask is per element, so element k being skipped says
        nothing about k+1, and a prefetch under k's mask breaks the chain for
        everything after a masked element.  But the transfer reads
        `glb_m0 = &m0[batchId0 * stride]`, which is bound inside the guard, so
        the binding has to come out with it.

        Only where computing the address does not itself read memory indexed by
        the element.  `Addressing.PTR_BASED` loads `m0[batchId0]` out of a
        pointer array before offsetting it, and a masked element is one the
        caller told us not to process: nothing in the interface promises that
        its pointer is valid to dereference.  Strided addressing multiplies an
        index by a stride and promises nothing that could be broken.

        Only the leading run, because the guard is one block and what leaves it
        has to be a prefix.
        """
        from .ptr_manip import GetElementPtr
        if not self._context.get_user_options().enable_wrap_loads:
            return set()
        out = set()
        for instr in self._region:
            if not isinstance(instr, GetElementPtr):
                break
            if instr.dereferences_the_batch():
                break
            out.add(id(instr))
        return out

    def prologue_index(self) -> str:
        """The index a peeled iteration should use.

        The loop variable does not exist before the loop, so a peeled iteration
        cannot name it.  The generator binds ``batchId_start`` ahead of the
        loop, and ``batchId1`` beside it as ``batchId_start`` clamped into
        range -- this returns the clamped one, and the distinction is not
        cosmetic.

        ``batchId_start`` is ``threadIdx.y + blockDim.y * blockIdx.x``, bounded
        by the launch geometry and not by the element count.  The grid is sized
        ``min(occupancy, numElements)`` *blocks* of ``blockDim.y`` rows, so at
        100 elements and 16 rows the last thread starts at 1599: the threads
        whose start is past the end are the common case, not the edge.  They
        never enter the loop, which is why the loop body never noticed --- but
        a peeled iteration runs *before* the guard, so it dereferences that
        index unconditionally.  With strided addressing that reads past the
        batch; with ``Addressing.PTR_BASED`` it reads a pointer past the end of
        the pointer array and then follows it.

        Clamping sends those threads to element 0, whose value they load and
        never use.
        """
        return f'{GeneralLexicon.BATCH_ID_NAME}1'

    def _num_elements(self) -> str:
        return f'{GeneralLexicon.NUM_ELEMENTS}{self._section_index}'

    def _block_id(self, block: Optional[str] = None) -> str:
        """The same start expression the generator spells, and parenthesised
        for the same reason: it is a sum handed on as an operand, and whoever
        splices it decides the precedence."""
        lexic = self._vm.get_lexic()
        if block is None:
            block = lexic.block_idx_x
        return f'({lexic.thread_idx_y} + {lexic.block_dim_y} * ({block}))'

    def _size_guard(self) -> str:
        return f'{self._batch(0)} < {self._num_elements()}'

    def _flag_guard(self, writer):
        """The per-element mask, as one definition where the IR can hold it.

        One definition and not a `bool` reassigned under an `if`: two
        assignments to one name is not a value, so the `if (allowed)` around
        the body would have to be raw text as well, and the body could say
        nothing about the condition it runs under.

        `OPTIONAL` spells the null check as a conditional expression for the
        same reason.  `?:` short-circuits, so `flags0[batchId0]` is still read
        only when the pointer is non-null, and it is still one value.
        `REQUIRED` has no null to check: the parameter has no default, so the
        caller supplied a pointer.
        """
        if self._carried_flags is not None:
            return self._word_flag(writer, self._carried_flags[0], 'allowed')
        flags = f'{GeneralLexicon.FLAGS_NAME}{self._section_index}'
        read = f'static_cast<bool>({flags}[{{0}}])'
        if self._flags is FlagMode.OPTIONAL:
            read = f'{flags} == nullptr ? true : {read}'
        if hasattr(writer, 'decl_expr') and self._induction is not None:
            from tensorforge.backend.pir.core import BOOL, Effect, MemSpace
            return writer.decl_expr(
                'const bool allowed', read,
                BOOL, None, args=(self._induction,), kind=Effect.READ,
                space=MemSpace.GLOBAL, hint='allowed', extern='allowed')
        writer(f'const bool allowed = '
               f'{read.format(self._batch(0))};')
        return 'allowed'

    def _lookahead_bindings(self, writer, bound: dict = None) -> None:
        """Bind batchid1..N as clamped element indices, for prefetching.

        Arithmetic and not an expression over text: an index is a `select` over
        a comparison, which is three ops the IR already knows, so there is
        nothing left for a raw expression to buy.  What it costs is everything
        a pass would want here -- CSE folds the two additions in one clamp into
        one statement and the `n`-th clamp's addend into the `n+1`-th's, and a
        consumer moved to another element is a substitution on an operand
        rather than a rewrite of a string.

        The stride and the element count stay text.  They are kernel
        parameters and grid queries, uniform by construction and opaque to the
        IR either way; as operands of an op they are literals it carries
        without reading, which is the same thing a name would have been and
        needs no seam to say so.

        No `extern`, so these are the IR's values with the macro layer's name
        only as a hint.  Nothing spells them any more: the addresses that used
        to take them as operands, which is what the frame above is for.

        The first one escapes, and only it.  The loop names its successor index
        in an *attribute* --- `wrap_prefetch` reads it to rewrite a transfer to
        the next element --- and an attribute is not an operand, so the use
        chain does not see it and `dce` would take the definition away from
        under a pass that has not run yet.  That is what `escapes` says: this
        is referenced from somewhere the graph does not model.  The rest are
        ordinary values and now disappear where nothing reads them, which is
        the two dead clamps every kernel used to carry.
        """
        if not (hasattr(writer, 'op') and self._induction is not None):
            for n in range(1, self._lookahead + 1):
                prev = self._batch(n - 1)
                writer(f'const auto {self._batch(n)} = '
                       f'{prev} + {self._stride} < {self._num_elements()} ? '
                       f'{prev} + {self._stride} : {prev};')
            return
        from tensorforge.backend.pir.core import SIZE, BOOL
        prev = self._induction
        first = None
        for n in range(1, self._lookahead + 1):
            # Clamped rather than wrapped: the last iterations of the loop ask
            # for an element past the end, and the answer that costs nothing is
            # the one they already hold -- a valid address they prefetch and
            # never read.
            ahead = writer.op('add', SIZE, prev, self._stride,
                              hint=f'ahead{n}')
            inside = writer.op('lt', BOOL, ahead, self._num_elements(),
                               hint=f'inbatch{n}')
            prev = writer.op('select', SIZE, inside, ahead, prev,
                             hint=self._batch(n), escapes=(n == 1))
            if bound is not None:
                bound[self._batch(n)] = prev
            if first is None:
                first = prev
        self._first_lookahead = first

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
        head, guarded, tail = self._split_guard()
        for instr in head:
            instr.gen_code(writer)
        # The next element's flag, read here and not where the tail first
        # needs it: there it is a load its user waits on at once, the same
        # chase as the pointer `WrapLoads` binds at the head for that reason.
        next_flag = None
        if self._flags is not FlagMode.ABSENT and any(
                getattr(i, '_guard_by_own_flag', False) for i in tail):
            if self._carried_flags is not None:
                next_flag = self._word_flag(writer, self._carried_flags[1],
                                            'allowed_next')
            else:
                next_flag = self._element_flag(
                    writer, self._tail_element(writer), 'allowed_next')
        if self._flags is FlagMode.ABSENT:
            # Nothing to skip against, so no condition and no block.  The
            # split above still holds: `head` is what has to run for every
            # element, and running it first keeps the order the pipelining
            # pass arranged whether or not a guard follows it.
            #
            # But without the block, nothing keeps the compiler from hoisting
            # the body's invariant loads out of the loop -- see
            # `Lexic.loop_body_fence`.  Only where there is a loop to hoist
            # out of.
            fence = self._vm.get_lexic().loop_body_fence()
            if fence and self._mode is not LoopMode.SINGLE:
                if hasattr(writer, 'decl_expr'):
                    # It touches nothing the IR models; it only has to stay.
                    writer(fence, accesses=())
                else:
                    writer(fence)
            self._emit_guarded(writer, guarded)
            self._emit_own_flagged(writer, tail,
                                   lambda: self._tail_element(writer),
                                   'allowed_next')
            return
        cond = self._flag_guard(writer)
        # A real `Op.IF` where the condition is a value.  A raw block would
        # make the whole body one opaque region as far as any pass is
        # concerned -- `wrap_prefetch` looks into the loop, finds the guard,
        # and would report no transfers because none are *its* statements.
        guard = (writer.if_(cond) if hasattr(writer, 'if_')
                 and not isinstance(cond, str) else writer.If(cond))
        with guard:
            self._emit_guarded(writer, guarded)
        self._emit_own_flagged(writer, tail,
                               lambda: self._tail_element(writer),
                               'allowed_next', cond=next_flag)

    def _emit_own_flagged(self, writer, instrs, element, name,
                          cond=None) -> None:
        """Emit `instrs`; those that follow their element's pointer, under that
        element's flag.

        For the transfers `WrapLoads` issues for another element than the one
        the body is on -- the peel for the first, the tail for the next -- and
        only where the address comes out of a pointer array.  Such a transfer
        is outside the guard on purpose, so a masked element still prefetches
        its successor; but the successor may itself be masked, and nothing in
        the interface promises the pointer of an element the caller told us to
        skip.  Reading the array entry is fine, the index is clamped into
        range; following it is not.  So the transfer, and nothing else, runs
        under the flag of the element it fetches.  A barrier between two such
        transfers stays unconditional, and a skipped transfer still commits:
        an empty group, the same on every lane of the multiplication, which is
        what every lane's count has to agree on.

        `element` is called only if something here needs the flag, since in
        the prologue asking for the index emits a binding.  `cond` is that flag
        where the caller has read it already.
        """
        instrs = list(instrs)
        i = 0
        while i < len(instrs):
            instr = instrs[i]
            i += 1
            if (self._flags is FlagMode.ABSENT
                    or not getattr(instr, '_guard_by_own_flag', False)):
                instr.gen_code(writer)
                continue
            if cond is None:
                cond = self._element_flag(writer, element(), name)
            if (hasattr(instr, '_copy_predicate') and hasattr(writer, 'copy_async')
                    and not isinstance(cond, str)):
                # A shared transfer: its copies take the flag as a predicate
                # rather than sit in a block.  A copy in a block is issued on
                # one path only as far as `asyncmem` can tell, and the loop
                # carrying its token would lose the steady state its waits are
                # counted against.  The predicate is a real branch per copy,
                # so the pointer is still not followed for a masked element.
                instr._copy_predicate = cond
                try:
                    instr.gen_code(writer)
                finally:
                    instr._copy_predicate = None
                continue
            # The instructions after it that sit under the same flag share its
            # block: one branch rather than one per instruction, and the
            # statements side by side where the lowering can take them as one
            # (the prefetch hints, which ESIMD asks for in one gather).
            group = [instr]
            while (i < len(instrs)
                   and getattr(instrs[i], '_guard_by_own_flag', False)
                   and not (hasattr(instrs[i], '_copy_predicate')
                            and hasattr(writer, 'copy_async'))):
                group.append(instrs[i])
                i += 1
            guard = (writer.if_(cond) if hasattr(writer, 'if_')
                     and not isinstance(cond, str) else writer.If(cond))
            with guard:
                for member in group:
                    member.gen_code(writer)

    def _element_flag(self, writer, index, name):
        """`flags[index]`, spelled the way `_flag_guard` spells the current one."""
        flags = f'{GeneralLexicon.FLAGS_NAME}{self._section_index}'
        read = f'static_cast<bool>({flags}[{{0}}])'
        if self._flags is FlagMode.OPTIONAL:
            read = f'{flags} == nullptr ? true : {read}'
        if hasattr(writer, 'decl_expr') and not isinstance(index, str):
            from tensorforge.backend.pir.core import BOOL, Effect, MemSpace
            return writer.decl_expr(
                f'const bool {name}', read, BOOL, None, args=(index,),
                kind=Effect.READ, space=MemSpace.GLOBAL, hint=name,
                extern=name)
        writer(f'const bool {name} = {read.format(index)};')
        return name

    def _carries_flags(self, writer) -> bool:
        """Does the loop hand the mask from one iteration to the next?

        Where it carries prefetches across the back edge, and there is a mask.
        Read at the head, the element's flag is a load its guard waits on at
        once, and on AMD that wait is `vmcnt(0)`: the counter retires in
        order, so it also waits for every copy the previous tail issued, and
        the pipeline those copies were meant to be is gone -- with a mask
        passed, `chain_three` keeps one wait per iteration, and it is that
        one.  So the words ride the loop instead, like the tokens: this
        element's and the next one's come in, and the one two ahead is read
        at the head and handed on.  A word is read one iteration before it
        is first needed, and whatever waits for it waits for loads long done.

        The word and not the `bool`: a comparison right behind the load is a
        use right behind the load, and the wait comes back with it.  The
        conversion is made where the flag is used.

        Only on the structured path, which has iteration arguments, and only
        with a successor index to count on from.
        """
        return (self._flags is not FlagMode.ABSENT and bool(self._wrap_prologue)
                and self._lookahead >= 1 and hasattr(writer, 'for_'))

    def _element_word(self, writer, index, name):
        """`flags[index]` as the word it is -- see `_carries_flags`.

        One per element, so laid out like the element index itself: every lane
        of the multiplication holds the same word.  The loop declares the words
        it carries from that, and the ESIMD lowering refuses a value it cannot
        tell the spread of.
        """
        from tensorforge.backend.pir.core import (SCALAR_LAYOUT, Effect,
                                                  MemSpace, ScalarType)
        from tensorforge.common.basic_types import Datatype
        flags = f'{GeneralLexicon.FLAGS_NAME}{self._section_index}'
        read = f'{flags}[{{0}}]'
        if self._flags is FlagMode.OPTIONAL:
            # `1` and not `1u`: the conditional converts it to the word's
            # type anyway, and the oracle that reads kernels in the tests
            # parses C literals without suffixes.
            read = f'{flags} == nullptr ? 1 : {read}'
        return writer.decl_expr(
            f'const uint32_t {name}', read, ScalarType(Datatype.U32), None,
            args=(index,), kind=Effect.READ, space=MemSpace.GLOBAL, hint=name,
            extern=name, layout=SCALAR_LAYOUT)

    def _word_flag(self, writer, word, name):
        """A carried flag word as the condition it stands for."""
        from tensorforge.backend.pir.core import BOOL
        return writer.decl_expr(f'const bool {name}', 'static_cast<bool>({0})',
                                BOOL, None, args=(word,), hint=name,
                                extern=name)

    def _clamped_successor(self, writer, index, hint):
        """`index + stride`, clamped the way the lookahead bindings clamp.

        The same clamp and not a simpler one: the word read for an element
        is the predicate of the copy that follows that element's pointer, so
        it has to name the element the copy will be issued for -- including
        at the end of the batch, where the clamp holds the index in place.
        """
        from tensorforge.backend.pir.core import BOOL, SIZE
        ahead = writer.op('add', SIZE, index, self._stride, hint=f'{hint}Ahead')
        inside = writer.op('lt', BOOL, ahead, self._num_elements(),
                           hint=f'{hint}In')
        return writer.op('select', SIZE, inside, ahead, index, hint=hint)

    def _tail_element(self, writer):
        """The element the tail prefetches: this loop's clamped successor."""
        bound = BatchLoop.indices_in(writer)
        if bound is not None and self._batch(1) in bound:
            return bound[self._batch(1)]
        return self._batch(1)

    def _prologue_element(self, writer):
        """The element the peel fetches: `prologue_index`, bound before the loop."""
        if hasattr(writer, 'batch_id'):
            return writer.batch_id(1)
        return self.prologue_index()

    def _emit_guarded(self, writer, guarded) -> None:
        """The part of the region that a mask, if there is one, may skip."""
        if AbstractInstruction._shared_body:
            # Already inside one -- opened by `gen_code` around the loop.
            for instr in guarded:
                instr.gen_code(writer)
        elif self._wide_bodies():
            # one body for every instruction of the region
            budget = max((i.temp_shmem() for i in guarded), default=0)
            AbstractInstruction.build_shared_body(
                self._context, writer,
                lambda _builder: [instr.gen_code(writer) for instr in guarded],
                scratch=budget)
        else:
            for instr in guarded:
                instr.gen_code(writer)

    def _wide_bodies(self) -> bool:
        """One PIR body for the whole region, or one per instruction."""
        return self._context.get_user_options().wide_bodies

    def gen_code(self, writer) -> None:
        # Deliberately no writer.Scope() and no comment: the loop used to be
        # emitted inline by the generator, and adding either would change the
        # generated text.
        if self._structured_loop(writer):
            # One body for the whole section, with the loop *inside* it.
            #
            # Until now the builder was opened by `_emit_body`, one level
            # further in, so every body sat within the loop and none could
            # name it.  That is the 2% `tools/macro_surface.py` measures and
            # the reason a transfer cannot be moved to the previous iteration:
            # `can_reorder` licenses swaps inside a body, and nothing licenses
            # a move across a back edge made of Writer text.
            budget = self.temp_shmem()
            AbstractInstruction.build_shared_body(
                self._context, writer, self.gen_code_inner, scratch=budget)
            return
        self.gen_code_inner(writer)

    def _structured_loop(self, writer) -> bool:
        """Should this loop be a construct in the IR rather than Writer text?

        Wherever there is a loop at all.  `PERSISTENT` is an `Op.FOR` over the
        stride; `LAUNCHCTRL` is an `Op.WHILE`, whose successor is the result
        of `cursor.next(queue)` in its own body and which leaves through an
        `Op.EXIT` when that answers -1.  `SINGLE` has no loop, so there is
        nothing to put anywhere.

        What the construct buys is the same in both cases and is not
        cosmetic.  `batchId0` is a value, so the size guard is an `Op.IF` over
        it and the body's reads of it are operands rather than text.  The
        region a pass walks is the whole loop rather than the part between the
        braces, which is what lets a prologue be expressed and a move across
        the back edge be licensed by something other than luck.  And the loop
        states how far entering its body is agreed across threads, so `verify`
        can decide barrier legality from the form instead of from a comment.

        Not when a body is already open: `shared_body` nests, and a second
        one would put the loop inside the body it is meant to contain.
        """
        if self._mode is LoopMode.SINGLE:
            return False
        if not self._wide_bodies():
            return False
        if AbstractInstruction._shared_body:
            return False
        # `for_` is the builder's; the Writer has only the raw `For`.  Testing
        # for `alloc` would not do it -- `Writer` carries a `VarAlloc` under
        # that name, so the check passed for neither and the path never ran
        # while the corpus dutifully reported no drift.
        return not hasattr(writer, 'for_')

    def gen_code_inner(self, writer) -> None:
        if self._grouped():
            self._gen_grouped(writer)
            return
        if self._mode is LoopMode.PERSISTENT:
            # TODO: OMP target
            # TODO: maybe iterate over adjacent elements? (for indirect pointers)
            self._declare_stage_counter(writer)
            # Before the loop, not merely before the guard.  The window is the
            # same for every element, and a peeled transfer is emitted outside
            # the loop -- with the declaration inside it, the prologue names a
            # value whose `extern` binding happens later and the result renders
            # but does not compile.
            self._declare_windows_early(writer, list(self._region))
            # The words the loop starts with: the first element's, which is
            # also what the peel is predicated on, and its successor's.
            carry = self._carries_flags(writer)
            peel_flag = None
            if carry:
                first = self._prologue_element(writer)
                word_first = self._element_word(writer, first, 'flagWordFirst')
                word_second = self._element_word(
                    writer, self._clamped_successor(writer, first, 'flagSecond'),
                    'flagWordSecond')
                if any(getattr(i, '_guard_by_own_flag', False)
                       for i in self._wrap_prologue):
                    peel_flag = self._word_flag(writer, word_first,
                                                'allowed_peel')
            self._emit_own_flagged(writer, self._wrap_prologue,
                                   lambda: self._prologue_element(writer),
                                   'allowed_peel', cond=peel_flag)
            if hasattr(writer, 'for_'):
                # `extern` and `ctype` because the name and the type are the
                # macro layer's: `batchId0` is spelled out by the lookahead
                # bindings, the flag guard and every `access_address` in the
                # body, and it is `size_t` because it is compared against
                # `numElements`.
                from tensorforge.backend.pir.core import (SIZE,
                                                          Uniformity)
                # Neither `extern` nor `ctype`.  The name is nobody's business
                # now that every reader of the index takes it as an operand,
                # and the width is the induction value's own -- an override on
                # the header widened the variable and left everything computed
                # from it back at `int32_t`.
                # Each wrapped shared transfer's tokens ride the loop: the
                # peel's are the first iteration's arguments, the tail's are
                # yielded as the next one's, and the wait at the consumer names
                # them -- see `carried_tokens`.  A transfer whose peel issued
                # nothing structured carries nothing, and its wait drains.
                inits: list = []
                types: list = []
                spans: list = []
                for transfer in self._carried_transfers():
                    toks = transfer._peel.tokens_for(writer)
                    if not toks:
                        continue
                    spans.append((transfer, len(inits), len(toks)))
                    inits.extend(toks)
                    types.extend(t.type for t in toks)
                flags_at = len(inits)
                if carry:
                    inits.extend((word_first, word_second))
                    types.extend((word_first.type, word_second.type))
                table = BatchLoop._push_carried(writer)
                try:
                    with writer.for_(self._start, self._num_elements(),
                                     self._stride, hint=self._batch(0),
                                     index_type=SIZE,
                                     peel_index=self.prologue_index(),
                                     uniform=Uniformity.MULT,
                                     inits=tuple(inits),
                                     types=tuple(types)) as loop:
                        self._loop_handle = loop
                        for transfer, at, n in spans:
                            table[id(transfer)] = loop.iter_args[at:at + n]
                        # The induction *value*, not just its name.  Anything
                        # inside that mentions `batchId0` has to say so as an
                        # operand, or the IR sees a computation with no inputs and
                        # hoists it out of the loop that defines the thing it
                        # reads -- which is what happened the first time, silently
                        # and only in the generated text.
                        self._induction = loop.induction
                        try:
                            with BatchLoop.batch_indices(writer) as bound:
                                bound[self._batch(0)] = loop.induction
                                self._lookahead_bindings(writer, bound)
                                # The first lookahead binding is what this loop
                                # calls the next element, and `wrap_prefetch` needs
                                # exactly that: it moves a transfer one iteration
                                # earlier and has no way to know how the traversal
                                # clamps.
                                loop._next_index = self._first_lookahead
                                if carry:
                                    own, nxt = loop.iter_args[flags_at:flags_at + 2]
                                    word_ahead = self._element_word(
                                        writer, self._clamped_successor(
                                            writer, self._tail_element(writer),
                                            'flagAhead'),
                                        'flagWordAhead')
                                    self._carried_flags = (own, nxt)
                                self._emit_body(writer)
                                self._carried_flags = None
                                self._advance_stage_counter(writer)
                                if spans or carry:
                                    yielded = []
                                    for transfer, at, n in spans:
                                        toks = transfer.tokens_for(writer)
                                        if len(toks) != n:
                                            raise InternalError(
                                                f'{transfer} issued {len(toks)} '
                                                f'copies in the loop and {n} in '
                                                f'its peel; the tokens it carries '
                                                f'cannot line up')
                                        yielded.extend(toks)
                                    if carry:
                                        yielded.extend((nxt, word_ahead))
                                    loop.yield_(*yielded)
                        finally:
                            self._induction = None
                            self._carried_flags = None
                    for transfer, at, n in spans:
                        table[id(transfer)] = loop.results[at:at + n]
                    for instr in self._wrap_epilogue:
                        instr.gen_code(writer)
                finally:
                    BatchLoop._pop_carried()
                return
            with writer.For(f'size_t {self._batch(0)} = {self._start}; '
                            f'{self._batch(0)} < {self._num_elements()}; '
                            f'{self._batch(0)} += {self._stride}'):
                self._lookahead_bindings(writer)
                self._emit_body(writer)
                self._advance_stage_counter(writer)
            for instr in self._wrap_epilogue:
                instr.gen_code(writer)
        elif self._mode is LoopMode.LAUNCHCTRL:
            self._declare_stage_counter(writer)
            self._declare_windows_early(writer, list(self._region))
            depth = self._queue_depth
            writer(f'__shared__ tensorforge::ClusterLaunchQueue<{depth}> '
                   f'launchQueue{self._section_index};')
            # The cursor is a *local*: it is per-thread state, and every thread
            # of the block advances it identically.  In shared memory the
            # parity would be one word 128 threads flip, which is a race whose
            # symptom is a wait for a phase that has already gone past.
            writer(f'tensorforge::ClusterLaunchCursor<{depth}> '
                   f'launchCursor{self._section_index};')
            writer(f'launchCursor{self._section_index}'
                   f'.start(launchQueue{self._section_index});')
            if hasattr(writer, 'while_'):
                self._queried_loop(writer)
                return
            writer(f'size_t {self._batch(0)} = {self._block_id()};')
            with writer.While('true'):
                with writer.If(self._size_guard()):
                    self._emit_body(writer)
                self._advance_stage_counter(writer)
                # Outside the size guard, deliberately.  `next` contains the
                # block barrier that separates one element's shared memory
                # from the next one's, and the guard is per element: the rows
                # of a block hold different elements, so a barrier inside it
                # is reached by some rows and not others.  That does not
                # reliably deadlock -- `bar.sync` pairs arrivals by barrier
                # *number*, so the rows that skipped rendezvous at the next
                # one instead and the loop limps on one barrier out of step,
                # reusing the tile an iteration early.  Measured both ways: a
                # block barrier under the guard hangs outright once a flag
                # mask makes the rows disagree persistently.
                writer(f'const int nextBlock{self._section_index} = '
                       f'launchCursor{self._section_index}'
                       f'.next(launchQueue{self._section_index});')
                with writer.If(f'nextBlock{self._section_index} < 0'):
                    writer('break;')
                writer(f'{self._batch(0)} = '
                       f'{self._block_id(f"nextBlock{self._section_index}")};')
        else:
            writer(f'const size_t {self._batch(0)} = {self._block_id()};')
            with writer.If(self._size_guard()):
                self._emit_body(writer)

    def _queried_loop(self, builder) -> None:
        """The launch-control traversal, as an `Op.WHILE` over the work queue.

        Three things are values here rather than text, and each of them is a
        fact some pass needs.

        The element index is the loop's induction, `MULT`-uniform because the
        rows of a block hold different elements -- so the size guard is an
        `Op.IF` over it and every read of it in the body is an operand.

        The hand-off is an `Op.CALL` carrying `Effect.BARRIER` and its access
        to the queue.  `next` contains a block barrier, and a pass that cannot
        see it would be free to move a shared-memory access across the point
        where one element's tile stops being live.  The access is keyed on the
        queue by name, so it provably does not conflict with the arena.

        The exit is where the trip count is decided, and its condition is
        `BLOCK`-uniform: every thread reads the same cancel response out of
        shared memory behind that barrier, so all of them leave on the same
        iteration.  `_entry_uniformity` reads exactly that off the exit, which
        is what makes a block barrier legal in this body -- and it stays
        illegal under the size guard, whose condition is only `MULT`-uniform.
        """
        from tensorforge.backend.pir.core import (SIZE, INDEX, BOOL,
                                                  Access, Effect, MemSpace,
                                                  Uniformity)

        index = self._section_index
        queue = f'launchQueue{index}'
        lexic = self._vm.get_lexic()

        with builder.while_(self._block_id(), hint=self._batch(0),
                            index_type=SIZE,
                            uniform=Uniformity.MULT) as loop:
            self._loop_handle = loop
            self._induction = loop.induction
            # Only `batchId0`: this traversal binds no lookahead, because the
            # next element is whatever the queue answers and there is nothing
            # to compute it from ahead of time.  So an address naming
            # `batchId1` in here is naming the *prologue's* binding, which is a
            # different element -- and it now says so rather than resolving
            # against a name that happens to be in scope.
            try:
                with BatchLoop.batch_indices(builder) as bound:
                    bound[self._batch(0)] = loop.induction
                    self._queried_body(builder, loop, index, queue, lexic)
            finally:
                self._induction = None

    def _queried_body(self, builder, loop, index, queue, lexic) -> None:
        """The body of the queried traversal, once its index is published."""
        from tensorforge.backend.pir.core import (INDEX, SIZE, BOOL, Access,
                                                  Effect, MemSpace, Uniformity)
        if True:
            if True:
                guard = builder.op('lt', BOOL, loop.induction,
                                   self._num_elements(), hint='inrange')
                with builder.if_(guard):
                    self._emit_body(builder)
                self._advance_stage_counter(builder)
                # Outside the size guard, deliberately.  The barrier `next`
                # carries separates one element's shared memory from the next
                # one's, and the guard is per element: the rows of a block
                # hold different elements, so a barrier inside it is reached
                # by some rows and not others.  That does not reliably
                # deadlock -- `bar.sync` pairs arrivals by barrier *number*,
                # so the rows that skipped rendezvous at the next one instead
                # and the loop limps on one barrier out of step, reusing the
                # tile an iteration early.
                nxt = builder.call(
                    f'launchCursor{index}.next', INDEX, queue,
                    hint='next', pure=False, movable=False,
                    effect=Effect.BARRIER,
                    accesses=(Access(Effect.READ | Effect.WRITE,
                                     MemSpace.SHARED, queue),),
                    uniform=Uniformity.BLOCK, materialize=True)
                loop.exit_when(builder.op('lt', BOOL, nxt, 0, hint='drained'))
                # The successor index, in the same shape the initial one has.
                # `threadIdx.y` is the multiplication's index and says so, so
                # the sum comes out `MULT` without anyone claiming it.
                offset = builder.op('mul', INDEX, lexic.block_dim_y, nxt,
                                    hint='row')
                loop.yield_(builder.op('add', SIZE, builder.thread_id('y'),
                                       offset, hint=self._batch(0)))

    def _gen_grouped(self, writer) -> None:
        """One traversal for a whole group of rows.

        The loop is driven by the group leader, whose index is the lowest in
        the group, so it runs as often as the row that has the most to do and
        no row is cut short.  Every row of the group therefore reaches the same
        barriers the same number of times, which is the whole point: the rows
        share a wave, so a barrier cannot separate them and each one has to
        arrive.
        """
        if (self._mults_per_block is not None
                and self._mults_per_block % self._group_size):
            raise InternalError(
                f'a block of {self._mults_per_block} multiplications does not '
                f'hold whole groups of {self._group_size}')
        # The group a wave-collective instruction asked for is traversed as
        # IR; the block-wide group keeps the text it has always been emitted as.
        if (self._narrow_group and self._mode is LoopMode.PERSISTENT
                and hasattr(writer, 'for_')):
            self._gen_grouped_ir(writer)
            return
        self._declare_lane(writer)
        self._declare_stage_counter(writer)
        self._declare_windows_early(writer, list(self._region))
        if self._mode is LoopMode.PERSISTENT:
            with writer.For(f'size_t {self._group_batch()} = '
                            f'{self._group_start()}; '
                            f'{self._group_batch()} < {self._num_elements()}; '
                            f'{self._group_batch()} += {self._stride}'):
                mask = self._declare_row_element(writer)
                self._lookahead_bindings(writer)
                with elementmask.element_mask(None, mask):
                    self._emit_guarded(writer, list(self._region))
                self._advance_stage_counter(writer)
            return
        writer(f'const size_t {self._group_batch()} = {self._group_start()};')
        with writer.If(f'{self._group_batch()} < {self._num_elements()}'):
            mask = self._declare_row_element(writer)
            with elementmask.element_mask(None, mask):
                self._emit_guarded(writer, list(self._region))

    def _gen_grouped_ir(self, writer) -> None:
        """`_gen_grouped`'s persistent traversal, as IR.

        Spelled as text, the lane, the mask, the element and the lookahead
        clamps were statements that declared no accesses, so every pass
        reasoning about the body -- the scratch check among them -- had to
        assume each touched everything.  As values they are arithmetic and one
        declared read of the flags, and the loop is one the IR can see.

        The start stays text, as it does for the per-row loop: a bound carries
        no uniformity then, and the induction states it -- the group's, which
        is what a wave-wide rendezvous inside needs.
        """
        from tensorforge.backend.pir.core import (BOOL, SIZE, Effect, MemSpace,
                                                  Uniformity)
        lexic = self._vm.get_lexic()
        self._declare_stage_counter(writer)
        self._declare_windows_early(writer, list(self._region))
        lane = writer.op('rem', SIZE, writer.thread_id('y'), self._group_size,
                         hint=self._lane())
        start = (f'({lexic.thread_idx_y} - {lexic.thread_idx_y} % '
                 f'{self._group_size}) + {lexic.block_dim_y} * '
                 f'({lexic.block_idx_x})')
        with writer.for_(start, self._num_elements(), self._stride,
                         hint=self._group_batch(), index_type=SIZE,
                         uniform=Uniformity.MULTGROUP) as loop:
            self._loop_handle = loop
            group = loop.induction
            row = writer.op('add', SIZE, group, lane, hint='row')
            # One named declaration, because every global write spells the
            # mask as text (`Symbol.store`), and the flags read folded into it
            # as the text had it: `&&` keeps a row past the end from reading.
            cond = '{0} < ' + self._num_elements()
            kind, space = Effect.NONE, None
            if self._flags is not FlagMode.ABSENT:
                flags = f'{GeneralLexicon.FLAGS_NAME}{self._section_index}'
                read = f'static_cast<bool>({flags}[{{0}}])'
                if self._flags is FlagMode.OPTIONAL:
                    read = f'({flags} == nullptr || {read})'
                cond = f'{cond} && {read}'
                kind, space = Effect.READ, MemSpace.GLOBAL
            active = writer.decl_expr(f'const bool {self._active()}', cond,
                                      BOOL, None, kind=kind, space=space,
                                      args=(row,), hint=self._active(),
                                      extern=self._active())
            batch = writer.op('select', SIZE, active, row, group,
                              hint=self._batch(0))
            self._induction = batch
            try:
                with BatchLoop.batch_indices(writer) as bound:
                    bound[self._batch(0)] = batch
                    self._lookahead_bindings(writer, bound)
                    loop._next_index = self._first_lookahead
                    with elementmask.element_mask(active, self._active()):
                        self._emit_guarded(writer, list(self._region))
                    self._advance_stage_counter(writer)
            finally:
                self._induction = None

    def __str__(self) -> str:
        return (f'batchloop.{self._mode.value} '
                f'[{self._num_elements()}] '
                f'uniform={self.uniform_scope().name.lower()} '
                f'({len(self._region)} instructions)')
