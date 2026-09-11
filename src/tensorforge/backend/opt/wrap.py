# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Prefetch across the back edge: the transfer for element ``k + 1``, issued at
the tail of iteration ``k``.

``Pipeline`` advances a transfer by whole iterations and needs a copy of the
buffer per stage.  This moves it the way ``MoveLoads`` would if the loop were
unrolled once: the transfer for element ``k + 1`` travels up from its place in
iteration ``k + 1``, across the back edge, and stops after the last
instruction of iteration ``k`` that touches its buffer.  The write then
follows the last read, so one buffer is enough -- no rotation, no stage
index::

    before:  [ l1 c1 l2 c2 ]
    after:   l1 l2 | [ c1 c2 l1' l2' ]

with ``l1'`` reading element ``k + 1`` -- the loop's clamped lookahead index --
and the peeled ``l1`` the thread's first element, clamped likewise.

Everything below the last instruction that touches the buffer is crossed by
construction, so the tail of the body is a legal place, and it is the one that
keeps the transfer outside the per-element flag guard: a masked element has to
go on prefetching its successor, which a mid-body placement under the guard
cannot.  The same placement for register and shared destinations.  Registers
used to be placed by *slot* -- ``slots.py`` still has the accounting of what a
distance costs -- which put the transfer mid-body, under the guard, and gave a
body with a single compute nothing to move.  Placing by dependence has neither
problem.

Which transfers go is ``move_distance``, the number ``MoveLoads`` travels by.
In the loop unrolled once, the ``j``-th transfer of an iteration moves up past
``d`` loads, and that runs off the top of the body -- into the previous
iteration -- exactly when ``j < d``.  So at ``d = 1`` the first transfer wraps
and the rest are ``MoveLoads``' pipeline inside the body; a larger ``d`` wraps
more of them, and one at least the number of transfers wraps every one it may.
Where a wrapped transfer lands does not depend on ``d``: it goes to the tail,
the one place outside the flag guard.  ``wrap_distance`` is not read.

A register destination needs little from the rest of the pipeline: it is
thread-private, so a barrier is no obstacle and none is needed; its loads are
plain loads, so there is no wait to place.  What it does need is its
declaration out of the loop -- a register declared in the body is a fresh
object every iteration -- which goes into the loop's prologue with the peel.

A shared destination needs more, and gets it:

* the barriers -- between the last read and the wrapped write, and between
  the wait at the head and the first read.  ``SyncThreadsOpt`` places both,
  the second because it treats a write still unfenced at the end of a loop
  body as carried into the next iteration;
* the buffer kept live across the back edge while the copy is in flight --
  ``LivenessAnalysis`` iterates to a fixed point over the loop, so the region
  allocator gives nobody else its offset in between;
* a wait that does not depend on the issue having been emitted first --
  ``LoadWait`` drains for a wrapped transfer, because in program order the
  wait now precedes the issue;
* the peeled copy and the drain after the loop inside the loop's own body --
  ``BatchLoop`` emits them there, after the windows are declared and around
  the ``for``.  A transfer is only a ``copy.async`` where its window and its
  pointer are values of the body it is emitted in, and a peel ahead of that
  body names a window the body binds only later: it renders as text, through
  a pipeline object nothing declares.  The drain is there because the last
  iteration prefetches a clamped element nobody reads, into memory the next
  section may reuse.

And both kinds, where the address comes out of a pointer array: the transfer
under the flag of the element it fetches -- the next one's in the tail, the
first one's in the peel.  The array entry is read unconditionally, the clamp
keeps it in range; the pointer is followed only for an element the caller did
not mask.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.instructions.memory.load import (GlbToRegLoader,
                                                          GlbToShrLoader,
                                                          LoadWait)
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.symbol import Symbol
from tensorforge.common.helper import Addressing

from .abstract import AbstractTransformer, Context
from .slots import SlotModel, Transfer


class TailWrap(NamedTuple):
    transfer: AbstractInstruction       # GlbToShrLoader or GlbToRegLoader
    producer: GetElementPtr
    alloc: Optional[RegisterAlloc]      # a register destination's, hoisted


class WrapLoads(AbstractTransformer):
    """Issue each transfer for the next element at the tail of the body."""

    def __init__(self,
                 context: Context,
                 instructions: List[AbstractInstruction],
                 distance: int = 1):
        super(WrapLoads, self).__init__(context, instructions)
        if distance < 1:
            raise ValueError(f'move distance must be >= 1, got {distance}')
        self._distance = distance
        self.rejected: List[Tuple[object, str]] = []
        self.wrapped: List[str] = []

    # ------------------------------------------------------------------ #

    def apply(self) -> None:
        # Nothing is added to this stream: peels and drains are emitted by the
        # loop, inside its own body -- see the module docstring.
        for instr in self._instrs:
            if isinstance(instr, BatchLoop):
                instr.replace_region(0, self._wrap_loop(instr))

    # ------------------------------------------------------------------ #

    def _wrap_loop(self, loop: BatchLoop) -> List[AbstractInstruction]:
        body = list(loop.region)
        if loop._mode is LoopMode.SINGLE:
            # one iteration, so there is no next element to prefetch and no
            # back edge to wrap across
            return body
        if loop._mode is LoopMode.LAUNCHCTRL:
            # There *is* a next element, and this pass cannot name it.  It
            # prefetches through `index_name(1)`, which the strided loop binds
            # as `batchId0 + stride`; the queue hands out whatever CTA the
            # launcher cancels, so that name is some other block's element.
            # The transfer would land in the buffer the next iteration reads
            # and every element after the first would be computed from the
            # wrong operands -- no crash, wrong numbers.  `Generator` refuses
            # the combination outright, and the reason lives here because this
            # is what depends on it.
            return body
        if loop._group_size > 1:
            # A group of rows traversed in lockstep (`BatchLoop._gen_grouped`)
            # emits its region and nothing else: no peel, no drain, no carried
            # tokens.  A wrapped transfer would be issued for every element
            # but the first, and the first iteration would read a buffer
            # nothing filled -- no crash, wrong numbers (`local_flux` at 16
            # lanes on the MMA path: 7 % off).  Nor is `index_name(1)` a row's
            # next element there: the group's element is `active ? row :
            # group`, and an inactive row's successor would be its group's.
            # Refused wherever the loop may be grouped: whether a block-wide
            # group is depends on the block's multiplication count, which is
            # settled after the passes run.
            for t in SlotModel(body).run().transfers[:self._distance]:
                dest = t.load.defs()[0] if t.load.defs() else None
                self._reject(getattr(dest, 'name', '?'),
                             'the loop traverses a group of rows in lockstep, '
                             'which emits no peel for the first element')
            return body

        # One pointer to element k + 1, and one to the first element, per
        # source -- see `_pointers`.  Per loop: another loop binds its own.
        self._pointers_for = {}
        self._names = set()

        # The slot model only enumerates the transfers and their consumers;
        # placement is by dependence.  Each one wrapped goes to the tail, so
        # the ones still to be planned keep their place ahead of it.
        # The first `distance` transfers, in body order: the ones whose move
        # by that many loads runs across the back edge.  Counted before any is
        # moved, since moving one to the tail would reorder the rest.
        for t in SlotModel(body).run().transfers[:self._distance]:
            plan = self._plan(loop, body, t)
            if plan is not None:
                self._apply(loop, body, plan)
        return body

    # ------------------------------------------------------------------ #

    def _plan(self, loop: BatchLoop, body,
              t: Transfer) -> Optional[TailWrap]:
        """Whether ``MoveLoads`` could take this transfer into the previous iteration.

        Unrolled once, the transfer for element ``k + 1`` starts at its place in
        iteration ``k + 1`` and walks up.  It first crosses everything ahead of
        it in its own iteration -- if any of that touches its buffer, it never
        leaves, and ``MoveLoads`` has already put it where it belongs.  Past the
        back edge it walks up iteration ``k`` until the last instruction that
        touches the buffer, which stops it; everything below that is crossed by
        construction, so the tail of the body is a legal place, and the one
        that keeps it outside the flag guard.
        """
        load = t.load
        dest = load.defs()[0] if load.defs() else None
        name = getattr(dest, 'name', '?')
        shared = isinstance(load, GlbToShrLoader)
        if not shared and not isinstance(load, GlbToRegLoader):
            return self._reject(name, 'neither a global-to-shared nor a '
                                      'global-to-register transfer')
        if t.first_use_slot is None:
            return self._reject(name, 'loaded value is never read in this body')
        alloc = None
        if shared:
            if not self._context.get_user_options().wide_bodies:
                # The peel and the drain have to share a body with the loop,
                # or the transfers fall back to text -- see the module
                # docstring.
                return self._reject(name, 'the loop is not one body '
                                          '(wide_bodies is off), so the peel '
                                          'cannot be issued as a copy into '
                                          'the window it fills')
            if getattr(load, '_stages', 1) > 1:
                return self._reject(name, 'buffer is rotated; the stage it '
                                          'fills is chosen per iteration, and '
                                          'wrapping would fill the wrong one')
            if not hasattr(load, '_ctor_kwargs'):
                return self._reject(name, 'transfer cannot be cloned for the '
                                          'peel')
        else:
            alloc = next((i for i in body if isinstance(i, RegisterAlloc)
                          and i._dest is dest), None)
            if alloc is None:
                return self._reject(
                    name, 'no RegisterAlloc for the destination; the '
                          'declaration has to leave the loop or the value does '
                          'not survive the back edge')
            if alloc._init_value not in (None, 0):
                return self._reject(name, 'buffer is declared with a non-zero '
                                          'initialiser, which a hoisted '
                                          'declaration would apply once')
        producer = self._producer(body, load._src)
        if producer is None:
            return self._reject(
                name, 'source pointer is not computed by a GetElementPtr in '
                      'this body, so there is no element index to advance')
        if isinstance(producer._batch_offset, str):
            return self._reject(name, 'source pointer already names an index '
                                      'verbatim; already pipelined')
        writers = [i for i in body
                   if any(o is dest for o in i.defs())
                   and not isinstance(i, (RegisterAlloc, LoadWait))]
        if len(writers) > 1:
            kinds = ', '.join(sorted({type(i).__name__ for i in writers}))
            return self._reject(
                name, f'buffer is written {len(writers)} times per iteration '
                      f'({kinds}); a wrapped buffer must hold one element for '
                      f'the whole iteration')
        at = body.index(load)
        # The source, not only the buffer.  A store names the pointer it writes
        # through among its `defs()`, and a store ahead of the transfer in its
        # own iteration is a write the transfer has to see: `d += ...; out +=
        # d * c` reads `d` back after writing it.  Moved to the previous
        # iteration's tail, the transfer read element k + 1 before k + 1 wrote
        # it -- `sliced_write_view`, `accumulate_then_read` and two more, all
        # computing from the stale value.  A batch-invariant source is one
        # address for every element, so there any store in the body counts.
        src = load._src
        batch_invariant = getattr(src.obj, 'addressing', None) is Addressing.NONE
        region = body if batch_invariant else body[:at]
        writer = next((i for i in region
                       if i is not load and self._writes_source(i, src, producer)),
                      None)
        if writer is not None:
            return self._reject(
                name, f'{type(writer).__name__} writes the tensor the transfer '
                      f'reads{" (batch-invariant)" if batch_invariant else " ahead of it"}, '
                      f'so the transfer would read the element before the write')
        for instr in body[:at]:
            if self._blocks(load, dest, instr, shared, alloc):
                return self._reject(
                    name, f'{type(instr).__name__} ahead of the transfer '
                          f'touches its buffer'
                          f'{" or orders shared memory" if shared else ""}, '
                          f'so the transfer cannot leave its own iteration')
        return TailWrap(transfer=load, producer=producer, alloc=alloc)

    @staticmethod
    def _writes_source(instr, src, producer) -> bool:
        """Does `instr` write the memory `src` points into?

        Through the same pointer, or through another binding of the same
        tensor.  The binding itself -- the `GetElementPtr` that names `src`, or
        any other -- computes an address and writes nothing.
        """
        if instr is producer or isinstance(instr, GetElementPtr):
            return False
        return any(d is src or (getattr(d, 'obj', None) is not None
                                and d.obj is src.obj)
                   for d in instr.defs())

    @staticmethod
    def _blocks(load, dest, instr, shared: bool, alloc) -> bool:
        """May ``load``, retargeted to element ``k + 1``, not cross ``instr``?

        ``MoveLoads._conflicts`` with one difference: the moved transfer reads
        a pointer to the *next* element, bound right in front of it, so this
        element's pointer binding is no dependence -- the binding; a store
        through it is, and `_writes_source` asks that separately.  What is
        left here is the buffer -- anything reading or writing it -- and what ``MoveLoads``
        never takes a transfer across: an instruction that does not say what it
        touches, and, for a shared destination only, a barrier.  A barrier
        orders what other threads did to shared memory; a register is this
        thread's own.  The register's own declaration is not an obstacle
        either: it leaves the loop with the transfer.
        """
        if instr is load or instr is alloc:
            return False
        if instr.barrier_scope() is not None:
            return shared
        if not instr.describes_dataflow():
            return True
        return any(s is dest for s in tuple(instr.defs()) + tuple(instr.uses()))

    def _apply(self, loop: BatchLoop, body: List[AbstractInstruction],
               plan: TailWrap) -> None:
        """Rewrite one transfer, and give the loop its peel (and drain)."""
        transfer = plan.transfer
        old_src = transfer._src
        dest = transfer.defs()[0]
        shared = isinstance(transfer, GlbToShrLoader)

        ahead, ahead_ptr, peeled, peeled_ptr = self._pointers(loop, plan)
        if shared:
            peeled_load = GlbToShrLoader(**{**transfer._ctor_kwargs,
                                            'src': peeled})
            # Registered at the end of the buffer's user list, and left there:
            # the in-loop transfer stays the first user, so it keeps declaring
            # the window, in the loop's body, where `_declare_windows_early`
            # puts it -- and the peel, emitted in that same body right after,
            # writes through a window the body knows.  Making the peel the
            # first user instead moved the declaration into a body of its own,
            # and both transfers fell back to text.
        else:
            peeled_load = GlbToRegLoader(context=self._context,
                                         src=peeled,
                                         dest=dest,
                                         num_threads=transfer._num_threads,
                                         linearize=transfer._linearize,
                                         src_bbox=transfer._bbox,
                                         src_offset=transfer._offset)

        # Retarget the body transfer in place: the LoadWait that MoveLoads left
        # at the consumer keeps pointing at this object, which is what orders
        # the consumer after it.
        self._drop_user(old_src, transfer)
        transfer._src = ahead
        if shared:
            transfer._ctor_kwargs['src'] = ahead
        ahead.add_user(transfer)
        transfer._wrapped = True
        if shared:
            # The loop carries this transfer's tokens, and the peel's are the
            # first iteration's: `BatchLoop` needs to know which peel that is.
            transfer._peel = peeled_load
        if plan.producer.dereferences_the_batch():
            # A pointer array: the transfer follows the pointer of the element
            # it fetches, and that pointer is promised only for an element the
            # caller did not mask.  `BatchLoop` puts it under that element's
            # flag -- the transfer, not the address, since reading the array
            # entry is in range.  Refusing instead left `local_flux`, whose
            # operands all come out of pointer arrays, entirely unwrapped.
            transfer._guard_by_own_flag = True
            peeled_load._guard_by_own_flag = True

        body.remove(transfer)
        if plan.alloc is not None:
            body.remove(plan.alloc)
        if ahead_ptr is not None:
            # The next element's address at the head of the body, not next to
            # the transfer at its tail.  Out of a pointer array it is a load,
            # and bound right before the copy it is a load the copy waits on
            # at once: on AMD a `vmcnt(0)` per wrapped transfer, since the
            # counter retires in order and the pointer is the newest load.  At
            # the head it has the whole body to arrive.  Outside the guard like
            # the transfer, and as safe: the entry is read at the clamped index,
            # in range whatever the mask; only following it needs the flag.
            body.insert(0, ahead_ptr)
            loop.mark_unguarded([ahead_ptr])
        body.append(transfer)
        loop.mark_unguarded_tail([transfer])

        # The binding may now feed nothing -- but a store names the pointer it
        # writes through among its `defs()`, not its `uses()`.  Asking `uses()`
        # alone took `glb_m0` away from under the store to `C` once the load of
        # `C` for an accumulation had been moved to `wrap_glb_m0`, and the
        # kernel did not compile.
        if not any(any(x is old_src for x in tuple(i.uses()) + tuple(i.defs()))
                   for i in body):
            body.remove(plan.producer)
            self._drop_user(old_src, plan.producer)

        hoisted = [plan.alloc] if plan.alloc is not None else []
        bound = [peeled_ptr] if peeled_ptr is not None else []
        loop.add_wrap_prologue(hoisted + bound + [peeled_load])
        if shared:
            loop.add_wrap_epilogue([LoadWait(transfer)])
        self.wrapped.append(
            f'{getattr(dest, "name", "?")} [{"shr" if shared else "reg"}]')

    def _pointers(self, loop: BatchLoop, plan: TailWrap):
        """`(ahead, ahead_ptr, peeled, peeled_ptr)` for this transfer's source.

        The two bindings -- to element k + 1, the loop's lookahead index,
        clamped so the last iteration prefetches a valid address it never
        reads; and to the thread's first element, clamped likewise -- are made
        once per source symbol.  A second transfer out of the same pointer, `C`
        read into registers and into shared memory, say, reuses them; the
        instructions come back as `None` so it does not bind them again.
        Binding them per transfer declared `peel_glb_m0` twice in one scope,
        which does not compile -- and each is a 64-bit pointer held across the
        loop, so a second copy is two registers for nothing.

        Names stay unique even so: two distinct symbols may carry the same
        name, and then the second pair gets a suffix.
        """
        old_src = plan.transfer._src
        known = self._pointers_for.get(id(old_src))
        if known is not None:
            return known[0], None, known[1], None
        base = old_src.name
        suffix = ''
        while f'wrap_{base}{suffix}' in self._names:
            suffix = f'_{len(self._names)}'
        self._names.update({f'wrap_{base}{suffix}', f'peel_{base}{suffix}'})

        ahead = Symbol(f'wrap_{base}{suffix}', old_src.stype, old_src.obj)
        ahead.data_view = old_src.data_view
        ahead_ptr = GetElementPtr(self._context,
                                  src=plan.producer._src,
                                  dest=ahead,
                                  include_extra_offset=plan.producer._include_extra_offset,
                                  batch_offset=1)
        peeled = Symbol(f'peel_{base}{suffix}', old_src.stype, old_src.obj)
        peeled.data_view = old_src.data_view
        peeled_ptr = GetElementPtr(self._context,
                                   src=plan.producer._src,
                                   dest=peeled,
                                   include_extra_offset=plan.producer._include_extra_offset,
                                   batch_offset=loop.prologue_index())
        self._pointers_for[id(old_src)] = (ahead, peeled)
        return ahead, ahead_ptr, peeled, peeled_ptr

    def _reject(self, name, reason) -> None:
        self.rejected.append((name, reason))
        return None

    @staticmethod
    def _producer(body, sym) -> Optional[GetElementPtr]:
        for instr in body:
            if isinstance(instr, GetElementPtr) and any(d is sym
                                                        for d in instr.defs()):
                return instr
        return None

    @staticmethod
    def _drop_user(sym, instr) -> None:
        users = sym.get_user_list()
        while instr in users:
            users.remove(instr)

    # ------------------------------------------------------------------ #

    def report(self) -> str:
        lines = [f'wrap: {len(self.wrapped)} transfer(s) wrapped, '
                 f'{len(self.rejected)} rejected']
        for name in self.wrapped:
            lines.append(f'  + {name}')
        for name, reason in self.rejected:
            lines.append(f'  - {name}: {reason}')
        return '\n'.join(lines)
