# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""``dest = reduce(op, src, dims)`` -- shape-changing, so not elementwise.

Two emission paths, split on where the contracted axes sit relative to the
thread-distributed one:

* **non-lead** -- every contracted axis is a sequential axis.  Each lane owns
  one point of the kept iteration space and folds the contracted axes into a
  register.  No lane ever reads another lane's value, so there is no shuffle,
  no shared memory and no barrier.  This is the path implemented here.

* **lead** -- a contracted axis *is* the thread-distributed one, so the fold
  crosses lanes.  Within one wave that is a shuffle butterfly; across waves the
  waves' partials meet in a scratch tile in shared memory, with a rendezvous of
  the multiplication around it (`_fold_across_waves`).  A width that is not a
  whole number of waves is still refused: its lanes are interleaved with other
  multiplications' and are not the contiguous run the second stage indexes.
"""

from typing import List, Sequence

from tensorforge.backend.symbol import (LeadLoop, Loop, SymbolType,
                                        SymbolView, Variable, add_offset,
                                        write_loops)
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError
from tensorforge.common.operation import ReductionOperator

from . import ComputeInstruction
from .crosslane import CrossLaneFold


class ReductionInstruction(CrossLaneFold, ComputeInstruction):
    def __init__(self,
                 context: Context,
                 dest: SymbolView,
                 src: SymbolView,
                 dims: List[int],
                 operation: ReductionOperator,
                 prefer_align: bool,
                 num_threads: int,
                 permute: List[int] = None):
        super(ReductionInstruction, self).__init__(context)
        self._dest = dest
        self._op = src
        self._dims = sorted(dims)
        self._operation = operation
        self._permute = permute
        self._prefer_align = prefer_align
        self._num_threads = num_threads
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self.registers = None

        # Dropping the contracted axes renumbers what is left, but not what
        # the destination *is*: it is declared at the rank it keeps, so the box
        # it states is the box this operation writes.
        self.claim_destination(self._dest)
        self.check_addressable([self._dest, self._op], 'reduction')

        for view in (dest, src):
            view.symbol.add_user(self)

    # -- data flow ------------------------------------------------------- #

    def defs(self):
        return (self._dest.symbol,)

    def uses(self):
        return (self._op.symbol,)

    def get_operands(self):
        return [self._op.symbol]

    # -- iteration space ------------------------------------------------- #

    @property
    def _dtype(self):
        """What the reduction runs in: its operand's datatype.

        It was the kernel's floating-point type throughout -- accumulator,
        neutral element, combine and exchange -- so `all(B >= C)`, an `And`
        over booleans, instantiated `ReductionOperation<float, And>`, whose
        `&&` does not exist for `float`: no target compiled it.  The kernel's
        type is only the answer where the operand states none.
        """
        dtype = (getattr(self._op.symbol.obj, 'datatype', None)
                 or self._context.fp_type)
        # Narrower than a register, a value is widened for the exchange: the
        # cross-lane moves are defined in whole 32-bit words (`hip.h` asserts
        # it, and gfx1150 refused `all` over booleans), and 0/1 in an int is
        # what `&`, `|` and `^` fold correctly.  The store narrows it again.
        if dtype in (Datatype.BOOL, Datatype.I8, Datatype.I16):
            return Datatype.I32
        return dtype

    def _kept(self) -> List[int]:
        rank = self._op.bbox.rank()
        return [i for i in range(rank) if i not in self._dims]

    # -- emission -------------------------------------------------------- #

    def gen_code_inner(self, writer: Writer):
        src_lead = self.lead_dim(self._op)
        kept = self._kept()

        if src_lead in self._dims:
            self._check_cross_lane_is_available()
            self._lead_reduction(writer, kept, src_lead)
            return

        self._check_dest_lead(kept, src_lead)
        self._nonlead_reduction(writer, kept, src_lead)

    def _check_dest_lead(self, kept: Sequence[int], src_lead: int) -> None:
        """The kept lead axis has to land on the destination's lead axis.

        Dropping the contracted axes renumbers what is left, so the source's
        lead axis sits at `kept.index(src_lead)` in the destination.  If the
        destination declares a different one, the lane that computed a value
        and the lane that stores it are not the same lane, and the result is
        wrong in a way no shape check upstream would catch.
        """
        if self._dest.bbox.rank() != len(kept):
            # A full reduction may legitimately land in a rank-1 buffer of one
            # element (`ReductionDescr` allows it); that shape has no lead axis
            # to agree with, and it only arises when every axis is contracted,
            # which the caller has already rejected.
            return
        expected = kept.index(src_lead)
        actual = self.lead_dim(self._dest)
        if actual != expected:
            raise InternalError(
                f'reduction: {self._op.symbol.name} leads on axis {src_lead}, '
                f'which is axis {expected} of the destination after dropping '
                f'{self._dims}, but {self._dest.symbol.name} leads on axis '
                f'{actual}')

    def _nonlead_reduction(self, writer: Writer, kept: Sequence[int],
                           src_lead: int) -> None:
        """Kept axes outside, contracted axes folded into a register inside."""
        loopstack = []
        for i in kept:
            lo, hi = 0, self._op.bbox.size(i)
            if i == src_lead:
                loopstack.append(LeadLoop(f'k{i}', lo, hi, self._num_threads, 1,
                                          unroll=False))
            else:
                loopstack.append(Loop(f'k{i}', lo, hi, 1, unroll=False))

        write_loops(self._context, writer, loopstack, self._body(writer, kept))

    def _check_cross_lane_is_available(self) -> None:
        """A cross-lane fold needs whole waves once it spans more than one.

        The exchange in `tensorforge_device` is a shuffle, and a shuffle
        reaches one wave (`_reach`).  Above that the waves' partials meet in
        shared memory -- one slot per wave, a rendezvous, a fold over the
        slots -- which `_fold_across_waves` emits and `temp_shmem` reserves
        for.  Both index the waves of one multiplication as a contiguous run
        of lanes.

        That run is what a width between two multiples of the wave does not
        have: `MultLayout` interleaves such widths with the other
        multiplications in the same wave (gcd units), so lane `l` of a
        multiplication is not thread `l`, and neither the slot index nor the
        rendezvous count is expressible.  `CudaLexic.has_sync_mult` says the
        same thing from the other side -- `barrier.sync` counts threads in
        whole warps.
        """
        vul = self._reach()
        if self._num_threads > vul and self._num_threads % vul:
            raise InternalError(
                f'reduction: a cross-lane fold over {self._num_threads} '
                f'threads spans {self._num_threads / vul:.2f} waves of {vul}. '
                f'The second stage meets the waves of one multiplication in '
                f'shared memory, which needs their lanes contiguous -- a '
                f'width that leaves a partial wave is interleaved with the '
                f'other multiplications of that wave instead.')

    def temp_shmem(self) -> int:
        """One slot per wave, per multiplication, for the super-wave fold.

        Declared even though `_check_cross_lane_is_available` currently
        refuses that case: the budget is read before any body is built, the
        figure is a property of the thread count alone, and stating it here
        keeps the reservation and the use in one place for when the barrier
        arrives.  `tempShrMem` is already striped by `threadIdx.y`, so this is
        per multiple and not per block.
        """
        if not self._contracts_lead():
            return 0
        return self._wave_slots()

    def _contracts_lead(self) -> bool:
        return self.lead_dim(self._op) in self._dims

    def _lead_reduction(self, writer: Writer, kept: Sequence[int],
                        src_lead: int) -> None:
        """The lead axis is contracted, so the fold crosses lanes.

        Three steps, and the middle one is why this cannot reuse `LeadLoop`:

        1. each lane folds the contracted axes it owns into a partial.  Lanes
           that own no element -- the lead extent need not fill the thread
           count -- take the neutral element instead of being skipped;
        2. one cross-lane all-reduce over the partials;
        3. lane 0 stores.

        Step 1's `if`/`else` is load-bearing.  `LeadLoop` would emit a bare
        guard, and a shuffle under a guard is undefined behavior whenever the
        mask names a lane that did not reach it: with a lead extent of 16 and
        32 threads, half the warp would sit outside the region while the other
        half asked it for a value.  `LeadLoop.neutral` looks like it addresses
        this, but the pass that would consume the attribute (`if_convert`) is
        not in the default pipeline and does not read it.  Selecting the
        neutral element explicitly makes every lane arrive at the exchange
        with something defined, which is what the exchange requires.

        Kept axes wrap all three steps as sequential loops.  They cannot stay
        thread-distributed: the lanes are spoken for by the axis being
        contracted, and the whole of steps 2 and 3 has to run once per kept
        point.  The destination is therefore written by one lane for every
        element, which is as uncoalesced as it sounds and is the price of
        contracting the axis the hardware distributes.
        """
        loopstack = [Loop(f'k{i}', 0, self._op.bbox.size(i), 1, unroll=False)
                     for i in kept]

        def inner(varlist):
            index = dict(zip(kept, varlist))
            self._fold_across_lanes(writer, index, kept, varlist, src_lead)

        if loopstack:
            write_loops(self._context, writer, loopstack, inner)
        else:
            inner([])

    def _fold_across_lanes(self, writer: Writer, index: dict,
                           kept: Sequence[int], varlist, src_lead: int) -> None:
        from tensorforge.backend.pir.core import BOOL

        lead = self._lane(writer)
        partial = self._lane_partial(writer, index, src_lead, lead)
        total = self._fold_across_waves(
            writer,
            self._cross_lane(writer, partial, self._exchange_width(src_lead)),
            lead)

        # In SPMD `reduction` is an all-reduce, so every lane holds the answer
        # and letting them all write would be a race on one address rather
        # than a disagreement.  Still a race, so it is guarded.
        #
        # Under an explicit vector there is nobody to guard against.  The
        # reduction *collapses* the lane axis -- its result is one value in
        # one work-item -- so there is no second writer, and a guard would be
        # `if (mask)` over a comparison against a vector lane index, which is
        # not a branch condition at all.
        #
        # Nor into registers, where there is no address to race on: every lane
        # writes its own copy, and every copy has to hold the answer.  A
        # temporary's image is written to its buffer by every lane at once
        # (`StoreRegToShr`: one value, the same on every lane) or read where
        # it is.  Guarded, lane 0 alone held it, and the other lanes stored
        # their zeros to the address lane 0 stored the answer to: SeisSol's
        # damage step takes the `max` of ten temporaries this way, and read
        # back whichever lane won.  `_exchange_width` makes the all-reduce
        # reach every lane for it.
        if writer._explicit_simd() or self._into_registers():
            self._dest.symbol.store(writer, self._context, total,
                                    self._dest_index(kept, varlist), False)
            return
        with writer.if_(writer.op('eq', BOOL, lead, 0, hint='w')):
            self._dest.symbol.store(writer, self._context, total,
                                    self._dest_index(kept, varlist), False)

    def _into_registers(self) -> bool:
        """Is the destination one value in registers, a copy on every lane?

        Without axes only.  With kept axes the fold runs them as sequential
        loops, and a register image spread over the lanes is not addressed by
        a sequential index at all -- `Symbol.access_address` takes it for the
        slot -- so there is nothing a guard could make right or wrong there.
        """
        return (self._dest.bbox.rank() == 0
                and self._dest.symbol.stype in (SymbolType.Register,
                                                SymbolType.Scratch))

    def _slots(self, src_lead: int) -> int:
        """How many elements of the lead axis one lane owns.

        `LeadIndex(slot, threads, stride)` addresses one of them.  Folding only
        slot 0 -- which is what a single `LeadIndex(0, ...)` does -- silently
        drops every element past the first `num_threads`, and the answer that
        comes out is a sum over the part that fitted.
        """
        extent = self._op.bbox.size(src_lead)
        return (extent + self._num_threads - 1) // self._num_threads

    def _lane_partial(self, writer: Writer, index: dict, src_lead: int, lead):
        """This lane's fold over every slot of the lead axis that it owns.

        The slot count is a compile-time constant, so the slots are unrolled
        here rather than emitted as a loop: only the last one can be ragged,
        and unrolling means the guard is emitted for that one alone instead of
        being evaluated on every iteration.

        A ragged slot is an `if`/`else` yielding the neutral element, not a
        bare guard.  The exchange downstream is a shuffle, and a shuffle
        reached by only part of the wave is undefined -- so every lane has to
        leave here with a defined value even when it owns nothing.
        """
        from tensorforge.backend.pir.core import BOOL, ScalarType

        acc_type = ScalarType(self._dtype)
        extent = self._op.bbox.size(src_lead)
        rest = [d for d in self._dims if d != src_lead]

        def fold(slot, valid=None):
            from tensorforge.backend.symbol import LeadIndex
            inner = dict(index)
            inner[src_lead] = LeadIndex(slot, self._num_threads, 1,
                                        valid=valid)
            return self._fold_axes(writer, inner, rest, 0)

        acc = None
        for slot in range(self._slots(src_lead)):
            lo = slot * self._num_threads
            if lo + self._num_threads <= extent:
                contrib = fold(slot)
            elif writer._explicit_simd():
                # A lane-varying condition is a mask here, not a branch: the
                # load is held to the lanes that own an element (`valid`, the
                # full-lane tail's mechanism), and the rest take the neutral
                # element by selection.
                owned = extent - lo
                contrib = writer.op(
                    'select', acc_type,
                    writer.op('lt', BOOL, lead, owned, hint='own'),
                    fold(slot, valid=owned),
                    writer.const(self._neutral(), acc_type), hint='own')
            else:
                # The guard has to contain the load, not just select after it:
                # a lane past the end would otherwise read out of bounds.
                guard = writer.if_else(
                    writer.op('lt', BOOL, lead, extent - lo, hint='own'),
                    types=(acc_type,))
                with guard.then():
                    guard.yield_(fold(slot))
                with guard.otherwise():
                    guard.yield_(writer.const(self._neutral(), acc_type))
                contrib = guard.result
            acc = contrib if acc is None else self._combine(writer, acc,
                                                            contrib)
        return acc

    def _exchange_width(self, src_lead: int) -> int:
        """How many lanes the fold actually has to cross.

        `num_threads` is the upper bound, not the answer.  When the lead axis
        fits in one slot, only lanes `0 .. extent-1` hold anything; the rest
        arrive with the neutral element and reducing them in costs a butterfly
        step that cannot change the result.  Rounding the extent up to a power
        of two keeps the groups aligned, so lane 0's group still covers every
        lane that holds data -- and lane 0 is the one that stores.

        16 over 32 threads is four exchanges instead of five.  With more than
        one slot every lane holds data and the full width is the answer.  So
        it is for a destination in registers, whatever the extent: there every
        lane keeps the answer (`_fold_across_lanes`), and a lane outside lane
        0's group would keep its own group's fold of neutral elements.
        """
        # And never wider than one exchange reaches: past that the fold is in
        # two stages, and this is the first of them (`_fold_across_waves`).
        reach = self._reach()
        if self._slots(src_lead) > 1 or self._into_registers():
            return min(self._num_threads, reach)
        extent = self._op.bbox.size(src_lead)
        width = 1
        while width < extent:
            width <<= 1
        return min(width, self._num_threads, reach)

    def _body(self, writer: Writer, kept: Sequence[int]):
        def inner(varlist):
            # `write_loops` hands over one variable per loop it wrote, in the
            # order the loops were pushed -- i.e. per *kept* axis.  The
            # contracted axes get their variables from the fold below.
            index = dict(zip(kept, varlist))
            value = self._fold(writer, index, 0)
            if value is None:
                raise InternalError(
                    f'reduction: {self._op.symbol.name} yielded no value for '
                    f'its load; the sparse path has no reduction lowering yet')
            self._dest.symbol.store(writer, self._context, value,
                                    self._dest_index(kept, varlist), False)

        return inner

    def _fold(self, writer: Writer, index: dict, depth: int):
        return self._fold_axes(writer, index, self._dims, depth)

    def _fold_axes(self, writer: Writer, index: dict, axes: Sequence[int],
                   depth: int):
        """Nest one `for` per contracted axis, carrying the accumulator.

        The accumulator is a loop-carried SSA value rather than a declared
        register mutated in place: `Op.ACCUM` lowers to `+=` and so speaks only
        for `AddOperator`, while `min`, `max` and `*` need the operator's own
        combine.  Carrying it keeps all five operators on one path and leaves
        the fold visible to the passes instead of hidden behind a name.
        """
        from tensorforge.backend.pir.core import ScalarType

        if depth == len(axes):
            return self._load(writer, index)

        axis = axes[depth]
        lo, hi = 0, self._op.bbox.size(axis)
        acc_type = ScalarType(self._dtype)

        if lo >= hi:
            # An empty contraction is the neutral element by definition.  It
            # cannot come out of the loop below, which would never run.
            return writer.const(self._neutral(), acc_type)

        loop = writer.for_(lo, hi, 1, inits=(self._neutral(),),
                           types=(acc_type,), unroll=True, hint=f'r{axis}')
        with loop:
            index[axis] = Variable(str(loop.induction), Datatype.I32,
                                   loop.induction)
            inner = self._fold_axes(writer, index, axes, depth + 1)
            loop.yield_(self._combine(writer, loop.iter_args[0], inner))
        del index[axis]
        return loop.result

    def _neutral(self):
        """The operator's identity, as a literal of the reduction's dtype.

        The operator answers per type, so the integer case no longer needs
        rejecting here: `min` over `I32` starts at `INT32_MAX`, not at an
        infinity that the type cannot hold.
        """
        # The raw Python value, not `fp.literal(...)` of it: the emitter calls
        # `literal` itself on a CONST's value.  Formatting here too worked by
        # accident for the infinities, since `float('-INFINITY')` parses, and
        # not at all for `0.0f`, which does not.
        return self._operation.neutral(self._dtype)

    def _combine(self, writer: Writer, acc, value):
        from tensorforge.backend.pir.core import ScalarType
        irop = self._operation.irop()
        if irop is None:
            raise InternalError(
                f'reduction: {self._operation} has no pseudo-IR op to combine '
                f'with')
        return writer.op(irop, ScalarType(self._dtype), acc, value,
                         hint='r', pure=True)

    def _load(self, writer: Writer, index: dict):
        addr = [index[i] for i in range(self._op.bbox.rank())]
        value = self._op.symbol.load(writer, self._context, None,
                                     self._offset(self._op, addr), False)
        if value is None:
            # Checked here rather than where the fold ends, because a load
            # nested inside a sequential axis hands its None to `_combine`
            # instead of returning it: the loop result is a value either way,
            # so the miss is invisible one frame up.  `Symbol.load` answers
            # None for every structured load under `simd_mode`, and the fold
            # built `max(acc, None)` out of it.
            raise InternalError(
                f'reduction: {self._op.symbol.name} has no structured load on '
                f'this backend, so there is no value to fold. The reduction '
                f'needs one; a named-variable load would not give the fold '
                f'something it can carry across the loop.')
        return value

    def _dest_index(self, kept: Sequence[int], varlist) -> List:
        if self._dest.bbox.rank() == len(kept):
            return self._offset(self._dest, list(varlist))
        # Full reduction into a one-element sink: the kept space is empty, so
        # there is nothing to index by.
        return self._offset(self._dest, [0] * self._dest.bbox.rank())

    @staticmethod
    def _offset(view: SymbolView, varlist: List) -> List:
        """Fold the bounding box's lower corner into each index -- into a
        `LeadIndex` itself, which a `VarOffset` may not wrap."""
        return [add_offset(varlist[i], o)
                for i, o in enumerate(view.bbox.lower())]

    def __str__(self):
        return (f'{self._dest.symbol.name} = {self._operation}'
                f'({self._op.symbol.name}, dims={self._dims})')
