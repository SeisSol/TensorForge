# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Folding one value out of what the lanes hold between them.

Two operations ask for this and they are not the same operation.  A reduction
contracts the thread-distributed axis, so each lane folds the elements it owns
and the lanes then fold those partials.  A contraction to one value multiplies
operands and sums over indices that have nothing to do with the lanes -- until
one of its operands is a register image, which is spread over them, and the
sum over that index becomes the same question.

So the fold lives here rather than in either: the wave-wide exchange, the
rendezvous for a multiplication wider than one wave, and the slot count both
the reservation and the emission have to agree on.

What a host provides:

* ``_context`` and ``_num_threads`` -- the geometry;
* ``_dtype`` -- what is folded, which is the operand's type and not the
  kernel's (`ReductionInstruction._dtype`: `all()` over booleans folds ints);
* ``_operation`` -- an operator that can name itself to the lexic
  (``operation()``) and combine two values (``irop()``);
* ``_combine(writer, acc, value)`` -- the combining step, because an operator
  that has no pseudo-IR op says so in its own words.
"""

from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import InternalError


class CrossLaneFold:
    """The exchange, and the rendezvous where the exchange does not reach."""

    def _reach(self) -> int:
        """How far one exchange reaches (`Lexic.exchange_reach`): the wave,
        a sub-group the kernel states, or under ESIMD the whole vector."""
        vm = self._context.get_vm()
        return vm.get_lexic().exchange_reach(self._num_threads,
                                             vm.get_hw_descr())

    def _lane(self, writer: Writer):
        """Which element of the distributed dimension this lane is at.

        `writer.lane_index`, not `threadIdx.x % num_threads` spelled out.  The
        arithmetic here was the SPMD answer written as though it were the only
        one -- and it is not: an explicitly vectorized lowering holds every
        element at once, so the answer is a vector of all `num_threads`
        indices rather than one of them.

        Same call `LeadLoop._lead` makes, so `cse` still merges the two.
        """
        return writer.lane_index(self._num_threads, 1, hint='lead')

    def _wave_slots(self) -> int:
        """Slots the second stage needs: one per wave, per multiplication.

        Zero where one exchange reaches every lane.  `temp_shmem` states this
        before any body exists and `_fold_across_waves` uses it while emitting,
        so the two read it from one place.
        """
        reach = self._reach()
        return max(0, self._num_threads // reach) if self._num_threads > reach \
            else 0

    def _cross_lane(self, writer: Writer, partial, width: int):
        """The all-reduce over `width` lanes, as the lexic spells it.

        A call into `tensorforge_device`, not a butterfly built here: both
        backends already define `tensorforge::reduction` under the same name,
        `multilinear`'s lead-dimension fold wants the identical exchange, and a
        backend whose sub-group reduction is a single library call -- SYCL's
        `reduce_over_group`, say -- overrides one method instead of growing a
        second lowering.
        """
        from tensorforge.backend.pir.core import ScalarType

        lexic = self._context.get_vm().get_lexic()
        if lexic.exchange_xor('{0}', 1) is not None:
            # No all-reduce to call, only the exchange (SPMD SYCL): the
            # butterfly, one step per bit, combined by the operator's `irop`.
            if width & (width - 1):
                raise InternalError(
                    f'cross-lane fold: a butterfly over {width} lanes; it '
                    f'pairs lanes by their bits, which needs a power of two')
            fp = ScalarType(self._dtype)
            acc, mask = partial, 1
            while mask < width:
                other = writer.rawexpr(lexic.exchange_xor('{0}', mask), acc,
                                       type_=fp, hint='x', pure=True,
                                       movable=False, crosslane=True)
                acc = self._combine(writer, acc, other)
                mask <<= 1
            return acc
        text = lexic.reduction('{0}', self._operation.operation(),
                               self._dtype, width, subblock=1)
        return writer.rawexpr(text, partial,
                              type_=ScalarType(self._dtype),
                              hint='red', pure=True, movable=False)

    def _fold_across_waves(self, writer: Writer, total, lead):
        """The second stage: the waves' partials meet in shared memory.

        Each wave has folded its own lanes by shuffle; its first lane writes
        that partial into a slot of the scratch tail `temp_shmem` reserved,
        and after a rendezvous of the multiplication every lane reads every
        slot and folds them.  Every lane, not one: this is an all-reduce, and
        a destination in registers keeps a copy per lane.

        Two barriers rather than one.  The second is what makes the stores
        visible to the reads.  The first is for whatever loop encloses this:
        it runs the whole sequence again, and without it a lane that has
        arrived at the next round's store would overwrite a slot another lane
        is still reading.
        """
        waves = self._wave_slots()
        if waves <= 1:
            return total

        from tensorforge.backend.pir.core import BOOL, INDEX, MemSpace

        reach = self._reach()
        slots = writer.alloc(self._dtype, (waves,), MemSpace.SHARED,
                             hint='fold')
        wave = writer.op('div', INDEX, lead, writer.const(reach, INDEX),
                         hint='wave')
        lane = writer.op('rem', INDEX, lead, writer.const(reach, INDEX),
                         hint='inwave')
        writer.barrier('mult', threads=self._num_threads)
        with writer.if_(writer.op('eq', BOOL, lane, 0, hint='w')):
            writer.store(slots, total, wave)
        writer.barrier('mult', threads=self._num_threads, handoff=True)
        out = None
        for i in range(waves):
            part = writer.load(slots, writer.const(i, INDEX), hint='fold')
            out = part if out is None else self._combine(writer, out, part)
        return out
