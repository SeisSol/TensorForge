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
  crosses lanes.  Within one wave that is a shuffle butterfly; across waves it
  needs a scratch tile in shared memory plus a barrier.  Not implemented; the
  guard below says so rather than emitting something plausible and wrong.
"""

from typing import List, Sequence

from tensorforge.backend.symbol import (LeadLoop, Loop, SymbolView, Variable,
                                        VarOffset, write_loops)
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError
from tensorforge.common.operation import ReductionOperator

from . import ComputeInstruction


class ReductionInstruction(ComputeInstruction):
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

    @staticmethod
    def _lead_dim(view: SymbolView) -> int:
        """Which axis of `view` is distributed over threads.

        Read off the symbol rather than assumed to be axis 0, so that when
        `Symbol.lead_dims` becomes authoritative for every symbol type -- it is
        already the truth for register tiles, set by `multilinear_builder`, and
        already consulted by `Symbol.load`/`store` -- this path follows without
        an edit.  The instruction-local `self._lead_dims = [0]` that
        `ElementwiseInstruction` and the multilinear path carry is the same
        fact stated a second time, and the two can disagree.
        """
        lead = getattr(view.symbol, 'lead_dims', [0])
        if len(lead) != 1:
            raise InternalError(
                f'reduction: {view.symbol.name} declares {len(lead)} lead '
                f'dimensions; exactly one is supported')
        return lead[0]

    def _kept(self) -> List[int]:
        rank = self._op.bbox.rank()
        return [i for i in range(rank) if i not in self._dims]

    # -- emission -------------------------------------------------------- #

    def gen_code_inner(self, writer: Writer):
        src_lead = self._lead_dim(self._op)
        kept = self._kept()

        if src_lead in self._dims:
            raise InternalError(
                f'reduction over the lead dimension is not implemented yet: '
                f'{self._operation} over axes {self._dims} contracts axis '
                f'{src_lead} of {self._op.symbol.name}, which is distributed '
                f'over threads. A cross-lane fold is needed here.')

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
        actual = self._lead_dim(self._dest)
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
        """Nest one `for` per contracted axis, carrying the accumulator.

        The accumulator is a loop-carried SSA value rather than a declared
        register mutated in place: `Op.ACCUM` lowers to `+=` and so speaks only
        for `AddOperator`, while `min`, `max` and `*` need the operator's own
        combine.  Carrying it keeps all five operators on one path and leaves
        the fold visible to the passes instead of hidden behind a name.
        """
        from tensorforge.backend.pir.core import ScalarType

        if depth == len(self._dims):
            return self._load(writer, index)

        axis = self._dims[depth]
        lo, hi = 0, self._op.bbox.size(axis)
        acc_type = ScalarType(self._context.fp_type)

        if lo >= hi:
            # An empty contraction is the neutral element by definition.  It
            # cannot come out of the loop below, which would never run.
            return writer.const(self._neutral(), acc_type)

        loop = writer.for_(lo, hi, 1, inits=(self._neutral(),),
                           types=(acc_type,), unroll=True, hint=f'r{axis}')
        with loop:
            index[axis] = Variable(str(loop.induction), Datatype.I32,
                                   loop.induction)
            inner = self._fold(writer, index, depth + 1)
            loop.yield_(self._combine(writer, loop.iter_args[0], inner))
        del index[axis]
        return loop.result

    def _neutral(self):
        """The operator's identity, as a literal of the kernel's dtype.

        `MinOperator`/`MaxOperator` return `-+math.inf`, which an integer
        kernel has no representation for; that is rejected here rather than
        spelled into integer code.
        """
        value = self._operation.neutral()
        fp = self._context.fp_type
        if value in (float('inf'), float('-inf')) and \
                fp not in (Datatype.F16, Datatype.F32, Datatype.F64):
            raise InternalError(
                f'reduction: {self._operation} has no representable neutral '
                f'element in {fp}')
        return fp.literal(value)

    def _combine(self, writer: Writer, acc, value):
        from tensorforge.backend.pir.core import ScalarType
        irop = self._operation.irop()
        if irop is None:
            raise InternalError(
                f'reduction: {self._operation} has no pseudo-IR op to combine '
                f'with')
        return writer.op(irop, ScalarType(self._context.fp_type), acc, value,
                         hint='r', pure=True)

    def _load(self, writer: Writer, index: dict):
        addr = [index[i] for i in range(self._op.bbox.rank())]
        return self._op.symbol.load(writer, self._context, None,
                                    self._offset(self._op, addr), False)

    def _dest_index(self, kept: Sequence[int], varlist) -> List:
        if self._dest.bbox.rank() == len(kept):
            return self._offset(self._dest, list(varlist))
        # Full reduction into a one-element sink: the kept space is empty, so
        # there is nothing to index by.
        return self._offset(self._dest, [0] * self._dest.bbox.rank())

    @staticmethod
    def _offset(view: SymbolView, varlist: List) -> List:
        """Fold the bounding box's lower corner into each index."""
        return [VarOffset(varlist[i], o) if o else varlist[i]
                for i, o in enumerate(view.bbox.lower())]

    def __str__(self):
        return (f'{self._dest.symbol.name} = {self._operation}'
                f'({self._op.symbol.name}, dims={self._dims})')
