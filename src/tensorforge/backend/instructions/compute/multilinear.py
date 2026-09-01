# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Union
import math
from . import ComputeInstruction
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.backend.symbol import VecIndex, SymbolType, add_offset, Symbol, SymbolView, DataView, Loop, LeadLoop, write_loops, LeadIndex, LinearizedLoop, Immediate
from tensorforge.common.exceptions import InternalError, GenerationError
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.operation import ReductionOperator
from typing import Union, List
from tensorforge.common.basic_types import Datatype
from tensorforge.backend.pir.core import MemSpace
from tensorforge.backend.instructions.abstract_instruction import _explicit_simd
from tensorforge.backend.writer import Writer

from tensorforge.common.matrix.tensor import Tensor

from .primitives import nvidia as nvidia
from .primitives import amd as amd
from .primitives import intel as intel
from .matmul import MatmulOperands
from .strategy import (ComputeShape, Strategy, choose_strategy,
                       is_contraction, legal_strategies)

#: Which module owns the matrix paths for a vendor.  One row per target, and
#: every question the dispatch asks goes to the same row -- so what gets
#: emitted and what gets reserved for it cannot be answered by two different
#: tables that then disagree.
_VENDOR_MODULES = {
    'amd': amd,
    'nvidia': nvidia,
    'intel': intel,
}


def _vendor_module(context):
    return _VENDOR_MODULES.get(context.get_vm().get_hw_descr().vendor)

import numpy as np

from copy import copy

def _contiguous_first_axis(sym) -> bool:
    """Whether axis 0 of this symbol is the one adjacent in memory.

    The condition a wide load along that axis rests on, and it is a property
    of the *view* rather than of the tensor: a transposed operand has the same
    tensor and a stride that makes the values `ld` apart.
    """
    try:
        return sym.data_view.get_dim_strides()[0] == 1
    except Exception:
        return False


class MultilinearInstruction(ComputeInstruction):
    def __init__(self,
               context: Context,
               dest: Symbol,
               ops: List[SymbolView],
               target: List[List[int]],
               prev: Union[None, Symbol],
               next: Union[None, Symbol],
               productOperation: ReductionOperator,
               sumOperation: ReductionOperator,
               dest_obj,
               num_threads: int,
               blockcount: int=1,
               theta: int=0,
               lead_width: int=1,
               k_width: int=1,
               prev_offset=None):
        super(MultilinearInstruction, self).__init__(context)
        self._dest = dest
        self._ops = ops
        self._target = target
        self._productOperation = productOperation
        self._sumOperation = sumOperation
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self._num_threads = num_threads
        #: Adjacent lead-dimension elements per lane.  The lane count is
        #: already reduced to match, so `threads * lead_width` is what the
        #: lane count was before -- see `vectorize.lead_threads_and_width`.
        self._lead_width = lead_width
        #: Reduction steps one body covers.  The innermost `k` loop steps by
        #: this, and the body emits that many products into one accumulator
        #: before writing it back -- so the destination is read and written
        #: once per group instead of once per step, and the operand whose
        #: contiguous axis *is* `k` is loaded once as a vector of that width.
        self._k_width = k_width
        self._blockcount = blockcount
        # origin of the lead loop, chosen by the builder so that a register-
        # resident operand's lane assignment lines up (see
        # MultilinearBuilder._lead_origin_shift).  Only relative offsets
        # matter, so shifting it is free apart from at most one extra slot.
        self._theta = theta
        self._prev = prev
        self._prev_offset = prev_offset
        self._next = next
        self._dest_obj = dest_obj

        assert num_threads % blockcount == 0

        self.registers = None
        if dest.stype != SymbolType.Register:
            raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
        else:
            self._dest = dest

        for op in self._ops:
            op.symbol.add_user(self)
        dest.add_user(self)

        self._scalar = []
        ops2 = []
        target2 = []
        for op, target in zip(self._ops, self._target):
            if len(target) == 0:
                self._scalar += [op]
            else:
                ops2 += [op]
                target2 += [target]

        self._ops = ops2
        self._target = target2

        self._analyze()

    def _eff_offset(self, i, j):
        """Operand offset as seen from the shifted lead origin."""
        return self._ops[i].offset[j] - (self._theta
                                         if self._target[i][j] == 0 else 0)

    def _check_offsets(self):
        """A slicing offset is a logical->storage shift, never a loop bound.

        Global and shared-memory operands build their address from the whole
        index expression, so the offset rides along as a `VarOffset` and needs
        no further thought.

        Registers distribute the lead dimension across lanes: element `s` sits
        in lane `s % T`, slot `s // T`.  A shift of `q*T` is a change of slot
        with the lane untouched and so is expressible as an address; a
        remainder would move data between lanes and needs a shuffle, which an
        address cannot do.  `Symbol.build_address` applies the whole-block part
        via `unwrap_lead`; anything else has to be consumed at load time, which
        is what GlbToRegLoader does for a fresh staging load.

        The destination needs no check: `_idest` accumulates in logical
        coordinates and StoreRegToGlb folds `dest.offset` in on the way out.
        """
        for op in self._ops:
            if all(o == 0 for o in op.offset):
                continue
            if op.symbol.stype not in (SymbolType.Register, SymbolType.Scratch):
                continue

            i = self._ops.index(op)
            view = op.symbol.data_view
            threads = op.symbol.num_threads
            for j, o in enumerate(op.offset):
                if self._target[i][j] == 0 and self._eff_offset(i, j) % threads != 0:
                    raise GenerationError(
                        f'{op.symbol.name}: lead-dimension slicing offset {o} '
                        f'is not a multiple of {threads}. Only whole '
                        f'thread-blocks can be re-indexed on a register-'
                        f'resident operand; a remainder is a cross-lane move '
                        f'and has to be consumed by the staging load instead.')
                lo, hi = op.bbox.lower()[j] + o, op.bbox.upper()[j] + o
                have = view.get_bbox()
                if lo < have.lower()[j] or hi > have.upper()[j]:
                    raise GenerationError(
                        f'{op.symbol.name}: sliced range [{lo},{hi}) in dim {j} '
                        f'is not covered by what the registers hold '
                        f'([{have.lower()[j]},{have.upper()[j]}))')

    @staticmethod
    def _same_box(a, b):
        return (a.rank() == b.rank()
                and list(a.lower()) == list(b.lower())
                and list(a.upper()) == list(b.upper()))

    def _analyze(self):
        self._check_offsets()

        # The destination decides how many indices this operation writes; the
        # operands only say which of them they carry.  Deriving the rank from
        # the operands alone drops any index none of them mentions --- which is
        # exactly what a broadcast is: `t4[32x3] = t2[32]` has one operand
        # targeting `[0]`, and dimension 1 vanished, so the loop nest ran over
        # `n0` only and wrote one slot per lead block instead of three.  An
        # operand that lacks an index is read at the same address for every
        # value of it, which is the broadcast; the index still has to exist.
        targetrank = self._dest_obj.bbox.rank()
        for i, op in enumerate(self._ops):
            for j in range(op.bbox.rank()):
                targetrank = max(self._target[i][j] + 1, targetrank)
        self._ns = [(-math.inf, math.inf)] * targetrank
        preKs = {}
        self._opdim_to_nks = []
        sparseK = {}
        self._sparseN = [False] * targetrank
        for i, op in enumerate(self._ops):
            opdim = [''] * op.bbox.rank()
            for j in range(op.bbox.rank()):
                # logical coordinates on both sides --- op.offset deliberately
                # does not appear here, and neither does the storage bbox: both
                # are addressing concerns, resolved in Symbol.access_address.
                lower = op.bbox.lower()[j]
                upper = op.bbox.upper()[j]
                if self._target[i][j] < 0:
                    if self._target[i][j] not in preKs:
                        preKs[self._target[i][j]] = (lower, upper)
                        sparseK[self._target[i][j]] = False
                    preKs[self._target[i][j]] = (max(preKs[self._target[i][j]][0], lower), min(preKs[self._target[i][j]][1], upper))
                    opdim[j] = f'k{-self._target[i][j] - 1}'
                    sparseK[self._target[i][j]] |= op.symbol is not None and op.symbol.obj is not None and not op.symbol.obj.is_dense()
                else:
                    self._ns[self._target[i][j]] = (max(self._ns[self._target[i][j]][0], lower), min(self._ns[self._target[i][j]][1], upper))
                    opdim[j] = f'n{self._target[i][j]}'
                    self._sparseN[self._target[i][j]] |= op.symbol is not None and op.symbol.obj is not None and not op.symbol.obj.is_dense()

            self._opdim_to_nks += [opdim]

        for i in range(len(self._ns)):
            # honest intersection with the destination's range.  The previous
            # form clamped a *size* against a range, which only coincides with
            # this one while dest.bbox.lower() == 0.
            self._ns[i] = (max(self._ns[i][0], self._dest_obj.bbox.lower()[i]),
                           min(self._ns[i][1], self._dest_obj.bbox.upper()[i]))

        # move the whole lead loop into the shifted origin; every participant's
        # effective offset drops by theta in turn (_eff_offset)
        if self._theta and self._ns:
            self._ns[0] = (self._ns[0][0] + self._theta,
                           self._ns[0][1] + self._theta)

        self._ks = [0] * len(preKs)
        self._sparseK = [False] * len(preKs)
        for i in range(len(preKs)):
            assert -i-1 in preKs
            self._ks[i] = preKs[-i-1]
            self._sparseK[i] = sparseK[-i-1]

        iterate_dimensions = []
        loads = []
        reductions = []
        self._is_log = False

        # TODO: do not really optimize here anything any more (on a higher level)... Just generate code
        # i.e.: what can be loaded in early/late, do

        # TODO: handle offsets

        self._idest = copy(self._dest)
        self._idest.name = f'i{self._dest.name}'
        self._idest.data_view = DataView(shape = [u - l for l,u in self._ns], permute=[i for i in range(targetrank)])
        self._idest.data_view._bbox = BoundingBox([l for l,_ in self._ns],
                                                  [u for _,u in self._ns])
        self._iregs = 1
        if len(self._ns) > 0:
            self._iregs = -(-self._ns[0][1] // self._num_threads) - self._ns[0][0] // self._num_threads
            # The third copy of the slot-count formula, and the third place
            # that has to agree with the addressing side about how many
            # entries one slot takes.  `DataView.lead_lanes` is that number:
            # one per slot when the lane is the thread, `num_threads` when the
            # work-item holds the whole wave.
            self._iregs *= DataView.lead_lanes(
                None, _explicit_simd(self._context), self._num_threads)
        for l,u in self._ns[1:]:
            self._iregs *= u - l

        if self._theta:
            # the accumulator is indexed by the shifted lead loop, so it takes
            # the shifted view; inheriting an unshifted one from the
            # surrounding symbols would put the store in the wrong origin
            self._dest.data_view = self._idest.data_view
        else:
            # `prev`/`next` are adopted for their *layout*: when the result is
            # handed to or taken from a register image, the accumulator has to
            # be indexed the way that image is, or the transfer between them is
            # off.  Two things disqualify a neighbour.  A global or shared
            # symbol describes the whole buffer, not the box this operation
            # writes.  And a register image is only usable if it is an image of
            # the *same* box: `_deferred_stores` is keyed by symbol name and
            # lives for the whole kernel, so what a later operation finds there
            # may have been staged for a read of a much wider region.  In the
            # poroelastic space-time predictor a one-row-one-column write
            # picked up the image of the whole 32x13x4 tensor and inherited its
            # box; the accumulator then claimed elements it never computed, and
            # the store wrote all of them --- reading past the end of the
            # register array on the way.
            for neighbour in (self._prev, self._next):
                if (neighbour is not None
                        and neighbour.stype in (SymbolType.Register,
                                                SymbolType.Scratch)
                        and neighbour.data_view is not None
                        and self._same_box(neighbour.data_view.get_bbox(),
                                           self._idest.data_view.get_bbox())):
                    self._dest.data_view = neighbour.data_view
            if self._dest.data_view is None:
                self._dest.data_view = self._idest.data_view

        # From the destination symbol, not assumed to be axis 0.  These are
        # destination axis indices: an N axis in `_lead_dims` becomes a
        # `LeadLoop`, everything else a sequential `Loop`.  A K axis would be
        # written `-i-1`, which is the encoding the test at the head of the K
        # nest reads -- nothing produces one today, since a contraction axis
        # spread across the lanes is the cross-lane fold that `_leading_dim`
        # never got.
        #
        # The value is [0] for every case in the corpus, because
        # `MultilinearDescr._lead_dim` aligns the thread count to the
        # destination's axis 0 and nothing sets the destination's `lead_dims`
        # to anything else.  Taking it from the symbol is what makes the two
        # statements one: `multilinear_builder` already sets `lead_dims` on a
        # staged register image, and `Symbol.load` addresses through it.
        self._lead_dims = [self.lead_dim(self._dest)]

    def gen_code_inner(self, writer: Writer):
        # A comment touches nothing.  Left conservative it would be read as
        # touching every buffer, and this one sits above them all.
        writer.Comment(f'{self._ns} {self._ks}')

        if len(self._scalar) == 0 and self._prev is None and self._next is None and self._idest.data_view == self._dest.data_view:
            self._vdest = self._dest
        elif hasattr(writer, 'alloc') and callable(getattr(writer, 'alloc')):
            # Same shape `RegisterAlloc` stopped emitting as text, from a
            # different site.  It matters beyond its own node count: this was
            # the last raw declaration inside a compute body, and
            # `flatten_scopes` keeps the `{ }` around any region whose raw
            # text declares a C++ name -- so one line here held a wall around
            # every multilinear in the corpus.
            #
            # The name stays via `extern` for the same reason as there: the
            # accumulation still spells `ir2` out in places this does not
            # reach yet.
            value = writer.alloc(self._dest.get_fptype(), (self._iregs,),
                                 MemSpace.REGISTER, hint=self._idest.name,
                                 extern=self._idest.name, init='{}')
            self._idest.set_pir_buffer(writer, value)
            self._vdest = self._idest
        else:
            writer(f'{self._dest.get_fptype()} {self._idest.name}[{self._iregs}]{"{}"};',
                   accesses=())
        if not self._nonleading_dim_test(writer):
            self._nonleading_dim(writer)
        if len(self._ns) == 0:
            self._leading_dim(writer)
        self._apply_linear(writer)

    def _nonleading_dim(self, writer: Writer):
        loopstack = []

        # TODO: preload values where necessary (i.e. no N in there)
        # Also, postpone multiplications until necessary

        # thread_mask: TODO
        # writer(f'int32_t n0 = {self._vm.get_lexic().thread_idx_x} % {self._ns[0]};')
        # writer(f'int32_t n1a = {self._vm.get_lexic().thread_idx_x} / {self._ns[0]};')
        # n1i = self._num_threads // self._ns[0]
        # writer(f'int32_t n{i} = dimmin + n1a; n{i} < {dimmax}; n{i} += {n1i}')

        # (for broadcasting)
        force_unroll = True #self._context.get_vm().get_hw_descr().vendor == 'amd'

        matrixK = 1

        loopmap = {}

        outerLoops = []

        # TODO: linearize
        for i, (dimmin, dimmax) in enumerate(self._ks):
            loopmap[f'k{i}'] = len(loopstack) + len(outerLoops)
            if -i-1 not in self._lead_dims:
                step = matrixK if i == len(self._ks) - 1 else 1
                if (i == len(self._ks) - 1 and self._k_width > 1
                        and (self._sparseK[i] or force_unroll)):
                    # Unrolled only.  The last group of a ragged extent is
                    # shorter than the rest, and knowing *how much* shorter is
                    # what lets the body emit the right number of products.
                    # In an unrolled loop the induction value is a Python
                    # integer and the answer is a compile-time count; in a
                    # real `for` it would be a runtime comparison per step,
                    # which costs more than the loads it saves.
                    step *= self._k_width
                loop = [Loop(f'k{i}', dimmin, dimmax, step, unroll=self._sparseK[i] or force_unroll)]
                if self._sparseK[i] or force_unroll or True:# and False:
                    loopstack += loop
                else:
                    outerLoops += loop

        #loopstack += [LinearizedLoop(outerLoops)]

        stride = 1
        threads = self._num_threads
        for i, (dimmin, dimmax) in enumerate(self._ns):
            loopmap[f'n{i}'] = len(loopstack) + len(outerLoops) #- 1
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=self._sparseN[i] or force_unroll)]
            else:
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride,
                                       unroll=self._sparseN[i] or force_unroll,
                                       width=self._lead_width)]
                threads //= max(1, -(-(dimmax - dimmin) // self._lead_width))
                stride *= dimmax - dimmin

        def nonlead_writer(varlist):
            """Fully structured: loads hand back values, products and the
            accumulation are IR ops, no name leaves the block.  Returns False
            if any operand cannot produce a value (the sparse path declares
            and then conditionally assigns a named variable, which is not SSA);
            the caller then falls back."""
            from tensorforge.backend.pir.core import ScalarType
            from tensorforge.backend.symbol import lead_width_of
            # From the *indices*, not from `self._lead_width`.  The lead loop
            # peels the components no whole vector covers and hands them over
            # as plain element indices, so the last body of a ragged
            # dimension is scalar while every earlier one is wide.  Reading
            # the instruction's field instead made the peeled body build a
            # vector type over a scalar element -- a splat of the tail and a
            # wide store past the end of it.
            width = lead_width_of(
                [varlist[loopmap[f'n{i}']] for i, _ in enumerate(self._ns)])
            ftype = (ScalarType(self._idest.get_fptype()) if width == 1
                     else ScalarType(self._idest.get_fptype(), width))
            steps, kslot = self._k_group(varlist, loopmap)

            # One vector load per operand whose contiguous axis is the
            # reduction, hoisted out of the group: `B[k,n] .. B[k+V-1,n]` are
            # adjacent, so the `V` steps below take their operand from
            # components of a single load instead of from `V` loads.
            packs = self._k_packs(writer, varlist, loopmap, kslot, len(steps))

            prod = None
            for c, kval in enumerate(steps):
                terms = []
                for i, op in enumerate(self._ops):
                    if (i, c) in packs:
                        v = packs[(i, c)]
                    else:
                        idx = [add_offset(varlist[loopmap[nk]]
                                          if nk != kslot else kval,
                                          self._eff_offset(i, j))
                               for j, nk in enumerate(self._opdim_to_nks[i])]
                        v = op.symbol.load(writer, self._context, None, idx,
                                           False)
                    if v is None:
                        # zero; no data
                        return
                    terms.append(self._splat(writer, ftype, v))
                if len(terms) == 0:
                    # also zero
                    return
                term = terms[0]
                for i in range(1, len(terms)):
                    term = self._emit_binop(writer, ftype,
                                            self._productOperation,
                                            term, terms[i])
                # Accumulated inside the group, so the destination is read and
                # written once rather than once per reduction step.  Sound
                # because the sum operation is associative over the reduction
                # axis by construction -- it is the same operation the loop
                # itself is folding with.
                prod = term if prod is None else self._emit_binop(
                    writer, ftype, self._sumOperation, prod, term)
            if prod is None:
                return
            ns = [varlist[loopmap[f'n{i}']] for i, _ in enumerate(self._ns)]
            value = self._vdest.load(writer, self._context, None, ns, False)
            if value is None:
                assert False
            total = self._emit_binop(writer, ftype, self._sumOperation,
                                     value, prod)
            self._vdest.store(writer, self._context, total, ns, False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def _k_group(self, varlist, loopmap):
        """The reduction values this body covers, and which slot they fill.

        `(values, slot)`.  At `k_width == 1` that is the single value the loop
        handed over and the behaviour is unchanged.  Wider, the loop steps by
        `k_width` and this expands the base into the group -- clipped at the
        extent, so a ragged reduction simply gets a shorter last group.  No
        guard and no masking: unlike the lead dimension, a reduction has no
        lanes to leave half-valid, the leftover steps are just fewer terms in
        the same sum.
        """
        if not self._ks:
            return [None], None
        slot = f'k{len(self._ks) - 1}'
        base = varlist[loopmap[slot]]
        if self._k_width == 1 or not isinstance(base, Immediate):
            return [base], slot
        _, kmax = self._ks[-1]
        first = base._value
        return ([Immediate(first + c, base._type)
                 for c in range(min(self._k_width, kmax - first))], slot)

    def _k_packs(self, writer, varlist, loopmap, kslot, steps):
        """One wide load per operand contiguous along the reduction axis.

        Returns `{(operand, step): value}` for the operands that could be
        loaded once for the whole group.  An operand qualifies when the
        reduction is its *own* leading dimension -- then the `steps` values it
        contributes are adjacent in memory and one load fetches them all.
        `A`, whose leading dimension is `m`, does not qualify: its reduction
        values are `ldA` apart and no load reaches them together.

        This removes loads, not splats.  Each component still feeds its own
        product against its own `A`, and each still needs its own broadcast
        into the vector width.
        """
        packs = {}
        if steps < 2 or kslot is None:
            return packs
        for i, op in enumerate(self._ops):
            nks = self._opdim_to_nks[i]
            if not nks or nks[0] != kslot or len(nks) < 2:
                continue
            sym = op.symbol
            if not _contiguous_first_axis(sym):
                continue
            base = varlist[loopmap[kslot]]
            idx = [add_offset(VecIndex(base, steps) if j == 0
                              else varlist[loopmap[nk]],
                              self._eff_offset(i, j))
                   for j, nk in enumerate(nks)]
            v = sym.load(writer, self._context, None, idx, False)
            if v is None or getattr(v.type, 'length', None) != steps:
                continue
            for c in range(steps):
                packs[(i, c)] = writer.extract(v, c)
        return packs

    def _splat(self, writer, ftype, v):
        """A scalar operand broadcast into every component of the vector.

        `B` in `C[m,n] += A[m,k] B[k,n]` is not indexed by the lead dimension,
        so it loads one element while `A` and `C` load `lead_width` of them.
        Multiplying a vector by a scalar is not something the generated types
        do -- and on CUDA it would not compile at all -- so the scalar is
        packed into a vector of itself.

        This is where the packing overhead lives, and it is the reason the
        width is not free even where the registers are: one `{b, b}` per
        distinct `B` value.  It pays because the pack is loop-invariant in the
        lead dimension while the FMAs it feeds are not, so LICM hoists it out
        of exactly the loop that multiplies it.
        """
        if ftype.length is None:
            return v
        if getattr(getattr(v, 'type', None), 'length', None) is not None:
            return v
        return writer.pack(ftype, *([v] * ftype.length), hint='splat')

    def _emit_binop(self, writer, ftype, operator, a, b):
        """`operator` as an IR op if it has one, else its format string."""
        name = operator.irop()
        if name is not None:
            return writer.op(name, ftype, a, b, hint='p')
        return writer.rawexpr(operator.format('{0}', '{1}'), a, b,
                              type_=ftype, hint='p', pure=True, movable=True)

    def _second_operand_is_sparse(self):
        """Whether the B operand takes the sparse access path.

        The same expression decides three things -- which accessor `B` gets,
        whether `matmul` declines, and whether shared memory is reserved --
        so it is written once.
        """
        obj = self._ops[1].symbol.obj
        return bool(obj) and (not obj.is_dense()
                              or self._ops[1].symbol.data_view.shape[0] < 16)

    def _strategy(self) -> Strategy:
        """Which arrangement computes this operation.

        Derived on each call rather than stored.  Two callers ask -- the
        emission below, and `temp_shmem` before any body exists -- and the
        answer has to be the same for both: a reservation made for one
        arrangement and an emission of another is either a buffer nobody
        writes or an overrun.  Everything it reads is fixed by `_analyze`, so
        deriving it twice cannot disagree with itself the way two stored
        copies can.
        """
        module = _vendor_module(self._context)
        if module is None or not is_contraction(len(self._ops),
                                                self._lead_width):
            return Strategy.GENERIC
        shape = ComputeShape(threads=self._num_threads,
                             dtype=self._idest.datatype,
                             sparse=self._second_operand_is_sparse(),
                             explicit_simd=_explicit_simd(self._context))
        offered = module.strategies(shape, self._context)
        return choose_strategy(legal_strategies(offered),
                               self._context.get_vm().get_hw_descr().vendor)


    def _nonleading_dim_test(self, writer: Writer):
        # if len(self._ks) == 0 and len(self._ops) == 1:
        #     return False

        strategy = self._strategy()

        if strategy is not Strategy.GENERIC:
            K = 1
            N = 1
            M = 1

            for mi, mx in self._ks:
                K *= mx - mi

            for mi, mx in self._ns[1:]:
                N *= mx - mi

            M *= -(-(self._ns[0][1] - self._ns[0][0]) // self._num_threads)
            Mx = (self._ns[0][1] - self._ns[0][0])

            def unwindJ(j):
                idx = [None]
                for mi, mx in self._ns[1:]:
                    size = mx - mi
                    idx += [j % size + mi]
                    j //= size
                return idx

            def unwindI(i):
                size = -(-(self._ns[0][1] - self._ns[0][0]) // self._num_threads)
                idx = [LeadIndex(i % size + self._ns[0][0] // self._num_threads, self._num_threads, 1)]
                return idx

            if len(self._ks) == 0:
                # outer product
                kx = 0
            else:
                # TODO: remove
                kx = self._ks[0][0]

            def unwindK(k, full):
                ks00 = self._ks[0][0] if len(self._ks) > 0 else 0
                ks01 = self._ks[0][1] if len(self._ks) > 0 else 1

                size = ks01 - ks00
                if full:
                    idx = [k % size + ks00]
                else:
                    sizeL = -(-(size + kx) // self._num_threads)
                    idx = [LeadIndex(k % sizeL + ks00 // self._num_threads, self._num_threads, 1)]
                k //= size
                for mi, mx in self._ks[1:]:
                    size = mx - mi
                    idx += [k % size + mi]
                    k //= size
                return idx

            def unwindOp(i, j, k, opid, full):
                iidx = unwindI(i)
                jidx = unwindJ(j)
                kidx = unwindK(k, full)
                idx = []

                if opid is None:
                    nks = [f'n{i}' for i in range(len(self._ns))]
                else:
                    nks = self._opdim_to_nks[opid]
                for nk in nks:
                    if nk == 'n0':
                        idx += [iidx[0]]
                    elif nk[0] == 'k':
                        idx += [kidx[int(nk[1:])]]
                    else:
                        idx += [jidx[int(nk[1:])]]
                if opid is not None:
                    # same rule as the loop path: logical index in, storage
                    # index out.  add_offset folds the (usual) zero away.
                    idx = [add_offset(x, self._eff_offset(opid, d))
                           for d, x in enumerate(idx)]
                return idx

            def C(writer, var, i, j):
                self._vdest.store(writer, self._context, var, unwindOp(i, j, 0, None, False), False)

            if self._second_operand_is_sparse():
                def sparse(k, j):
                    if self._ops[1].symbol.obj and not self._ops[1].symbol.obj.is_dense():
                        return self._ops[1].symbol.obj.linear_index(unwindOp(0, j, k, 1, True)) is not None
                    return True
            else:
                sparse = None

            def B(writer, var, j, k):
                # `var is None` asks for the value itself rather than a name
                # the caller allocated.  A vendor intrinsic needs the former:
                # an operand without a definition point has no def-use edge
                # back to the read that produced it.
                if sparse:
                    res = self._ops[1].symbol.load_linear(writer, self._context, var, k)
                    return res if var is None else True
                with writer.speculative() as spec:
                    res = self._ops[1].symbol.load(writer, self._context, var, unwindOp(0, j, k, 1, False), False)
                    if not res:
                        spec.discard()
                return res

            def A(writer, var, i, k):
                with writer.speculative() as spec:
                    res = self._ops[0].symbol.load(writer, self._context, var, unwindOp(i, 0, k, 0, True), False)
                    if not res:
                        spec.discard()
                return res

            ops = MatmulOperands(
                A=A, B=B, C=C, sparse=sparse,
                lead_slots=M, lead_elements=Mx, n=N, k=K, kx=kx,
                threads=self._num_threads, dtype=self._idest.datatype)

            # A path may find out mid-emission that it cannot serve the shape,
            # and saying so has to leave the body as it found it -- otherwise
            # the generic nest below writes a second set of products on top of
            # a partial one, and both are emitted.
            with writer.speculative() as spec:
                taken = _vendor_module(self._context).matmul(
                    writer, ops, self._context, strategy)
                if not taken:
                    spec.discard()
            return taken
        return False

    def _apply_linear(self, writer: Writer):
        if len(self._scalar) == 0 and self._prev is None and self._next is None and self._idest.data_view == self._dest.data_view:
            # no linear needed
            return

        from tensorforge.backend.pir.core import ScalarType
        ftype = ScalarType(self._idest.get_fptype())

        if len(self._scalar) > 0:
            scalar_var = self._scalar[0].symbol.load(writer, self._context, None, [], False)
            assert scalar_var is not None

            for scalar in self._scalar[1:]:
                scalar_add = scalar.symbol.load(writer, self._context, None, [], False)
                scalar_var = self._emit_binop(writer, ftype, self._productOperation, scalar_add, scalar_var)
                assert scalar_var is not None

        loopstack = []
        loopmap = {}

        # TODO: not fully ideal; might need only a copy paritally (i.e. use the original dimmin/dimmax)
        stride = 1
        threads = self._num_threads
        for i, (dimmin, dimmax) in enumerate(self._ns):
            loopmap[f'n{i}'] = len(loopstack)
            dimmin = self._dest.data_view.get_bbox().lower()[i]
            dimmax = self._dest.data_view.get_bbox().upper()[i]

            dimmini = self._idest.data_view.get_bbox().lower()[i]
            dimmaxi = self._idest.data_view.get_bbox().upper()[i]

            unroll = dimmini != dimmin or dimmaxi != dimmax
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=unroll)]
            else:
                # Same width as `_apply_nonlead`.  This nest walks the *same*
                # register image -- it is the beta/prologue pass over the
                # destination -- so a cyclic walk here and a blocked one there
                # disagree about which lane owns which element.
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride,
                                       unroll=unroll, width=self._lead_width)]
                threads //= max(1, -(-(dimmax - dimmin) // self._lead_width))
                stride *= dimmax - dimmin

        def _dim_covered(i, var):
            """Is position `var` of this dim inside idest's coverage?

            Static shortcut first: if idest's bounds for this dimension already
            contain dest's *whole* iteration range, the answer is yes no matter
            where in that range the current lane sits --- true statically, no
            need to inspect `var` at all.

            The dynamic fallback (`.lead()`, a block-start value: `nonlead *
            block`, always a multiple of the block size) only agrees with true
            per-lane containment while idest's lower bound is itself a multiple
            of that block size.  A theta-shifted accumulator's bounds need not
            be: theta is chosen mod num_threads for lane alignment, but here
            `block` is this loop's own per-dimension stride factor, which can
            differ.  Comparing a block-start against raw, non-block-aligned
            bounds silently answered `False` for a lead dimension whose
            coverage was in fact exact, which is exactly the static case above
            already resolves --- so this fallback is only reached for the
            genuinely partial-overlap case it was written for.
            """
            lo_i = self._idest.data_view.get_bbox().lower()[i]
            hi_i = self._idest.data_view.get_bbox().upper()[i]
            lo_d = self._dest.data_view.get_bbox().lower()[i]
            hi_d = self._dest.data_view.get_bbox().upper()[i]
            if lo_i <= lo_d and hi_i >= hi_d:
                return True
            if not isinstance(var, (Immediate, LeadIndex)):
                return True
            if isinstance(var.nonlead(), (str,)):
                return True
            return lo_i <= int(var.lead()) and hi_i > int(var.lead())

        def nonlead_writer(varlist):
            needsLoad = all(_dim_covered(i, varlist[loopmap[f'n{i}']]) for i,_ in enumerate(self._ns))
            if needsLoad:
                valvar = self._vdest.load(writer, self._context, None, [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
            else:
                valvar = writer.const(
                    self._sumOperation.neutral(self._context.fp_type),
                    ftype)

            if len(self._scalar) > 0:
                valvar = self._emit_binop(writer, ftype, self._productOperation, valvar, scalar_var)
            if self._prev is not None:
                oldvalue = self._prev.load(writer, self._context, None, [add_offset(varlist[loopmap[f'n{i}']], self._prev_offset[i]) if self._prev_offset else varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
                valvar = self._emit_binop(writer, ftype, self._sumOperation, oldvalue, valvar)

            self._dest.store(writer, self._context, valvar, [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def _cublasdx_nonleadim_dim(self, writer: Writer):
        assert self._is_log
        with writer.Scope():
            # a _tiny_ bit hacky... But ok.
            writer('using namespace cublasdx;')

            m = 0
            n = 0
            k = 0

            num_threads = self._num_threads

            gemm_traits = []
            gemm_traits += [f'Size<{m}, {n}, {k}>']
            gemm_traits += ['Function<function::MM>']
            gemm_traits += ['Type<type::real>']

            transpose = lambda isTrue: 'transpose_mode::transposed' if isTrue else 'transpose_mode::non_transposed'
            gemm_traits += [f'TransposeMode<{transpose(False)}, {transpose(False)}>']
            gemm_traits += [f'Precision<{self._vm.fp_as_str()}>']

            # gemm_traits += [f'LeadingDimension<A,B,C>']

            sm = self._vm.get_hw_descr().model[3:]
            smprint = f'{sm}0'
            gemm_traits += [f'SM<{smprint}>']
            gemm_traits += ['Block']
            gemm_traits += [f'Block_Dim<{num_threads}>']
            traittype = '+'.join(f'{trait}()' for trait in gemm_traits)
            writer(f'using GemmType = decltype({traittype});')

            # currently, the alpha, beta are handled when storing back to global memory
            writer(f'GemmType().execute(1, {self._op1.name}, {self._op2.name}, 1, {self._dest.name});')

    def _leading_dim(self, writer: Writer):
        with writer.Scope():
            loopstack = []
            for i, (dimmin, dimmax) in enumerate(self._ns[1:]):
                loop = writer.For(f'int32_t n{i+1} = {dimmin}; n{i+1} < {dimmax}; ++n{i+1}', True)
                loop.__enter__()
                loopstack += [loop]

            self._idest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
            #writer(f'auto* shmAddr = &{self._shr_mem.name}[{self._shr_mem_offset}];')
            self._reduction(writer)
            write(f'value = tensorforge::reduction<tensorforge::ReductionOperation<{self._fp_as_str}, tensorforge::Op::Sum>, {self._num_threads}, 1, {self._fp_as_str}>(value);')
            # self._butterfly_reduction_loop(writer, max_array_length = 32, amd = False)
            #writer(f'{self._fp_as_str} newvalue = shmAddr[{sublane_address}];')
            self._idest.store(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)

            for loop in loopstack[::-1]:
                loop.__exit__(None, None, None)

    def get_operands(self):
        inops = [op.symbol for op in self._ops] + [op.symbol for op in self._scalar]
        if self._prev is None:
            return inops
        else:
            return inops + [self._prev]

    def __str__(self):
        return f'{self._dest.name} = {self._sumOperation}({f" {self._productOperation} ".join(op.symbol.name for op in self._ops)}) {self._sumOperation} {self._prev}' # TODO: dimensions

    def temp_shmem(self):
        """What the path this operation will take needs staged.

        Asked before any body is built, so it has to reach the same conclusion
        as the dispatch does later from the same two questions: whether a
        matrix path is taken at all, and which module owns it.  Naming the
        vendor here a second time is what lets the two answers drift, and a
        reservation that disagrees with the emission is either a buffer nobody
        writes or an overrun.
        """
        strategy = self._strategy()
        if strategy is Strategy.GENERIC:
            return 0
        return _vendor_module(self._context).scratch(strategy,
                                                     self._idest.datatype)
