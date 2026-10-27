# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import math
from . import ComputeInstruction
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.backend.symbol import slots_for, VecIndex, SymbolType, add_offset, Symbol, SymbolView, DataView, Loop, LeadLoop, write_loops, LeadIndex, Immediate, passed_by_value
from tensorforge.common.exceptions import InternalError, GenerationError
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.operation import ReductionOperator
from typing import Union, List, Tuple
from tensorforge.common.basic_types import Addressing, Datatype, Residence
from tensorforge.backend.pir.core import INDEX, MemSpace
from tensorforge.backend.instructions.abstract_instruction import _explicit_simd

from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.matrix import packing as packing_mod
from tensorforge.common.matrix.packing import KINDS

from .matmul import MatmulOperands
from .paths import matrix_paths
from .strategy import (ComputeShape, Span, Strategy, choose_strategy, covers,
                       is_contraction, lead_layout, legal_strategies, whole)


def _vendor_module(context):
    """The target's matrix module (`paths.matrix_paths`), or None."""
    return matrix_paths(context.target)

import dataclasses
import hashlib
import itertools

import numpy as np

from copy import copy


@dataclasses.dataclass(frozen=True)
class _EllSource:
    """Where a gathered step reads the second operand (`_ell_source`):
    `memory`, or a `register` image moved across `span` lanes, its element
    `k` in slot `slots[k]` of lane `lanes[k]`."""
    kind: str
    span: int = 0
    slots: dict = None
    lanes: dict = None


@dataclasses.dataclass(frozen=True)
class _EllPlan:
    """A product gathered in sliced ELL (`_ell_plan`).

    `steps` maps a slot of lanes to its steps, each a column, a weight and
    the register image's slot it reads (None in memory): `('k', k)` a column
    every lane reads, `('table', at)` one per lane, at `at` in the table
    `cols` -- a reduction index in memory, a source lane in a register image
    -- and `('one', None)`, `('k', w)` or `('table', at)` in `weights`
    alike.  `tables` is what `module_tables` hands out."""
    source: _EllSource
    steps: dict
    cols: str = None
    weights: str = None
    tables: dict = None


def _module_table_name(prefix: str, datatype, values) -> str:
    """A name that says which numbers it holds.

    yateto appends every kernel of a file to one translation unit, so a
    name at namespace scope is shared by all of them: two kernels with the
    same numbers then have one definition between them
    (`Generator.module_table_definitions`), and two with different ones do
    not collide."""
    sha = hashlib.new('md5', usedforsecurity=False)
    sha.update(repr((str(datatype), tuple(values))).encode())
    return f'tf_{prefix}_{sha.hexdigest()[:16]}'


def _contiguous_first_axis(sym) -> bool:
    """Whether axis 0 of this symbol is the one adjacent in memory.

    The condition a wide load along that axis rests on, and it is a property
    of the *view* rather than of the tensor: a transposed operand has the same
    tensor and a stride that makes the values `ld` apart.

    Adjacency is asked in elements.  The stride a contiguous axis carries is
    the size of one element, which is `elem_parts` scalars and only sometimes
    one of them; comparing against the literal would call a decomposed
    operand strided and cost it the wide load it is in fact entitled to.
    """
    try:
        view = sym.data_view
        return view.get_dim_strides()[0] == view.elem_parts
    except Exception:
        return False


def _roll_count(dimmin, dimmax, step, k_roll, k_unroll_max):
    """The unroll count of a rolled reduction, or 0 to unroll it whole.

    `k_roll` where it is asked for.  Otherwise a reduction longer than
    `k_unroll_max` steps (`Options.k_unroll_max`) is rolled by the largest
    divisor of its step count that is no larger, so that the rolled loop runs
    whole groups -- and by the cap itself where the count is prime.  Whether
    it can be rolled at all is `_rollable`'s question, not this one's.
    """
    if k_roll:
        return k_roll
    steps = (dimmax - dimmin) // step if step > 0 else 0
    if not k_unroll_max or steps <= k_unroll_max:
        return 0
    best = max(d for d in range(1, k_unroll_max + 1) if steps % d == 0)
    return best if best > 1 else k_unroll_max

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
               prev_offset=None,
               whole_prev: bool = False):
        super(MultilinearInstruction, self).__init__(context)
        #: `prev` is a register image of the destination's whole box -- the
        #: one the previous term of an accumulation chain left behind.  The
        #: result is then written over all of it, the product where this term
        #: reaches and `prev` elsewhere, so the image stays whole.
        self._whole_prev = whole_prev
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
        # origin of the lead loop, the builder's: where a pending result it
        # reads in place has its rows, or else the destination's offset
        # modulo a round of the lanes; the operands are staged to match
        # (MultilinearBuilder._lead_origin_shift).  Only relative offsets
        # matter, so shifting it is free apart from at most one extra slot.
        self._theta = theta
        # origin of the first reduction loop, chosen in `_analyze` on the same
        # grounds (see _reduction_origin_shift)
        self._kappa = 0
        self._prev = prev
        self._prev_offset = prev_offset
        self._next = next
        self._dest_obj = dest_obj
        # Shared memory one multiplication owns, set by the generator for
        # each build of the body, and whether the build asked for it; see
        # `set_mult_stride`.
        self._mult_stride = None
        self._mult_stride_asked = False

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
        self._has_epilogue = self.needs_epilogue()

    def _eff_offset(self, i, j):
        """Operand offset as seen from the shifted lead and reduction origins."""
        return (self._ops[i].offset[j]
                - (self._theta if self._target[i][j] == 0 else 0)
                - (self._kappa if self._target[i][j] == -1 else 0))

    def _reduction_origin_shift(self):
        """Pick the origin of the first reduction loop, as `theta` does for `n0`.

        The reduction's first index is distributed across lanes wherever it is
        read a block at a time (`unwindK` hands it out as a `LeadIndex`), so a
        register-resident operand that carries it pins the origin modulo T
        exactly like one carrying the lead index does: its element `s` sits in
        lane `s % T` whatever the loop would prefer.  An intermediate written
        from a slice starting at row 35 is laid out from lane 3 on; read back
        as the reduced operand of the next product by a loop starting at 0, it
        would be three lanes off -- a remainder no address can apply.
        Starting the loop where the data already is costs nothing: every
        operand's effective offset along the reduction drops by the same
        amount, and the ones in memory take any.

        Register operands that disagree modulo T would need a shuffle; the
        origin then stays unshifted, and they are handled as they would be
        without the shift.  An explicit-SIMD lowering has the lane in the type
        and selects any remainder (`Symbol.build_address`), so it keeps its
        origin as well: a shifted window would only split its vector loads in
        two.
        """
        threads = self._num_threads
        if not threads or _explicit_simd(self._context):
            return 0
        pins = set()
        for i, op in enumerate(self._ops):
            if op.symbol is None or op.symbol.stype not in (SymbolType.Register,
                                                            SymbolType.Scratch):
                continue
            for j, target in enumerate(self._target[i]):
                if target == -1 and j < len(op.offset):
                    pins.add(op.offset[j] % threads)
        return next(iter(pins)) if len(pins) == 1 else 0

    def _lane_readable_by_reduction(self, i):
        """Whether operand `i` can be read across the lanes by `(k, n)`.

        That is how every vendor path reads the second operand: `unwindK` hands
        out the first reduction index as a `LeadIndex`, so lane `s` reads the
        element at `k = s`.  An operand in memory takes any such index.  One in
        registers holds element `s` of its distributed dimension in lane `s %
        T` and no other, so it can be read that way only where that is the
        reduction index, from a whole block on.  Spread by another of its
        dimensions -- a matrix held a row per lane and read here by column --
        the element a lane needs is in the registers of another lane, at an
        index of its own, which no shuffle reads; and a shift by part of a
        block moves the data between the lanes.  The nest takes both: it reads
        such an operand an element at a time and broadcasts it.
        """
        op = self._ops[i]
        if op.symbol is None or op.symbol.stype not in (SymbolType.Register,
                                                        SymbolType.Scratch):
            return True
        reduced = [d for d, nk in enumerate(self._opdim_to_nks[i]) if nk == 'k0']
        spread = list(getattr(op.symbol, 'lead_dims', None) or [])
        if spread and spread != reduced:
            return False
        span = self._num_threads * self._lead_width
        return not span or all(self._eff_offset(i, d) % span == 0 for d in reduced)

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
        # targeting `[0]`, and dimension 1 would vanish, so the loop nest would
        # run over `n0` only and write one slot per lead block instead of
        # three.  An operand that lacks an index is read at the same address
        # for every value of it, which is the broadcast; the index still has
        # to exist.
        targetrank = self._dest_obj.bbox.rank()
        for i, op in enumerate(self._ops):
            for j in range(op.bbox.rank()):
                targetrank = max(self._target[i][j] + 1, targetrank)
        self._ns = [(-math.inf, math.inf)] * targetrank
        preKs = {}
        self._opdim_to_nks = []
        sparseK = {}
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

            self._opdim_to_nks += [opdim]

        for i in range(len(self._ns)):
            # intersection with the destination's range -- a range against a
            # range; clamping a size against a range would agree with it only
            # while dest.bbox.lower() == 0.
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

        # and the first reduction loop into its own, likewise (_eff_offset)
        self._kappa = self._reduction_origin_shift() if self._ks else 0
        if self._kappa:
            self._ks[0] = (self._ks[0][0] + self._kappa,
                           self._ks[0][1] + self._kappa)

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
            # The slot-count formula is called, not restated, like the
            # addressing side and the other allocation site: three statements
            # of one rule would let the width reach one of them and not the
            # others.  `DataView.lead_lanes` stays a separate factor -- how
            # many entries one slot takes is a question about the lowering,
            # not about the distribution.
            self._iregs = slots_for(self._ns[0][0], self._ns[0][1],
                                    self._num_threads, self._lead_width)
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
            # off.  Two things disqualify a neighbor.  A global or shared
            # symbol describes the whole buffer, not the box this operation
            # writes.  And a register image is only usable if it is an image of
            # the *same* box: the residency is keyed by symbol name and lives
            # for the whole kernel, so what a later operation finds there may
            # have been staged for a read of a much wider region.  In the
            # poroelastic space-time predictor a one-row-one-column write would
            # pick up the image of the whole 32x13x4 tensor and inherit its
            # box; the accumulator would then claim elements it never computes,
            # and the store would write all of them --- reading past the end of
            # the register array on the way.
            #
            # A whole image of the destination (`whole_prev`) is adopted even
            # where this term's own box is narrower: the result *is* the
            # image, updated where the term reaches, so it takes the image's
            # box, and the epilogue copies `prev` into the rest.
            def inside(a, b):
                return (a.rank() == b.rank() and all(
                    lb <= la and ua <= ub for la, ua, lb, ub in
                    zip(a.lower(), a.upper(), b.lower(), b.upper())))

            for neighbor in (self._prev, self._next):
                if (neighbor is not None
                        and neighbor.stype in (SymbolType.Register,
                                                SymbolType.Scratch)
                        and neighbor.data_view is not None
                        and (self._same_box(neighbor.data_view.get_bbox(),
                                            self._idest.data_view.get_bbox())
                             or (neighbor is self._prev and self._whole_prev
                                 and inside(self._idest.data_view.get_bbox(),
                                            neighbor.data_view.get_bbox())))):
                    self._dest.data_view = neighbor.data_view
            if self._dest.data_view is None:
                self._dest.data_view = self._idest.data_view

        # From the destination symbol, not assumed to be axis 0.  These are
        # destination axis indices: an N axis in `_lead_dims` becomes a
        # `LeadLoop`, everything else a sequential `Loop`.  A K axis would be
        # written `-i-1`, which is the encoding the test at the head of the K
        # nest reads -- nothing produces one, since a contraction axis spread
        # across the lanes needs a cross-lane fold, which this instruction does
        # not do.
        #
        # The value is [0] for every case in the corpus, because
        # `MultilinearDescr._lead_dim` aligns the thread count to the
        # destination's axis 0 and nothing sets the destination's `lead_dims`
        # to anything else.  Taking it from the symbol is what makes the two
        # statements one: `multilinear_builder` already sets `lead_dims` on a
        # staged register image, and `Symbol.load` addresses through it.
        self._lead_dims = [self.lead_dim(self._dest)]

    def _known_zero(self, i, idx) -> bool:
        """Is operand `i` zero at every cell one step of the loop reads?

        Asked of a batch-constant operand the description gives numbers for
        (`Tensor.has_values`) -- SeisSol's global matrices, whose values yateto
        passes along -- and answerable only where every index is a number: the
        reduction's, and the lane slot's, which the contraction unrolls.  A
        slot covers `block` lanes of `width` elements each, and the product is
        left out only if all of them read a zero; one row with a value keeps
        the step, since the lanes issue it together.

        Cells outside the rows the loop iterates are not read -- a ragged slot
        guards them off -- so they neither keep a step nor allow one to go.
        `skip_known_zeros` turns the question off.
        """
        if not self._context.get_user_options().skip_known_zeros:
            return False
        symbol = self._ops[i].symbol
        obj = getattr(symbol, 'obj', None)
        # The tensor itself, or a verbatim copy of it -- the section's
        # preloaded image, which `GlobalLoaderBuilder` lays out exactly as the
        # tensor: an index then is a cell of the tensor, which is what its
        # values are indexed by.  A copy staged for one operation is addressed
        # in its own window, shifted, and is left alone rather than
        # translated back.
        if (not self._indexed_as_tensor(symbol) or not isinstance(obj, Tensor)
                or obj.addressing is not Addressing.NONE or not obj.has_values()):
            return False
        data = obj.get_values()
        axes = []
        for j, x in enumerate(idx):
            if isinstance(x, Immediate):
                x = x.nonlead()
            if isinstance(x, (int, np.integer)):
                cells = [int(x)]
            elif (isinstance(x, LeadIndex)
                  and isinstance(x.nonlead(), (int, np.integer))):
                width, block = x.width, x._block
                base = width * block * int(x.nonlead()) + x.offset()
                cells = [base + width * lane + c for lane in range(block)
                         for c in range(width)]
                target = self._target[i][j]
                if target >= 0:
                    # the rows this contraction iterates, in the operand's
                    # coordinates: the loop's range moved by the offset
                    lo, hi = self._ns[target]
                    shift = self._eff_offset(i, j)
                    cells = [r for r in cells if lo + shift <= r < hi + shift]
            else:
                return False
            if any(not 0 <= r < data.shape[j] for r in cells):
                return False
            axes.append(cells)
        return all(data[cell] == 0 for cell in itertools.product(*axes))

    def _identity_reduction(self):
        """The diagonal of the first operand, where it is the whole of it.

        A batch-constant first operand the description gives numbers for,
        whose every row this contraction iterates has its nonzeros at the
        reduction step equal to the row -- the identity pattern, weighted or
        not.  The sum over the reduction is then the one product
        `A[i,i] B[i,j]`, and each lane reads only its own row of each
        operand: nothing crosses a lane.  Computed the usual way it is the
        dense product, because a step of the reduction is issued by every
        lane at once and only one of them holds a nonzero there -- SeisSol's
        face orientation `fP(1)` is exactly this, at every order, and as a
        dense product it was 8 % (o8) to 19 % (o4) of the flops
        `neighboringFlux[16]` executes on sm_120.

        Returns the diagonal's values in the contraction's rows, or None.
        The lead's range inside the reduction's, so every lane's diagonal
        step exists and no lane reads a row the reduction does not cover;
        the rest is `_lane_side_values`'s.  `identity_reduction` turns it
        off.
        """
        if not self._context.get_user_options().identity_reduction:
            return None
        value = self._lane_side_values()
        if value is None:
            return None
        (lo, hi), (klo, khi) = self._ns[0], self._ks[0]
        if hi <= lo or not (klo <= lo and hi <= khi):
            return None
        diagonal = []
        for r in range(lo, hi):
            for k in range(klo, khi):
                v = value(r, k)
                if k == r:
                    if v is None:
                        return None
                    diagonal.append(v)
                elif v:
                    return None
        return diagonal

    def _lane_side_values(self):
        """The first operand's numbers by row and reduction step, where the
        lanes run along its rows and the description gives them.

        What `_identity_reduction` and `_ell_plan` start from: one reduction,
        the output's lead on the first operand's other dimension and not on
        the second operand, a batch-constant first operand indexed as its
        tensor, one element per lane -- and a sum of products, since a zero
        absorbs `mul` and is neutral to `add` and to no other pair this
        instruction may carry.

        Returns `value(r, k)` in the contraction's coordinates -- the number,
        or None for a cell outside the operand's box -- or None.
        `skip_known_zeros` turns it off.
        """
        if not self._context.get_user_options().skip_known_zeros:
            return None
        if (self._productOperation.irop() != 'mul'
                or self._sumOperation.irop() != 'add'):
            return None
        if (len(self._ops) != 2 or len(self._ks) != 1
                or 0 not in self._lead_dims
                or sorted(self._opdim_to_nks[0]) != ['k0', 'n0']
                or 'k0' not in self._opdim_to_nks[1]
                or 'n0' in self._opdim_to_nks[1]):
            return None
        symbol = self._ops[0].symbol
        obj = getattr(symbol, 'obj', None)
        if (not self._indexed_as_tensor(symbol) or not isinstance(obj, Tensor)
                or obj.addressing is not Addressing.NONE
                or not obj.has_values()):
            return None
        if self._lead_width != 1:
            return None
        data = obj.get_values()
        if data.ndim != 2:
            return None
        nks = self._opdim_to_nks[0]
        lead, red = nks.index('n0'), nks.index('k0')
        o_lead, o_red = self._eff_offset(0, lead), self._eff_offset(0, red)

        def value(r, k):
            cell = [0, 0]
            cell[lead], cell[red] = r + o_lead, k + o_red
            if not all(0 <= cell[d] < data.shape[d] for d in (0, 1)):
                return None
            return float(data[tuple(cell)])
        return value

    def _ell_plan(self):
        """The first operand's nonzeros lane by lane, where gathering them is
        fewer steps than the reduction: sliced ELL.

        The lanes run along the rows of `A`, and a step of the usual nest is
        issued by every lane of a slot at once -- so a slot costs one step
        per reduction index any of its rows has a nonzero at, however few
        each row has (`_known_zero` drops only the steps no row needs).  A
        lane can instead take its own row's nonzeros in turn: step `t` reads
        `B` at the column of the lane's `t`-th nonzero and multiplies by it.
        A slot then costs as many steps as its fullest row has nonzeros, the
        rows with fewer padded with zero weights -- ELLPACK sliced by the
        slot's lanes, SELL-C with C the lanes.

        `B` is read at a row of each lane's own, so the read is a gather:
        at a per-lane address where `B` is in memory, across the lanes where
        it is a register image (`_ell_source`).  An image over several slots
        is gathered a slot at a time -- a lane's nonzeros in each slot of
        `B` are steps of their own, since one gather moves one register --
        which keeps a lane's sum in the order of the reduction.

        Columns and weights the lanes of a step share are immediates -- a
        permutation without weights needs no multiply, a column every lane
        reads is the usual broadcast -- and the others come from a table in
        the generated file, each lane's entry at its own position
        (`module_tables`).  The weights too, though the operand may reside
        in memory: they are the description's, as a broadcast operand's are
        under `inline_constants`, and the kernel is still handed the
        operand.

        Taken where it halves the steps at least, and no row has more than
        `ell_gather` nonzeros.  A step of the gather reads two numbers more
        than the nest's, and an address of its own: on sm_120 `volume`'s
        `kDivM(0)` at order 6, 35 steps for 55, was 9 % slower gathered, where
        products with a third of the steps or fewer were not.  The nest's
        steps are all of the reduction where it rolls -- its step is a value
        then, and no zero is known of it -- which is where the gather is
        worth most: `rT` at order 8, 120 steps a slot rolled against 20 to 36
        gathered, and `neighboringFlux` twice as fast.  Not under an
        explicitly vectorized lowering,
        whose lanes are one vector that a table is not indexed by.  The
        identity takes precedence (`_identity_reduction`): it needs neither
        a table nor a read across the lanes.

        Returns an `_EllPlan`, or None.
        """
        most = self._context.get_user_options().ell_gather
        block = self._num_threads
        if not most or not block or _explicit_simd(self._context):
            return None
        value = self._lane_side_values()
        if value is None:
            return None
        (lo, hi), (klo, khi) = self._ns[0], self._ks[0]
        if hi <= lo or khi <= klo:
            return None
        source = self._ell_source(klo, khi)
        if source is None:
            return None
        rows = {}
        for r in range(lo, hi):
            rows[r] = [(k, v) for k in range(klo, khi)
                       if (v := value(r, k))]
        # The register image's slot each nonzero is read from; memory has
        # one group for everything.
        group = ((lambda k: None) if source.kind == 'memory'
                 else (lambda k: source.slots[k]))
        groups = sorted({group(k) for k in range(klo, khi)},
                        key=lambda g: -1 if g is None else g)
        # The steps the nest would issue instead: a rolled reduction all of
        # them, since its step is a value and no zero is known of it.
        options = self._context.get_user_options()
        step = self._k_width if self._k_width > 1 else 1
        rolled = self._rollable(
            len(self._ks) - 1, klo, khi, step,
            _roll_count(klo, khi, step, options.k_roll, options.k_unroll_max))
        slots, dense, sparse = {}, 0, 0
        for s in range(lo // block, (hi - 1) // block + 1):
            live = [r for r in range(s * block, (s + 1) * block)
                    if lo <= r < hi]
            slots[s] = live
            dense += (-(-(khi - klo) // step) if rolled
                      else len({k for r in live for k, _ in rows[r]}))
            sparse += sum(max(sum(1 for k, _ in rows[r] if group(k) == g)
                              for r in live) for g in groups)
        widest = max(len(entries) for entries in rows.values())
        if not widest or widest > most or 2 * sparse > dense:
            return None

        cols, weights, steps = [], [], {}
        for s, live in slots.items():
            lanes = range(s * block, (s + 1) * block)
            plan = []
            for g in groups:
                mine = {r: [e for e in rows[r] if group(e[0]) == g]
                        for r in live}
                for t in range(max(len(e) for e in mine.values())):
                    held = {r: mine[r][t] for r in live if t < len(mine[r])}
                    ks = sorted({k for k, _ in held.values()})
                    if len(ks) == 1:
                        col = ('k', ks[0])
                    else:
                        # A row without a `t`-th nonzero, or a lane past the
                        # rows, reads any column the step reads -- its
                        # weight is zero, or nothing stores what it computes.
                        col = ('table', len(cols))
                        for r in lanes:
                            k = held[r][0] if r in held else ks[0]
                            cols.append(k if source.kind == 'memory'
                                        else source.lanes[k])
                    ws = {w for _, w in held.values()}
                    if len(held) == len(live) and len(ws) == 1:
                        w = ws.pop()
                        weight = ('one', None) if w == 1.0 else ('k', w)
                    else:
                        weight = ('table', len(weights))
                        weights.extend(held[r][1] if r in held else 0.0
                                       for r in lanes)
                    plan.append((col, weight, g))
            steps[s] = plan
        tables, names = {}, {}
        fptype = self._ops[0].symbol.get_fptype()
        for key, datatype, values in (('cols', Datatype.I32, cols),
                                      ('weights', fptype, weights)):
            if values:
                name = _module_table_name(f'ell{key[0]}', datatype, values)
                tables[name] = (datatype, tuple(values))
                names[key] = name
        return _EllPlan(source, steps, names.get('cols'),
                        names.get('weights'), tables)

    def _ell_source(self, klo, khi):
        """Where a gathered step reads the second operand, or None.

        Memory, where `B` is dense and in its logical order there -- what a
        shifted read asks of an operand (`Symbol.load`), and for the same
        reason: the address is a value, and a packing or a prepared order
        would want it translated.  A register image, where its distributed
        dimension is the reduction: each lane reads its own register of a
        slot, and the target moves it to the lane that wants it
        (`Lexic.gather`).  Which slot is the step's (`_ell_plan`), since a
        gather moves one register and does not choose among them.
        """
        symbol = self._ops[1].symbol
        obj = symbol.obj
        kpos = self._opdim_to_nks[1].index('k0')
        if symbol.stype in (SymbolType.SharedMem, SymbolType.Global,
                            SymbolType.Batch):
            view = symbol.data_view
            if (not obj.is_dense() or view is None
                    or view.elem_parts > 1 or view.part_plane
                    or getattr(obj, 'storage_order', None) is not None
                    or getattr(obj, 'simt_interleave', None) is not None
                    or getattr(obj, 'slot_major', None) is not None
                    or getattr(obj, 'packing', None) is not None):
                return None
            return _EllSource('memory')
        if symbol.stype is not SymbolType.Register:
            return None
        if list(symbol.lead_dims) != [kpos] or symbol.lead_width != 1:
            return None
        layout = symbol.register_layout()
        span = ((layout.block_span() if layout is not None else None)
                or self._num_threads)
        if self._context.target.lexic.gather('{0}', '{1}', span) is None:
            return None
        block = symbol.lead_block(kpos)
        shift = self._eff_offset(1, kpos)
        slots, lanes = {}, {}
        for k in range(klo, khi):
            cell = [0] * len(self._opdim_to_nks[1])
            cell[kpos] = k + shift
            lanes[k] = symbol.reading_lane(cell)
            if lanes[k] is None:
                return None
            # The slot `load` reads a fixed element from (its broadcast).
            slots[k] = (k + shift) // block
        return _EllSource('register', span, slots, lanes)

    def _ell_term(self, writer, plan, lead, step, varlist, loopmap):
        """One step of a gathered product: `B` at each lane's column, times
        the lane's weight -- or as it is, where every lane's weight is one."""
        from tensorforge.backend.pir.core import ScalarType
        (ckind, col), (wkind, weight), slot = step
        fp = ScalarType(self._ops[0].symbol.get_fptype())
        lane = []

        def entry(table, at, type_, hint):
            # A step's run of the table holds one entry per lane, in the
            # lanes' order: the index is the lane and a constant.
            if not lane:
                lane.append(writer.lane_offset(lead._block, lead._stride,
                                               hint='lane'))
            index = (writer.op('add', INDEX, lane[0], at, hint='ell')
                     if at else lane[0])
            return writer.rawexpr(f'{table}[{{0}}]', index, type_=type_,
                                  pure=True, movable=True, hint=hint)

        symbol = self._ops[1].symbol
        nks = self._opdim_to_nks[1]
        kpos = nks.index('k0')
        shift = self._eff_offset(1, kpos)
        index = [None if nk == 'k0'
                 else add_offset(varlist[loopmap[nk]], self._eff_offset(1, j))
                 for j, nk in enumerate(nks)]
        if ckind == 'k':
            index[kpos] = add_offset(col, shift)
            b = symbol.load(writer, self._context, None, index, False)
        elif plan.source.kind == 'memory':
            # The row as this axis's address, a value per lane.  Carried by
            # a lead index (`vaddr`) rather than an integer index so that
            # the value keeps the lanes' layout: it is one number per lane,
            # not one number for all of them.
            index[kpos] = LeadIndex(0, lead._block, lead._stride,
                                    offset=shift,
                                    vaddr=entry(plan.cols, col, INDEX, 'col'))
            b = symbol.load(writer, self._context, None, index, False)
        else:
            index[kpos] = LeadIndex(slot, symbol.lead_block(kpos), 1)
            own = symbol.load(writer, self._context, None, index, False)
            if own is None:
                return None
            text = self._context.target.lexic.gather('{0}', '{1}',
                                                     plan.source.span)
            b = writer.rawexpr(text, own, entry(plan.cols, col, INDEX, 'src'),
                               type_=own.type, hint='g', pure=True,
                               movable=True, crosslane=True)
        if b is None or wkind == 'one':
            return b
        w = (writer.const(weight, fp) if wkind == 'k'
             else entry(plan.weights, weight, fp, 'w'))
        return writer.op('mul', fp, w, b, hint='p')

    def module_tables(self):
        """Numbers this instruction reads by name from the generated file,
        `{name: (datatype, values)}` -- the generator defines them
        (`Generator.module_table_definitions`)."""
        return dict(getattr(self, '_module_tables', {}))

    def _diagonal_term(self, writer, row, b, diagonal):
        """`A[i,i] B[i,j]` for the lane holding row `i`, from the numbers.

        The weight is not read from the operand: the description gives it,
        and a packed operand has no address for a cell whose column is a
        lane's.  A literal per row, the lane's row selecting it -- the
        pattern `Symbol._encode_lead_values` writes for a number in the code,
        and like that one it does not depend on the column, so the columns of
        a slot share it.

        The four cases the identity pattern has: all ones, the second
        operand's row as it is; ones and zeros, a select of it; weights with
        zeros among them, the product selected, so that a zero is no product
        with a `NaN` -- the contract `skip_known_zeros` keeps everywhere; and
        weights only, the product.
        """
        from tensorforge.backend.pir.core import BOOL, ScalarType
        fp = ScalarType(self._ops[0].symbol.get_fptype())
        lo = self._ns[0][0]
        values = set(diagonal)
        if values == {1.0}:
            return b
        zero = writer.const(0.0, fp)
        if len(values) == 1:
            weight = writer.const(diagonal[0], fp)
        else:
            weight = None
            for at, value in enumerate(diagonal):
                if value == 0:
                    continue
                cond = writer.op('eq', BOOL, row, lo + at, hint='diag')
                weight = writer.op('select', fp, cond, writer.const(value, fp),
                                   weight if weight is not None else zero,
                                   hint='weight')
            if weight is None:
                return zero
        if 0.0 not in values:
            return writer.op('mul', fp, weight, b, hint='p')
        live = writer.op('ne', BOOL, weight, zero, hint='live')
        if values <= {0.0, 1.0}:
            return writer.op('select', fp, live, b, zero, hint='p')
        return writer.op('select', fp, live,
                         writer.op('mul', fp, weight, b, hint='p'), zero,
                         hint='p')

    @staticmethod
    def _indexed_as_tensor(symbol) -> bool:
        """Is an index into `symbol` a cell of its tensor?

        The tensor in global memory, or a copy made verbatim -- the section's
        preloaded image of a batch-constant operand (`verbatim`, set where
        the copy is made).  A per-operation staging window is not: its origin
        is wherever the window starts.

        And an operand whose numbers are in the code (`SymbolType.Data`, as
        `inline_constants` makes one): it has no memory at all, and a read of
        it is `obj.value(index)`, the number at the cell the index names.
        Asked by type rather than by the `inlined` flag, which the generator
        sets on the symbol it decides about and an instruction holding a copy
        does not see.  Left out, every zero of such an operand was
        multiplied in as `x * 0.0f`, which a compiler may not fold --
        `NaN * 0` is `NaN` -- so `C = X S` with a permutation `S` cost half
        of what a dense `S` does.
        """
        return (symbol.stype in (SymbolType.Global, SymbolType.Data)
                or bool(getattr(symbol, 'verbatim', False)))

    def _zero_block(self, e_lo, e_hi, k_lo, k_hi, span) -> bool:
        """`_known_zero` for a matrix path: is `A` zero over a block?

        `e` counts lead elements from the first slot the path reads, `k`
        contraction steps from the first it walks -- the coordinates of
        `MatmulOperands.A`, whose `unwindI`/`unwindK` this mirrors.  Rows past
        the ones this contraction keeps and steps past its depth are padding
        the path fills with zeros, so they neither keep a block nor allow one
        to go; a block with nothing else left is all padding and goes.
        """
        if not self._context.get_user_options().skip_known_zeros:
            return False
        symbol = self._ops[0].symbol
        obj = getattr(symbol, 'obj', None)
        if (not self._indexed_as_tensor(symbol) or not isinstance(obj, Tensor)
                or obj.addressing is not Addressing.NONE or not obj.has_values()
                or len(self._ks) != 1
                or sorted(self._opdim_to_nks[0]) != ['k0', 'n0']):
            return False
        nks = self._opdim_to_nks[0]
        lead, red = nks.index('n0'), nks.index('k0')
        lo, hi = self._ns[0]
        base = (lo // span) * span
        rows = [base + e + self._eff_offset(0, lead)
                for e in range(e_lo, e_hi) if lo <= base + e < hi]
        kfirst, klast = self._ks[0]
        ks = [kfirst + k + self._eff_offset(0, red)
              for k in range(k_lo, k_hi) if k < klast - kfirst]
        if not rows or not ks:
            return True
        data = obj.get_values()
        if (min(rows) < 0 or max(rows) >= data.shape[lead]
                or min(ks) < 0 or max(ks) >= data.shape[red]):
            return False
        block = (data[np.ix_(rows, ks)] if lead == 0
                 else data[np.ix_(ks, rows)])
        return not np.any(block)

    def gen_code_inner(self, writer: Writer):
        # A comment touches nothing.  Left conservative it would be read as
        # touching every buffer, and this one sits above them all.
        writer.Comment(f'{self._ns} {self._ks}')

        if self.needs_epilogue() != self._has_epilogue:
            # decided when the builder appended the epilogue, or not; a view
            # that changed since would leave the two disagreeing
            raise InternalError(
                f'{self}: whether it needs an epilogue changed after the '
                f'builder decided it')
        if not self._has_epilogue:
            self._vdest = self._dest
        elif hasattr(writer, 'alloc') and callable(getattr(writer, 'alloc')):
            # The same structured alloc as `RegisterAlloc`'s, from a different
            # site.  It matters beyond its own node count: `flatten_scopes`
            # keeps the `{ }` around any region whose raw text declares a C++
            # name, so a raw declaration here would hold a wall around every
            # multilinear.
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

    def _nonleading_dim(self, writer: Writer):
        self._offer_simt_order()
        loopstack = []

        # TODO: preload values where necessary (i.e. no N in there)
        # Also, postpone multiplications until necessary

        # Every loop is unrolled, which is what lets an operand that lacks an
        # index be read once and broadcast.  A count here rolls the reduction
        # instead -- see `_rollable` and the option.
        opts = self._context.get_user_options()
        k_roll, k_unroll_max = opts.k_roll, opts.k_unroll_max

        loopmap = {}

        diagonal = self._identity_reduction()
        ell = self._ell_plan() if diagonal is None else None
        self._module_tables = dict(ell.tables) if ell is not None else {}
        for i, (dimmin, dimmax) in enumerate(self._ks):
            loopmap[f'k{i}'] = len(loopstack)
            if diagonal is not None or ell is not None:
                # Collapsed: the one live step of a lane is its own row,
                # which the body takes from the lead index -- or gathered,
                # each step a column of the lane's own (`_ell_plan`).
                continue
            if -i-1 not in self._lead_dims:
                step = 1
                if i == len(self._ks) - 1 and self._k_width > 1:
                    # Unrolled only.  The last group of a ragged extent is
                    # shorter than the rest, and knowing *how much* shorter is
                    # what lets the body emit the right number of products.
                    # In an unrolled loop the induction value is a Python
                    # integer and the answer is a compile-time count; in a
                    # real `for` it would be a runtime comparison per step,
                    # which costs more than the loads it saves.
                    step *= self._k_width
                roll = _roll_count(dimmin, dimmax, step, k_roll, k_unroll_max)
                rolled = (i == len(self._ks) - 1
                          and self._rollable(i, dimmin, dimmax, step, roll))
                loopstack += [Loop(f'k{i}', dimmin, dimmax, step,
                                   unroll=not rolled,
                                   pragma=roll if rolled else True)]

        stride = 1
        threads = self._num_threads
        for i, (dimmin, dimmax) in enumerate(self._ns):
            loopmap[f'n{i}'] = len(loopstack)
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=True)]
            else:
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride,
                                       unroll=True,
                                       width=self._lead_width,
                                       full_lane=self._full_lane_tail(
                                           i, dimmin, dimmax))]
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
            # the instruction's field instead would make the peeled body build
            # a vector type over a scalar element -- a splat of the tail and a
            # wide store past the end of it.
            width = lead_width_of(
                [varlist[loopmap[f'n{i}']] for i, _ in enumerate(self._ns)])
            ftype = (ScalarType(self._idest.get_fptype()) if width == 1
                     else ScalarType(self._idest.get_fptype(), width))
            if diagonal is not None:
                # The reduction step a lane reads is the row it holds.
                steps, kslot = [varlist[loopmap['n0']]], 'k0'
            elif ell is not None:
                # The slot's steps, each a column and a weight per lane.
                lead = varlist[loopmap['n0']]
                if not isinstance(lead.nonlead(), (int, np.integer)):
                    raise InternalError(
                        f'{self}: a gathered product over a slot the kernel '
                        f'computes; its steps are the slot\'s own')
                steps, kslot = ell.steps.get(int(lead.nonlead()), []), 'k0'
            else:
                steps, kslot = self._k_group(varlist, loopmap)
            # A vector body folds its last factor in as one `fma` op rather
            # than a multiply and an add.  The compiler contracts the pair
            # anyway, so the arithmetic is the same; what the op adds is a
            # name the target can spell itself -- the paired FMA of sm_100
            # is reached only through `__ffma2_rn`, and nvcc does not form it
            # from two scalar ones.  The scalar body keeps the multiply and the
            # add.
            fused = (ftype.length is not None
                     and self._productOperation.irop() == 'mul'
                     and self._sumOperation.irop() == 'add')

            # One vector load per operand whose contiguous axis is the
            # reduction, hoisted out of the group: `B[k,n] .. B[k+V-1,n]` are
            # adjacent, so the `V` steps below take their operand from
            # components of a single load instead of from `V` loads.
            packs = ({} if diagonal is not None or ell is not None else
                     self._k_packs(writer, varlist, loopmap, kslot, len(steps)))

            prods = []
            for c, kval in enumerate(steps):
                if ell is not None:
                    term = self._ell_term(writer, ell, lead, kval, varlist,
                                          loopmap)
                    if term is None:
                        return
                    prods.append(term)
                    continue
                idxs = [[add_offset(varlist[loopmap[nk]]
                                    if nk != kslot else kval,
                                    self._eff_offset(i, j))
                         for j, nk in enumerate(self._opdim_to_nks[i])]
                        for i in range(len(self._ops))]
                if any(self._known_zero(i, idx) for i, idx in enumerate(idxs)):
                    # A factor the description says is zero at every cell
                    # this step reads: no load, and no product to add.
                    continue
                terms = []
                if diagonal is not None:
                    # The lane's one product, its weight from the numbers
                    # (`_diagonal_term`); the first operand is not read.
                    b = self._ops[1].symbol.load(writer, self._context, None,
                                                 idxs[1], False)
                    if b is None:
                        return
                    row = kval.build(writer, self._context)
                    prods.append(self._diagonal_term(writer, row, b,
                                                     diagonal))
                    continue
                for i, op in enumerate(self._ops):
                    if (i, c) in packs:
                        v = packs[(i, c)]
                    else:
                        v = op.symbol.load(writer, self._context, None,
                                           idxs[i], False)
                    if v is None:
                        # zero; no data
                        return
                    terms.append(self._splat(writer, ftype, v))
                if len(terms) == 0:
                    # also zero
                    return
                if fused and len(terms) >= 2:
                    prods.append(terms)
                    continue
                term = terms[0]
                for i in range(1, len(terms)):
                    term = self._emit_binop(writer, ftype,
                                            self._productOperation,
                                            term, terms[i])
                prods.append(term)
            if not prods:
                return
            ns = [varlist[loopmap[f'n{i}']] for i, _ in enumerate(self._ns)]
            value = self._vdest.load(writer, self._context, None, ns, False)
            if value is None:
                assert False
            # Folded into the destination one product at a time, and the
            # destination is still read and written once per group.  Not
            # `dest + (p0 + p1)`: without reassociation that is a multiply, a
            # fused multiply-add and an add for two products, where
            # `(dest + p0) + p1` contracts into two fused multiply-adds.  It
            # is also the order `k_width == 1` sums in, so the width does not
            # change the rounding.
            total = value
            for term in prods:
                if isinstance(term, list):
                    head = term[0]
                    for factor in term[1:-1]:
                        head = self._emit_binop(writer, ftype,
                                                self._productOperation,
                                                head, factor)
                    total = writer.op('fma', ftype, head, term[-1], total,
                                      hint='p')
                    continue
                total = self._emit_binop(writer, ftype, self._sumOperation,
                                         total, term)
            self._vdest.store(writer, self._context, total, ns, False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def _k_group(self, varlist, loopmap):
        """The reduction values this body covers, and which slot they fill.

        `(values, slot)`.  At `k_width == 1` that is the single value the loop
        handed over.  Wider, the loop steps by `k_width` and this expands the
        base into the group -- clipped at the extent, so a ragged reduction
        simply gets a shorter last group.  No guard and no masking: unlike the
        lead dimension, a reduction has no lanes to leave half-valid, the
        leftover steps are just fewer terms in the same sum.
        """
        if not self._ks:
            return [None], None
        slot = f'k{len(self._ks) - 1}'
        base = varlist[loopmap[slot]]
        if self._k_width == 1:
            return [base], slot
        if not isinstance(base, Immediate):
            # A rolled reduction: the loop variable is the group's first step
            # and the rest follow it.  No clipping is needed, since `_rollable`
            # only rolls an extent that divides into whole groups.
            return [add_offset(base, c) for c in range(self._k_width)], slot
        _, kmax = self._ks[-1]
        first = base._value
        return ([Immediate(first + c, base._type)
                 for c in range(min(self._k_width, kmax - first))], slot)

    def _full_lane_tail(self, i, dimmin, dimmax) -> bool:
        """May the ragged end of lead dimension `i` compute on every lane?

        Only where the lanes past the end are padding.  A memory access holds
        itself to the lanes that hold data (`LeadIndex.valid`), so the
        question is only about the registers the body *writes*: the
        accumulator's image.  If that image is exactly this loop's window,
        what follows the window in its last slot is the allocation's rounding
        up to whole slots; if it is larger -- an accumulation into a slice of
        an existing image, as the space-time predictor does with its theta
        shift -- those lanes are the next rows, and writing them is wrong.
        So: no lead origin shift, and the image's box equals the window.

        Under ESIMD the guard costs a narrower vector (`LeadLoop._narrow`);
        under SPMD a branch around the whole block, which the scheduler cannot
        look across.
        """
        opts = self._context.get_user_options()
        if not getattr(opts, 'full_lane_tails', False):
            return False
        if self._theta or self._lead_width != 1:
            return False
        idest = self._idest
        if idest.stype in (SymbolType.Register, SymbolType.Scratch):
            box = idest.data_view.get_bbox()
            if (box.lower()[i], box.upper()[i]) != (dimmin, dimmax):
                return False
        return True

    def _rollable(self, i, dimmin, dimmax, step, k_roll):
        """May reduction loop `i` be a real loop instead of an unrolled one?

        Only when asked (`Options.k_roll`), and only where a runtime value of
        `k` is something every access it reaches can take.  A register image
        cannot: its slots are named at compile time, and a runtime index puts
        the whole array into local memory.  So every operand this loop indexes
        has to live in memory.  A sparse reduction needs its pattern at
        compile time, and a ragged extent a clipped last group, which only an
        unrolled loop can count.
        """
        if not k_roll or self._sparseK[i]:
            return False
        if step <= 0 or (dimmax - dimmin) % step != 0 or dimmax - dimmin <= step:
            return False
        slot = f'k{i}'
        memory = (SymbolType.Global, SymbolType.Batch, SymbolType.SharedMem)
        for j, op in enumerate(self._ops):
            if slot in self._opdim_to_nks[j] and op.symbol.stype not in memory:
                return False
        return True

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
            if (sym.stype == SymbolType.Register
                    and getattr(sym, 'linear_runs', None)):
                # A register image filled from packed storage holds the
                # pattern's entries where `store_linear` put them, and is read
                # one fixed entry at a time (`Symbol._linear_image_load`):
                # adjacent steps are not adjacent there, and there is no
                # vector of them to load.  Its steps are read one by one --
                # packed, `derivative` at k_width 2 would not generate at all.
                continue
            if not self._pack_is_aligned(i, sym, steps, writer):
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

    def _pack_is_aligned(self, i: int, sym, steps: int, writer=None) -> bool:
        """Whether a group of `steps` may be read as one vector from `sym`.

        Contiguity says the values are adjacent; it does not say the vector
        starts where a vector may start, and a wide access that begins
        mid-vector is a fault rather than a slower access.  SeisSol's damage
        step reads `epsTotal[125, 6]`: the 125 are adjacent, so contiguity
        alone would take the pack -- and every second column begins at element
        125, which is odd, so the kernel would die on a misaligned address at
        `k_width` 2 and 4.

        Four conditions, and they are different facts.  The base has to be
        *provably* aligned to the whole group (`Symbol.linear_align_bytes`,
        through `vectorize.reduction_vector_width` -- the rule the lead width
        already asks, with the reduction's own answer that divisibility does
        not matter here).  Every other axis has to move the window by a whole
        number of groups, because the index that moves it is a loop variable
        and one value of it landing mid-vector is enough.  And the group's
        first element has to be one: the reduction need not start at zero and
        the view need not either.

        And the buffer must not be permuted against bank conflicts.  In the
        damage step that would give `*(VectorT<float,2>*)&s0[21]`, where 21 is
        the image of 20 under the staging buffer's `xor4`, with the other three
        conditions all true -- and they cannot catch it: every one of them is
        about the tensor's layout, and the permutation is not in the layout.
        It is applied to the index inside `PirBuilder`, once, to the first
        element of the access; the rest of a vector reads past it unpermuted.
        See `PirBuilder._check_width`, which refuses such an access outright
        for the callers that have no choice.
        """
        from tensorforge.backend.instructions.memory.vectorize import (
            reduction_vector_width)

        if writer is not None and getattr(sym, 'swizzled', None) is not None \
                and sym.swizzled(writer, steps):
            return False
        try:
            strides = list(sym.data_view.get_dim_strides())
        except Exception:                # a symbol without a view: no promise
            return False
        if any(stride % steps for stride in strides[1:]):
            return False
        start = self._eff_offset(i, 0) + (self._ks[-1][0] if self._ks else 0)
        if start % steps:
            return False
        obj = getattr(sym, 'obj', None)
        dense = obj.is_dense() if obj is not None and hasattr(obj, 'is_dense') \
            else True
        return reduction_vector_width(steps, sym.get_fptype().size(),
                                      sym.linear_align_bytes(), 1, dense,
                                      cap=steps) >= steps

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

        The writer of a register-resident operand
        (`multilinear_builder.py:_make_load_op`) asks it too, and the two have
        to agree, or the operand is *written* under one element-to-lane map and
        *read* under another.  The flat fill `GlbToRegLoader` emits puts
        storage element `f` on lane `f % T`, slot `f // T`; ordinary addressing
        puts element `(i0, i1)` on lane `i0 % T`, slot
        `i0 // T + i1 * ceil(e0 / T)`.  Those are the same map only when
        `e0 % T == 0`.  A second clause here, such as
        `data_view.shape[0] < 16`, would decide on a size against a constant
        the writer does not consult -- and a plain `C = A @ B` with `K = 20`
        over 16 lanes would read six output columns out of register slots
        nobody wrote, and get exact zeros.

        So both sides ask this one question, and `is_dense` is the whole of
        it: the flat fill is for the operands stored compressed, which are
        exactly the ones whose cells have no dimension-wise addresses to begin
        with.  Keeping the two in step is what the predicate is for; which
        vector width either map is walked at is a separate question and stays
        open to both.
        """
        obj = self._ops[1].symbol.obj
        return bool(obj) and not obj.is_dense()

    def _output_extent(self) -> int:
        """Columns the second index spans, flattened.

        Needed before emission as well as during it: a plan is laid out over
        this, and `convergence_scope` asks for the plan.
        """
        n = 1
        for mi, mx in self._ns[1:]:
            n *= mx - mi
        return n

    def settle_storage(self, pack_level: int = 0, param_room=None) -> None:
        """Offer `A` its storage order now, before anything is sized from it.

        The offer is made where the operation is emitted, and that is too late
        for an operand the section prologue copies into shared memory: the
        copy is sized, and the arena laid out, before any body exists.  So the
        generator asks here, once the stream is final, with the arguments
        emission will use -- and emission then finds the order in place, which
        is what the offers' idempotence is for.  The matrix path offers its
        fragment order, the nest its SIMT interleave (`_offer_simt_order`):
        offered only at emission, the interleave would outgrow the preloaded
        image, which is sized for the dense operand, and the reads would run
        into the next image.
        """
        if not self._ops or len(self._ns) == 0:
            return
        #: What a column table may still take of the kernel's arguments,
        #: shared with every other offer of this walk
        #: (`Generator._param_room`).
        self._param_room = param_room
        plan = self._plan()
        # Three offers, one operand, and which goes first is a policy --
        # stated here rather than falling out of the order the code happens
        # to be in.
        #
        # The SIMT interleave and the vendor's fragment order *grow* the
        # buffer and buy a wide load with it; a packing shrinks it.  How
        # far apart the two are depends on the order, and at the orders
        # that matter it is not close: over `elastic-linearck`, the
        # interleave takes 15 of 118 constant operands at order 3 and 79 at
        # order 8, and the stored total goes from +10.0 % to +17.4 % of the
        # box where a packing puts it at -31.6 % and -23.7 %.  Through the
        # generator at order 8 the interleave takes every operand of the
        # six largest kernels and they come out 22.5 % larger; at order 3
        # it takes almost none and they come out 32 % smaller.
        #
        # So the default goes to the packing, and the interleave keeps what
        # a caller asks for: `prepare_operands` passed explicitly, or a
        # packing that declines because there is nothing to leave out.  Its
        # own measured win -- `local_flux` at 8 lanes, merged and rolled,
        # -31.4 % -- is one configuration and not a rule, and a caller who
        # is in it says so.
        opts = self._context.get_user_options()
        asked = self._context.get_asked_options().asked()
        packing_first = (opts.pack_operands != 'off'
                         and 'prepare_operands' not in asked)
        if packing_first:
            self._offer_packing(pack_level)
            if getattr(self._ops[0].symbol.obj, 'packing', None) is not None:
                return
        if plan[0].strategy is Strategy.GENERIC:
            self._offer_simt_order(settling=True)
            self._offer_packing(pack_level)
            return
        depth = math.prod(mx - mi for mi, mx in self._ks)
        self._offer_order(_vendor_module(self._context),
                          self._ops[0].symbol.obj,
                          self._ns[0][1] - self._ns[0][0], depth)
        if plan[0].strategy is Strategy.DPP:
            # The DPP chain reads `A` by coordinate, so a packing addresses
            # it exactly as it does the nest's.
            self._offer_packing(pack_level)

    def _offer_simt_order(self, settling=False):
        """Let a batch-constant `A` be stored so a lane reads its rows in vectors.

        The generic nest distributes the lead dimension cyclically: lane `l`
        holds rows `l, l + T, l + 2T, ...`, which are `T` elements apart in a
        column-major `A` and so arrive one scalar load each.  Stored instead
        with each lane's rows side by side, in groups of up to one 16-byte
        vector, a lane reads `group` rows with one aligned load, and the
        lanes of a group still read one contiguous run between them.  The multiply and
        every other operand keep the cyclic layout; only where `A` sits in
        memory changes, and the host packs it from `storage_map`.

        The conditions of `_offer_order`, for the same reasons -- the caller
        asked, the operand is `Addressing.NONE` and dense -- plus what the
        address needs (`_reads_interleavable`): the row index is the lead
        dimension of a nest at width 1, and every read starts its rows on a
        slot.  Not the whole tensor, and not rows that fill the lanes: a read
        that starts on a slot names whole slots of the order, whatever its
        extent, and a ragged last slot is padding, which a full-lane tail
        reads as zeros.  SeisSol reads its operators in boxes -- `volume` rows
        1 to 54 of 64, the derivative fewer each order -- and a condition on
        the whole tensor would offer none of them.

        Nor only one reader.  The order is the tensor's, not this operation's:
        `Symbol._interleaved_load` addresses every read of it, so another
        multiplication may read it as long as it is a nest of the same lanes
        that reads it the same way -- the derivative reads each operator once
        per order.  A matrix path reads fragments of an order it prepared
        itself, and a loader copies by coordinate; either one refuses.

        A stand-in of a merged run is not a buffer: what is stored are its
        members, one per iteration, and the one body reads whichever the
        counter selects -- so every member is stored alike, or none is, and
        the stand-in carries the addressing.
        """
        if not self._context.get_user_options().prepare_operands:
            return
        if not self._ops:
            return
        sym = self._ops[0].symbol
        a_obj = getattr(sym, 'obj', None)
        if not isinstance(a_obj, Tensor):
            return
        variant = getattr(a_obj, 'is_variant', False)
        members = (tuple(getattr(a_obj, 'variant_members', ())) if variant
                   else (a_obj,))
        if (not members or a_obj.addressing is not Addressing.NONE
                or not a_obj.is_dense()
                or any(getattr(m, 'residence', Residence.MEMORY)
                       is not Residence.MEMORY for m in members)):
            return
        shape = tuple(int(x) for x in a_obj.get_actual_shape())
        if len(shape) != 2 or not self._reads_interleavable(sym, a_obj):
            return
        threads = self._num_threads
        # The prologue's copy into shared memory is not a reader: it moves the
        # storage as it lies (`GlbToShrLoader._verbatim`), and the image it
        # writes is read in the order the host stored (`Symbol.verbatim`).
        if any(user is not self
               and not (getattr(user, '_verbatim', False)
                        and user.get_dest() is sym)
               and not (isinstance(user, MultilinearInstruction)
                        and user._num_threads == threads
                        and user._reads_interleavable(sym, a_obj))
               for user in sym.get_user_list()):
            return
        rows, cols = shape
        # `Tensor.alignment` is deliberately not consulted.  It describes
        # the layout the frontend gave the operand, and preparing *replaces*
        # that layout: what the kernel reads is a buffer the host writes
        # from the offering, aligned as a fresh allocation is.  The offering
        # has no field to ask for it with -- yateto refuses one it does not
        # know by name (`ExportFactory._grantOffering`) -- so it is an
        # agreement between the two sides rather than a request, and this is
        # where it is written down.
        widest = 16 // a_obj.datatype.size()
        if widest < 2 or threads < 1:
            return
        slots = -(-rows // threads)
        # As wide as a lane's rows need, and no wider.  A vector past the rows
        # a lane holds is padding every load carries: 64 rows over 32 lanes
        # are two a lane, and a group of four would read half zeros -- the
        # operator stored at twice its size, and at one row a lane four times.
        # Of the widths that take the fewest loads, the narrowest: two rows a
        # lane in one 8-byte pair, three still in one 16-byte group (a quarter
        # of it padding, for one load rather than two), and one row in none --
        # the plain layout already hands the lanes of a column one run.
        group = min((w for w in (1, 2, 4, 8, 16) if w <= widest),
                    key=lambda w: (-(-slots // w), w))
        if group < 2:
            return
        groups = -(-slots // group)
        ld = groups * group * threads
        # A preloaded image is sized and laid out once the stream is settled
        # (`settle_storage`, where this is asked first).  Asked later -- a
        # matrix plan that declined at emission -- an order that stores more
        # than the dense operand would outgrow that image, so then only one
        # of the operand's own size.
        preloaded = any(getattr(user, '_verbatim', False)
                        and user.get_dest() is sym
                        for user in sym.get_user_list())
        if not settling and preloaded and ld != rows:
            return
        order = []
        for k in range(cols):
            for g in range(groups):
                for lane in range(threads):
                    for c in range(group):
                        row = lane + threads * (g * group + c)
                        order.append(row + rows * k if row < rows else -1)
        # Idempotent, and the same for every member: an order already there
        # is accepted only if it is this one -- a peeled first iteration reads
        # a member directly and may have stored it so already -- and anything
        # else (a fragment order, another geometry) refuses the whole run.
        for member in members:
            if (member.addressing is not Addressing.NONE
                    or not member.is_dense()
                    or tuple(int(x) for x in member.get_actual_shape()) != shape
                    or list(member.get_bbox().lower())
                    != list(a_obj.get_bbox().lower())
                    or (member.storage_order is not None
                        and tuple(member.storage_order) != tuple(order))):
                return
        for member in members:
            member.storage_order = order
            member.simt_interleave = (threads, group, ld)
        if variant:
            a_obj.simt_interleave = (threads, group, ld)

    def _reads_interleavable(self, sym, a_obj) -> bool:
        """Whether this multiplication reads `A` the way the SIMT interleave
        addresses it (`Symbol._interleaved_load`).

        As its own lead operand, at width 1 over its lanes, with the row index
        the lead dimension of the nest; with its rows starting on a slot --
        both the box's own start and this read's offset whole slots, since a
        lane vector that starts inside a slot is spread over two groups; and
        by the generic nest, which reads `A` through `Symbol.load`.  A matrix
        path reads the fragments of an order it prepared for itself, and one
        that found this one would take it for that.
        """
        if (_explicit_simd(self._context) or self._lead_width != 1
                or not self._ops or self._ops[0].symbol is not sym):
            return False
        if self._opdim_to_nks[0] != ['n0', 'k0'] or 0 not in self._lead_dims:
            return False
        threads = self._num_threads
        if (threads < 1 or a_obj.get_bbox().lower()[0] % threads
                or self._eff_offset(0, 0) % threads):
            return False
        return self._plan()[0].strategy is Strategy.GENERIC

    def _reads_packable(self, sym, a_obj) -> bool:
        """Whether this multiplication reads `A` the way a packing addresses
        it (`Symbol._packed_index`).

        As its own lead operand, with the row index the lead dimension of a
        generic nest or of AMD's DPP chain -- the conditions of
        `_reads_interleavable`, minus the one it has that this does not need:
        the rows need not start on a slot, because a packing is addressed by
        an offset on the lane and not by whole groups, and requiring the
        alignment is precisely what would throw the saving away.

        A lane that holds several adjacent rows (`Options.lead_vectorize`,
        AMD's width-2 DPP) is served by a packing whose runs are whole
        granules of that width, which the offer asks for below; what the
        read then needs on top is that its own window starts on one.

        What it has instead is the column.  A packed slot is `off_c + r -
        start_c`, and `off_c` and `start_c` are the column's own numbers, so
        an unrolled reduction gives them as integers and a rolled one does
        not.  That is a restriction on the *form* and not on the read, and
        `_offer_packing` applies it there: a rolled reduction may have a
        uniform packing, whose slot is affine in the column and needs no
        per-column number at all; any other form needs the numbers read at
        run time, which is `Options.pack_tables`, and without that it may
        not have one.
        """
        if (_explicit_simd(self._context) or not self._ops
                or self._ops[0].symbol is not sym):
            return False
        if self._opdim_to_nks[0] != ['n0', 'k0'] or 0 not in self._lead_dims:
            return False
        if self._num_threads < 1 or self._lead_width < 1:
            return False
        if self._eff_offset(0, 0) % self._lead_width:
            # A window that starts inside a lane's own run of rows: the
            # predicate that holds the access off is per lane and cannot
            # mask within one (`Symbol._packed_index` refuses it).
            return False
        if self._plan()[0].strategy not in (Strategy.GENERIC, Strategy.DPP):
            # The nest and AMD's DPP chain, both of which read `A` by
            # coordinate -- which is what a packing changes the address of
            # and nothing else.  The matrix paths are out: they read
            # fragments of an order they prepared for themselves, and a
            # packed operand has no fragments, only the same cells with the
            # empty ones left out.
            return False
        return True

    def _bitset_fits(self, a_obj, rows: int, threads: int) -> bool:
        """Whether every row a lane slot reaches fits in one 32-bit word.

        A bitset slot is a population count below the lane's own row, and the
        shift that masks it is undefined at 32 (`Symbol._bitset_index`
        refuses there).  The highest row any lane reaches is the last slot's
        last lane, which is what this counts -- not the row count, which for
        a 33-row operand over 32 lanes would say yes to a read of row 63.

        That count holds only while a slot starts on a multiple of the lane
        count, so this asks for what the other two forms deliberately do not:
        the box's own start and the read's offset in whole slots.  Without
        them lane 0 begins inside a slot and the last lane runs `rows +
        threads - 2` rows up -- 56 for a 24-row operand over 32 lanes, which
        is where the shift stops being defined.
        """
        if threads < 1 or rows < 1:
            return False
        if (int(a_obj.get_bbox().lower()[0]) % threads
                or self._eff_offset(0, 0) % threads):
            return False
        return threads * -(-rows // threads) <= 32

    def _reduction_is_rolled(self) -> bool:
        """Whether the innermost reduction will be a real loop.

        The same question `gen_code_inner` answers when it builds the loop
        stack, asked before there is one: settling the storage happens first,
        and an offer that assumed an unrolled reduction would be wrong by the
        time the body is written.

        `k_roll` being zero is not the answer: `k_unroll_max` rolls a long
        reduction unasked, and `_roll_count` is where the two meet.  Reading
        the option instead of asking it is what made a 230-step reduction
        look unrolled, and the packed read then wanted a column the loop
        does not have at build time.
        """
        if not self._ks:
            return False
        opts = self._context.get_user_options()
        i = len(self._ks) - 1
        if -i - 1 in self._lead_dims:
            return False
        dimmin, dimmax = self._ks[i]
        step = self._k_width if self._k_width > 1 else 1
        roll = _roll_count(dimmin, dimmax, step, opts.k_roll,
                           opts.k_unroll_max)
        return self._rollable(i, dimmin, dimmax, step, roll)

    def _offer_packing(self, level: int = 0):
        """Let a batch-constant `A` be stored without the box cells its own
        numbers never fill.

        The bounding box is the smallest rectangle around the non-zeros, and
        a rectangle around a triangle is half padding: SeisSol's operators
        hold 11545 numbers in 20627 cells.  The padding is read, multiplied
        by zero and added, and -- the reason this is worth a storage decision
        rather than an arithmetic one -- it is *stored*, which is what
        decides whether an operand fits in shared memory.

        Three forms (`matrix.packing`), and the option names which: `rows`
        drops the rows no column fills, `runs` stores each column between its
        own ends, `bits` stores the non-zeros alone.  `auto` takes the
        smallest one the reads can address, which today is `runs` -- the
        bitset needs a population count in the address and is offered only
        when asked for by name.

        The conditions of `_offer_simt_order`, for the same reasons: the
        order is the tensor's and not this operation's, so every reader has
        to address it the same way or none may; a stand-in's members are
        stored alike or not at all; and an operand already carrying an order
        keeps it.  Unlike the interleave this needs the operand's *numbers*,
        since what it discards is what they say is absent -- without them
        `nonzero_pattern` reports the whole box present and the packing
        saves nothing, which `pack` returns as `None`.
        """
        kind = self._context.get_user_options().pack_operands
        if kind == 'off' or not self._ops:
            return
        fit = kind == 'fit'
        sym = self._ops[0].symbol
        a_obj = getattr(sym, 'obj', None)
        if not isinstance(a_obj, Tensor):
            return
        variant = getattr(a_obj, 'is_variant', False)
        members = (tuple(getattr(a_obj, 'variant_members', ())) if variant
                   else (a_obj,))
        if (not members or a_obj.addressing is not Addressing.NONE
                or any(getattr(m, 'residence', Residence.MEMORY)
                       is not Residence.MEMORY for m in members)):
            return
        if not self._reads_packable(sym, a_obj):
            return
        threads = self._num_threads
        # A verbatim copy into shared memory moves the storage as it lies and
        # is read back the same way, so it is not a second opinion about the
        # order; anything else that reads the operand has to address it as
        # this does.
        grain = int(self._lead_width)
        if any(user is not self
               and not (getattr(user, '_verbatim', False)
                        and user.get_dest() is sym)
               and not (isinstance(user, MultilinearInstruction)
                        and user._num_threads == threads
                        and user._lead_width == grain
                        and user._reads_packable(sym, a_obj))
               for user in sym.get_user_list()):
            return
        shape = tuple(int(x) for x in a_obj.get_actual_shape())
        lower = list(a_obj.get_bbox().lower())
        for member in members:
            if (member.addressing is not Addressing.NONE
                    or not member.is_dense()
                    or not member.has_values()
                    or member.storage_parts != 1
                    or (member.storage_order is not None
                        and member.packing is None)
                    or member.simt_interleave is not None
                    or member.slot_major is not None
                    or tuple(int(x) for x in member.get_actual_shape()) != shape
                    or list(member.get_bbox().lower()) != lower):
                return
        # The forms to try, tightest first, falling through to the looser
        # ones where a tighter is refused rather than merely unprofitable
        # (`packing.fit_kinds` says why that matters).
        kinds = (packing_mod.fit_kinds(level) if fit
                 else tuple(reversed(KINDS)) if kind == 'auto'
                 else (kind,))
        # A rolled reduction hands the column over as a loop variable, so
        # only a form whose slot is affine in it can be addressed
        # (`Symbol._uniform_packed_index`).  Checked on the packing rather
        # than on the kind, because `runs` is uniform whenever the columns
        # happen to agree and there is no reason to refuse that.
        #
        # Asked of *every* reader and not only of this one.  The order is
        # the tensor's: an operand offered a form here is read in that form
        # by each of them, and one whose own reduction rolls would meet a
        # column it cannot fold into an address.  SeisSol's order-8
        # derivative is the case -- one unrolled reader offering `runs` to
        # five readers of which the later ones roll.
        rolled = any(
            user._reduction_is_rolled()
            for user in (self,) + tuple(
                u for u in sym.get_user_list()
                if isinstance(u, MultilinearInstruction) and u is not self))
        # With `pack_tables` the per-column numbers are read at run time
        # instead, so a rolled reader no longer holds every reader to the
        # forms that need none (`Symbol._tabled_packed_index`).  The value
        # names the home: the kernel's arguments, the generated file, or
        # `auto` -- the arguments while they have room.
        home = str(self._context.get_user_options().pack_tables)
        if home not in ('off', 'param', 'module', 'auto'):
            raise GenerationError(
                f'pack_tables={home!r}: a column table lives in the '
                f'arguments (param), in the generated file (module), or '
                f'nowhere (off); auto takes the first that has room')
        tables = home != 'off'
        # One packing over all the members at once, not one apiece: a
        # stand-in's body addresses whichever member the counter names, so
        # two members packed differently would read one another's slots
        # (`packing.pack_union`).
        chosen = None
        for name in kinds:
            if name == 'bits' and (grain != 1
                                   or not self._bitset_fits(a_obj, shape[0],
                                                            threads)):
                continue
            chosen = packing_mod.pack_union(members, name, grain)
            if (chosen is not None and rolled and not chosen.uniform
                    and not tables):
                chosen = None
                continue
            if chosen is not None:
                break
        if chosen is None:
            return
        # One table for the operand, named after the symbol the readers
        # reach it through -- a stand-in's members share a packing, so they
        # share the numbers that address it.  Only where a reader actually
        # computes its column: otherwise every number is an immediate and
        # the parameter would be read by nothing.
        table = (self._table_name(sym)
                 if rolled and chosen.needs_table else None)
        # Where it lives: the arguments while they have room, and the
        # generated file once they do not.  Not a question about the form --
        # the numbers are the same either way, and so is the read.
        #
        # Claimed before the offer can turn out to be a repeat, because the
        # room is what the *other* operands are then budgeted against: a
        # second walk over the stream re-offers some operands and not
        # others (`Generator._settle_storage` retries at a tighter level),
        # and one that keeps the table it was granted still holds the
        # argument space it was granted with.
        in_param = table is not None and self._claim_table(chosen, table,
                                                           home)
        if all(member.packing == chosen for member in members):
            return
        order = chosen.order()
        for member in members:
            # Cleared first: the setter checks an order against the box, and
            # a retry at a tighter form is replacing one packing with
            # another rather than adding to it.
            member.storage_order = None
            member.packing = None
            member.storage_order = order
            member.packing = chosen
            member.packing_table = table
            member.packing_table_param = in_param
        if variant:
            a_obj.packing = chosen
            a_obj.packing_table = table
            a_obj.packing_table_param = in_param

    def _table_name(self, sym) -> str:
        return f'{sym.name}_pack'

    def _claim_table(self, packing, name: str, home: str = 'auto') -> bool:
        """Take the table's bytes out of what the kernel's arguments have
        left, and say whether they had them.

        The argument space is small where it is smallest -- 4 kB on sm_86
        and gfx942, 2 kB on PVC -- and three tables of a 96-column operator
        come to 3.4 kB between them, so this is a real bound: on sm_86 the
        order-8 derivative's third operator is exactly the one that does not
        fit.  What it decides is the table's *home* and not the operand's
        form; a table turned away from the arguments is a namespace-scope
        array in the generated file (`Lexic.module_constant`), which the
        kernel indexes with the same expression.

        Charged once per operand: a second reader is granted the table the
        first one was granted, not a second copy of it.  Without a room to
        ask -- an offer made outside `Generator._settle_storage`, which is
        the only caller that knows what the arguments have left -- the
        arguments are assumed to have room, which is what they had before
        there was a bound at all.

        `home` overrides the question: `param` and `module` each name one
        home for every table and are there to be measured against each
        other, since which of the two is cheaper is not obvious from the
        spelling.
        """
        if home == 'module':
            return False
        room = getattr(self, '_param_room', None)
        if room is None or home == 'param':
            return True
        if name in room['given']:
            return room['given'][name]
        need = 4 * len(packing.columns_table())
        fits = need <= room['left']
        room['given'][name] = fits
        if fits:
            room['left'] -= need
        return fits

    def _offer_order(self, module, a_obj, lead, depth):
        """Let the target store `A` in the order it will read it.

        Asked here rather than at construction because here is where it is
        settled: the plan is chosen, the module that will emit is known, and
        every instruction exists, so the question "does anything else read
        this operand" has an answer.  Earlier it does not, and marking an
        operand a second reader then addresses by row and column is not a
        missed optimization but a wrong kernel.

        Four conditions, and each one is a way the marking could be wrong
        rather than merely unhelpful:

        * The caller asked.  Preparing an operand moves work onto whoever
          fills the buffer, so it is `Options.prepare_operands` and not a
          vendor rule --- a host that cannot run the packer cannot use the
          kernel, and only the caller knows.
        * `Addressing.NONE`.  The permutation is applied once to a buffer the
          whole batch shares; per-element data would have to be permuted per
          element, which is the transform being avoided.
        * Nothing else reads it.  The order is this operation's shape and this
          target's fragment layout, so two readers are two orders.
        * The target has one to state.  `prepared_order` answers `None` where
          it does not --- a wave it is not written for, an accumulator with no
          entry --- and a refusal has to be an answer here, not an exception.

        Idempotent: an operand already carrying an order keeps it, so
        regenerating does not re-decide.
        """
        if not self._context.get_user_options().prepare_operands:
            return
        if self._lead_width != 1:
            # A packed lead operand is read as vectors of rows, and an order
            # stated for single rows would be read by coordinate -- wrongly.
            return
        if getattr(a_obj, 'residence', Residence.MEMORY) is not Residence.MEMORY:
            # An operand whose numbers are in the instruction stream or in a
            # kernel argument has no buffer, and every one of these orders is
            # a statement about a buffer.  Offered one, the matrix path asks
            # for a slot, the read declines for want of an address, and the
            # accessor hands `extract` the nothing it got back.
            return
        if not isinstance(a_obj, Tensor) or a_obj.storage_order is not None:
            # Only a tensor has a storage convention.  A register-resident
            # operand is already in whatever order the pass that produced it
            # left it in, and there is no buffer for a host to write.
            return
        if a_obj.addressing is not Addressing.NONE or not a_obj.is_dense():
            return
        if len(a_obj.get_actual_shape()) != 2:
            return
        # The prologue's copy into shared memory is not a reader: it moves the
        # storage as it lies, whatever the order (`GlbToShrLoader._verbatim`).
        # Nor is another multiplication that reads a slot-major order the way
        # this one does: that order is the tensor's and not this operation's
        # -- the address is rewritten in `Symbol.load`, for every read -- so
        # the derivative's five reads of one operator all take it.  Only a
        # target whose orders can be that is asked with other readers about.
        sym = self._ops[0].symbol
        readers = [user for user in sym.get_user_list() if user is not self
                   and not (getattr(user, '_verbatim', False)
                            and user.get_dest() is sym)]
        if module is None or (readers and not module.ORDERS_EVERY_READER):
            return
        order = module.prepared_order(tuple(a_obj.get_actual_shape()),
                      self._idest.get_fptype(), self._context,
                      columns=self._output_extent(), lead=lead, depth=depth,
                      threads=self._num_threads)
        if order is None:
            return
        slot_major = getattr(order, 'slot_major', None)
        if any(not (slot_major is not None
                    and self._reads_slot_major(user, sym, a_obj, slot_major))
               for user in readers):
            return
        # A stand-in of a merged run is not a buffer: its members are, one per
        # iteration, and the one body reads whichever the counter selects in
        # the stand-in's order -- so every member is stored alike or none is,
        # as in `_offer_simt_order`.  A member already carrying this order is
        # accepted: a peeled first iteration reads it directly and stores it so.
        variant = getattr(a_obj, 'is_variant', False)
        members = (tuple(getattr(a_obj, 'variant_members', ())) if variant
                   else (a_obj,))
        shape = tuple(int(x) for x in a_obj.get_actual_shape())
        if not members:
            return
        # What an order may say beyond which cell a slot holds: how the kernel
        # addresses it where that is not a fragment read (`Tensor.slot_major`),
        # and how many scalars an element takes where the target reads the
        # halves a matrix instruction multiplies rather than splitting them.
        # An order that says nothing about parts leaves them to whoever set
        # them -- a caller may store an operand split before any order exists.
        parts = getattr(order, 'parts', None)
        if (parts is not None and parts != 1
                and any(reader._plan()[0].strategy is not Strategy.MATRIX
                        for reader in [self] + readers)):
            # The halves are what a matrix instruction multiplies; any other
            # arrangement multiplies the floats, and would read the first
            # half as one -- so every reader has to be one.
            parts = 1
        if slot_major is not None and not self._slot_major_fits(a_obj):
            return
        for member in members:
            if (member.addressing is not Addressing.NONE
                    or not member.is_dense()
                    or tuple(int(x) for x in member.get_actual_shape()) != shape
                    or (member.storage_order is not None
                        and tuple(member.storage_order) != tuple(order))
                    or (parts is not None
                        and member.storage_parts not in (1, parts))):
                return
        # Parts planar with it: the order puts a lane's fragments side by side,
        # and the instruction wants each part's in one register group
        # (`Tensor.storage_planar`).
        for target in members + ((a_obj,) if variant else ()):
            target.storage_order = order
            target.storage_planar = True
            if slot_major is not None:
                target.slot_major = slot_major
            if parts is not None and parts != 1:
                target.storage_parts = parts

    def _slot_major_fits(self, a_obj) -> bool:
        """Whether every read of `A` here names whole slots of its rows.

        A slot-major operand (`Tensor.slot_major`) is addressed by the lane
        vector: slot `s` of column `k` is one run of `threads` rows.  A read
        whose rows start inside a slot -- a lead origin, a slicing offset or a
        box that is not a whole number of slots from the rows' start -- has no
        such run, and nor does a lead index of another width.  The column is
        free: every column of every slot is stored.
        """
        return (self._lead_width == 1
                and self._opdim_to_nks[0] == ['n0', 'k0']
                and a_obj.get_bbox().lower()[0] % self._num_threads == 0
                and self._eff_offset(0, 0) % self._num_threads == 0)

    @staticmethod
    def _reads_slot_major(user, sym, a_obj, slot_major) -> bool:
        """Whether `user`, another reader of `A`, reads a slot-major order as
        the one offering it would: as its own lead operand, over the same
        lanes, in whole slots."""
        return (isinstance(user, MultilinearInstruction)
                and bool(user._ops) and user._ops[0].symbol is sym
                and user._num_threads == slot_major[0]
                and user._slot_major_fits(a_obj))

    def _shape(self) -> ComputeShape:
        """What the choice and the reservation are both made from.

        One place, because they have to agree: a plan chosen from one reading
        of the operation and a buffer sized from another is either memory
        nobody writes or an overrun.
        """
        return ComputeShape(threads=self._num_threads,
                            accumulator=self._idest.get_fptype(),
                            sparse=self._second_operand_is_sparse(),
                            explicit_simd=_explicit_simd(self._context),
                            lead=self._ns[0][1] - self._ns[0][0],
                            depth=math.prod(mx - mi for mi, mx in self._ks),
                            a_parts=getattr(self._ops[0].symbol.obj,
                                            'storage_parts', 1)
                            if self._ops and self._ops[0].symbol.obj else 1,
                            lead_layout=lead_layout(self._num_threads,
                                                    self._lead_width),
                            a_uniform=self._a_uniform())

    def _a_uniform(self) -> bool:
        """Whether the lead operand is the same for every multiplication."""
        a_obj = (self._ops[0].symbol.obj
                 if self._ops and self._ops[0].symbol is not None else None)
        return (a_obj is not None
                and getattr(a_obj, 'addressing', None) is Addressing.NONE)

    def set_mult_stride(self, stride) -> None:
        """The shared memory one multiplication owns, in elements, as far as
        the generator knows it when it builds the body.

        Read by a matrix path whose warp holds several multiplications and
        reads its neighbors' tiles (`nvidia._warp_group`).  The arena is laid
        out after the body, so the first build is told a guess, and a body
        that asked (`asked_mult_stride`) is built again with the answer.
        """
        self._mult_stride = stride
        self._mult_stride_asked = False

    def asked_mult_stride(self) -> bool:
        """Whether the body built since `set_mult_stride` read the stride."""
        return self._mult_stride_asked

    def _ask_mult_stride(self) -> int:
        self._mult_stride_asked = True
        return self._mult_stride

    def convergence_scope(self):
        """How far the threads have to run in step for the plan to be legal.

        An NVIDIA matrix instruction is the warp's collective: every lane
        issues it together.  Where a multiplication is narrower than the warp,
        its neighbors in the warp are part of that and have to take the same
        trips through the batch loop -- which the loop's grouped traversal
        guarantees and `verify` checks it against.  The target says so
        (`convergence`); a target that says nothing asks nothing.
        """
        module = _vendor_module(self._context)
        if module is None or not is_contraction(len(self._ops)):
            return None
        ask = module.convergence
        shape = self._shape()
        levels = [level for level in (ask(span.strategy, shape)
                                      for span in self._plan())
                  if level is not None]
        return max(levels) if levels else None

    def _window_declines(self) -> bool:
        """Whether the output's window is one the matrix paths cannot place.

        At lead width `w` a slot is `threads * w` elements, and that is the
        unit every lead and contraction index of those paths divides by.  At
        width one the span is the thread count.  A packed operand whose lead
        or contraction does not start on a whole vector, or a lead that does
        not start on a slot, goes to the nest, which peels.

        And how many slots the window *touches*, against how many the paths
        count.  They count `ceil(extent / span)` and put the window at slot
        `start // span`, so a start inside a slot is rounded away twice: a
        32-row window at row 9 over 32 lanes touches two slots and is emitted
        as one, with its first nine rows read from before the buffer and its
        last nine never computed -- which the destination then stored from
        registers nothing had assigned.  A window that *fits* inside one
        slot is not affected, which is why this counts rather than asking
        whether the start is on a slot: the lane bounds clip such a window
        and always did.  At width above one the first condition is the
        stronger one and this adds nothing.

        Asked by `_plan` as well as by the emission.  Asked by the emission
        alone, the plan still said the matrix path, `settle_storage` stored
        `A` in k-quads for it, and the nest then read that buffer by
        coordinate (`packed_boxed_runs` on gfx942).
        """
        width = self._lead_width
        span = self._num_threads * width
        if not span or not self._ns:
            return False
        ks00 = self._ks[0][0] if len(self._ks) > 0 else 0
        if width > 1 and (self._ns[0][0] % span or ks00 % width
                          or ks00 >= span):
            return True
        lo, hi = self._ns[0]
        return hi > lo and ((hi - 1) // span - lo // span + 1
                            != -(-(hi - lo) // span))

    def _plan(self) -> Tuple[Span, ...]:
        """Which arrangements compute this operation, over which columns.

        Derived on each call rather than stored.  Two callers ask -- the
        emission below, and `convergence_scope` before any body exists -- and
        the answer has to be the same for both: a traversal driven for one
        arrangement and an emission of another is a matrix instruction some
        lanes of the warp do not reach.  Everything it reads is fixed by
        `_analyze`, so deriving it twice cannot disagree with itself the way
        two stored copies can.
        """
        n = self._output_extent()
        module = _vendor_module(self._context)
        if module is None or not is_contraction(len(self._ops)):
            return whole(Strategy.GENERIC, n)
        # A second operand whose numbers are in the code has no memory: the
        # vendor paths read B through it -- spread over the lanes and
        # broadcast (AMD), as fragments, staged -- and it has none of those.
        # The nest reads it as literals.
        if self._ops[1].symbol.stype == SymbolType.Data:
            return whole(Strategy.GENERIC, n)
        # Nor is one passed by value spread over the lanes: it is a scalar to
        # every product, read at one index by all of them -- from the constant
        # bank or into an SGPR -- which is what the nest does with it.  Spread,
        # its index is the lane's, and the struct it is passed as is copied
        # into private memory to be indexed at all.
        if passed_by_value(self._ops[1].symbol):
            return whole(Strategy.GENERIC, n)
        # A window the matrix paths cannot place: the nest computes it, and
        # says so here, before `settle_storage` stores `A` in an order only
        # the matrix paths read.
        if self._window_declines():
            return whole(Strategy.GENERIC, n)
        shape = self._shape()
        chosen = choose_strategy(
            legal_strategies(module.strategies(shape, self._context)),
            module.ORDER)
        if chosen is Strategy.GENERIC:
            return whole(Strategy.GENERIC, n)
        plan = module.plan(chosen, shape, n, self._context)
        if not covers(plan, n):
            raise InternalError(
                f'{self._context.target.hw.vendor} planned '
                f'{plan} for {n} columns, which does not compute each of them '
                f'exactly once')
        return plan

    def _nonleading_dim_test(self, writer: Writer):
        # if len(self._ks) == 0 and len(self._ops) == 1:
        #     return False

        plan = self._plan()

        # Every vendor path is written for `C[m,n] += A[m,k] B[k,n]`: the lead
        # index on A and on the destination, B read across the lanes by
        # `(k, n)`.  A second operand that carries the lead -- the broadcast
        # operand first, `t[i,j] = s * M[i,j]` -- is read per lane instead, and
        # the sparse test and the broadcast reads take its lead index for a
        # number.  The nest computes it.
        if len(self._ops) > 1 and 'n0' in self._opdim_to_nks[1]:
            return False
        if len(self._ops) > 1 and not self._lane_readable_by_reduction(1):
            return False
        # A first operand that is the identity pattern has one live step per
        # lane, and a different one in each: a matrix path issues every step
        # on every lane and would compute the dense product.  The nest takes
        # the one product a lane needs and reads nothing across the lanes
        # (`_identity_reduction`).  A first operand with few nonzeros per row
        # is the same case with more than one live step: the nest gathers
        # them (`_ell_plan`).
        if (self._identity_reduction() is not None
                or self._ell_plan() is not None):
            return False

        if plan[0].strategy is not Strategy.GENERIC:
            K = 1
            N = 1
            M = 1

            for mi, mx in self._ks:
                K *= mx - mi

            for mi, mx in self._ns[1:]:
                N *= mx - mi

            # The window conditions, which `_plan` asks as well
            # (`_window_declines`).
            if self._window_declines():
                return False
            width = self._lead_width
            span = self._num_threads * width
            ks00 = self._ks[0][0] if len(self._ks) > 0 else 0
            M *= -(-(self._ns[0][1] - self._ns[0][0]) // span)
            Mx = (self._ns[0][1] - self._ns[0][0])

            def unwindJ(j):
                idx = [None]
                for mi, mx in self._ns[1:]:
                    size = mx - mi
                    idx += [j % size + mi]
                    j //= size
                return idx

            def unwindI(i):
                size = -(-(self._ns[0][1] - self._ns[0][0]) // span)
                idx = [LeadIndex(i % size + self._ns[0][0] // span,
                                 self._num_threads, 1, width=width)]
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
                    sizeL = -(-(size + kx) // span)
                    # The block holding the window's first element starts at
                    # a whole span, `ks00 % span` elements before it.  The
                    # paths pair step `s` with lane `s % threads` and never
                    # use those lanes; a block read still touches them, and
                    # in memory they are not the operand's (`LeadIndex`).
                    head = (ks00 % span) // width if k % sizeL == 0 else 0
                    idx = [LeadIndex(k % sizeL + ks00 // span,
                                     self._num_threads, 1, width=width,
                                     first=head)]
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

            def A(writer, var, i, k, part=0, parts=1):
                """One part of `A` at element `(i, k)`.

                `part` is zero for an operand stored as one scalar per
                element, which is every operand a frontend describes.  Where
                the operand is stored prepared the parts sit adjacent --- the
                part index is the innermost stride --- so the element is the
                same place and only the addend differs.  Which part this is,
                is the caller's to say; what the addend means for the width,
                the layout claim and the broadcast is `load`'s, and stays
                there.
                """
                with writer.speculative() as spec:
                    res = self._ops[0].symbol.load(writer, self._context, var,
                                                   unwindOp(i, 0, k, 0, True),
                                                   False, part=part, parts=parts)
                    if not res:
                        spec.discard()
                return res

            def A_slot(writer, slot, parts=1, shift=None, width=1):
                """One fragment of a pre-ordered `A`, named by storage slot.

                A prepared operand has no coordinate addressing left, and that
                is the point rather than a limitation: its slots *are* the
                order a lane reads them in, so the address is what the staged
                read computed --- one memory away, with nothing between.

                A single lead index, whose `nonlead` names the slot within the
                lane and whose lane term the class supplies.  The remaining
                axes fold away: the view still describes the operand's shape,
                because everything else that asks it --- the bound, the batch
                stride, the element type --- still wants that answer, and
                only this one read does not.

                They fold away at the box's lower corner, not at zero.  The
                address subtracts that corner from every coordinate, and a
                slot is already counted from the stored window's start: an
                axis indexed at zero over a box from column 1 moved every
                read a column back -- `C = A[:, 1:17] B[1:17, :]` read row
                `l - 4` of its k-quads at sixteen lanes, and lanes 0 to 3
                read sixteen floats before the buffer (gfx942, MI300A).

                The lead axis is the other way round: a lane holds the row of
                its own position, counted from zero, and the stored order
                counts rows from the box's start, so the read does want the
                corner taken off -- in the order's units, `width` slots a
                row, where the address takes it off once.  A box from row 9
                read its k-quads nine floats back where they are thirty-six,
                misaligned and wrong (`packed_boxed_runs` on gfx942, where
                the operand is prepared even though the product declines the
                matrix path).  At width one the two agree, which is every
                other prepared order.
                """
                threads = self._num_threads
                view = self._ops[0].symbol.data_view
                rank = view.rank()
                # `width` slots per lane, adjacent: the lane's fragments of one
                # tile, read as one access per part.  That needs the parts
                # planar -- adjacent parts would interleave the slots -- which
                # is how a fragment-ordered operand is stored.
                if width > 1 and parts > 1 and not view.part_plane:
                    raise InternalError(
                        f'{width} adjacent slots of a {parts}-part operand need '
                        f'its parts planar, and they are adjacent')
                # `shift` is a slot count added at run time: a warp shared by
                # several multiplications reads a fragment over all of its
                # lanes, and the lanes past this multiplication's own are
                # `shift` slots of it further on.
                nonlead = slot // (threads * width)
                value = (None if shift is None else
                         writer.op('add', INDEX, shift, nonlead, hint='slot'))
                lower = view.get_dim_offsets()
                index = ([LeadIndex(nonlead, threads, 1, value=value, width=width,
                                    offset=lower[0] * (1 - width))]
                         + [lower[d] for d in range(1, rank)])
                with writer.speculative() as spec:
                    res = self._ops[0].symbol.load(writer, self._context, None,
                                                   index, False, parts=parts)
                    if not res:
                        spec.discard()
                return res


            def B_direct(kblock, nblock):
                """Whether `B_frag` can address every fragment of this shape.

                Asked once, before anything is emitted, because the answer has
                to hold for the whole span: a path that reads some fragments
                from memory and stages the rest would need both, and the tile
                it was trying to remove back.

                Everything it checks is a way the fragment's coordinate would
                stop being `base + lane`, which is the only form a `LeadIndex`
                has.  A second contraction or output axis puts a term between
                the two; a transposed operand swaps which lane bits mean what;
                a slice offset shifts the base off its block, and then no
                `nonlead` names it.  Each of those is refused rather than
                approximated -- the staging path below is correct for all of
                them, and being slower is not the same as being wrong.
                """
                if len(self._ks) != 1 or len(self._ns) != 2:
                    return False
                if self._opdim_to_nks[1] != ['k0', 'n1']:
                    return False
                if any(self._eff_offset(1, d) != 0 for d in range(2)):
                    return False
                if (self._ks[0][0] % kblock or self._ns[1][0] % nblock):
                    return False
                # And the banks.  The fragment reads `(t % kblock) + stride *
                # (t / kblock)`, so the `nblock` lane groups land `stride`
                # apart and collide whenever `stride` shares too much with the
                # bank count: `gcd(stride, 32)` values of `stride * g` repeat,
                # giving `gcd / kblock`-way conflicts.  A window whose row is a
                # multiple of 32 puts every group in the same banks -- 8-way,
                # against the 2-way the staged path reaches with its swizzle.
                #
                # Refused rather than swizzled because the buffer is not this
                # instruction's to permute: it is the staging window some
                # loader filled, and a permutation applied on only one of its
                # two sides is not a permutation.  What lifts this is the
                # window carrying a layout the reader can ask about -- the same
                # thing `strategies` says is missing for fragments.
                strides = self._ops[1].symbol.data_view.get_dim_strides()
                banks = 32
                if math.gcd(int(strides[1]), banks) > kblock * 2:
                    return False
                return True

            def B_frag(writer, j, k, kblock, nblock):
                """One `B` fragment, at the distribution the instruction reads.

                The staging tile it replaces exists for an addressing reason
                and not a hardware one: the coordinate accessor hands lane `t`
                the element at lead index `k + t`, while the fragment wants
                `B[k + t % kblock][n + t / kblock]`.  Both of those are plain
                lane distributions, which is exactly what `LeadIndex` says --
                `block` and `stride` describe which lane holds what -- so the
                address is expressible and every lane reads its own elements
                out of memory.  Nothing crosses lanes, so nothing has to be
                redistributed through shared memory.

                `B_direct` is the guard, and it is the caller's to ask first.
                """
                assert B_direct(kblock, nblock)
                index = [LeadIndex((k + self._ks[0][0]) // kblock, kblock, 1),
                         LeadIndex((j + self._ns[1][0]) // nblock, nblock,
                                   kblock)]
                with writer.speculative() as spec:
                    res = self._ops[1].symbol.load(writer, self._context, None,
                                                   index, False)
                    if not res:
                        spec.discard()
                return res

            def A_zero(e_lo, e_hi, k_lo, k_hi):
                """Whether `A` is known to be zero over lead elements
                `[e_lo, e_hi)` and contraction steps `[k_lo, k_hi)`
                (`MatmulOperands.A_zero`, `_zero_block`)."""
                return self._zero_block(e_lo, e_hi, k_lo, k_hi, span)

            a_obj = self._ops[0].symbol.obj
            self._offer_order(_vendor_module(self._context),
                              a_obj, Mx, K)
            uniform = self._a_uniform()
            # One contraction axis, one scalar an element, dense and in
            # memory: then step `k + p` is `p` strides of that axis further,
            # which is all `A_wave` adds to the address.  A register image
            # has no address to add to.
            k_stride = None
            view = self._ops[0].symbol.data_view
            if (uniform and len(self._ks) == 1 and view is not None
                    and 'k0' in self._opdim_to_nks[0]
                    and getattr(a_obj, 'storage_parts', 1) == 1
                    and a_obj.is_dense()
                    # in its logical order: a prepared buffer is not strided
                    and getattr(a_obj, 'storage_order', None) is None
                    and getattr(a_obj, 'simt_interleave', None) is None
                    and self._ops[0].symbol.stype in (SymbolType.SharedMem,
                                                      SymbolType.Batch,
                                                      SymbolType.Global)):
                k_stride = view.get_dim_strides()[
                    self._opdim_to_nks[0].index('k0')]

            def A_wave(writer, i, k, mults):
                """`A(i, k + p)` in the lanes of the `p`-th of `mults`
                multiplications sharing a wave (`MatmulOperands.A_wave`).

                The caller keeps `k + mults - 1` inside the contraction; a
                masked-off row reads too, at its clamped element, which is the
                same operand since it does not depend on the element.  `None`
                where the steps would wrap around the contraction
                (`unwindK` takes `k` modulo its extent), and the caller then
                reads each step itself.
                """
                size = self._ks[0][1] - self._ks[0][0]
                if k % size + mults > size:
                    return None
                p = writer.op('rem', INDEX, writer.thread_id('y'), mults,
                              hint='m')
                shift = writer.op('mul', INDEX, p, k_stride, hint='r')
                with writer.speculative() as spec:
                    res = self._ops[0].symbol.load(writer, self._context, None,
                                                   unwindOp(i, 0, k, 0, True),
                                                   False, shift=shift)
                    if not res:
                        spec.discard()
                return res
            ops = MatmulOperands(
                A=A, B=B, C=C, sparse=sparse,
                a_parts=getattr(a_obj, 'storage_parts', 1) if a_obj else 1,
                B_frag=B_frag, B_direct=B_direct,
                a_uniform=uniform,
                A_wave=A_wave if k_stride is not None else None,
                lockstep=self.convergence_scope() is not None,
                a_shared=self._ops[0].symbol.stype is SymbolType.SharedMem,
                a_resident=self._ops[0].symbol.stype in (SymbolType.Register,
                                                         SymbolType.Scratch),
                mult_stride=self._ask_mult_stride,
                A_slot=(A_slot if a_obj is not None
                        and getattr(a_obj, 'storage_order', None) is not None
                        # An interleaved operand is ordered for the nest's
                        # lanes, not for a fragment: read by coordinate.
                        and getattr(a_obj, 'simt_interleave', None) is None
                        # Nor is a packed one.  It is the same `(row,
                        # column)` operand with the empty cells left out,
                        # so its slots are not fragments and there is no
                        # quad to read: `Symbol._packed_index` gives the
                        # address, and the path reads it by coordinate like
                        # any other operand.  Offered here, `quad_width`
                        # would make every read four steps wide over a
                        # buffer that has no such run.
                        and getattr(a_obj, 'packing', None) is None
                        else None),
                A_zero=A_zero if width == 1 else None,
                lead_slots=M, lead_elements=Mx, n=N, k=K, kx=kx,
                threads=self._num_threads, lead_width=width,
                a=self._ops[0].symbol.get_fptype(),
                b=self._ops[1].symbol.get_fptype(),
                accumulator=self._idest.get_fptype())

            # A path may find out mid-emission that it cannot serve the shape,
            # and saying so has to leave the body as it found it -- otherwise
            # the generic nest below writes a second set of products on top of
            # a partial one, and both are emitted.
            with writer.speculative() as spec:
                module = _vendor_module(self._context)
                taken = all(module.matmul(writer, ops, self._context, span)
                            for span in plan)
                if not taken:
                    spec.discard()
            return taken
        return False

    def get_operands(self):
        # The scalar factors and the previous value are the epilogue's.
        return [op.symbol for op in self._ops]

    def needs_epilogue(self) -> bool:
        """Whether the accumulator is not the destination.

        A scalar factor, a previous value to add, a neighbor whose layout the
        destination takes, or a destination box the accumulated range does not
        match -- any of them, and the result goes through
        `MultilinearEpilogue` rather than straight into the destination.
        """
        return not (len(self._scalar) == 0 and self._prev is None
                    and self._next is None
                    and self._idest.data_view == self._dest.data_view)

    def epilogue(self):
        """The instruction that writes the destination from this one's
        accumulator, or None where the accumulator already is it."""
        if not self._has_epilogue:
            return None
        from .epilogue import MultilinearEpilogue
        return MultilinearEpilogue(self._context, self._idest, self._dest,
                                   self._scalar, self._prev,
                                   self._prev_offset, self._ns,
                                   self._lead_dims, self._num_threads,
                                   self._lead_width, self._productOperation,
                                   self._sumOperation,
                                   empty_product=len(self._ops) == 0,
                                   beside=self._beside())

    def _beside(self):
        """What the accumulator may hold beside its range, by lead dimension:
        `(before, past)`, whether the lanes before the window and those past
        it hold what the products there came to rather than the zeros the
        accumulator is declared with.

        Only where the window computes on every lane (`_full_lane_tail`).
        Before it, always: a memory read there may stay whole
        (`Symbol._memory_first`), and a register image may hold rows of its
        own.  Past it, where an operand that carries the lead index reads no
        zero there (`_zero_past`)."""
        out = {}
        for i in self._lead_dims:
            if i >= len(self._ns) or not self._full_lane_tail(i, *self._ns[i]):
                continue
            past = any(i in target and not self._zero_past(op, target.index(i))
                       for op, target in zip(self._ops, self._target))
            out[i] = (True, past)
        return out

    @staticmethod
    def _zero_past(region, axis) -> bool:
        """Does the operand `region` read a zero past its rows along its
        `axis`?  Memory does: a read there is held off the lanes past the
        rows (`LeadIndex.valid`).  A register image where its own rows end
        there, and whose producer said that it holds a zero past them
        (`zero_past`: a staging, or a multiplication that leaves those lanes
        alone or wrote zeros there); not where it holds rows past the
        window."""
        sym = region.symbol
        if sym.stype not in (SymbolType.Register, SymbolType.Scratch):
            return True
        if not getattr(sym, 'zero_past', False):
            return False
        end = region.offset[axis] + region.bbox.upper()[axis]
        return sym.data_view.get_bbox().upper()[axis] <= end

    def defs(self):
        # the accumulator, where an epilogue writes the destination from it
        return (self._idest,) if self._has_epilogue else (self._dest,)

    def __str__(self):
        product = f" {self._productOperation} ".join(op.symbol.name for op in self._ops)
        if self._has_epilogue:
            return f'{self._idest.name} = {self._sumOperation}({product})'
        # Where there is no epilogue this form stays exactly as it is, down to
        # the `None` standing for the absent previous value: the comment is
        # part of every kernel's source, and so of its name.
        return f'{self._dest.name} = {self._sumOperation}({product}) {self._sumOperation} {self._prev}' # TODO: dimensions
