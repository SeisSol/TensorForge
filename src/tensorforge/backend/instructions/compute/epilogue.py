# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a multiplication does with its result once it has one.

`C = alpha * (A x B) + C'`.  The contraction accumulates `A x B` over the range
its operands support; this writes the destination from it -- scaled by the
product of the scalar operands, added onto the destination's previous value,
and over the destination's whole box, with the sum's neutral element where the
accumulated range does not reach.  GEMM libraries call it the epilogue.

A contraction whose every operand is a scalar -- a broadcast, `t[i] = s[]` --
has moved all of them here and accumulated nothing.  What it would have
accumulated is the empty product, the product's neutral element, and it is the
same at every point of the box; reading the accumulator instead read the sum's
neutral element, and every broadcast came out zero.

It was the tail of `MultilinearInstruction` (`_apply_linear`), which made the
accumulator and the destination two views inside one instruction.  Apart, the
contraction writes exactly what it accumulates, and this is the one place the
two boxes meet.
"""

from typing import List, Optional

from tensorforge.backend.symbol import (Immediate, LeadIndex, LeadLoop, Loop,
                                        add_offset, write_loops)
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context

from . import ComputeInstruction


def _emit_binop(writer, ftype, operator, a, b):
    """`operator` as an IR op if it has one, else its format string."""
    name = operator.irop()
    if name is not None:
        return writer.op(name, ftype, a, b, hint='p')
    return writer.rawexpr(operator.format('{0}', '{1}'), a, b,
                          type_=ftype, hint='p', pure=True, movable=True)


def _splat(writer, ftype, v):
    """A scalar broadcast into every component of a lead-width vector.

    The same packing `MultilinearInstruction._splat` does for `B`: the
    generated vector types have no scalar-times-vector.
    """
    if ftype.length is None:
        return v
    if getattr(getattr(v, 'type', None), 'length', None) is not None:
        return v
    return writer.pack(ftype, *([v] * ftype.length), hint='splat')


class MultilinearEpilogue(ComputeInstruction):
    """`dest = scalars * accumulator (+ prev)` over the destination's box.

    Registers no users of its own: the accumulator shares its user list with
    the destination it was copied from, and a consumer's count of that list
    decides other things (a nontemporal store, for one).
    """

    def __init__(self,
                 context: Context,
                 accumulator,
                 dest,
                 scalars: List,
                 prev,
                 prev_offset: Optional[List],
                 ranges: List,
                 lead_dims: List[int],
                 num_threads: int,
                 lead_width: int,
                 product_operation,
                 sum_operation,
                 empty_product: bool = False):
        super().__init__(context)
        self._acc = accumulator
        self._dest = dest
        self._scalars = list(scalars)
        self._prev = prev
        self._prev_offset = prev_offset
        self._ns = list(ranges)
        self._lead_dims = list(lead_dims)
        self._num_threads = num_threads
        self._lead_width = lead_width
        self._productOperation = product_operation
        self._sumOperation = sum_operation
        #: The contraction had no tensor operand, so what it accumulated is
        #: the empty product rather than anything in the accumulator.
        self._empty_product = empty_product
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self.registers = None

    def defs(self):
        return (self._dest,)

    def get_operands(self):
        out = [self._acc] + [s.symbol for s in self._scalars]
        return out + ([self._prev] if self._prev is not None else [])

    def gen_code_inner(self, writer: Writer):
        from tensorforge.backend.pir.core import ScalarType
        ftype = ScalarType(self._acc.get_fptype())

        if len(self._scalars) > 0:
            scalar_var = self._scalars[0].symbol.load(writer, self._context, None, [], False)
            assert scalar_var is not None

            for scalar in self._scalars[1:]:
                scalar_add = scalar.symbol.load(writer, self._context, None, [], False)
                scalar_var = _emit_binop(writer, ftype, self._productOperation, scalar_add, scalar_var)
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

            dimmini = self._acc.data_view.get_bbox().lower()[i]
            dimmaxi = self._acc.data_view.get_bbox().upper()[i]

            unroll = dimmini != dimmin or dimmaxi != dimmax
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=unroll)]
            else:
                # Same width as the contraction's nest.  This one walks the
                # *same* register image -- it is the beta/prologue pass over
                # the destination -- so a cyclic walk here and a blocked one
                # there disagree about which lane owns which element.
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride,
                                       unroll=unroll, width=self._lead_width)]
                threads //= max(1, -(-(dimmax - dimmin) // self._lead_width))
                stride *= dimmax - dimmin

        def _dim_covered(i, var):
            """Is position `var` of this dim inside the accumulator's coverage?

            Static shortcut first: if the accumulator's bounds for this
            dimension already contain dest's *whole* iteration range, the
            answer is yes no matter where in that range the current lane sits
            --- true statically, no need to inspect `var` at all.

            The dynamic fallback (`.lead()`, a block-start value: `nonlead *
            block`, always a multiple of the block size) only agrees with true
            per-lane containment while the accumulator's lower bound is itself
            a multiple of that block size.  A theta-shifted accumulator's
            bounds need not be: theta is chosen mod num_threads for lane
            alignment, but here `block` is this loop's own per-dimension stride
            factor, which can differ.  Comparing a block-start against raw,
            non-block-aligned bounds silently answered `False` for a lead
            dimension whose coverage was in fact exact, which is exactly the
            static case above already resolves --- so this fallback is only
            reached for the genuinely partial-overlap case it was written for.
            """
            lo_i = self._acc.data_view.get_bbox().lower()[i]
            hi_i = self._acc.data_view.get_bbox().upper()[i]
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
            from tensorforge.backend.symbol import lead_width_of
            # The body's own width, read off the indices exactly as the
            # contraction's nest reads it.  The loads already take theirs from
            # there and come back wide; typing the arithmetic with the
            # instruction's scalar type instead made the sum a scalar, and the
            # store wrote a vector into one register slot -- on CUDA an error,
            # on HIP the same, since a GNU vector does not narrow either.
            width = lead_width_of(
                [varlist[loopmap[f'n{i}']] for i, _ in enumerate(self._ns)])
            btype = (ftype if width == 1
                     else ScalarType(self._acc.get_fptype(), width))
            needsLoad = all(_dim_covered(i, varlist[loopmap[f'n{i}']]) for i,_ in enumerate(self._ns))
            if self._empty_product:
                valvar = _splat(writer, btype, writer.const(
                    self._productOperation.neutral(self._context.fp_type),
                    ftype))
            elif needsLoad:
                valvar = self._acc.load(writer, self._context, None, [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
            else:
                valvar = _splat(writer, btype, writer.const(
                    self._sumOperation.neutral(self._context.fp_type),
                    ftype))

            if len(self._scalars) > 0:
                valvar = _emit_binop(writer, btype, self._productOperation, valvar,
                                     _splat(writer, btype, scalar_var))
            if self._prev is not None:
                oldvalue = self._prev.load(writer, self._context, None, [add_offset(varlist[loopmap[f'n{i}']], self._prev_offset[i]) if self._prev_offset else varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
                valvar = _emit_binop(writer, btype, self._sumOperation, oldvalue, valvar)

            self._dest.store(writer, self._context, valvar, [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def __str__(self):
        parts = [self._acc.name] + [s.symbol.name for s in self._scalars]
        text = f' {self._productOperation} '.join(parts)
        if self._prev is not None:
            text += f' {self._sumOperation} {self._prev.name}'
        return f'{self._dest.name} = {text}'
