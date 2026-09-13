# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A contraction whose destination has no axes: one value per element.

`rhoInv[] = materialParameters[z] * pickRhoInv[z]` has nothing to spread over
the lanes.  The destination is one number, and so is everything a lane would
hold of it.  Every lane computes it, from the same operands, in sequential
loops over the contracted indices: the value is uniform, needs no exchange and
no barrier, and stays in a register unless an operation that cannot read one
asks for it.

The multilinear path distributes its destination's axis 0, which this
destination does not have, so it is not bent to fit.  The loops here are the
contraction written out, which is all a lead of one element would have left of
it anyway.
"""

from typing import List

from tensorforge.backend.symbol import SymbolType, Variable, add_offset
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError
from tensorforge.common.operation import AddOperator, MulOperator

from . import ComputeInstruction


class ScalarContractionInstruction(ComputeInstruction):
    def __init__(self,
                 context: Context,
                 dest,
                 ops: List,
                 target,
                 add: bool,
                 num_threads: int):
        super().__init__(context)
        if dest.bbox.rank() != 0:
            raise InternalError(
                f'scalar contraction: {dest.symbol.name} has axes; it is the '
                f'multilinear path\'s')
        self._dest = dest
        self._ops = list(ops)
        self._target = [list(t) for t in target]
        self._add = add
        self._num_threads = num_threads
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self.registers = None
        self.claim_destination(self._dest)
        self.check_addressable(self._ops + [self._dest], 'scalar contraction')
        seen = set()
        for view in [self._dest] + self._ops:
            if id(view.symbol) not in seen:
                seen.add(id(view.symbol))
                view.symbol.add_user(self)

        #: Every contracted index, over what all of its operands support.
        self._ranges = {}
        for view, axes in zip(self._ops, self._target):
            for j, axis in enumerate(axes):
                if axis >= 0:
                    raise InternalError(
                        f'scalar contraction: {view.symbol.name} is indexed '
                        f'by destination axis {axis}, and the destination has '
                        f'none')
                lo, hi = view.bbox.lower()[j], view.bbox.upper()[j]
                prev = self._ranges.get(axis)
                self._ranges[axis] = ((max(prev[0], lo), min(prev[1], hi))
                                      if prev else (lo, hi))
        self._axes = sorted(self._ranges, reverse=True)

    def defs(self):
        return (self._dest.symbol,)

    def uses(self):
        out = tuple(v.symbol for v in self._ops)
        return out + ((self._dest.symbol,) if self._add else ())

    def get_operands(self):
        return [v.symbol for v in self._ops]

    def gen_code_inner(self, writer: Writer):
        from tensorforge.backend.pir.core import ScalarType
        fp = ScalarType(self._context.fp_type)
        total = self._fold(writer, {}, 0, fp)
        if self._add:
            old = self._load(writer, self._dest, [])
            total = writer.op(AddOperator().irop(), fp, old, total, hint='s',
                              pure=True)
        self._dest.symbol.store(writer, self._context, total, [], False)

    def _fold(self, writer: Writer, index: dict, depth: int, fp):
        """One `for` per contracted index, carrying the sum."""
        if depth == len(self._axes):
            return self._product(writer, index, fp)
        axis = self._axes[depth]
        lo, hi = self._ranges[axis]
        if lo >= hi:
            return writer.const(0.0, fp)
        loop = writer.for_(lo, hi, 1, inits=(0.0,), types=(fp,), unroll=True,
                           hint=f's{-axis - 1}')
        with loop:
            index[axis] = Variable(str(loop.induction), Datatype.I32,
                                   loop.induction)
            inner = self._fold(writer, index, depth + 1, fp)
            loop.yield_(writer.op(AddOperator().irop(), fp,
                                  loop.iter_args[0], inner, hint='s',
                                  pure=True))
        del index[axis]
        return loop.result

    def _product(self, writer: Writer, index: dict, fp):
        value = None
        for view, axes in zip(self._ops, self._target):
            factor = self._load(writer, view, [index[a] for a in axes])
            value = factor if value is None else writer.op(
                MulOperator().irop(), fp, value, factor, hint='s', pure=True)
        return value if value is not None else writer.const(1.0, fp)

    def _load(self, writer: Writer, view, coords):
        """`view` at `coords`, in its own index space."""
        from tensorforge.backend.pir.core import ScalarType
        if view.symbol.stype == SymbolType.Scalar:
            # a named factor, passed by value
            return writer.rawexpr(view.symbol.name,
                                  type_=ScalarType(self._context.fp_type),
                                  hint='s', pure=True, movable=True)
        offset = list(getattr(view, 'offset', None) or [0] * len(coords))
        value = view.symbol.load(writer, self._context, None,
                                 [add_offset(c, o)
                                  for c, o in zip(coords, offset)], False)
        if value is None:
            raise InternalError(
                f'scalar contraction: {view.symbol.name} has no structured '
                f'load on this backend, so there is no value to multiply')
        return value

    def __str__(self):
        ops = ' × '.join(v.symbol.name for v in self._ops)
        return f'{self._dest.symbol.name} {"+" if self._add else ""}= {ops}'
