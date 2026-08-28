# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""One scalar operation applied pointwise over a tensor.
"""

from typing import List, Sequence, Union

import numpy as np

from tensorforge.backend.scopes import Scopes
from tensorforge.backend.symbol import (LeadLoop, Loop, Symbol, SymbolView,
                                        write_loops, VarOffset)
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.operation import Operation

from . import ComputeInstruction


ScalarLike = (int, float, np.integer, np.floating)


class ElementwiseInstruction(ComputeInstruction):
    def __init__(self,
                 context: Context,
                 op: Operation,
                 dest: SymbolView,
                 srcs: Sequence[Union[SymbolView, int, float]],
                 prefer_align: bool,
                 num_threads: int):
        super(ElementwiseInstruction, self).__init__(context)
        self._op = op
        self._dest = dest
        self._srcs = list(srcs)
        self._prefer_align = prefer_align
        self._num_threads = num_threads
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self.registers = None

        for view in self._tensor_srcs() + [self._dest]:
            if not isinstance(view.symbol.obj, Tensor):
                raise InternalError('elementwise: operand is not a tensor')

        # From the symbols, not assumed to be axis 0.  Every operand shares
        # the destination's shape -- `ElementwiseDescr` checks that -- so
        # iteration axis i is axis i of each of them, and the lane axis has to
        # be the one they all declare.  `Symbol.load` and `Symbol.store`
        # already address through `lead_dims`; iterating a different axis here
        # would spread one and address the other.
        self._lead_dims = [self.shared_lead_dim(
            [self._dest] + self._tensor_srcs(), 'elementwise')]

        # The iteration space is the destination's, not something unified over
        # a list of assignments: elementwise means the operands share its shape,
        # which ElementwiseDescr checks at construction.  SymbolView carries
        # exactly the (symbol, bbox) pair that optree's TensorVar duplicated.
        bbox = self._dest.bbox
        self._ks = [(0, bbox.size(i)) for i in range(bbox.rank())]

        seen = set()
        for view in [self._dest] + self._tensor_srcs():
            if id(view.symbol) not in seen:
                seen.add(id(view.symbol))
                view.symbol.add_user(self)

    # -- data flow ------------------------------------------------------- #

    def _tensor_srcs(self) -> List[SymbolView]:
        return [s for s in self._srcs if not isinstance(s, ScalarLike)]

    def defs(self):
        return (self._dest.symbol,)

    def uses(self):
        return tuple(v.symbol for v in self._tensor_srcs())

    def get_operands(self):
        # Previously this returned [] with a "TODO: for now", which made every
        # elementwise operand invisible to liveness and to barrier insertion.
        # It now agrees with uses().
        return [v.symbol for v in self._tensor_srcs()]

    # -- emission -------------------------------------------------------- #

    def gen_code_inner(self, writer: Writer):
        loopstack = []
        for i, (dimmin, dimmax) in enumerate(self._ks):
            if i in self._lead_dims:
                loopstack.append(LeadLoop(f'k{i}', dimmin, dimmax,
                                          self._num_threads, 1, unroll=False))
            else:
                loopstack.append(Loop(f'k{i}', dimmin, dimmax, 1, unroll=False))

        write_loops(self._context, writer, loopstack, self._body(writer))

    @staticmethod
    def _index(view: SymbolView, varlist) -> List:
        # optree emitted `(n{k} + bbox.lower()[k])` as text, which forced a
        # named `n{k}` variable and left the address arithmetic opaque.  A
        # `VarOffset` carries the loop value itself, so the offset folds and
        # the whole address becomes IR.
        return [VarOffset(varlist[i], o) if o else varlist[i]
                for i, o in enumerate(view.bbox.lower())]

    def _body(self, writer: Writer):
        from tensorforge.backend.pir.core import ScalarType

        def inner(varlist):
            operands = []
            args = []
            for src in self._srcs:
                if isinstance(src, ScalarLike):
                    operands.append(self._context.fp_type.literal(src))
                    continue
                v = src.symbol.load(writer, self._context, None,
                                    self._index(src, varlist), False)
                if v is None:
                    return self._body_named(writer, varlist)
                operands.append('{%d}' % len(args))
                args.append(v)

            # get_operation always takes two values; unary ops pass '' as the
            # second.  Handing it placeholders instead of names keeps the
            # operand order but lets the emitter fill in whatever the value
            # ends up being called -- or inline it entirely.
            padded = operands + [''] if len(operands) == 1 else operands
            lexic = self._context.get_vm().get_lexic()
            text = lexic.get_operation(self._op, self._context.fp_type, *padded)
            result = writer.rawexpr(text, *args,
                                    type_=ScalarType(self._context.fp_type),
                                    hint='e', pure=True, movable=True)
            self._dest.symbol.store(writer, self._context, result,
                                    self._index(self._dest, varlist), False)

        return inner

    def _body_named(self, writer, varlist):
        """Fallback for operands that cannot yield a value (sparse loads)."""
        for i, _ in enumerate(self._ks):
            writer(f'auto n{i} = {varlist[i].write(self._context)};')
        operands: List[str] = []
        counter = 0
        for src in self._srcs:
            if isinstance(src, ScalarLike):
                operands.append(self._context.fp_type.literal(src))
                continue
            var = f'v{counter}'
            counter += 1
            src.symbol.load(writer, self._context, var,
                            [f'(n{i} + {o})'
                             for i, o in enumerate(src.bbox.lower())], False)
            operands.append(var)
        padded = operands + [''] if len(operands) == 1 else operands
        lexic = self._context.get_vm().get_lexic()
        result = f'v{counter}'
        writer(f'const auto {result} = '
               f'{lexic.get_operation(self._op, self._context.fp_type, *padded)};')
        self._dest.symbol.store(writer, self._context, result,
                                [f'(n{i} + {o})'
                                 for i, o in enumerate(self._dest.bbox.lower())],
                                False)

    def __str__(self):
        def render(s):
            return s.symbol.name if isinstance(s, SymbolView) else str(s)
        args = ', '.join(render(s) for s in self._srcs)
        return f'{self._dest.symbol.name} = {self._op.name.lower()}({args})'
