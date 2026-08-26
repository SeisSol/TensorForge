# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from . import ComputeInstruction
from tensorforge.backend.symbol import Symbol, SymbolView
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError
from tensorforge.common.operation import ReductionOperator
from typing import List


class ReductionInstruction(ComputeInstruction):
    """``dest = reduce(op, src, dims)``.

    NOTE: the emission path is still a skeleton -- ``_nonlead_reduction`` was
    never reachable, because ``__init__`` was a bare ``pass`` and left every
    field undefined, and the generator called it with five arguments against an
    eight-argument signature.  The constructor is now real, so the instruction
    participates in the data-flow interface and ``gen_code`` fails with a clear
    message instead of an ``AttributeError`` on an unset field.
    """

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
        self._dims = list(dims)
        self._operation = operation
        self._permute = permute
        self._prefer_align = prefer_align
        self._num_threads = num_threads
        self._is_ready = True
        self._gemm_meta_data = None
        self.registers = None

        for view in (dest, src):
            view.symbol.add_user(self)

    def defs(self):
        return (self._dest.symbol,)

    def uses(self):
        return (self._op.symbol,)

    def get_operands(self):
        return [self._op.symbol]

    def gen_code_inner(self, writer: Writer):
        raise InternalError(
            f'reduction is not implemented yet: cannot emit '
            f'{self._operation} over axes {self._dims}')

    def _nonlead_reduction(self, writer: Writer):
        with writer.If(self.gen_mask_threads(self._op.data_view.get_lead_dim())):
            loopstack1 = []
            loopstack2 = []
            for i, dimlen in enumerate(self._op.data_view.get_nonlead_dims()):
                if i not in self._dims:
                    loop = writer.For(f'int32_t k{i} = 0; k{i} < {dimlen}; ++k{i}', True)
                    loop.__enter__()
                    loopstack1 += [loop]

            writer(f'{self._fp_as_str} value = 0;')

            for i, dimlen in enumerate(self._op.data_view.get_nonlead_dims()):
                if i in self._dims:
                    loop = writer.For(f'int32_t k{i} = 0; k{i} < {dimlen}; ++k{i}')
                    loop.__enter__()
                    loopstack2 += [loop]

            address = self._op.data_view.get_address(lead_idx=self._vm.get_lexic().thread_idx_x, nonlead_idx=['k{i}' for i in range(len(self._op.data_view.get_nonlead_dims()))])
            writer(f'{self._fp_as_str} input = {self._op.name}[{address}]')
            writer(f'value = {self._operation.write("value", "input")};')

            for loop in loopstack2[::-1]:
                loop.__exit__(None, None, None)

            res_access = '' if self._dest.obj.size == 1 else '[k]'
            writer(f'{self._dest.name}{res_access} = value;')

            for loop in loopstack1[::-1]:
                loop.__exit__(None, None, None)

    def _lead_reduction(self, writer: Writer):
        # for now: only shmem, and maybe SYCL reductions
        pass

    def __str__(self):
        return (f'{self._dest.symbol.name} = {self._operation}'
                f'({self._op.symbol.name}, dims={self._dims})')
