from abc import abstractmethod
from typing import Union
import enum
import math
from kernelforge.common.matrix.tensor import Tensor
from . import ComputeInstruction
from kernelforge.backend.symbol import SymbolType, Symbol, DataView
from kernelforge.common.exceptions import InternalError
from kernelforge.backend.writer import Writer
from kernelforge.common.context import Context
from kernelforge.common.operation import ReductionOperator
from typing import Union, List

class MultilinearInstruction(ComputeInstruction):
    def __init__(self,
               context: Context,
               dest: Symbol,
               ops: List[Symbol],
               target: List[List[int]],
               productOperation: ReductionOperator,
               sumOperation: ReductionOperator,
               prefer_align: bool,
               num_threads: int):
        super(MultilinearInstruction, self).__init__(context)
        self._dest = dest
        self._ops = ops
        self._target = target
        self._productOperation = productOperation
        self._sumOperation = sumOperation
        self._prefer_align = prefer_align
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self._num_threads = num_threads

        self.registers = None
        if dest.stype != SymbolType.Register:
            raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
        else:
            self._dest = dest

        for op in self._ops:
            if not isinstance(op.obj, Tensor):
                raise InternalError('gemm: op1 is not a matrix')

            op.add_user(self)
        dest.add_user(self)

        self._analyze()

    def _analyze(self):
        targetrank = 0
        for i, op in enumerate(self._ops):
            for j in range(len(op.data_view.shape)):
                targetrank = max(self._target[i][j] + 1, targetrank)
        self._ns = [(-math.inf, math.inf)] * targetrank
        preKs = {}
        self._opdim_to_nks = []
        for i, op in enumerate(self._ops):
            opdim = [''] * len(op.data_view.shape)
            for j in range(len(op.data_view.shape)):
                if self._target[i][j] < 0:
                    if self._target[i][j] not in preKs:
                        preKs[self._target[i][j]] = (op.data_view.get_bbox().lower()[j], op.data_view.get_bbox().upper()[j])
                    preKs[self._target[i][j]] = (max(preKs[self._target[i][j]][0], op.data_view.get_bbox().lower()[j]), min(preKs[self._target[i][j]][1], op.data_view.get_bbox().upper()[j]))
                    opdim[j] = f'k{-self._target[i][j] - 1}'
                else:
                    self._ns[self._target[i][j]] = (max(self._ns[self._target[i][j]][0], op.data_view.get_bbox().lower()[j]), min(self._ns[self._target[i][j]][1], op.data_view.get_bbox().upper()[j]))
                    opdim[j] = f'n{self._target[i][j]}'
            self._opdim_to_nks += [opdim]
        self._ks = [0] * len(preKs)
        for i in range(len(preKs)):
            assert -i-1 in preKs
            self._ks[i] = preKs[-i-1]
        
        # TODO: handle offsets
        self._dest.data_view = DataView(shape = [u - l for l,u in self._ns], permute=[i for i in range(targetrank)])

    def gen_code_inner(self, writer: Writer):
        with writer.If(self.gen_mask_threads(self._dest.data_view.shape[0])):
            loopstack = []

            for i, (dimmin, dimmax) in enumerate(self._ns[1:]):
                writer.insert_pragma_unroll()
                loop = writer.For(f'int n{i+1} = {dimmin}; n{i+1} < {dimmax}; ++n{i}')
                loop.__enter__()
                loopstack += [loop]

            for i, (dimmin, dimmax) in enumerate(self._ks):
                loop = writer.For(f'int k{i} = {dimmin}; k{i} < {dimmax}; ++k{i}')
                loop.__enter__()
                loopstack += [loop]

            for i, op in enumerate(self._ops):
                op.load(writer, self._context, f'data{i}', [self._vm.get_lexic().thread_idx_x if nk == 'n0' else nk for nk in self._opdim_to_nks[i]], False)
                if i > 0:
                    writer(f'{self._fp_as_str} prod{i} = {self._productOperation.format(f"prod{i-1}", f"data{i}")};')
                else:
                    writer(f'{self._fp_as_str} prod{i} = data{i};')
            if len(self._ops) > 0:
                self._dest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
                writer(f'{self._fp_as_str} newvalue = {self._sumOperation.format("value", f"prod{len(self._ops)-1}")};')
                self._dest.store(writer, self._context, 'newvalue', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)

            for loop in loopstack[::-1]:
                loop.__exit__(None, None, None)

    def get_operands(self):
        return self._ops

    def __str__(self):
        return f'{self._dest.name} = {self._sumOperation}({",".join(op.name for op in self._ops)})' # TODO: dimensions
