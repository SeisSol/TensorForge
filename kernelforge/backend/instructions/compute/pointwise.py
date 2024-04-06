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
from kernelforge.generators.optree import Assignment, writeAssignments

class MultilinearInstruction(ComputeInstruction):
    def __init__(self,
               context: Context,
               assignments: List[Assignment],
               productOperation: ReductionOperator,
               sumOperation: ReductionOperator,
               prefer_align: bool,
               num_threads: int):
        super(MultilinearInstruction, self).__init__(context)
        self._assignments = assignments
        self._productOperation = productOperation
        self._sumOperation = sumOperation
        self._prefer_align = prefer_align
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self._num_threads = num_threads

        self.registers = None

        # TODO: get index list
        seen_tensors = set()
        for assignment in self._assignments:
          assignment.assignSymbols(self.TODO)
          for tensor in assignment.symbols():
            if tensor not in seen_tensors:
              tensor.add_user(self)
              seen_tensors.add(tensor)
              if not isinstance(op.obj, Tensor):
                raise InternalError('gemm: op1 is not a matrix')

    def gen_code_inner(self, writer: Writer):
        self._assignment_loop(writer)

    def _assignment_loop(self, writer: Writer):
        loopstack = []

        for i, (dimmin, dimmax) in enumerate(self._ns):
            if i not in self._lead_dims:
                writer.insert_pragma_unroll()
                loop = writer.For(f'int n{i} = {dimmin}; n{i} < {dimmax}; ++n{i}')
                loop.__enter__()
                loopstack += [loop]

        writeAssignments(self._assignments, writer, self._context)

        for loop in loopstack[::-1]:
            loop.__exit__(None, None, None)

    def get_operands(self):
        return self._ops

    def __str__(self):
        return f'{self._dest.name} = {self._sumOperation}({",".join(op.name for op in self._ops)})' # TODO: dimensions
