from . import ComputeInstruction
from tensorforge.backend.symbol import Symbol
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.operation import ReductionOperator
from typing import List


class ReductionInstruction(ComputeInstruction):
    def __init__(self,
               context: Context,
               dest: Symbol,
               op: Symbol,
               permute: List[int],
               dims: List[int],
               operation: ReductionOperator,
               prefer_align: bool,
               num_threads: int):
        pass

    def _analyze(self):
        
        pass

    def gen_code_inner(self, writer: Writer):
        self._nonlead_reduction()
        self._lead_reduction()
    
    def _nonlead_reduction(self, writer: Writer):
        with writer.If(self.gen_mask_threads(self._op.data_view.get_lead_dim())):
            loopstack1 = []
            loopstack2 = []
            for i, dimlen in enumerate(self._op.data_view.get_nonlead_dims()):
                if i not in self._dims:
                    writer.insert_pragma_unroll()
                    loop = writer.For(f'int k{i} = 0; k{i} < {dimlen}; ++k{i}')
                    loop.__enter__()
                    loopstack1 += [loop]
            
            writer(f'{self._fp_as_str} value = 0;')

            for i, dimlen in enumerate(self._op.data_view.get_nonlead_dims()):
                if i in self._dims:
                    loop = writer.For(f'int k{i} = 0; k{i} < {dimlen}; ++k{i}')
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

    def get_operands(self):
        return [self._op]

    def __str__(self):
        return f'{self._dest.name} = {self._operation}({self._op.name}, {self._dims})' # TODO: dimensions
