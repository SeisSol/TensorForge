from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
from abc import abstractmethod


class ShrMemBasedSparseDenseGemm(AbstractInstruction):
    def __init__(self, **kwargs):
        super(ShrMemBasedSparseDenseGemm, self).__init__(kwargs['vm'])
        self._trans_a = kwargs['trans_a']
        self._trans_b = kwargs['trans_b']
        self._op1 = kwargs['op1']
        self._op2 = kwargs['op2']
        self._dest = kwargs['dest']
        self._num_threads = kwargs['num_threads']
        self._mat_a = kwargs['mat_a']

        if self._op1.stype == SymbolType.Batch:
            raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

        if self._op2.stype == SymbolType.Batch:
            raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

        if self._dest.stype != SymbolType.Register:
            raise InternalError('gemm: `dest` must be a register obj.')

        self._is_ready = True

    def _get_inner_loop(self, writer, op2_value):
        op1_data_view = self._op1.data_view
        writer.Pragma('unroll')
        with writer.For(f'int m = 0; m < {op1_data_view.rows}; ++m'):
            if self._trans_a:
                raise Exception("TODO")
            op1_addr = f'm + {op1_data_view.rows} * k'

            res_access = '' if self._dest.obj.size == 1 else '[m]'
            writer(f'{self._dest.name}{res_access} += {op2_value} * {self._op1.name}[{op1_addr}];')

    def gen_code(self, writer):
        value_var = 'value'
        op1_data_view = self._op1.data_view
        op2_data_view = self._op2.data_view
        thread_idx_x = self._vm.get_lexic().thread_idx_x
        with writer.If(self.gen_mask_threads(op2_data_view.lead_dim)):
            writer(f'{self._vm.fp_as_str()} {value_var};')

            writer.Emptyline()
            if not self._trans_a:
                for k in range(0, op2_data_view.rows):
                    non_zeros = self._mat_a.get_coo_col_major()[k]
                    if len(non_zeros) == 0:
                        continue

                    if self._trans_b:
                        op2_addr = f'{k} * {op2_data_view.lead_dim} + {thread_idx_x}'
                    else:
                        op2_addr = f'{k} + {thread_idx_x} * {op2_data_view.rows}'
                    writer(f'{value_var} = {self._op2.name}[{op2_addr}];')

                    writer.Emptyline()
                    self._get_inner_loop_sparse_with_a_col(writer, value_var, k, non_zeros, self._mat_a.get_values())
            else:
                for k in range(0, op2_data_view.rows):
                    non_zeros = self._mat_a.get_coo_row_major()[k]
                    if len(non_zeros) == 0:
                        continue

                    if self._trans_b:
                        op2_addr = f'{k} * {op2_data_view.lead_dim} + {thread_idx_x}'
                    else:
                        op2_addr = f'{k} + {thread_idx_x} * {op2_data_view.rows}'
                    writer(f'{value_var} = {self._op2.name}[{op2_addr}];')

                    writer.Emptyline()
                    self._get_inner_loop_sparse_with_a_col(writer, value_var, k, non_zeros, self._mat_a.get_values())               

    def _get_inner_loop_sparse_with_a_col(self, writer, op2_value, col_id, non_zeros, val_a=None):
        # Iterate the first column first then the second etc. (coo_b[0] if col major, otherwise coo_b[1] if row major)
        # As we iterate we need to find the element in the real ordering (coordiantes)
        # This function iterates a column until the end
        if self._trans_a:
            if len(non_zeros) > 0:
                value_known = val_a != None
                transposed_row_id = col_id
                writer.Comment(f"Mul begin col {transposed_row_id} (would be a row of non transposed matrix)")

                
                for col_id in non_zeros:
                    iter = self._mat_a.find_1d_offset(transposed_row_id, col_id)
                    res_access = f"[{col_id}]"

                    if not value_known:
                        writer(f'{self._dest.name}{res_access} += {op2_value} * {self._op1.name}[{iter}];')
                    else:
                        writer(f'{self._dest.name}{res_access} += {op2_value} * {val_a[iter]}{self._vm.get_real_literal()};')

                writer.Comment(f"Mul end col {transposed_row_id} (would be a row of non transposed matrix)")
                writer.Emptyline()
        else:
            if len(non_zeros) > 0:
                value_known = val_a != None
                writer.Comment(f"Mul begin col {col_id}")

                for row_id in non_zeros:
                    iter = self._mat_a.find_1d_offset(row_id, col_id)
                    res_access = f"[{row_id}]"

                    if not value_known:
                        writer(f'{self._dest.name}{res_access} += {op2_value} * {self._op1.name}[{iter}];')
                    else:
                        writer(f'{self._dest.name}{res_access} += {op2_value} * {val_a[iter]}{self._vm.get_real_literal()};')

                writer.Comment(f"Mul end col {col_id}")
                writer.Emptyline()

    def __str__(self) -> str:
        return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'


class RegisterOnlySparseDenseGemm(AbstractInstruction):
    def __init__(self, **kwargs):
        super(RegisterOnlySparseDenseGemm, self).__init__(kwargs['vm'])
        self._trans_a = kwargs['trans_a']
        self._trans_b = kwargs['trans_b']
        self._op1 = kwargs['op1']
        self._op2 = kwargs['op2']
        self._dest = kwargs['dest']
        self._num_threads = kwargs['num_threads']
        self._mat_a = kwargs['mat_a']

        if self._op1.stype == SymbolType.Batch:
            raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

        if self._op2.stype == SymbolType.Batch:
            raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

        if self._dest.stype != SymbolType.Register:
            raise InternalError('gemm: `dest` must be a register obj.')

        self._is_ready = True

    def gen_code(self, writer):
        raise Exception("Register Only Sparse x Dense Matrix Implementation is not yet implemented")

    def __str__(self) -> str:
        return f'{self._dest.name} = rb_gemm({self._op1.name}, {self._op2.name})'
