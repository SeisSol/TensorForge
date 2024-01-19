from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
from abc import abstractmethod


class ShrMemBasedDenseSparseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    super(ShrMemBasedDenseSparseGemm, self).__init__(kwargs['vm'])
    self._trans_a = kwargs['trans_a']
    self._trans_b = kwargs['trans_b']
    self._op1 = kwargs['op1']
    self._op2 = kwargs['op2']
    self._dest = kwargs['dest']
    self._num_threads = kwargs['num_threads']
    self._mat_b = kwargs['mat_b']

    if self._op1.stype == SymbolType.Batch:
      raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

    if self._op2.obj.get_values() == None and self._op2.stype == SymbolType.Batch:
      raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

    if self._dest.stype != SymbolType.Register:
      raise InternalError('gemm: `dest` must be a register obj.')

    if self._trans_b:
      raise Exception('Sparse Matrix is not supported to be transposed (provide the storage order with the list of coordinates).')

    self._is_ready = True

  def gen_code(self, writer):
    value_var = 'value'
    op1_data_view = self._op1.data_view
    op2_data_view = self._op2.data_view
    thread_idx_x = self._vm.get_lexic().thread_idx_x
    with writer.If(self.gen_mask_threads(op1_data_view.rows)):
      writer(f'{self._vm.fp_as_str()} {value_var};')

      writer.Emptyline()
      for k in range(0, op1_data_view.columns):
        if not self._trans_b:
          non_zeros = self._mat_b.get_coo_per_row()[k]
        else:
          non_zeros = self._mat_b.get_coo_per_col()[k]
        if len(non_zeros) == 0:
          continue

        op1_addr = f'{thread_idx_x} + {k} * {op1_data_view.lead_dim}'
        writer(f'{value_var} = {self._op1.name}[{op1_addr}];')
        writer.Emptyline()

        self._get_inner_loop_sparse_with_a_row(writer, value_var, k, self._mat_b.get_values())

  def _get_inner_loop_sparse_with_a_row(self, writer, op1_value, row_id, val_b=None):
    # Iterate the first column first then the second etc. (coo_b[0] if col major, otherwise coo_b[1] if row major)
    # As we iterate we need to find the element in the real ordering (coordiantes)
    # This function iterates a column until the end
    if not self._trans_b:
      non_zeros = self._mat_b.get_coo_per_row()[row_id]
    else:
      non_zeros = self._mat_b.get_coo_per_col()[row_id]
    if len(non_zeros) > 0:
      value_known = val_b != None
      writer.Comment(f"Mul begin col {row_id}")

      if not self._trans_b:
        for col_id in non_zeros:
          it = self._mat_b.find_1d_offset(row_id, col_id)
          res_access = f"[{col_id}]"

          if not value_known:
            writer(f'{self._dest.name}{res_access} += {op1_value} * {self._op2.name}[{it}];')
          else:
            writer(
              f'{self._dest.name}{res_access} += {op1_value} * {val_b[it]}{self._vm.get_real_literal()};')
      else:
        col_id = row_id
        for row_id in non_zeros:
          it = self._mat_b.find_1d_offset(row_id, col_id)
          res_access = f"[{row_id}]"

          if not value_known:
            writer(f'{self._dest.name}{res_access} += {op1_value} * {self._op2.name}[{it}];')
          else:
            writer(
              f'{self._dest.name}{res_access} += {op1_value} * {val_b[it]}{self._vm.get_real_literal()};')

      writer.Comment(f"Mul end col {row_id}")
      writer.Emptyline()

  def __str__(self) -> str:
    return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'


class RegisterOnlyDenseSparseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    super(RegisterOnlyDenseSparseGemm, self).__init__(kwargs['vm'])

  def gen_code(self, writer):
    raise Exception("Register Only Sparse-by-Dense Matrix Implementation is not supported.")

  def __str__(self) -> str:
    return f'{self._dest.name} = rb_gemm({self._op1.name}, {self._op2.name})'
