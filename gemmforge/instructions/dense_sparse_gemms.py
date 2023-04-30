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
    self._sparse_a = kwargs['sparse_a']
    self._sparse_b = kwargs['sparse_b']
    self._coo_a = kwargs['coo_a']
    self._coo_b = kwargs['coo_b']
    self._val_a = kwargs['val_a']
    self._val_b = kwargs['val_b']

    if self._coo_b == None:
      raise InternalError("For Dense x Sparse Kernel we need coordinates of Matrix b")

    if self._op1.stype == SymbolType.Batch:
      raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

    if self._op2.stype == SymbolType.Batch:
      raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

    if self._dest.stype != SymbolType.Register:
      raise InternalError('gemm: `dest` must be a register obj.')

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
        op1_addr = f'{thread_idx_x} + {k} * {op1_data_view.lead_dim}'
        writer(f'{value_var} = {self._op1.name}[{op1_addr}];')
        writer.Emptyline()

        self._get_inner_loop_sparse_with_a_row(writer, value_var, k, self._coo_b[1][k], self._val_b)

  def _get_inner_loop_sparse_with_a_row(self, writer, op1_value, row_id, non_zeros, val_b=None):
    # Iterate the first column first then the second etc. (coo_b[0] if col major, otherwise coo_b[1] if row major)
    # As we iterate we need to find the element in the real ordering (coordiantes)
    # This function iterates a column until the end
    if len(non_zeros) > 0:
      value_known = val_b != None
      writer.Comment(f"Mul begin col {row_id}")
      coordinates = self._coo_b[2]

      for col_id in non_zeros:
        (i, j) = (row_id, col_id)
        iter = 0
        for (_i, _j) in coordinates:
          if i == _i and j == _j:
            break
          iter += 1

        res_access = f"[{col_id}]"
        if not value_known:
            writer(f'{self._dest.name}{res_access} += {op1_value} * {self._op2.name}[{iter}];')
        else:
            writer(f'{self._dest.name}{res_access} += {op1_value} * {val_b[iter]};')
      writer.Comment(f"Mul end col {row_id}")
      writer.Emptyline()

  def __str__(self) -> str:
    return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'


class RegisterOnlyDenseSparseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    raise Exception("TODO!")

  def gen_code(self, writer):
    raise Exception("TODO!")

  def __str__(self) -> str:
    raise Exception("TODO!")