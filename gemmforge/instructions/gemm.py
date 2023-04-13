from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
from abc import abstractmethod


class GenericGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    super(GenericGemm, self).__init__(kwargs['vm'])
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
    self.counter = 0

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
      if self._sparse_b:
        for k in range(0, op1_data_view.columns):
          # with writer.For(f'int k = 0; k < {op1_data_view.columns}; ++k'):
          op1_addr = f'{thread_idx_x} + {k} * {op1_data_view.lead_dim}'
          if len(self._coo_b[0][k]) > 0:
            writer(f'{value_var} = {self._op1.name}[{op1_addr}];')
            writer.Emptyline()

            #if not self._val_b:
            self._get_inner_loop_sparse(writer, value_var, k, self._coo_b[0][k], False, None)
            #else:
            #  self._get_inner_loop_sparse(writer, value_var, k, self._coo_b[0][k], True, self._val_b) 
      else:
        with writer.For(f'int k = 0; k < {op1_data_view.columns}; ++k'):
          op1_addr = f'{thread_idx_x} + k * {op1_data_view.lead_dim}'
          writer(f'{value_var} = {self._op1.name}[{op1_addr}];')

          writer.Emptyline()
          self._get_inner_loop(writer, value_var)

  def _get_inner_loop(self, writer, op1_value):
    op2_data_view = self._op2.data_view
    writer.Pragma('unroll')
    with writer.For(f'int n = 0; n < {self._dest.obj.size}; ++n'):
      if self._trans_b:
        op2_addr = f'n + {op2_data_view.lead_dim} * k'
      else:
        op2_addr = f'k + {op2_data_view.lead_dim} * n'

      res_access = '' if self._dest.obj.size == 1 else '[n]'
      writer(f'{self._dest.name}{res_access} += {op1_value} * {self._op2.name}[{op2_addr}];')

  def _get_inner_loop_sparse(self, writer, op1_value, col_id, non_zeros, value_known=False, val_b=None):
    if len(non_zeros) > 0:
      op2_data_view = self._op2.data_view
      writer.Comment(f"Mul begin col {col_id}")
      for non_zero in non_zeros:
        """
        # If the sparse would be saved as if the full array then we would do it like here
        if self._trans_b:
            op2_addr = f'{non_zero} + {op2_data_view.lead_dim} * {col_id}'
        else:
            op2_addr = f'{col_id} + {op2_data_view.lead_dim} * {non_zero}'
        """

        # Counter is in this case the number of the non-zero element
        # If we found a matrix address (i,j) that corresponds to (non_zero, col_id)
        # Then we write its offset in the array that will be provided and access it
        counter = 0
        if self._trans_b:
          coo_to_use = self._coo_b[0] #col major iter coo
          val_b_to_use = self._val_b[0]
          col = 0
          found = False
          for row_ids in coo_to_use:
            for row in row_ids:
              if row == non_zero and col == col_id:
                found = True
              if not found:
                counter += 1
            col += 1
          if not found:
            raise Exception("oh no")
        else:
          coo_to_use = self._coo_b[1] #row major iter coo
          val_b_to_use = self._val_b[1]
          row = 0
          found = False
          for col_ids in coo_to_use:
            for col in col_ids:
              if row == non_zero and col == col_id:
                found = True
              if not found:
                counter += 1
            row += 1
          if not found:
            raise Exception("oh no")

        res_access = f"[{non_zero}]"
        if not value_known:
            writer(f'{self._dest.name}{res_access} += {op1_value} * {self._op2.name}[{counter}];')
        else:
            writer(f'{self._dest.name}{res_access} += {op1_value} * {val_b_to_use[counter]};')
      writer.Comment(f"Mul end col {col_id}")
      writer.Emptyline()

  def __str__(self) -> str:
    return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'
