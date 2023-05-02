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
            writer(f'{self._dest.name}{res_access} += {op1_value} * {val_b[iter]}{self._vm.get_real_literal()};')
      writer.Comment(f"Mul end col {row_id}")
      writer.Emptyline()

  def __str__(self) -> str:
    return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'


class RegisterOnlyDenseSparseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    super(RegisterOnlyDenseSparseGemm, self).__init__(kwargs['vm'])
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
    op1_variable = 'value1'
    op2_variable = 'value2'
    warp_idx_variable = 'wid'

    op1_data_view = self._op1.data_view
    op2_data_view = self._op2.data_view
    thread_idx_x = self._vm.get_lexic().thread_idx_x

    warp_id = self._vm._lexic.get_sub_group_id(self._vec_unit_length)
    writer(f'auto {warp_idx_variable} = {warp_id};')

    writer(f'{self._vm.fp_as_str()} {op1_variable};')
    writer(f'{self._vm.fp_as_str()} {op2_variable};')

    active_sub_group_mask = self._vm._lexic.active_sub_group_mask()
    if active_sub_group_mask:
      sub_group = 'mask'
      writer(f'auto {sub_group} = {active_sub_group_mask};')
    else:
      sub_group = None

    k_end = op1_data_view.rows if self._trans_a else op1_data_view.columns
    num_cols = op1_data_view.columns if self._trans_a else op1_data_view.rows

    writer.Emptyline()
    writer.Pragma('unroll')
    with writer.For(f'int k = 0; k < {k_end}; ++k'):
      with writer.If(self.gen_mask_threads(num_cols)):
        if self._trans_a:
          op1_addr = f'k + {thread_idx_x} * {op1_data_view.lead_dim}'
        else:
          op1_addr = f'{thread_idx_x} + k * {op1_data_view.lead_dim}'
        writer(f'{op1_variable} = {self._op1.name}[{op1_addr}];')

      start_tile_variable = 'startTileN'
      end_tile_variable = 'endTileN'

      writer.Emptyline()
      writer.Pragma('unroll')
      with writer.For(f'int {start_tile_variable} = 0; '
                      f'{start_tile_variable} < {self._dest.obj.size}; '
                      f'{start_tile_variable} += {self._vec_unit_length}'):
        writer(f'int shiftedWid = {start_tile_variable} + {warp_idx_variable};')
        with writer.If(f'shiftedWid < {self._dest.obj.size}'):
          if self._trans_b:
            op2_addr = f'shiftedWid + k * {op2_data_view.lead_dim}'
          else:
            op2_addr = f'shiftedWid * {op2_data_view.lead_dim} + k'
          writer(f'{op2_variable} = {self._op2.name}[{op2_addr}];')

          if active_sub_group_mask:
            writer(f'{sub_group} = {active_sub_group_mask};')

        writer.Emptyline()
        writer(f'int {end_tile_variable} = '
               f'{start_tile_variable} + {self._vec_unit_length};')

        writer(f'{end_tile_variable} = '
               f'({end_tile_variable} < {self._dest.obj.size}) '
               f' ? {end_tile_variable} : {self._dest.obj.size};')

        writer.Emptyline()
        self._get_inner_loop(writer,
                             op1_variable,
                             op2_variable,
                             start_tile_variable,
                             end_tile_variable,
                             sub_group)

  def _get_inner_loop(self,
                      writer,
                      op1_variable,
                      op2_variable,
                      start,
                      end,
                      sub_group):
    writer.Pragma('unroll')
    with writer.For(f'int n = {start}, broadcastIdx = 0; '
                    f'n < {end}; '
                    f'++n, ++broadcastIdx'):
      tmp_value = 'tmp'
      broadcast_sync = self._vm._lexic.broadcast_sync(op2_variable,
                                                      "broadcastIdx",
                                                      sub_group)
      writer(f'auto {tmp_value} = {broadcast_sync};')

      res_access = '' if self._dest.obj.size == 1 else '[n]'
      writer(f'{self._dest.name}{res_access} += {op1_variable} * {tmp_value};')

  def __str__(self) -> str:
    return f'{self._dest.name} = rb_gemm({self._op1.name}, {self._op2.name})'