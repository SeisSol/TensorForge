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

        self._get_inner_loop_sparse_with_a_row(writer, value_var, k, self._mat_b.get_coo_row_major()[k], self._mat_b.get_values())

  def _get_inner_loop_sparse_with_a_row(self, writer, op1_value, row_id, non_zeros, val_b=None):
    # Iterate the first column first then the second etc. (coo_b[0] if col major, otherwise coo_b[1] if row major)
    # As we iterate we need to find the element in the real ordering (coordiantes)
    # This function iterates a column until the end
    if len(non_zeros) > 0:
      value_known = val_b != None
      writer.Comment(f"Mul begin col {row_id}")
      coordinates = self._mat_b.get_coordinates()

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
    self._mat_b = kwargs['mat_b']

    if self._op1.stype == SymbolType.Batch:
      raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

    if self._op2.stype == SymbolType.Batch:
      raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

    if self._dest.stype != SymbolType.Register:
      raise InternalError('gemm: `dest` must be a register obj.')

    self._is_ready = True


  def gen_code(self, writer):
    self._val_b = self._mat_b.get_values()
    value_known = self._val_b != None
    op1_variable = 'value1'
    op2_variable = 'value2'
    warp_idx_variable = 'wid'
    self._vec_unit_length = self._vm.get_hw_descr().vec_unit_length

    op1_data_view = self._op1.data_view
    op2_data_view = self._op2.data_view
    thread_idx_x = self._vm.get_lexic().thread_idx_x

    warp_id = self._vm._lexic.get_sub_group_id(self._vec_unit_length)
    writer(f'auto {warp_idx_variable} = {warp_id};')

    writer(f'{self._vm.fp_as_str()} {op1_variable};')
    writer(f'{self._vm.fp_as_str()} {op2_variable};')
    writer(f'{self._vm.fp_as_str()} tmp;')
    writer(f'int shiftedWid;')

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
    for k in range(k_end):
      writer.Comment(f"Begin Column {k} of A")
      with writer.If(self.gen_mask_threads(num_cols)):
        if self._trans_a:
          op1_addr = f'{k} + {thread_idx_x} * {op1_data_view.lead_dim}'
        else:
          op1_addr = f'{thread_idx_x} + {k} * {op1_data_view.lead_dim}'
        writer(f'{op1_variable} = {self._op1.name}[{op1_addr}];')
    
      l = list()
      rb = list()
      rb.append(0)
      i = 0
      for row in self._mat_b.get_coo_row_major():
        l.append(len(row))
        if i!=0 and i!=len(self._mat_b.get_coo_row_major())-1:
          rb[i] = rb[i-1] + len(row)

      assert(len(l) == k_end)

      #non_zero_el_per_row = "non_zero_el_per_row"
      #s = f"int {non_zero_el_per_row}[{k_end}] = {'{'}"
      #for i in range(len(l)-1):
      #  s += (str(l[i]))
      #  s += (", ")
      #s += (str(l[len(l)-1]))
      #s += ("};")
      
      #writer(s) 

      writer.Emptyline()
      tileid = 0
      for start_tile_n in range(0, k_end, self._vec_unit_length):
        writer.Comment(f"Begin Tile {start_tile_n}..{start_tile_n+self._vec_unit_length}")
        writer(f'shiftedWid = {start_tile_n} + {warp_idx_variable};')
        
        # We need to load values of k'th row of k for that we need to find their offset in the sparse matrix
        non_zeros = self._mat_b.get_coo_row_major()[k]
        non_zeros_of_this_tile = non_zeros[start_tile_n:start_tile_n+self._vec_unit_length]

        if len(non_zeros_of_this_tile) == 0:
          continue

        it = 0
        non_zero_to_thread = list()
        for non_zero_col_id in non_zeros_of_this_tile:
          with writer.If(f'shiftedWid == {it}'):
            op2_addr = self._mat_b.find_1d_offset(k, non_zero_col_id)
            writer(f'{op2_variable} = {self._op2.name}[{op2_addr}];')
            non_zero_to_thread.append((non_zero_col_id, it))
          if it != len(non_zeros_of_this_tile) - 1:
            writer(f'else')
          it += 1

        if active_sub_group_mask:
          writer(f'{sub_group} = {active_sub_group_mask};')

        end_tile = start_tile_n + self._vec_unit_length
        if end_tile >= k_end:
          end_tile = k_end
        if end_tile > start_tile_n + len(non_zeros_of_this_tile):
          end_tile = start_tile_n + len(non_zeros_of_this_tile)
        
        broadcast_idx = 0
        for n in non_zeros_of_this_tile: #range(start_tile_n, end_tile):
          if not value_known:
            tmp_value = 'tmp'
            broadcast_sync = self._vm._lexic.broadcast_sync(op2_variable,
                                                            broadcast_idx,
                                                            sub_group)
            writer(f'{tmp_value} = {broadcast_sync};')
            res_access = f"[{n}]"
            writer(f'{self._dest.name}{res_access} += {op1_variable} * {tmp_value};')
          else:
            res_access = f"[{n}]"
            address = self._mat_b.find_1d_offset(k ,n)
            writer(f'{self._dest.name}{res_access} += {op1_variable} * {self._val_b[address]}{self._vm.get_real_literal()};')
          broadcast_idx += 1
        tileid += 1
        writer.Comment(f"End Tile {start_tile_n}..{start_tile_n+self._vec_unit_length}")
        if start_tile_n < k_end - self._vec_unit_length:
          writer.Emptyline()
      writer.Comment(f"End Column {k} of A")
      writer.Emptyline()

  def __str__(self) -> str:
    return f'{self._dest.name} = rb_gemm({self._op1.name}, {self._op2.name})'