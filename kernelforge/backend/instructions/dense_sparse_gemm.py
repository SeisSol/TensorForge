from .abstract_instruction import AbstractInstruction
from kernelforge.common.context import Context
from kernelforge.common.matrix.matrix import Matrix
from kernelforge.backend.symbol import Symbol, SymbolType, DataView
from kernelforge.common.exceptions import InternalError, GenerationError
from kernelforge.backend.writer import Writer
from abc import abstractmethod
from .gemm import Gemm

class ShrMemBasedDenseSparseGemm(AbstractInstruction, Gemm):
  def __init__(self,
               context: Context,
               trans_a: bool,
               trans_b: bool,
               op1: Symbol,
               op2: Symbol,
               dest: Symbol,
               prefer_align: bool,
               num_threads: int):
    super(ShrMemBasedDenseSparseGemm, self).__init__(context)
    self._trans_a = trans_a
    self._trans_b = trans_b
    self._op1 = op1
    self._op2 = op2
    self._prefer_align = prefer_align
    self._is_ready = True
    self._user_options = context.get_user_options()
    self._gemm_meta_data = None
    self._mat_b = op2.obj
    
    if dest.stype != SymbolType.Register:
      raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
    else:
      self._dest = dest

    if self._op1.stype == SymbolType.Batch:
      raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

    if self._op2.obj.get_values() == None and self._op2.stype == SymbolType.Batch:
      raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

    if self._dest.stype != SymbolType.Register:
      raise InternalError('gemm: `dest` must be a register obj.')

    if self._trans_b:
      raise Exception('Sparse Matrix is not supported to be transposed (provide the storage order with the list of coordinates).')

    op1.add_user(self)
    op2.add_user(self)
    dest.add_user(self)

    self._analyze()

    self._is_ready = True

  def gen_code(self, writer):
    value_var = 'value'
    op1_data_view = self._op1.data_view
    op2_data_view = self._op2.data_view
    thread_idx_x = self._vm.get_lexic().thread_idx_x
    with writer.If(self.gen_mask_threads(op1_data_view.get_dim_size(0))):
      writer(f'{self._vm.fp_as_str()} {value_var};')

      writer.Emptyline()
      for k in range(0, op1_data_view.get_dim_size(1)):
        if not self._trans_b:
          non_zeros = self._mat_b.get_coo_per_row()[k]
        else:
          non_zeros = self._mat_b.get_coo_per_col()[k]
        if len(non_zeros) == 0:
          continue

        op1_addr = f'{thread_idx_x} + {k} * {op1_data_view.get_lead_dim()}'
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

  def get_op1(self):
    return self._op1

  def get_op2(self):
    return self._op2

  def __str__(self) -> str:
    return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'
  
  def _analyze(self):
    self._op1_view = self._op1.data_view
    self._op2_view = self._op2.data_view

    self._is_layout_as_requested = self._op2_view.is_transposed == self._trans_b
    if self._is_layout_as_requested:
      self._n_range = self._op2_view.get_dim_size(1)
    else:
      self._n_range = self._op2_view.get_dim_size(0)

    self._m_range = self._op1_view.get_dim_size(0)
    num_dirty_rows = 0

    if self._prefer_align:
      op1_bbox = self._op1_view.get_bbox()
      self._op1_view = deepcopy(self._op1.data_view)
      aligned_begin, aligned_end = self._context.align_range(begin=op1_bbox[0], end=op1_bbox[2])

      if aligned_end > self._op1_view.get_lead_dim():
        aligned_end = self._op1_view.get_lead_dim()

      aligned_bbox = [aligned_begin, op1_bbox[1], aligned_end, op1_bbox[3]]

      self._op1_view.reset_bbox(aligned_bbox)
      num_dirty_rows = op1_bbox[0] - aligned_begin

      if not (aligned_begin == op1_bbox[0] and aligned_end == op1_bbox[2]):
        text = f'gemm aligned along `m` dim: '
        text += f'from [{op1_bbox[0]}, {op1_bbox[2]}] '
        text += f'to [{aligned_begin}, {aligned_end}]; '
        text += f'num. dirty rows in `result`: {num_dirty_rows}'
        self._gemm_meta_data = text

    dest_bbox = [0, 0, self._m_range, self._n_range]
    dest_bbox[0] += num_dirty_rows
    dest_bbox[2] += num_dirty_rows
    self._dest.data_view = DataView(rows=self._op1_view.get_dim_size(0),
                                    columns=self._n_range,
                                    is_transposed=False,
                                    bbox=dest_bbox)


class RegisterOnlyDenseSparseGemm(AbstractInstruction, Gemm):
  def __init__(self, **kwargs):
    super(RegisterOnlyDenseSparseGemm, self).__init__(kwargs['context'])

  def gen_code(self, writer):
    raise Exception("Register Only Sparse-by-Dense Matrix Implementation is not supported.")

  def __str__(self) -> str:
    return f'{self._dest.name} = rb_gemm({self._op1.name}, {self._op2.name})'
