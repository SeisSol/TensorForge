from kernelforge.common.context import Context
from kernelforge.common.matrix.matrix import Matrix
from kernelforge.backend.symbol import Symbol, SymbolType, DataView
from kernelforge.common.exceptions import InternalError, GenerationError
from kernelforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction
from copy import deepcopy
from .gemm import Gemm
from typing import List, Union

class Pointwise(ComputeInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               ops: List[Symbol],
               permute: List[List[int]],
               operation: Operation,
               prefer_align: bool,
               num_threads: int):
    super(Pointwise, self).__init__(context)
    self._ops = ops
    self._permute = permute
    self._prefer_align = prefer_align
    self._is_ready = True
    self._user_options = context.get_user_options()
    self._gemm_meta_data = None
    self._operation = operation

    self.registers = None
    if dest.stype != SymbolType.Register:
      raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
    else:
      self._dest = dest
    
    # TODO: check op type

    for op in ops:
        op.add_user(self)
    dest.add_user(self)

    self._analyze()

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

  def gen_code_inner(self, writer: Writer):
    if self._gemm_meta_data:
      writer(f'// meta: {self._gemm_meta_data}')

    with writer.If(self.gen_mask_threads(self._dest.data_view.get_lead_dim())):
      k_range = self._dest.data_view.get_nonlead_dim()
      writer.insert_pragma_unroll()
      with writer.For(f'int k = 0; k < {k_range}; ++k'):
        for i, op in enumerate(self._ops):
            address = op.data_view.get_address(lead_idx=self._vm.get_lexic().thread_idx_x, nonlead_idx='k')
            writer(f'{self._fp_as_str} value{i} = {op.name}[{address}];')
        writer(f'{self._fp_as_str} result = {self._write_operation()};')

        res_access = '' if self._dest.obj.size == 1 else '[k]'
        writer(f'{self._dest.name}{res_access} = result;')

  def _write_operation(self):
    fptype = self._context.fp_type
    return self._context.get_vm().get_lexic().get_operation(self.operation, fptype, 'value0', 'value1')

  def _check(self):
    view = self._dest.data_view

    for i, op in enumerate(self._ops):
      view_op = op.data_view

      if not view_op:
        raise InternalError(f'symbol data view has not been assign to operand {i}')
      
      if not view_op.permute == self.permute[i]:
        raise GenerationError(f'`operand {i} layout does not match the layout request by gemm instr.`')
      
      if not view == view_op:
        raise GenerationError('')

    if not view_op2:
      raise InternalError(f'gemm: symbol data view has not been assign to `op2`')

    is_requested_layout = view_op2.is_transposed == self._trans_b

    # layout op1 is transposed if necessary and layout has already been adjusted
    # Note: if a subsequent GEMM requires to change the current layout
    # the matrix is going to be reloaded to the shared memory
    k_range_op1 = view_op1.get_dim_size(1)

    # Note: we do not reload op2 to the shared memory if the current gemm op. requires
    # a different layout in contrast to the one that has already been loaded to the shared memory
    k_range_op2 = view_op2.get_dim_size(0) if is_requested_layout else view_op2.get_dim_size(1)

    if self._user_options.exact_contraction_length:
      if k_range_op1 != k_range_op2:
        print(view_op1)
        print(view_op2)
        raise GenerationError(f'gemm: mismatch of contraction length '
                              f'k_range_op1( {k_range_op1} ) != k_range_op2( {k_range_op2} )')

    op2_columns = view_op2.get_dim_size(0)
    if op2_columns > self._dest.obj.size:
      msg = f'{op2_columns} > {self._dest.obj.size}'
      # raise InternalError(f'gemm: contraction length is bigger than reg. size i.e, {msg}')

  def get_operands(self):
    return self._ops

  def __str__(self):
    return f'{self._dest.name} = {self._operation}({",".join(op.name for op in self._ops)})'
