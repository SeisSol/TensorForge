from tensorforge.common.context import Context
from tensorforge.common.matrix.matrix import Matrix
from tensorforge.backend.symbol import Symbol, SymbolType, DataView
from tensorforge.common.exceptions import InternalError, GenerationError
from tensorforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction
from copy import deepcopy


class CSA(AbstractInstruction):
  def __init__(self,
               context: Context,
               trans_a: bool,
               trans_b: bool,
               op1: Symbol,
               op2: Symbol,
               dest: Symbol,
               prefer_align: bool):
    super(CSA, self).__init__(context)
    self._trans_a = trans_a
    self._trans_b = trans_b
    self._op1 = op1
    self._prefer_align = prefer_align
    self._is_ready = True
    self._user_options = context.get_user_options()
    self._gemm_meta_data = None

    self.registers = None
    if dest.stype != SymbolType.Register:
      raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
    else:
      self._dest = dest

    if not isinstance(self._op1.obj, Matrix):
       raise InternalError('gemm: op1 is not a matrix')

    op1.add_user(self)
    dest.add_user(self)

    self._analyze()

  def _analyze(self):
    self._op1_view = self._op1.data_view

    self._m_range = self._op1_view.get_dim_size(0)
    self._n_range = self._op1_view.get_dim_size(1)
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
        text = f'csa aligned along `m` dim: '
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


  def gen_code(self, writer: Writer):
    self._check()
    writer.new_line()
    writer(f'// csa: {self._op1.name}')
    if self._gemm_meta_data:
      writer(f'// meta: {self._gemm_meta_data}')

    with writer.If(self.gen_mask_threads(self._op1_view.get_dim_size(0))):
      m_range = self._op1_view.get_dim_size(1)
      writer.insert_pragma_unroll()
      with writer.Block(f'for (int n = 0; n < {m_range}; ++n)'):
        address = self._op1_view.get_address(row_idx=self._vm.get_lexic().thread_idx_x, column_idx='n')

        op1_element = 'value'
        writer(f'{self._fp_as_str} {op1_element} = {self._op1.name}[{address}];')

        writer.new_line()
        dest_address = '' if self._dest.obj.size == 1 else '[n]'

        writer(f'{self._dest.name}{dest_address} = {op1_element};')

  def _check(self):
    view_op1 = self._op1.data_view
    if not view_op1:
      raise InternalError(f'symbol data view has not been assign to `op1`')

    if not view_op1.is_transposed == self._trans_a:
      raise GenerationError(f'`op1 layout does not match the layout request by gemm instr.`')

  def get_op1(self):
    return self._op1

  def __str__(self) -> str:
    return f'{self._dest.name} = csa({self._op1.name})'
