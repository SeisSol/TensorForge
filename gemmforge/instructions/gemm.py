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

      writer(f'{self._dest.name}[n] += {op1_value} * {self._op2.name}[{op2_addr}];')

  def __str__(self) -> str:
    return f'{self._dest.name} = gemm({self._op1.name}, {self._op2.name})'
