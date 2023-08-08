from abc import ABC, abstractmethod
from gemmforge.symbol_table import Symbol, SymbolType
from ..abstract_instruction import AbstractInstruction
from gemmforge.basic_types import ShrMemObject
from gemmforge.exceptions import InternalError


class AbstractShrMemWrite(AbstractInstruction):
  def __init__(self, vm):
    super().__init__(vm)
    self._shm_volume: int = 0
    self._shr_mem_offset = None

  def compute_shared_mem_size(self):
    return self._shm_volume

  def set_shr_mem_offset(self, offset: int):
    self._shr_mem_offset = offset
    self._is_ready = True

  def __str__(self) -> str:
    pass


class AbstractShrMemLoader(AbstractShrMemWrite):
  def __init__(self, **kwargs):
    super(AbstractShrMemLoader, self).__init__(kwargs['vm'])

    self._lexic = self._vm.get_lexic()
    self._dest: Symbol = kwargs['dest']
    self._src: Symbol = kwargs['src']
    self._shr_mem: ShrMemObject = kwargs['shr_mem']
    self._num_threads: int = kwargs['num_threads']
    self._load_and_transpose: bool = kwargs['load_and_transpose']
    self._manual_unroll_threshold: int = 4

    self._check()
    self._lead_dim = None
    self._align_shm_volume = None

    self._is_ready: bool = False

  def _assign(self, writer, shr_mem_address, glb_mem_address):
    lhs = f'{self._dest.name}[{shr_mem_address}]'
    rhs = f'{self._src.name}[{glb_mem_address}]'
    writer(f'{lhs} = {rhs};')

  def gen_code(self, writer) -> None:
    writer.Emptyline()

    lhs = f'{self._vm.fp_as_str()}* {self._dest.name}'
    rhs = f'{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = &{rhs};')

  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  def _check(self) -> None:
    if self._src.stype != SymbolType.Global:
      raise InternalError('shr-load: `src` operand is not in global mem.')

    if not self._src.data_view:
      raise InternalError('shm-factory: `src` operand must hava data_view attribute')

    if self._dest.stype != SymbolType.SharedMem:
      raise InternalError('shr-load: `dest` operand is not in shr. mem.')
