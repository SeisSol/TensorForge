from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from tensorforge.backend.writer import Writer
from typing import Union
from tensorforge.common.context import Context
from tensorforge.common.basic_types import GeneralLexicon

class MemoryInstruction(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._declare = True

  @abstractmethod
  def gen_code_inner(self, writer: Writer):
    pass
  
  def gen_code_declare(self, writer: Writer):
    pass

  def gen_code(self, writer: Writer):
    if self._declare:
      self.gen_code_declare(writer)
    with writer.Scope():
      writer.Comment(self.__str__())
      self.gen_code_inner(writer)

class AbstractShrMemWrite(MemoryInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._shm_volume: int = 0
    self._shr_mem_offset: Union[int, None] = 0
    self._declare = False
    self._global_offset = False
  
  def gen_code_declare(self, writer: Writer) -> None:
    if self._declare:
      lhs = f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
      if self._global_offset:
        rhs = f'{GeneralLexicon.TOTAL_SHR_MEM}[{self._shr_mem_offset}]'
      else:
        rhs = f'{self._shr_mem.name}[{self._shr_mem_offset}]'
      writer(f'{lhs} = &{rhs};')

  def compute_shared_mem_size(self) -> int:
    user_options = self._context.get_user_options()

    if user_options.align_shr_mem:
      size = self._context.align(self._shm_volume)
    else:
      size = self._shm_volume
    return size

  def set_shr_mem_offset(self, offset: int, first: bool, global_offset: bool) -> None:
    self._shr_mem_offset = offset
    self._is_ready = True
    self._declare = first
    self._global_offset = global_offset

  @abstractmethod
  def get_dest(self):
    pass
