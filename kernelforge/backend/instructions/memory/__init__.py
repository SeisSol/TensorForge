from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from kernelforge.backend.writer import Writer
from typing import List, Union
from kernelforge.common.context import Context

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

  def compute_shared_mem_size(self) -> int:
    user_options = self._context.get_user_options()

    if user_options.align_shr_mem:
      size = self._context.align(self._shm_volume)
    else:
      size = self._shm_volume
    return size

  def set_shr_mem_offset(self, offset: int, first: bool) -> None:
    self._shr_mem_offset = offset
    self._is_ready = True
    self._declare = first

  @abstractmethod
  def get_dest(self):
    pass
