from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from kernelforge.backend.writer import Writer

class ComputeInstruction(AbstractInstruction):
  @abstractmethod
  def get_operands(self):
    return []

  @abstractmethod
  def gen_code_inner(self, writer: Writer):
    pass
  
  def gen_code(self, writer: Writer):
    with writer.Scope():
      writer.Comment(self.__str__())
      self.gen_code_inner(writer)
