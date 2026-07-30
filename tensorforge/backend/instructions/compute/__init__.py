from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from tensorforge.backend.writer import Writer

class ComputeInstruction(AbstractInstruction):
  @abstractmethod
  def get_operands(self):
    return []

  @abstractmethod
  def gen_code_inner(self, writer: Writer):
    pass

  def gen_code(self, writer: Writer):
    def build(sink):
      with sink.Scope():
        sink.Comment(self.__str__())
        self.gen_ir(sink)
    self.through_pir(writer, build)
