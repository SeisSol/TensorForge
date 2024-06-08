
class ComputeInstruction:
  @abstractmethod
  def get_operands(self):
    return []

class Gemm(ComputeInstruction):
  pass
