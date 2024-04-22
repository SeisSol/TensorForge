from kernelforge.common.context import Context
from kernelforge.common.matrix.dense import Matrix
from kernelforge.backend.symbol import Symbol, SymbolType, DataView
from kernelforge.common.exceptions import InternalError, GenerationError
from kernelforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction
from copy import deepcopy

class ComputeInstruction:
  @abstractmethod
  def get_operands(self):
    return []

class Gemm(ComputeInstruction):
  pass
