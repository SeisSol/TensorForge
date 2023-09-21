from kernelforge.common.context import Context
from kernelforge.common.basic_types import FloatingPointType
from kernelforge.backend.symbol import Symbol, SymbolType
from kernelforge.backend.writer import Writer
from kernelforge.common.exceptions import InternalError
from .abstract_instruction import AbstractInstruction


class ClearRegisters(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol):
    super(ClearRegisters, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('ptr: operand `src` is not in registers')

    self._is_ready = True
    self._src = src
    src.add_user(self)

  def gen_code(self, writer: Writer):
    writer.new_line()
    writer(f'// clear registers')
    writer.insert_pragma_unroll()
    with writer.For(f'int i = 0; i < {self._src.obj.size}; ++i'):
      fp_prefix = 'f' if self._context.fp_type == FloatingPointType.FLOAT else ''
      writer(f'{self._src.name}[i] = 0.0{fp_prefix};')

  def __str__(self) -> str:
    return f'clear_regs {self._src.name}[{self._src.obj.size}];'
