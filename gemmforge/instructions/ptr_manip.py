from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.common import get_extra_offset_name
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection
from gemmforge.exceptions import GenerationError


class GetElementPtr(AbstractInstruction):
  def __init__(self,
               vm: VM,
               src,
               dest,
               include_extra_offset=True):
    super(GetElementPtr, self).__init__(vm)
    self._src = src
    self._dest = dest
    self._include_extra_offset = include_extra_offset
    self._is_ready = True

  def gen_code(self, writer):

    batch_obj = self._src.obj
    batch_addressing = batch_obj.addressing

    if self._include_extra_offset:
      extra_offset = f' + {get_extra_offset_name(self._src.name)}'
    else:
      extra_offset = ''

    address = ''
    if batch_addressing == "strided":
      main_offset = f'{GeneralLexicon.BATCH_ID} * {batch_obj.get_real_volume()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {sub_offset}{extra_offset}'
    elif batch_addressing == "pointer_based":
      main_offset = f'{GeneralLexicon.BATCH_ID}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset}][{sub_offset}{extra_offset}'
    elif batch_addressing == "none":
      address = f'{batch_obj.get_offset_to_first_element()}{extra_offset}'
    else:
      GenerationError(f'unknown addressing of {self._src.name}, given {batch_addressing}')

    rhs = f'&{self._src.name}[{address}]'

    lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
    lhs += f'{self._vm.fp_as_str()} * const __restrict__ {self._dest.name}'
    writer(f'{lhs} = {rhs};')

  def __str__(self) -> str:
    return f'{self._dest.name} = getelementptr_b2g {self._src.name};'
