from .abstract_instruction import AbstractInstruction
from kernelforge.common.vm.vm import VM
from kernelforge.common.aux import get_extra_offset_name, Addressing
from kernelforge.common.basic_types import GeneralLexicon, DataFlowDirection
from kernelforge.common.exceptions import GenerationError

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
      extra_offset = f' + {get_extra_offset_name(self._src)}'
    else:
      extra_offset = ''

    address = ''
    if batch_addressing == Addressing.STRIDED:
      main_offset = f'{GeneralLexicon.BATCH_ID_NAME} * {batch_obj.get_real_volume()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {sub_offset}{extra_offset}'
      rhs = f'&{self._src.name}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{self._vm.fp_as_str()} * const {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.PTR_BASED:
      main_offset = f'{GeneralLexicon.BATCH_ID_NAME}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset}][{sub_offset}{extra_offset}'
      src_suffix = '_ptr' if self._vm.get_lexic()._backend == 'targetdart' else ''
      rhs = f'&{self._src.name}{src_suffix}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{self._vm.fp_as_str()} * const {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.NONE:
      address = f'{batch_obj.get_offset_to_first_element()}{extra_offset}'
      rhs = f'&{self._src.name}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{self._vm.fp_as_str()} * const {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.SCALAR:
      rhs = f'{self._src.name}'
      lhs = f'{self._vm.fp_as_str()} {self._dest.name}'
    else:
      GenerationError(f'unknown addressing of {self._src.name}, given {batch_addressing}')

    writer(f'{lhs} = {rhs};')

  def __str__(self) -> str:
    return f'{self._dest.name} = getelementptr_b2g {self._src.name};'
