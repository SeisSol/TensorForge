from typing import Union
from tensorforge.common import Context
from tensorforge.backend.symbol import Symbol
from tensorforge.backend.exceptions import InternalError
from tensorforge.backend.writer import Writer
from .abstract_instruction import AbstractInstruction


class RegisterAlloc(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               size: int,
               init_value: Union[float, None]=None):
    super(RegisterAlloc, self).__init__(context)
    self._size = size
    self._init_value = init_value
    self._dest = dest
    self._is_ready = True
    dest.add_user(self)

  def gen_code(self, writer: Writer):
    if self._dest.obj.size < 1:
      raise InternalError('size of reg. obj must be at least 1')

    if self._dest.obj.size == 1:
      init_value = ''
      if isinstance(self._init_value, float):
        init_value = f' = {self._init_value}'
      result = f'{self._context.fp_as_str()} {self._dest.obj.name}{init_value};'
    else:
      init_values_list = ''
      if isinstance(self._init_value, float):
        real_literal = self._vm.get_real_literal()
        init_values = ', '.join([f'{str(self._init_value)}{real_literal}'] * self._dest.obj.size)
        init_values_list = f' = {{{init_values}}}'
      result = f'{self._context.fp_as_str()} {self._dest.obj.name}[{self._dest.obj.size}]{init_values_list};'
    writer(result)

  def __str__(self) -> str:
    return f'{self._dest.obj.name} = alloc_regs {self._dest.obj.size};'


class ShrMemAlloc(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               size: Union[int, None]):
    super(ShrMemAlloc, self).__init__(context)
    self._size = size
    self._dest = dest

    dest.add_user(self)

  def gen_code(self, writer: Writer):
    shrmem_obj = self._dest.obj
    common_shrmem = f'{GeneralLexicon.TOTAL_SHR_MEM}'
    common_shrmem_size = shrmem_obj.get_total_size()

    shr_mem_decl = lexic.declare_shared_memory_inline(name=common_shrmem,
                                                      precision=self._vm.fp_as_str(),
                                                      size=common_shrmem_size,
                                                      alignment=8)

    if shr_mem_decl:
      writer(f'{shr_mem_decl};')

    address = f'{shrmem_obj.get_size_per_mult()} * {self._vm.lexic.thread_idx_y}'
    writer(f'{self._fp_as_str} * {shrmem_obj.name} = &{common_shrmem}[{address}];')

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size():
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = alloc_shr [{self._dest.obj.get_total_size_as_str()}];'
