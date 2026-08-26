# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Union
from tensorforge.common.context import Context
from tensorforge.backend.symbol import Symbol
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.backend.writer import Writer
from tensorforge.backend.pir.core import MemSpace
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

  def gen_ir(self, writer: Writer):
    if self._dest.obj.size > 0:
      datatype = self._context.fp_type if self._dest.obj.datatype is None else self._dest.obj.datatype

      init_values_list = ''
      if isinstance(self._init_value, (float, int, bool)):
        if self._init_value == 0:
          init_values_list = "{}"
        else:
          real_literal = self._vm.get_real_literal()
          init_values = ', '.join([datatype.literal(self._init_value)] * self._dest.obj.size)
          init_values_list = f' = {{{init_values}}}'
      # Structured: the tile becomes a value the body can reason about
      # instead of a line of text that conflicts with everything.  The name
      # is still the macro layer's, because the consumers still spell it out
      # -- see IRBuilder.alloc's `extern`, and the count of buffers that will
      # stop needing it once symbol.py takes the value instead.
      #
      # `gen_ir` is already the inside of `through_pir`, so the sink is the
      # builder.  On the unmigrated path it is the Writer, which has no
      # structured alloc, so that path keeps emitting the line.
      if hasattr(writer, 'alloc') and callable(getattr(writer, 'alloc')):
        writer.alloc(datatype, (self._dest.obj.size,), MemSpace.REGISTER,
                     hint=self._dest.obj.name, extern=self._dest.obj.name,
                     init=init_values_list)
      else:
        writer(f'{datatype} {self._dest.obj.name}'
               f'[{self._dest.obj.size}]{init_values_list};')

  def __str__(self) -> str:
    return f'{self._dest.obj.name} = alloc_regs [{self._dest.obj.size}];'

class ShrMemAlloc(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               size: Union[int, None]):
    super(ShrMemAlloc, self).__init__(context)
    self._size = size
    self._dest = dest

    dest.add_user(self)

  def gen_ir(self, writer: Writer):
    shrmem_obj = self._dest.obj
    common_shrmem = f'{GeneralLexicon.TOTAL_SHR_MEM}'
    common_shrmem_size = shrmem_obj.get_total_size()
    if common_shrmem_size > 0:
      shr_mem_decl = self._vm.get_lexic().declare_shared_memory(name=common_shrmem,
                                                        precision=self._vm.fp_as_str())

      if shr_mem_decl:
        writer(f'{shr_mem_decl};')

      address = f'{shrmem_obj.get_size_per_mult()} * {self._vm.get_lexic().thread_idx_y} + {shrmem_obj.get_global_size()}'
      writer(f'{self._fp_as_str}* {shrmem_obj.name} = &{common_shrmem}[{address}];')
      writer(f'{self._fp_as_str}* tempShrMem = &{shrmem_obj.name}[{shrmem_obj.get_temp_offset()}];')

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size() is not None:
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = alloc_shr [{self._dest.obj.get_total_size_as_str()}];'
