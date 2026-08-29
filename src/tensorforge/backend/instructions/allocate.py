# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Union
from tensorforge.common.context import Context
from tensorforge.backend.symbol import Symbol
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.backend.writer import Writer
from tensorforge.backend.pir.core import Effect, MemSpace
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
        # Deliberately *not* over-aligned.  An earlier version declared this
        # `alignas(16)` so a wide access could be cast onto it; that was a
        # category error.  A register array lives in the private address
        # space, where AMDGPU interleaves the lanes at dword granularity --
        # `(private_addr / 4) * wave_size * 4 + lane * 4 + private_addr % 4`.
        # Four consecutive private dwords are `wave_size * 4` bytes apart in
        # the backing memory, so a 16-byte-aligned private address does not
        # name a contiguous 16 bytes and a wide private access cannot be one
        # transfer however it is aligned.  The alignment bought nothing there
        # and padded the scratch frame, which is occupancy.
        #
        # In the case that matters the array is promoted and there is no
        # address at all.  What the cast needs is to be well-defined C++, and
        # that is what the relaxed vector type is for -- see
        # `Symbol._linear_claim`.
        value = writer.alloc(datatype, (self._dest.obj.size,),
                             MemSpace.REGISTER,
                             hint=self._dest.obj.name,
                             extern=self._dest.obj.name,
                             init=init_values_list)
        # Publish it, so a consumer built into this same body addresses the
        # buffer as a value rather than by interpolating the name.  Consumers
        # in other bodies see None and keep using the name -- see
        # Symbol.pir_buffer.
        self._dest.set_pir_buffer(writer, value)
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
    """Bind the kernel's one shared arena, and the two windows into it.

    Not an allocation, despite the name.  `declare_shared_memory` reinterprets
    a pointer the launch configuration supplies -- dynamic shared memory on
    CUDA and HIP, and on SYCL an accessor declared in the kernel signature, so
    there it is nothing at all.  There is no array to declare and no size for
    `Op.ALLOC` to carry, which is why this is the one buffer in the section
    that keeps a name: it exists before any body does.

    The two windows are ordinary bindings, and were three bare statements --
    `Effect.UNKNOWN`, conflicting with every access in the body and pinning
    everything on both sides.  As values they declare what they touch and
    carry a def-use edge, so a scheduler knows a read through a window cannot
    rise above the binding of it.

    Recorded against the arena, not against themselves: `may_alias` treats
    distinct bases as never aliasing, so a window claiming its own identity
    would let a write through the arena reorder past a read through the
    window.
    """
    shrmem_obj = self._dest.obj
    common_shrmem = f'{GeneralLexicon.TOTAL_SHR_MEM}'
    common_shrmem_size = shrmem_obj.get_total_size()
    if common_shrmem_size <= 0:
      return

    lexic = self._vm.get_lexic()
    shr_mem_decl = lexic.declare_shared_memory(name=common_shrmem,
                                               precision=self._vm.fp_as_str())
    address = (f'{shrmem_obj.get_size_per_mult()} * {lexic.thread_idx_y} '
               f'+ {shrmem_obj.get_global_size()}')

    if not hasattr(writer, 'decl_expr'):
      if shr_mem_decl:
        writer(f'{shr_mem_decl};')
      writer(f'{self._fp_as_str}* {shrmem_obj.name} = &{common_shrmem}[{address}];')
      writer(f'{self._fp_as_str}* tempShrMem = &{shrmem_obj.name}[{shrmem_obj.get_temp_offset()}];')
      return

    from tensorforge.backend.pir.core import BufferType, MemSpace as _MemSpace

    def window(name, decl, text, size):
      return writer.decl_expr(
          decl, text.replace('{', '{{').replace('}', '}}'),
          BufferType(self._context.fp_type, (size,), _MemSpace.SHARED),
          self._dest, kind=Effect.READ, hint=name, extern=name,
          alias_root=self._dest)

    if shr_mem_decl:
      # The arena itself: a reinterpret of the launch's pointer, so its
      # declarator carries the cast and there is no type to render it from.
      decl, _, rhs = shr_mem_decl.partition(' = ')
      window(common_shrmem, decl, rhs, common_shrmem_size)

    window(shrmem_obj.name, f'{self._fp_as_str}* {shrmem_obj.name}',
           f'&{common_shrmem}[{address}]', shrmem_obj.get_size_per_mult())
    window('tempShrMem', f'{self._fp_as_str}* tempShrMem',
           f'&{shrmem_obj.name}[{shrmem_obj.get_temp_offset()}]',
           common_shrmem_size - shrmem_obj.get_temp_offset())

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size() is not None:
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = alloc_shr [{self._dest.obj.get_total_size_as_str()}];'
