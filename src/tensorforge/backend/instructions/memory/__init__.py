# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from tensorforge.backend.writer import Writer
from typing import Union
from tensorforge.common.context import Context
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.backend.pir.core import MemSpace

class MemoryInstruction(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._declare = True

  def write_base(self) -> str:
    """The pointer this instruction writes through.

    The symbol's own name for everything except a rotating shared-memory
    buffer, where the declared pointer addresses the stage consumers read and
    the transfer fills a different one.
    """
    return self._dest.name

  @abstractmethod
  def gen_code_inner(self, writer: Writer):
    pass

  def gen_code_declare(self, writer: Writer):
    pass

  def gen_ir(self, sink):
    # The declaration belongs outside the scope: the symbol it declares is
    # consumed by later instructions.
    if self._declare:
      self.gen_code_declare(sink)
    with sink.Scope():
      sink.Comment(self.__str__())
      # A rotating buffer needs its write-side alias here, inside the scope,
      # so it cannot clash with the consumer's pointer of the same name.
      gen_write_base = getattr(self, 'gen_write_base', None)
      if gen_write_base is not None:
        gen_write_base(sink)
      self.gen_code_inner(sink)

class AbstractShrMemWrite(MemoryInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._shm_volume: int = 0
    self._shr_mem_offset: Union[int, None] = 0
    self._declare = False
    self._global_offset = False
    # Multi-stage (rotating) buffer.  `_stages` copies are reserved back to
    # back and `_stage_expr` picks one at run time, so that iteration k can
    # write the stage iteration k+1 will read.
    #
    # Rotation is expressed here, as a property of the allocation, rather than
    # by the pipelining pass cloning the tensor object and renaming it to
    # `preload_*`: a clone is a second symbol that the allocator sizes
    # separately and liveness cannot relate to the original, which is why the
    # old MultiBuffer had to copy shared->shared to get the data back into the
    # buffer the consumers knew about.
    self._stages: int = 1
    self._stage_expr: Union[str, None] = None
    # When a buffer rotates, the stage the *consumer* reads and the stage this
    # transfer *writes* are different -- iteration k consumes stage k % d while
    # the advanced transfer fills stage (k + d - 1) % d.  The declaration
    # emitted by gen_code_declare names the consumer's view, because that is
    # the pointer every later instruction refers to by symbol name; the write
    # needs its own base, which `write_base()` provides.
    self._write_stage_expr: Union[str, None] = None

  def stage_size(self) -> int:
    """Size of one stage, i.e. what a single iteration needs."""
    user_options = self._context.get_user_options()
    if user_options.align_shr_mem:
      return self._context.align(self._shm_volume)
    return self._shm_volume

  def set_stages(self, stages: int, stage_expr: Union[str, None],
                 write_stage_expr: Union[str, None] = None) -> None:
    """``stage_expr`` selects the stage the declared pointer addresses -- the
    consumer's view.  ``write_stage_expr`` selects the stage this transfer
    fills; leave it unset when they coincide, which is every case except an
    advanced transfer."""
    if stages < 1:
      raise ValueError(f'a buffer needs at least one stage, got {stages}')
    if stages > 1 and not stage_expr:
      raise ValueError('a rotating buffer needs an expression selecting the '
                       'stage; without one every iteration writes stage 0')
    if write_stage_expr and stages < 2:
      raise ValueError('a separate write stage makes no sense on a '
                       'single-stage buffer')
    self._stages = stages
    self._stage_expr = stage_expr
    self._write_stage_expr = write_stage_expr

  def num_stages(self) -> int:
    return self._stages

  def _stage_offset(self, expr: Union[str, None] = None) -> str:
    expr = expr if expr is not None else self._stage_expr
    if self._stages == 1 or not expr:
      return f'{self._shr_mem_offset}'
    return f'{self._shr_mem_offset} + ({expr}) * {self.stage_size()}'

  def _arena(self) -> str:
    return (GeneralLexicon.TOTAL_SHR_MEM if self._global_offset
            else self._shr_mem.name)

  def rotates(self) -> bool:
    """Does this transfer fill a different stage than the declaration names?"""
    return bool(self._write_stage_expr
                and self._write_stage_expr != self._stage_expr)

  def write_base(self) -> str:
    """Base pointer this transfer writes through.

    Normally the symbol's own name.  When the buffer rotates, a scope-local
    alias, because the symbol's declaration addresses the stage the *consumers*
    read and writing through it would overwrite the data they are about to use.
    """
    return f'{self._dest.name}_w' if self.rotates() else self._dest.name

  def gen_write_base(self, writer: Writer) -> None:
    """Emit the write-side alias.  Call at the top of ``gen_code_inner``.

    ``MemoryInstruction.gen_ir`` puts the declaration outside a ``Scope()`` and
    the body inside one, so this alias is scope-local and cannot clash with the
    consumer's pointer of the same buffer.
    """
    if not self.rotates():
      return
    lhs = (f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} '
           f'{self.write_base()}')
    writer(f'{lhs} = &{self._arena()}'
           f'[{self._stage_offset(self._write_stage_expr)}];')

  def gen_code_declare(self, writer: Writer) -> None:
    if self._declare:
      offset = self._stage_offset()
      if (hasattr(writer, 'alloc') and callable(getattr(writer, 'alloc'))
          and self._stages == 1):
        # The window is a value, so a read through it declares what it
        # touches instead of naming it.  `extern` because the consumers still
        # spell `s0` out, same as the register tiles.
        #
        # Only where the offset is a number.  A rotating buffer's offset
        # carries the stage expression, and the scratch check orders windows
        # by their start to prove that two of them do not overlap -- a
        # symbolic start cannot be ordered against a numeric one, and letting
        # it through makes the check compare an int with a string rather than
        # decline.  The rotating case keeps the text until the offset itself
        # is a value.
        value = writer.alloc(self._dest.get_fptype(), (self.stage_size(),),
                             MemSpace.SHARED, hint=self._dest.name,
                             extern=self._dest.name,
                             arena=self._arena(), offset=int(offset),
                             restrict=self._vm.get_lexic().restrict_kw,
                             swizzle=self._swizzle())
        self._dest.set_pir_buffer(writer, value)
      else:
        lhs = f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
        writer(f'{lhs} = &{self._arena()}[{offset}];')

  #: Shared memory is 32 banks wide on both vendors, so there is nothing to
  #: gain from permuting over a longer period than that.
  _BANKS = 32

  def _swizzle(self):
    """A row permutation for this window, when one is both legal and useful.

    A tile written a row at a time and read a column at a time costs a bank
    cycle per lane: `s0[(threadIdx.x % 32) * 32]` puts every lane in bank 0
    with a different address, which `tools/bank_conflicts.py` reports as
    32-way.  Permuting the columns per row costs nothing and clears it, and
    leaves a row-wise access exactly as good -- it reaches the same banks in a
    different order.

    Only for a power-of-two row width, which is what makes `k ^ (n % width)`
    stay inside the row.  Everything else keeps the plain layout: a tile 13
    wide has no conflict worth this, and a permutation that carried across
    rows would be a different element, not a slower one.
    """
    from tensorforge.backend.pir.core import XorSwizzle
    view = self._dest.data_view
    if view is None or len(view.shape) < 2:
      return None

    # The width has to divide the *volume*, not merely be the row width.  The
    # permutation maps each block of `width` elements onto itself, so a buffer
    # whose last block is partial has indices that permute past its end -- 728
    # elements swizzled at 32 puts eight of them into the next window.  Shared
    # memory, silently, which is the worst failure available here.
    #
    # The largest power of two dividing the volume is safe by construction and
    # is also the better choice: a 16x16 tile takes 32 rather than 16 and its
    # column read goes 2-way to 1-way, and a 56x13 window takes 8 where the
    # row-width rule declined outright, 8-way to 4-way.
    #
    # An odd volume yields 1, which is no swizzle -- and that is the right
    # answer rather than a fallback.  A row width coprime with 32 already
    # spreads a column read over every bank, so permuting it would move
    # elements the plain layout had placed well: strides 9 and 13 are 1-way
    # untouched and 2- or 3-way under any width.
    volume = 1
    for n in view.shape:
      volume *= n
    width = 1
    while width * 2 <= self._BANKS and volume % (width * 2) == 0:
      width *= 2
    if width < 2:
      return None
    return XorSwizzle(width)

  def compute_shared_mem_size(self) -> int:
    # What the region allocator must reserve: every stage at once.  Returning
    # the per-stage size here would silently overlap the stages.
    return self._stages * self.stage_size()

  def set_shr_mem_offset(self, offset: int, first: bool, global_offset: bool) -> None:
    self._shr_mem_offset = offset
    self._is_ready = True
    self._declare = first
    self._global_offset = global_offset

  @abstractmethod
  def get_dest(self):
    pass
