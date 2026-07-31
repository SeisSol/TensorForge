from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from tensorforge.backend.writer import Writer
from typing import Union
from tensorforge.common.context import Context
from tensorforge.common.basic_types import GeneralLexicon

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
      lhs = f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
      writer(f'{lhs} = &{self._arena()}[{self._stage_offset()}];')

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
