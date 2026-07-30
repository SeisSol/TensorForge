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

  def stage_size(self) -> int:
    """Size of one stage, i.e. what a single iteration needs."""
    user_options = self._context.get_user_options()
    if user_options.align_shr_mem:
      return self._context.align(self._shm_volume)
    return self._shm_volume

  def set_stages(self, stages: int, stage_expr: Union[str, None]) -> None:
    if stages < 1:
      raise ValueError(f'a buffer needs at least one stage, got {stages}')
    if stages > 1 and not stage_expr:
      raise ValueError('a rotating buffer needs an expression selecting the '
                       'stage; without one every iteration writes stage 0')
    self._stages = stages
    self._stage_expr = stage_expr

  def num_stages(self) -> int:
    return self._stages

  def _stage_offset(self) -> str:
    if self._stages == 1 or not self._stage_expr:
      return f'{self._shr_mem_offset}'
    return f'{self._shr_mem_offset} + ({self._stage_expr}) * {self.stage_size()}'

  def gen_code_declare(self, writer: Writer) -> None:
    if self._declare:
      lhs = f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
      arena = (GeneralLexicon.TOTAL_SHR_MEM if self._global_offset
               else self._shr_mem.name)
      writer(f'{lhs} = &{arena}[{self._stage_offset()}];')

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
