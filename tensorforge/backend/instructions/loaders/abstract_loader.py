from abc import abstractmethod
from typing import Union
import enum
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.backend.instructions.store import AbstractShrMemWrite
from tensorforge.backend.symbol import SymbolType, Symbol
from tensorforge.common.exceptions import InternalError


class ShrMemLoaderType(enum.Enum):
  NOT_TRANSPOSED = 0
  TRANSPOSED = 1


class AbstractShrMemLoader(AbstractShrMemWrite):
  def __init__(self, **kwargs):
    super(AbstractShrMemLoader, self).__init__(kwargs['context'])
    self._dest = kwargs['dest']
    self._src = kwargs['src']
    self._shr_mem = kwargs['shr_mem']
    self._num_threads = kwargs['num_threads']
    self._load_and_transpose = kwargs['load_and_transpose']
    self._manual_unroll_threshold = 4

    self._check()
    self._lid_dim: Union[int, None] = None
    self._align_shm_volume: Union[int, None] = None
    self._tensor: Tensor = self._src.obj

    self._dest.add_user(self)
    self._src.add_user(self)
    self._shr_mem.add_user(self)
    self._is_ready: bool = False

  def gen_code(self, writer) -> None:
    writer.new_line()
    datatype = self._context.fp_type if self._dest.obj.datatype is None else self._dest.obj.datatype
    lhs = f'{datatype}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    rhs = f'{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = &{rhs};')

  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  @abstractmethod
  def get_loader_type(self) -> ShrMemLoaderType:
    pass

  def _check(self) -> None:
    if self._src.stype != SymbolType.Global:
      raise InternalError('shr-load: `src` operand is not in global mem.')

    if not isinstance(self._src.obj, Matrix):
      raise InternalError(f'shr-load: `src` operand is not a matrix, instead: {self._src.obj}')

    if self._dest.stype != SymbolType.SharedMem:
      raise InternalError('shr-load: `dest` operand is not in shr. mem.')

    if not isinstance(self._dest.obj, Matrix):
      raise InternalError(f'shr-load: `dest` operand is not a matrix, instead: {self._dest.obj}')

class NoLoadShrMemLoader(AbstractShrMemLoader):
  def __init__(self, **kwargs):
    super(NoLoadShrMemLoader, self).__init__(**kwargs)
    data_view = self._src.data_view
    self._shm_volume = data_view.rows * data_view.columns

    self._dest.data_view = deepcopy(self._src.data_view)
    self._dest.data_view.lead_dim = data_view.rows
