from tensorforge.common.context import Context
from tensorforge.backend.scopes import Scopes, Symbol
from tensorforge.common.matrix.matrix import Matrix
from tensorforge.backend.symbol import SymbolType, DataView
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.common.exceptions import InternalError
from tensorforge.common.basic_types import GeneralLexicon
from .abstract_builder import AbstractBuilder


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, context: Context, scopes: Scopes):
    super(GetElementPtrBuilder, self).__init__(context, scopes)

  def build(self, src: Symbol, include_extra_offset: bool = True):
    self._reset()
    if src.stype == SymbolType.Data:
      dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=SymbolType.Data,
                    obj=src.obj)
      dest.data_view = DataView(shape=src.obj.shape, permute=None, bbox=src.obj.get_bbox())
      self._scopes.add_symbol(dest)
    elif src.stype == SymbolType.Scalar:
      dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=SymbolType.Scalar,
                    obj=src.obj)
      dest.data_view = DataView(shape=[], permute=None)
      self._scopes.add_symbol(dest)
      self._instructions.append(GetElementPtr(self._context, src, dest, include_extra_offset))
    else:
      if src.stype != SymbolType.Batch:
        raise InternalError("src operand is not in a batch")

      if issubclass(Matrix, type(src.obj)):
        raise InternalError(f'src operand is not a matrix. Instead: {type(src.obj)}')

      dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=SymbolType.Global,
                    obj=src.obj)

      batched_matrix = src.obj
      dest.data_view = DataView(shape=batched_matrix.shape, permute=None, bbox=batched_matrix.get_bbox())

      self._scopes.add_symbol(dest)
      self._instructions.append(GetElementPtr(self._context, src, dest, include_extra_offset))
    src.add_user(self)
