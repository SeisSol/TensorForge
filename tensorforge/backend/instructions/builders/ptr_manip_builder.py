from tensorforge.common.context import Context
from tensorforge.backend.scopes import Scopes, Symbol
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.backend.symbol import SymbolType, DataView
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.common.exceptions import InternalError
from tensorforge.common.basic_types import GeneralLexicon
from .abstract_builder import AbstractBuilder


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, context: Context, scopes: Scopes):
    super(GetElementPtrBuilder, self).__init__(context, scopes)

  def build(self, src: Symbol, include_extra_offset: bool = True, batch_offset = 0):
    self._reset()

    dstype = src.stype

    if dstype not in (SymbolType.Scalar, SymbolType.Data):
      dstype = SymbolType.Global

    dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=dstype,
                    obj=src.obj)

    if src.stype not in (SymbolType.Scalar, SymbolType.Data):
      # TODO: remove this code path
      dest.data_view = DataView(shape=src.obj.shape, permute=None, bbox=src.obj.get_bbox())
    else:
      dest.data_view = DataView(shape=src.obj.shape, permute=None)
    self._scopes.add_symbol(dest)

    if src.stype != SymbolType.Data:
      self._instructions.append(GetElementPtr(self._context, src, dest, include_extra_offset, batch_offset))

    src.add_user(self)
