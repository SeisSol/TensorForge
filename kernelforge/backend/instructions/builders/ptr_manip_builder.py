from kernelforge.common.context import Context, VM
from kernelforge.backend.scopes import Scopes, Symbol
from kernelforge.common.matrix.dense import Matrix
from kernelforge.backend.symbol import SymbolType, DataView
from kernelforge.backend.instructions.ptr_manip import GetElementPtr
from kernelforge.common.exceptions import InternalError
from kernelforge.common.basic_types import GeneralLexicon
from .abstract_builder import AbstractBuilder


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, context: Context, scopes: Scopes):
    super(GetElementPtrBuilder, self).__init__(context, scopes)

  def build(self, src: Symbol, include_extra_offset: bool = True):
    self._reset()
    if src.stype != SymbolType.Batch:
      raise InternalError("src operand is not in a batch")

    if issubclass(Matrix, type(src.obj)):
      raise InternalError(f'src operand is not a matrix. Instead: {type(src.obj)}')

    dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                  stype=SymbolType.Global,
                  obj=src.obj)

    batched_matrix = src.obj
    dest.data_view = DataView(rows=batched_matrix.get_actual_num_rows(),
                              columns=batched_matrix.get_actual_num_cols(),
                              # lead_dim=batched_matrix.num_rows,
                              is_transposed=False)

    self._scopes.add_symbol(dest)
    self._instructions.append(GetElementPtr(self._context, src, dest, include_extra_offset))
    src.add_user(self)
