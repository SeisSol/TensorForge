from .abstract_builder import AbstractBuilder
from gemmforge.symbol_table import SymbolType, Symbol
from gemmforge.symbol_table import DataView
from gemmforge.instructions import GetElementPtr
from gemmforge.exceptions import InternalError


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, vm, symbol_table):
    super(GetElementPtrBuilder, self).__init__(vm, symbol_table)

  def build(self, src: Symbol, include_extra_offset: bool = True):
    self._reset()
    if src.stype != SymbolType.Batch:
      raise InternalError("src operand is not in a batch")

    dest = Symbol(name=f'glb_{src.name}',
                  stype=SymbolType.Global,
                  obj=src.obj)

    batched_matrix = src.obj
    dest.data_view = DataView(rows=batched_matrix.get_actual_num_rows(),
                              columns=batched_matrix.get_actual_num_cols(),
                              lead_dim=batched_matrix.num_rows,
                              is_transposed=False)

    self._symbol_table.add_symbol(dest)
    self._instructions.append(GetElementPtr(self._vm, src, dest, include_extra_offset))
