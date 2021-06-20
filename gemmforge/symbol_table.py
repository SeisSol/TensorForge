from .exceptions import InternalError
import enum


class SymbolType(enum.Enum):
  Batch = 1
  Global = 2
  SharedMem = 3
  Register = 4


class DataView:
  def __init__(self, rows: int, columns: int, lead_dim: int, is_transposed: bool):
    self.rows = rows
    self.columns = columns
    self.lead_dim = lead_dim
    self.is_transposed: bool = is_transposed

  def __str__(self):
    return f'rows: {self.rows}, cols: {self.columns}, lid: {self.lead_dim}, trans: {self.is_transposed}'


class Symbol:
  def __init__(self,
               name: str,
               stype: SymbolType,
               obj):
    self.name = name
    self.stype = stype
    self.obj = obj
    self.data_view = None

  def __str__(self):
    return f'name: {self.name}, type: {self.stype}'


class Scope:
  def __init__(self):
    self._symbols = {}
    
  def pop(self, obj):
    if obj in self._symbols:
      self._symbols.pop(obj)
    
  def __contains__(self, item):
    return item in self._symbols

  def __setitem__(self, obj, symbol: Symbol):
    self._symbols[obj] = symbol
    
  def __getitem__(self, obj):
    return self._symbols[obj]
  
  def values(self):
    return self._symbols.values()


class InverseSymbolTable:
  def __init__(self):
    self._global_scope = Scope()
    self._scopes = [self._global_scope]
    
  def add_scope(self):
    self._scopes.append(Scope())

  def add_symbol(self, symbol: Symbol):
    if symbol.obj in self._scopes[-1]:
      raise InternalError(f'symbol {symbol.name} has already in the current scope')
    else:
      self._scopes[-1][symbol.obj] = symbol
  
  def delete_symbol(self, obj):
    self._scopes.pop(obj)
    
  def print_scope(self, level=-1):
    if level > len(self._scopes):
      raise InternalError(f'level {level} exceeds num scopes equal to {len(self._scopes)}')

    for symbol in self._scopes[level]:
      print(symbol)
  
  def print_scopes(self):
    for level, scope in enumerate(self._scopes):
      print('*'*80)
      self.print_scope(level)

  @property
  def from_global(self):
    return self._global_scope
  
  def __getitem__(self, obj):
    for scope in reversed(self._scopes):
      if obj in scope:
        return scope[obj]
    raise InternalError(f'obj {id(obj)} has not been found')
