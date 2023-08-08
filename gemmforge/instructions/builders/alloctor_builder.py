from .abstract_builder import AbstractBuilder
from gemmforge.symbol_table import SymbolType, Symbol
from gemmforge.basic_types import RegMemObject, ShrMemObject
from gemmforge.instructions import RegisterAlloc, ShrMemAlloc
from gemmforge.basic_types import GeneralLexicon
from abc import abstractmethod


class AbstractAllocBuilder(AbstractBuilder):
  def __init__(self, vm, symbol_table):
    super(AbstractAllocBuilder, self).__init__(vm, symbol_table)
    self._obj = None

  @abstractmethod
  def _name_new_symbol(self):
    pass

  def get_resultant_obj(self):
    if not self._obj:
      raise NotImplementedError
    return self._obj


class ShrMemAllocBuilder(AbstractAllocBuilder):
  def __init__(self, vm, symbol_table):
    super(ShrMemAllocBuilder, self).__init__(vm, symbol_table)
    self._counter = 0

  def build(self, size=None):
    self._reset()
    name = self._name_new_symbol()
    self._obj = ShrMemObject(name, size)
    dest = Symbol(name=name,
                  stype=SymbolType.SharedMem,
                  obj=self._obj)

    self._symbol_table.add_symbol(dest)
    self._instructions.append(ShrMemAlloc(self._vm, dest, size))

  def _name_new_symbol(self):
    name = f'{GeneralLexicon.LOCAL_SHR_MEM}{self._counter}'
    self._counter += 1
    return name


class RegistersAllocBuilder(AbstractAllocBuilder):

  def __init__(self, vm, symbol_table):
    super(RegistersAllocBuilder, self).__init__(vm, symbol_table)
    self._counter = 0

  def build(self, size: int, init_value=None):
    self._reset()
    name = self._name_new_symbol()
    self._obj = RegMemObject(name, size)
    dest = Symbol(name,
                  SymbolType.Register,
                  self._obj)

    self._symbol_table.add_symbol(dest)
    self._instructions.append(RegisterAlloc(self._vm, dest, init_value))

  def _name_new_symbol(self):
    name = f'{GeneralLexicon.REG_NAME}{self._counter}'
    self._counter += 1
    return name
