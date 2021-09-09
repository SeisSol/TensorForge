from .abstract_generator import AbstractGenerator
from .symbol_table import InverseSymbolTable
from abc import ABC


class GemmLikeGenerator(AbstractGenerator, ABC):
  def __init__(self, vm):
    super(GemmLikeGenerator, self).__init__(vm)

    self._alpha = None
    self._beta = None

    self._symbol_table = InverseSymbolTable()
    self._instructions = []
