from abc import ABC


class AbstractBuilder(ABC):
  def __init__(self, vm, symbol_table):
    self._vm = vm
    self._symbol_table = symbol_table
    self._instructions = []

  def get_instructions(self):
    return self._instructions

  def _reset(self):
    self._instructions = []
