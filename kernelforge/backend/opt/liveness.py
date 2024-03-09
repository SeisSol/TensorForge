from typing import List, Dict, Set, Union
from copy import copy
from collections import OrderedDict
from kernelforge.backend.symbol import Symbol
from kernelforge.backend.instructions.compute import ComputeInstruction
from kernelforge.backend.instructions.memory import MemoryInstruction
from kernelforge.backend.symbol import SymbolType
from .abstract import AbstractOptStage, Context, AbstractInstruction


class LivenessAnalysis(AbstractOptStage):
  def __init__(self, context: Context, instructions: List[AbstractInstruction]):
    super(LivenessAnalysis, self).__init__(context)

    self._instrs: List[AbstractInstruction] = instructions
    self._map: Dict[int, Set[Symbol]] = OrderedDict()
    self._live_map: Union[Dict[int, Set[Symbol]], None] = None

  def apply(self) -> None:
    self._map = {len(self._instrs): set()}

    for index, instr in reversed(list(enumerate(self._instrs))):
      self._map[index] = copy(self._map[index + 1])
      if isinstance(instr, ComputeInstruction):
        self._check_use(index, instr)
      elif isinstance(instr, MemoryInstruction):
        self._check_define(index, instr)

    self._live_map = OrderedDict(reversed(list(self._map.items())))

  def get_live_map(self) -> Dict[int, Set[Symbol]]:
    return self._live_map

  def _check_use(self, index, instr: ComputeInstruction) -> None:
    operands = instr.get_operands()
    for operand in operands:
      if operand.stype == SymbolType.SharedMem:
        self._map[index].add(operand)

  def _check_define(self, index, instr) -> None:
    self._map[index].remove(instr.get_dest())
