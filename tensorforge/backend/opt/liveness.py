from typing import List, Dict, Set, Union
from copy import copy
from collections import OrderedDict
from tensorforge.backend.symbol import Symbol
from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.memory import AbstractShrMemWrite
from tensorforge.backend.instructions.memory.store import StoreShrMemToGlb
from tensorforge.backend.symbol import SymbolType
from .abstract import AbstractOptStage, Context, AbstractInstruction


class LivenessAnalysis(AbstractOptStage):
  def __init__(self, context: Context, instructions: List[AbstractInstruction]):
    super(LivenessAnalysis, self).__init__(context)

    self._instrs: List[AbstractInstruction] = instructions
    self._map: Dict[int, Set[Symbol]] = OrderedDict()
    self._live_map: Union[Dict[int, Set[Symbol]], None] = None

  def apply(self) -> None:
    forward_map = {-1: set()}

    for index, instr in list(enumerate(self._instrs)):
      forward_map[index] = copy(forward_map[index - 1])
      if isinstance(instr, AbstractShrMemWrite):
        self._check_define(forward_map, index, instr)

    backward_map = {len(self._instrs): set()}

    for index, instr in reversed(list(enumerate(self._instrs))):
      backward_map[index] = copy(backward_map[index + 1])
      if isinstance(instr, ComputeInstruction):
        self._check_use(backward_map, index, instr)
      if isinstance(instr, StoreShrMemToGlb):
        self._check_store(backward_map, index, instr)
    
    self._map = {}

    for index, instr in list(enumerate(self._instrs)):
      self._map[index] = forward_map[index].intersection(backward_map[index])

    self._live_map = OrderedDict(reversed(list(self._map.items())))

  def get_live_map(self) -> Dict[int, Set[Symbol]]:
    return self._live_map

  def _check_use(self, map, index, instr: ComputeInstruction) -> None:
    operands = instr.get_operands()
    for operand in operands:
      if operand.stype == SymbolType.SharedMem:
        map[index].add(operand)

  def _check_define(self, map, index, instr) -> None:
    map[index].add(instr.get_dest())
  
  def _check_store(self, map, index, instr) -> None:
    map[index].add(instr.get_src())
