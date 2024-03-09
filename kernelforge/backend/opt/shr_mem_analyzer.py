from typing import List, Dict, Union, Tuple, Set
from kernelforge.backend.instructions.memory import MemoryInstruction
from kernelforge.backend.symbol import Symbol
from kernelforge.backend.data_types import ShrMemObject
from kernelforge.common.exceptions import GenerationError
from .abstract import AbstractOptStage, Context
from .mem_region_allocation import Region


class ShrMemOpt(AbstractOptStage):
  def __init__(self,
               context: Context,
               shr_mem_obj: ShrMemObject,
               regions: List[Region],
               live_map: Dict[int, Set[Symbol]]):
    super(ShrMemOpt, self).__init__(context)

    self._shr_mem_obj: ShrMemObject = shr_mem_obj
    self._regions: List[Region] = regions
    self._live_map: Dict[int, Set[Symbol]] = live_map

  def apply(self) -> None:
    self._check_regions()

    max_memory, mem_per_region = self._compute_total_shr_mem_size()
    self._shr_mem_obj.set_size_per_mult(max_memory)

    offsets = self._compute_start_addresses(mem_per_region)
    self._assign_offsets(offsets)

  def _check_regions(self) -> None:
    for region in self._regions:
      for symbol in region:
        first_user = symbol.get_first_user()
        if not isinstance(first_user, MemoryInstruction):
          raise GenerationError(f'expected the first user of symbol {symbol.name} '
                                f'to be a subtype AbstractShrMemLoader or StoreRegToShr')

  def _compute_total_shr_mem_size(self) -> Tuple[int, List[int]]:
    max_memory: int = 0
    max_mem_per_region: List[int] = [0 for region in self._regions]
    for index, region in enumerate(self._regions):
      for symbol in region:
        instruction: MemoryInstruction = symbol.get_first_user()
        max_mem_per_region[index] = max(instruction.compute_shared_mem_size(),
                                        max_mem_per_region[index])
      max_memory += max_mem_per_region[index]
    return max_memory, max_mem_per_region

  def _compute_start_addresses(self, mem_per_region: List[int]) -> List[int]:
    num_regions: int = len(mem_per_region)
    offsets: List[int] = [0] * num_regions
    for index in range(1, num_regions):
      offsets[index] += mem_per_region[index - 1]
    return offsets

  def _assign_offsets(self, offsets: List[int]):
    for offset, region in zip(offsets, self._regions):
      for symbol in region:
        shr_mem_instr = symbol.get_first_user()
        shr_mem_instr.set_shr_mem_offset(offset)
