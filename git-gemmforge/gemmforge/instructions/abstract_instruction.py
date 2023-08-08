from gemmforge.vm import VM
from gemmforge.symbol_table import InverseSymbolTable
from abc import ABC, abstractmethod


class AbstractInstruction(ABC):
  def __init__(self, vm: VM):
    self._vm: VM = vm
    self._is_ready = False

  @abstractmethod
  def gen_code(self, writer) -> None:
    return None

  @abstractmethod
  def __str__(self) -> str:
    pass

  def is_ready(self) -> bool:
    return self._is_ready

  def gen_mask_threads(self, num_threads) -> str:
    return f'{self._vm.get_lexic().thread_idx_x} < {num_threads}'
