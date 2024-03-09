from abc import ABC, abstractmethod
from typing import Union, List
from kernelforge.common.context import Context, VM
from kernelforge.backend.writer import Writer


class AbstractInstruction(ABC):
  def __init__(self, context: Context):
    if not isinstance(context, Context):
      raise RuntimeError(f'received wrong type, expected Context, given {type(context)}')

    self._context = context
    self._vm: VM = context.get_vm()
    self._fp_as_str = context.fp_as_str()
    self._is_ready = False

  @abstractmethod
  def gen_code(self, writer: Writer) -> None:
    return None
  
  def get_headers(self) -> List[str]:
    return []

  def is_ready(self) -> bool:
    return self._is_ready

  @abstractmethod
  def __str__(self) -> str:
    pass

  def gen_mask_threads(self, num_threads) -> str:
    return f'{self._vm.get_lexic().thread_idx_x} < {num_threads}'

  def gen_range_mask_threads(self, begin, end) -> str:
    assert begin < end
    tid = self._vm.get_lexic().thread_idx_x
    if begin == 0:
      return f'{tid} < {end}'
    else:
      return f'({tid} >= {begin}) && ({tid} < {end})'
  
  # @abstractmethod
  def get_perfdata(self):
    pass
