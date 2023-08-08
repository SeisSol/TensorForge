from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.exceptions import InternalError


class SyncThreads(AbstractInstruction):
  def __init__(self, vm, num_threads_per_mult):
    super().__init__(vm)
    self._num_threads = num_threads_per_mult
    self._is_ready = True

  def gen_code(self, writer):
    writer(f'{self.__str__()}')

  def __str__(self) -> str:
    lexic = self._vm.get_lexic()
    if self._num_threads > self._vm.get_hw_descr().vec_unit_length:
      return f'{lexic.sync_threads()};'
    else:
      return f'{lexic.sync_vec_unit()};'

  def gen_mask_threads(self, num_threads) -> str:
    return ''
