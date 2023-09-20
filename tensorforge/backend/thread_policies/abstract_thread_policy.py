from tensorforge.vm import VM
from ..matrix import DenseMatrix
from abc import ABC, abstractmethod


class AbstractUniOpThreadPolicy(ABC):
  def __init__(self,
               vm: VM,
               num_threads: int,
               op1: DenseMatrix):
    self._vm = vm
    self._num_threads = num_threads
    self._op1 = op1

  @abstractmethod
  def get_num_ops_per_block(self):
    pass


class AbstractBinaryOpThreadPolicy(AbstractUniOpThreadPolicy):
  def __init__(self,
               vm: VM,
               num_threads: int,
               op1: DenseMatrix,
               op2: DenseMatrix):
    super().__init__(vm, num_threads, op1)
    self._op2 = op2

  @abstractmethod
  def get_num_ops_per_block(self):
    pass


class AbstractGemmLikeThreadPolicy(AbstractBinaryOpThreadPolicy):
  def __init__(self,
               vm: VM,
               num_threads: int,
               op1: DenseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(vm, num_threads, op1, op2)
    self._res = res

  @abstractmethod
  def get_num_ops_per_block(self):
    pass
