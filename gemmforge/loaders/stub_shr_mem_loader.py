from abc import ABC, abstractmethod
from .abstract_loader import AbstractShrMemLoader


class StubLoader(AbstractShrMemLoader):
  """A stub (zero) loader i.e. doesn't load anything. Purpose: to keep the same interface

  """

  def __init__(self, matrix, num_active_threads):
    super(StubLoader, self).__init__(matrix, num_active_threads)
    self.lid_dim = self.matrix.num_rows

  def compute_shared_mem_size(self):
      return 0

  def generate_scr(self, file, in_symbol):
    self.out_symbol = in_symbol
