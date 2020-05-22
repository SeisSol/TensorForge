from abc import ABC, abstractmethod
from .exceptions import GenerationError, InternalError
from .abstract_generator import AbstractGenerator


class GemmLikeGenerator(AbstractGenerator, ABC):

  def __init__(self, arch, precision):
    super(GemmLikeGenerator, self).__init__(arch, precision)

    self.alpha = None
    self.beta = None
    self.shr_mem_size_per_mult = None

    @abstractmethod
    def _get_total_shared_mem_size(self):
      pass