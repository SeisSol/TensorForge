from abc import ABC, abstractmethod
from .exceptions import GenerationError, InternalError
from .abstract_generator import AbstractGenerator


class GemmLikeGenerator(AbstractGenerator, ABC):
  def __init__(self, vm):
    super(GemmLikeGenerator, self).__init__(vm)
    
    self.alpha = None
    self.beta = None
    self.shr_mem_size_per_mult = None
