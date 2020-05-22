from abc import ABC, abstractmethod
from ..exceptions import GenerationError, InternalError
from ..abstract_generator import AbstractGenerator


class AbstractInitializer(AbstractGenerator):
  def __init__(self, arch, precision):
    super(AbstractInitializer, self).__init__(arch, precision)