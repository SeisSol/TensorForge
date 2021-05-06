from gemmforge.abstract_generator import AbstractGenerator
from gemmforge.vm import VM


class AbstractInitializer(AbstractGenerator):
  def __init__(self, vm: VM):
    super(AbstractInitializer, self).__init__(vm)
