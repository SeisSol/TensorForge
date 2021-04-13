from gemmforge.abstract_generator import AbstractGenerator


class AbstractInitializer(AbstractGenerator):
  def __init__(self, arch, precision):
    super(AbstractInitializer, self).__init__(arch, precision)
