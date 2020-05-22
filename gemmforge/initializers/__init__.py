from .abstract_initializer import ExactInitializer
from ..abstract_generator import AbstractGenerator


class StubInitializer(AbstractGenerator):
  def __init__(self, arch, precision):
    super(StubInitializer, self).__init__(arch, precision)

  def generate(self):
    self._generate_header()
    self._generate_kernel()
    self._generate_launcher()

  def _check(self):
    pass

  def _analyze(self):
    pass

  def _generate_kernel(self):
    self._kernel = ' '

  def _generate_launcher(self):
    self._launcher = ' '

  def _generate_header(self):
    self._header = ' '

  def _generate_base_name(self):
    pass

  def _get_func_params(self):
    pass

  def _get_func_args(self):
    pass

  def _get_block_dim_spec(self):
    pass

  def _get_grid_dim_spec(self):
    pass

  def func_call(self, args):
    return ''


def initializer_factory(init_value, matrix, arch, precision):
  if init_value == 1.0:
    return StubInitializer(arch, precision)
  else:
    return ExactInitializer(init_value, matrix, arch, precision)

