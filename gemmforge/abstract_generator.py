from abc import ABC, abstractmethod
from .exceptions import GenerationError, InternalError


class AbstractGenerator(ABC):
  PRECISION = ["float", "double"]
  PRECISION_TO_BYTES = {"float": 4, "double": 8}
  NUM_ELEMENTS_STR = "NumElements"
  ENCODING_LENGTH = 7

  def __init__(self, arch, precision):
    self.arch = arch
    if precision in AbstractGenerator.PRECISION:
      self.precision = precision
    else:
      raise ValueError("given precision: {}. "
                       "Allowed values: {}".format(precision,
                                                   ", ".join(AbstractGenerator.PRECISION)))

    self.base_name = None
    self.num_mult_per_block = None
    self.num_active_threads = None
    self.num_compute_threads = None
    self.max_num_regs_per_thread = None
    self._matrices = []

    self._kernel = None
    self._launcher = None
    self._header = None

    @abstractmethod
    def generate(self):
      pass

    @abstractmethod
    def _check(self):
      pass

    @abstractmethod
    def _analyze(self):
      pass

    @abstractmethod
    def _generate_kernel(self):
      pass

    @abstractmethod
    def _generate_launcher(self):
      pass

    @abstractmethod
    def _generate_header(self):
      pass

    @abstractmethod
    def _generate_base_name(self):
      pass

  def get_base_name(self):
    if self.base_name is not None:
      return self.base_name
    else:
      raise InternalError("base name hasn't been set yet")

  def get_kernel(self):
    if self._kernel is not None:
      return self._kernel
    else:
      raise InternalError("kernel hasn't been generated")

  def get_launcher(self):
    if self._launcher is not None:
      return self._launcher
    raise InternalError("launcher hasn't been generated")

  def get_launcher_header(self):
    if self._header is not None:
      return self._header
    raise InternalError("launcher header hasn't been generated")

  @abstractmethod
  def _get_func_params(self):
    params = [self._build_param(matrix) for matrix in self._matrices]
    params = ", ".join(params)
    return "{}, unsigned {}".format(params, AbstractGenerator.NUM_ELEMENTS_STR)

  @abstractmethod
  def _get_func_args(self):
    names = [f'{matrix.name}, {self._generate_extra_offset_symbol(matrix)}' for matrix in self._matrices]
    names = ", ".join(names)
    return "{}, {}".format(names, AbstractGenerator.NUM_ELEMENTS_STR)

  @abstractmethod
  def _get_block_dim_spec(self):
    if not (self.num_active_threads and self.num_mult_per_block):
      raise InternalError("kernel analysis hasn't been done yet")

  @abstractmethod
  def _get_grid_dim_spec(self):
    if not self.num_mult_per_block:
      raise InternalError("kernel analysis hasn't been done yet")

  def _build_param(self, matrix):
    sub_offset = f'int {self._generate_extra_offset_symbol(matrix)}'
    if matrix.is_mutable():
      return f'{self.precision} {matrix.ptr_type} {matrix.name}, {sub_offset}'
    else:
      if matrix.addressing == "none":
        return f'const {self.precision} {matrix.ptr_type} __restrict__ {matrix.name}, {sub_offset}'
      else:
        return f'const {self.precision} {matrix.ptr_type} {matrix.name}, {sub_offset}'

  def _generate_extra_offset_symbol(self, matrix):
    return  f'ExtraOffset{matrix.name}'