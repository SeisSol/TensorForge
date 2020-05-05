from abc import ABC, abstractmethod
from .exceptions import GenerationError, InternalError


class Generator(ABC):
  PRECISION = ["float", "double"]
  PRECISION_TO_BYTES = {"float": 4, "double": 8}
  TEAM_INDEX_STR = "(threadIdx.y + blockDim.y * blockIdx.x)"
  NUM_ELEMENTS_STR = "NumElements"
  ENCODING_LENGTH = 7

  def __init__(self, arch, precision):
    self.arch = arch
    if precision in Generator.PRECISION:
      self.precision = precision
    else:
      raise ValueError("given precision: {}. "
                       "Allowed values: {}".format(precision,
                                                   ", ".join(Generator.PRECISION)))

    self.base_name = None
    self.alpha = None
    self.beta = None
    self.num_mult_per_block = None
    self.shr_mem_size_per_mult = None
    self.num_active_threads = None
    self.num_compute_threads = None
    self.max_num_regs_per_thread = None
    self._matrices = []

    self._kernel = None
    self._launcher = None
    self._header = None

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
    def _get_total_shared_mem_size(self):
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

  def _get_func_params(self):
    params = [self._build_param(matrix) for matrix in self._matrices]
    params = ", ".join(params)
    return "{}, unsigned {}".format(params, Generator.NUM_ELEMENTS_STR)

  def _get_func_args(self):
    names = [matrix.name for matrix in self._matrices]
    names = ", ".join(names)
    return "{}, {}".format(names, Generator.NUM_ELEMENTS_STR)

  def _get_block_dim_spec(self):
    if self.num_active_threads and self.num_mult_per_block:
      return "Block({}, {}, 1)".format(self.num_active_threads,
                                       self.num_mult_per_block)
    else:
      raise InternalError("kernel analysis hasn't been done yet")

  def _get_grid_dim_spec(self):
    if self.num_mult_per_block:
      num_blocks = "({0} + {1} - 1) / {1}".format(Generator.NUM_ELEMENTS_STR,
                                                  self.num_mult_per_block)
      return "Grid({}, 1, 1)".format(num_blocks)
    else:
      raise InternalError("kernel analysis hasn't been done yet")

  def _build_param(self, matrix):
    if matrix.is_mutable():
      return "{} {} {}".format(self.precision, matrix.ptr_type, matrix.name)
    else:
      if matrix.addressing == "none":
        return "const {} {} __restrict__ {}".format(self.precision, matrix.ptr_type, matrix.name)
      else:
        return "const {} {} {}".format(self.precision, matrix.ptr_type, matrix.name)