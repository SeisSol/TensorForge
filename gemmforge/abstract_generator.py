from abc import ABC, abstractmethod
from .vm import VM
from .exceptions import GenerationError, InternalError
from .basic_types import DataFlowDirection
from .basic_types import GeneralLexicon
from .common import get_extra_offset_name


class AbstractGenerator(ABC):
  ENCODING_LENGTH = 7

  def __init__(self, vm: VM):
    self._vm = vm
    self._lexic = self._vm.get_lexic()
    self._hw_descr = self._vm.get_hw_descr()
    self._precision = self._vm.fp_as_str()
    self._is_set = False

    self._base_name = None
    self._num_ops_per_block = None
    self._num_active_threads = None
    self._num_compute_threads = None
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
    def _deduce_num_threads(self):
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

  def _check_if_set(self):
    if not self._is_set:
      raise GenerationError(f'call to generate before set. Please, set params first')

  def get_base_name(self):
    if self._base_name is not None:
      return self._base_name
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

  def get_element_size_guard(self, writer):
    team_index_str = self._lexic.batch_indexer_gemm()
    writer(f'unsigned {GeneralLexicon.BATCH_ID} = {team_index_str};')
    return f'{GeneralLexicon.BATCH_ID} < {GeneralLexicon.NUM_ELEMENTS}'

  def get_flag_guard(self, writer):
    writer(f'bool isFlagsProvided = ({GeneralLexicon.FLAGS_ID} != nullptr);')
    flag_value = f'static_cast<bool>({GeneralLexicon.FLAGS_ID}[{GeneralLexicon.BATCH_ID}])'
    writer(f'bool allowed = isFlagsProvided ? {flag_value} : true;')
    return 'allowed'

  @abstractmethod
  def _get_func_params(self):
    params = [self._build_param(matrix) for matrix in self._matrices]
    params = ", ".join(params)
    return f'{params}, unsigned {GeneralLexicon.NUM_ELEMENTS}, unsigned* {GeneralLexicon.FLAGS_ID}'

  @abstractmethod
  def _get_launcher_params(self, with_defaults):
    params = [self._build_param(matrix) for matrix in self._matrices]
    params = ", ".join(params)
    nullptr = ' = nullptr' if with_defaults else ''
    return "{}, unsigned {}, unsigned* {}{}, void* {}{}".format(params,
                                                                GeneralLexicon.NUM_ELEMENTS,
                                                                GeneralLexicon.FLAGS_ID,
                                                                nullptr,
                                                                GeneralLexicon.STREAM_PTR_STR,
                                                                nullptr)

  @abstractmethod
  def _get_func_args(self):
    names = [f'{matrix.name}, {self._generate_extra_offset_symbol(matrix)}' for matrix in self._matrices]
    names = ", ".join(names)
    return f'{names}, {GeneralLexicon.NUM_ELEMENTS}, {GeneralLexicon.FLAGS_ID}'

  @abstractmethod
  def _get_block_dim_spec(self):
    if not (self._num_active_threads and self._num_ops_per_block):
      raise InternalError("kernel analysis hasn't been done yet")

  @abstractmethod
  def _get_grid_dim_spec(self):
    if not self._num_ops_per_block:
      raise InternalError("kernel analysis hasn't been done yet")

  def _build_param(self, matrix):
    sub_offset = f'int {self._generate_extra_offset_symbol(matrix)}'
    precision = self._vm.fp_as_str()
    if matrix.direction == DataFlowDirection.SINK:
      return f'{precision} {matrix.ptr_type} {matrix.name}, {sub_offset}'
    else:
      if matrix.addressing == "none":
        return f'const {precision} {matrix.ptr_type} __restrict__ {matrix.name}, {sub_offset}'
      else:
        return f'const {precision} {matrix.ptr_type} {matrix.name}, {sub_offset}'

  def _generate_extra_offset_symbol(self, matrix):
    # TODO: remove this
    return get_extra_offset_name(matrix.name)