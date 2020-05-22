from ..abstract_generator import AbstractGenerator
from .. import constructs
from io import StringIO
import math
import hashlib


class ExactInitializer(AbstractGenerator):
  TEAM_INDEX_STR = "(threadIdx.z + blockDim.z * blockIdx.x)"

  def __init__(self, init_value, matrix, arch, precision):
    super(ExactInitializer, self).__init__(arch, precision)

    self.init_value = init_value

    self.matrix = matrix
    self.matrix._set_name('A')
    self.matrix._set_mutability(True)
    self._matrices = [self.matrix]

  def generate(self, base_name=None):
    self.base_name = base_name if base_name is not None else self._generate_base_name()

    self._check()
    self._analyze()

    self._generate_kernel()
    self._generate_header()
    self._generate_launcher()

  def _check(self):
    pass

  def _analyze(self):
    lid_dim_length = self.matrix.get_actual_num_rows()

    # we use active threads to add a single column
    num_vector_units_required = math.ceil(lid_dim_length / self.arch.vec_unit_length)
    self.num_compute_threads = lid_dim_length
    self.num_active_threads = num_vector_units_required * self.arch.vec_unit_length

    total_num_threas_per_op = self.num_active_threads * self.matrix.get_actual_num_cols()

    self.max_num_regs_per_thread = 10
    mults_wrt_num_regs = self.arch.max_reg_per_block / (total_num_threas_per_op * self.max_num_regs_per_thread)
    self.num_mult_per_block = max(int(mults_wrt_num_regs / self.arch.max_block_per_sm), 1)

  def _generate_kernel(self):
    global_symbols = {self.matrix.name: f'Glob{self.matrix.name}'}
    src = StringIO()
    with constructs.Cpp(src) as file:
      total_num_threas_per_op = self.num_active_threads * self.matrix.get_actual_num_cols()
      max_num_threads_per_block = total_num_threas_per_op * self.num_mult_per_block
      kernel_bounds = [max_num_threads_per_block]
      with file.Kernel(self.base_name, self._get_func_params(), kernel_bounds):
        with file.If("{} < {}".format(ExactInitializer.TEAM_INDEX_STR, AbstractGenerator.NUM_ELEMENTS_STR)):

          # declare ptrs for correct matrices
          file.VariableDeclaration("{}*".format(self.precision),
                                   global_symbols[self.matrix.name],
                                   self._get_global_matrix_ptr(self.matrix))

          # assign initial value to a matrix element
          with file.If("threadIdx.x < {}".format(self.matrix.get_actual_num_rows())):
              file.Assignment(f'{global_symbols[self.matrix.name]}[threadIdx.x]',
                              f'{self.init_value}')

      self._kernel = src.getvalue()

  def _generate_launcher(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      with file.Function(self.base_name, self._get_func_params()):
        file.VariableDeclaration("dim3", self._get_block_dim_spec())
        file.VariableDeclaration("dim3", self._get_grid_dim_spec())
        krnl_launch_param = "<<<Grid,Block>>>"
        file.Expression("kernel_{}{}({})".format(self.base_name,
                                                 krnl_launch_param,
                                                 self._get_func_args()))
        file.Expression("CHECK_ERR")
      self._launcher = src.getvalue()

  def _generate_header(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      file.FunctionDeclaration(self.base_name, self._get_func_params())
      content = src.getvalue()
    self._header = content

  def _generate_base_name(self):
    dim = f'm{self.matrix.get_actual_num_rows()}_{self.matrix.num_rows}'
    addressing = f'{self.matrix.addressing[0]}'

    result = hashlib.md5(f'{self.init_value}_{self.matrix.__str__()}'.encode())
    md5encoding = result.hexdigest()

    return "initialize_{}_{}_{}".format(dim,
                                        addressing,
                                        md5encoding[:AbstractGenerator.ENCODING_LENGTH])

  def _get_func_params(self):
    base_params = super(ExactInitializer, self)._get_func_params()
    if isinstance(self.init_value, float):
      return base_params
    else:
      return f'{self.precision} {self.init_value}, {base_params}'

  def _get_func_args(self):
    base_args = super(ExactInitializer, self)._get_func_args()
    if isinstance(self.init_value, float):
      return base_args
    else:
      return f'{self.init_value}, {base_args}'

  def _get_block_dim_spec(self):
    super(ExactInitializer, self)._get_block_dim_spec()
    return f'Block({self.num_active_threads}, {self.matrix.get_actual_num_cols()}, {self.num_mult_per_block})'

  def _get_grid_dim_spec(self):
    super(ExactInitializer, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(AbstractGenerator.NUM_ELEMENTS_STR,
                                                self.num_mult_per_block)
    return f'Grid({num_blocks}, 1, 1)'

  def _get_global_matrix_ptr(self, matrix):
    extra_offset_symbol = self._generate_extra_offset_symbol(matrix)
    offset_to_row = f'threadIdx.y * {matrix.num_rows}'

    if matrix.addressing == "strided":
      main_offset = "{} * {}".format(ExactInitializer.TEAM_INDEX_STR, matrix.get_real_volume())
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {} + {} + {}]".format(matrix.name,
                                             extra_offset_symbol,
                                             main_offset,
                                             sub_offset,
                                             offset_to_row)

    elif matrix.addressing == "pointer_based":
      main_offset = ExactInitializer.TEAM_INDEX_STR
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{}][{} + {} + {}]".format(matrix.name,
                                            main_offset,
                                            extra_offset_symbol,
                                            sub_offset,
                                            offset_to_row)

    else:
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {} + {}]".format(matrix.name, sub_offset, extra_offset_symbol, offset_to_row)

  def func_call(self, args):
    return f'{self.base_name}({", ".join(args)});'
