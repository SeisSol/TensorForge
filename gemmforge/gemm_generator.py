from . import constructs
from io import StringIO
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator
from .abstract_generator import AbstractGenerator as Generator
from .loaders import shm_mem_factory, StubLoader
import math
import hashlib


class GemmGenerator(GemmLikeGenerator):
  """ Generates GEMM GPU kernels: C = alpha * A * B + beta * C
  """
  TEAM_INDEX_STR = "(threadIdx.y + blockDim.y * blockIdx.x)"

  def __init__(self, arch, precision):
    super(GemmGenerator, self).__init__(arch, precision)
    self.mat_a = None
    self.mat_b = None
    self.mat_c = None
    self.mat_a_loader = None
    self.mat_b_loader = None

  def generate(self, mat_a, mat_b, mat_c, alpha, beta, base_name=None):
    self.mat_a = mat_a
    self.mat_a._set_name('A')
    self.mat_a._set_mutability(False)

    self.mat_b = mat_b
    self.mat_b._set_name('B')
    self.mat_b._set_mutability(False)

    self.mat_c = mat_c
    self.mat_c._set_name('C')
    self.mat_c._set_mutability(True)
    self._matrices = [self.mat_a, self.mat_b, self.mat_c]

    self.alpha = alpha
    self.beta = beta

    self.base_name = base_name if base_name is not None else self._generate_base_name()

    self._check()
    self._analyze()

    self._generate_kernel()
    self._generate_header()
    self._generate_launcher()

  def _generate_kernel(self):
    glob_symbols = {}
    for matrix in [self.mat_a, self.mat_b, self.mat_c]:
      glob_symbols[matrix.name] = "GlobMat{}".format(matrix.name)

    current_symbols = {}

    src = StringIO()
    with constructs.Cpp(src) as file:

      max_num_threads_per_block = self.num_active_threads * self.num_mult_per_block
      kernel_bounds = [max_num_threads_per_block]
      with file.Kernel(self.base_name, self._get_func_params(), kernel_bounds):
        with file.If("{} < {}".format(GemmGenerator.TEAM_INDEX_STR, Generator.NUM_ELEMENTS_STR)):

          # declare ptrs for correct matrices
          file.VariableDeclaration("const {}*".format(self.precision),
                                   glob_symbols[self.mat_a.name],
                                   self._get_global_matrix_ptr(self.mat_a))

          file.VariableDeclaration("const {}*".format(self.precision),
                                   glob_symbols[self.mat_b.name],
                                   self._get_global_matrix_ptr(self.mat_b))

          file.VariableDeclaration("{}*".format(self.precision),
                                   glob_symbols[self.mat_c.name],
                                   self._get_global_matrix_ptr(self.mat_c))

          # declare shared memory per kernel
          file.Expression("__shared__ {} Scratch[{}]".format(self.precision,
                                                             self._get_total_shared_mem_size()))

          # find address of matrix B within block shared memory
          shr_mem_address = "&Scratch[threadIdx.y * {}]".format(self.shr_mem_size_per_mult)
          file.VariableDeclaration("{}*".format(self.precision),
                                   self.mat_b_loader.get_output_symbol(),
                                   shr_mem_address)

          if self.mat_a.transpose:
            # find address of matrix A within block shared memory
            shr_mem_offset = self.mat_b_loader.compute_shared_mem_size()
            shr_mem_address = "&Scratch[threadIdx.y * {} + {}]".format(self.shr_mem_size_per_mult,
                                                                       shr_mem_offset)
            file.VariableDeclaration("{}*".format(self.precision),
                                     self.mat_a_loader.get_output_symbol(),
                                     shr_mem_address)

          # load matrices into shared memory
          self.mat_b_loader.generate_scr(file, glob_symbols[self.mat_b.name])
          self.mat_a_loader.generate_scr(file, glob_symbols[self.mat_a.name])
          file.Expression("__syncthreads()")

          # set up current compute symbols within the rest of the scope
          current_symbols[self.mat_b.name] = self.mat_b_loader.get_output_symbol()
          current_symbols[self.mat_a.name] = self.mat_a_loader.get_output_symbol()
          file.Emptyline()

          with file.If("threadIdx.x < {}".format(self.num_compute_threads)):
            # allocate a buffer for each cuda thread to hold computed results
            file.Emptyline()
            zero_fp_value = "0.0{}".format('f' if self.precision == "float" else '')
            file.ArrayDeclaration(self.precision,
                                  "Results",
                                  [zero_fp_value] * self.mat_c.get_actual_num_cols())

            file.VariableDeclaration(self.precision, "Value")


            # perform matrix multiplication
            # m, n, k - according to the BLAS documentation. Read BLAS spec.
            if self.mat_a.transpose:
              contraction_length = self.mat_a.get_actual_num_rows()
            else:
              contraction_length = self.mat_a.get_actual_num_cols()

            file.Emptyline()
            with file.For("int k = 0; k < {}; ++k".format(contraction_length)):
              first_operand = "{}[threadIdx.x + {} * k]".format(current_symbols[self.mat_a.name],
                                                                self.mat_a_loader.get_lid_dim())
              file.Assignment("Value", "{}".format(first_operand))

              """
              # EXPEREMENTAL
              # perform prefetch if possible
              if self.mat_a.addressing != 'none' and isinstance(self.mat_a_loader, StubLoader):
                # In other words, if matrix a is not going to be implicitly cached 
                # AND
                # it is going to reside on global memory without loading into the shared memory
                # (in case of matrix 'a' is transposed)
                next_addrs = "{} + threadIdx.x + {} * (k + 1)".format(current_symbols[self.mat_a.name],
                                                                      self.mat_a_loader.get_lid_dim())
                file.Expression(f'asm(" prefetch.global.L2 [ %0 ];" : : "l"({next_addrs}))')
              """

              file.Emptyline()
              file.Pragma("unroll")
              with file.For("int n = 0; n < {}; ++n".format(self.mat_c.get_actual_num_cols())):
                if self.mat_b.transpose:
                  second_operand = "{}[n + {} * k]".format(current_symbols[self.mat_b.name],
                                                           self.mat_b_loader.get_lid_dim())
                else:
                  second_operand = "{}[k + {} * n]".format(current_symbols[self.mat_b.name],
                                                           self.mat_b_loader.get_lid_dim())

                file.Accumulate("Results[n]",
                                "Value * {}".format(second_operand))

            # write results back to memory
            file.Emptyline()
            file.Pragma("unroll")
            with file.For("int n = 0; n < {}; ++n".format(self.mat_c.get_actual_num_cols())):
              rhs = "{}[threadIdx.x + {} * n]".format(glob_symbols[self.mat_c.name],
                                                        self.mat_c.num_rows)

              if self.alpha == 1.0:
                lhs = "Results[n]"
              else:
                if self.precision == "float" and isinstance(self.alpha, float):
                  lhs = f'{self.alpha}f * Results[n]'
                else:
                  lhs = f'{self.alpha} * Results[n]'

              if self.beta != 0.0:
                if self.beta == 1.0:
                  lhs += " + {}".format(rhs)
                else:
                  lhs += " + {} * {}".format("{}{}".format(self.beta, 'f' if self.precision == "float" else ''),
                                                           rhs)

              file.Assignment(rhs, lhs)

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

  def _check(self):
    try:
      # make sure that C is not transposed
      if self.mat_c.transpose:
        raise GenerationError("Cannot generate a matrix multiplication. "
                               "Matrix C is transposed")

      # check whether C and A match each other
      if self.mat_a.transpose:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (NoTrans) do not match")

      # check whether C and B match each other
      if self.mat_b.transpose:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (NoTrans) do not match")

      # check whether A and B match each other
      if self.mat_a.transpose:
        if self.mat_b.transpose:
          if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_cols():
            raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                  "Matrix A (Trans) and B (Trans) do not match")
        else:
          if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_rows():
            raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                  "Matrix A (Trans) and B (NoTrans) do not match")
      else:
        if self.mat_b.transpose:
          if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
            raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                  "Matrix A (NoTrans) and B (Trans) do not match")
        else:
          if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
            raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                  "Matrix A (NoTrans) and B (NoTrans) do not match")

    except GenerationError as error:
      matrices = {"A": self.mat_a, "B": self.mat_b, "C": self.mat_c}
      for name in matrices:
        print("Matrix {}:".format(name))
        print(matrices[name])
        print("=" * 80)

      raise error

  def _estimate_num_registers_per_mult(self, contraction_length):
    factor = GemmGenerator.PRECISION_TO_BYTES[self.precision] / 4
    return factor * (32 + contraction_length)

  def _analyze(self):
    if self.mat_a.transpose:
      lid_dim_length = self.mat_a.get_actual_num_cols()
    else:
      lid_dim_length = self.mat_a.get_actual_num_rows()

    num_vector_units_required = math.ceil(lid_dim_length / self.arch.vec_unit_length)
    self.num_compute_threads = lid_dim_length
    self.num_active_threads = num_vector_units_required * self.arch.vec_unit_length

    if self.mat_a.transpose:
      self.mat_a_loader = shm_mem_factory(matrix=self.mat_a,
                                          num_active_threads=self.num_active_threads,
                                          load_and_transpose=True)

    else:
      self.mat_a_loader = StubLoader(self.mat_a, self.num_active_threads)

    self.mat_b_loader = shm_mem_factory(matrix=self.mat_b,
                                        num_active_threads=self.num_active_threads,
                                        load_and_transpose=False)

    self.max_num_regs_per_thread = self._estimate_num_registers_per_mult(lid_dim_length)

    self.shr_mem_size_per_mult = self.mat_a_loader.compute_shared_mem_size() \
                                 + self.mat_b_loader.compute_shared_mem_size()

    shr_mem_bytes = self.shr_mem_size_per_mult * Generator.PRECISION_TO_BYTES[self.precision]
    mults_wrt_shr_mem = self.arch.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = self.arch.max_reg_per_block / (self.num_active_threads * self.max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))

    self.num_mult_per_block = max(int(mults_per_sm / self.arch.max_block_per_sm), 1)

  def _get_total_shared_mem_size(self):
    return self.shr_mem_size_per_mult * self.num_mult_per_block

  def _generate_base_name(self):
    if self.mat_a.transpose:
      dim1 = "m{}_{}".format(self.mat_a.get_actual_num_cols(), self.mat_a.num_rows)
      dim3 = "k{}".format(self.mat_a.get_actual_num_rows())
    else:
      dim1 = "m{}_{}".format(self.mat_a.get_actual_num_rows(), self.mat_a.num_rows)
      dim3 = "k{}".format(self.mat_a.get_actual_num_cols())

    if self.mat_b.transpose:
      dim2 = "n{}_{}".format(self.mat_b.get_actual_num_rows(), self.mat_b.num_rows)
    else:
      dim2 = "n{}_{}".format(self.mat_b.get_actual_num_cols(), self.mat_b.num_rows)

    dims = "{}_{}_{}".format(dim1, dim2, dim3)
    
    addressing = "{}{}{}".format(self.mat_a.addressing[0],
                                 self.mat_b.addressing[0],
                                 self.mat_c.addressing[0])

    traspose = "{}_{}".format("T" if self.mat_a.transpose else "NT",
                              "T" if self.mat_b.transpose else "NT")

    constants = "{}_{}".format(self.alpha, self.beta)

    result = hashlib.md5(("{}_{}{}{}".format(constants,
                                             self.mat_a.__str__(),
                                             self.mat_b.__str__(),
                                             self.mat_c.__str__()).encode()))
    md5encoding = result.hexdigest()
    prefix = 's' if self.precision == "float" else "d"

    # TODO: the below line is for debugging

    gemm_dims = f'm{self.mat_a.get_actual_num_rows()}_n{self.mat_b.get_actual_num_cols()}_k{self.mat_a.get_actual_num_cols()}'
    ldas = f'lda{self.mat_a.num_rows}_ldb{self.mat_b.num_rows}_ldc{self.mat_c.num_rows}'
    consts = f'alpha_{int(self.alpha)}_beta_{int(self.beta)}'
    """
    return "{}gemm_{}_{}_{}_{}".format(prefix,
                                       traspose,
                                       dims,
                                       addressing,
                                       md5encoding[:Generator.ENCODING_LENGTH])
    """
    return "{0}gemm_{1}_{2}_{3}_{4}_{5}_{6}".format(prefix,
                                                traspose,
                                                gemm_dims,
                                                ldas,
                                                consts,
                                                addressing,
                                                md5encoding[:Generator.ENCODING_LENGTH])

  def _get_func_params(self):
    base_params = super(GemmGenerator, self)._get_func_params()
    if isinstance(self.alpha, float):
      return base_params
    else:
      return f'{self.precision} {self.alpha}, {base_params}'

  def _get_func_args(self):
    base_args = super(GemmGenerator, self)._get_func_args()
    if isinstance(self.alpha, float):
      return base_args
    else:
      return f'{self.alpha}, {base_args}'


  def _get_block_dim_spec(self):
   super(GemmGenerator, self)._get_block_dim_spec()
   return f'Block({self.num_active_threads}, {self.num_mult_per_block}, 1)'

  def _get_grid_dim_spec(self):
    super(GemmGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(Generator.NUM_ELEMENTS_STR,
                                                self.num_mult_per_block)
    return f'Grid({num_blocks}, 1, 1)'

  def _get_global_matrix_ptr(self, matrix):

    extra_offset_symbol = self._generate_extra_offset_symbol(matrix)
    if matrix.addressing == "strided":
      main_offset = "{} * {}".format(GemmGenerator.TEAM_INDEX_STR, matrix.get_real_volume())
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {} + {}]".format(matrix.name,
                                        main_offset,
                                        sub_offset,
                                        extra_offset_symbol)

    elif matrix.addressing == "pointer_based":
      main_offset = GemmGenerator.TEAM_INDEX_STR
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{}][{} + {}]".format(matrix.name,
                                       main_offset,
                                       sub_offset,
                                       extra_offset_symbol)

    else:
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {}]".format(matrix.name, sub_offset, extra_offset_symbol)
