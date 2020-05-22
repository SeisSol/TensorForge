from . import constructs
from io import StringIO
from math import ceil
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator

"""
Generates Double Gemm kernels for GPU
"""
class TriadGenerator(GemmLikeGenerator):
  def __init__(self, arch, precision):
    super(TriadGenerator, self).__init__(arch, precision)
    self.mat_d = None
    self.mat_c = None
    self.mat_a = None
    self.mat_b = None

  def generate(self, mat_d, mat_a, mat_b, mat_c, alpha, beta, base_name=None):
    raise GenerationError("Triad operation has not been implemented yet")
    self.mat_d = mat_d
    self.mat_d._set_name('D')
    self.mat_d._set_mutability(True)

    self.mat_a = mat_a
    self.mat_a._set_name('A')
    self.mat_a._set_mutability(False)

    self.mat_b = mat_b
    self.mat_b._set_name('B')
    self.mat_b._set_mutability(False)

    self.mat_c = mat_c
    self.mat_c._set_name('C')
    self.mat_c._set_mutability(False)

    self._matrices = [self.mat_d, self.mat_c, self.mat_a, self.mat_b]

    self.alpha = alpha
    self.beta = beta

    self.base_name = base_name if base_name is not None else self._generate_base_name()

    self._check()
    self._analyze()

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

  def _generate_kernel(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      file.Include("gemmgen_aux.h")
      with file.Kernel(self.base_name, self._get_func_params()):
        with file.If("{} < {}".format(Generator.TEAM_INDEX_STR, Generator.NUM_ELEMENTS_STR)):
          """
          # declare ptrs for correct matrices
          file.VariableDeclaration("const {}*".format(self.precision),
                                   "MatA",
                                   self._get_global_matrix_ptr(self.mat_a))

          file.VariableDeclaration("const {}*".format(self.precision),
                                   "GlobMatB",
                                   self._get_global_matrix_ptr(self.mat_b))

          file.VariableDeclaration("{}*".format(self.precision),
                                   "MatC",
                                   self._get_global_matrix_ptr(self.mat_c))

          # declare shared memory
          '''
          file.ArrayDeclaration("extern __shared__ {}".format(self.precision),
                                "Scratch")
          '''
          file.Expression("__shared__ {} Scratch[{}]".format(self.precision,
                                                             self._get_total_shared_mem_size()))

          # find self-location within shared memory
          shr_mem_offset = "&Scratch[threadIdx.y * {}]".format(self.mat_b.get_actual_volume())
          file.VariableDeclaration("{}*".format(self.precision),
                                   "ShrMatB",
                                   shr_mem_offset)

          # TODO: Make decision how to load data into shared memory
          # somethimes a manual unroll performs better than compiler unroll
          # fill shared mem with data
          file.Emptyline()
          num_hops = int(self.mat_b.get_actual_volume() / self.num_active_threads)
          if num_hops > 2:
            file.Pragma("unroll")
            with file.For("int i = threadIdx.x; i < {}; i += blockDim.x".format(self.mat_b.get_actual_volume())):
              file.Assignment("ShrMatB[i]", "GlobMatB[i]")

          else:

            counter = 0
            while counter < num_hops:
              shr_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
              glob_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
              file.Assignment("ShrMatB[{}]".format(shr_mem_index),
                              "GlobMatB[{}]".format(glob_mem_index))
              counter += 1

            # the last hop to fill shared mem with data
            if (self.mat_b.get_actual_volume() % self.num_active_threads) != 0:
              residue = self.mat_b.get_actual_volume() - counter * self.num_active_threads
              with file.If("threadIdx.x < {}".format(residue)):
                shr_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
                glob_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
                file.Assignment("ShrMatB[{}]".format(shr_mem_index),
                                "GlobMatB[{}]".format(glob_mem_index))
          file.Expression("__syncthreads()")

          # allocate a buffer for each cuda thread
          # type, left, expresion=None
          file.Emptyline()
          zero_fp_value = "0.0{}".format('f' if self.precision == "float" else '')
          file.ArrayDeclaration(self.precision,
                                "Results",
                                [zero_fp_value] * self.mat_c.get_actual_num_cols())

          file.VariableDeclaration(self.precision, "Value")

          # perform matrix multiplication
          # m, n, k - according to the BLAS documentation. Read BLAS spec.
          file.Emptyline()
          with file.For("int k = 0; k < {}; ++k".format(self.mat_a.get_actual_num_cols())):
            first_operand = "MatA[threadIdx.x + {} * k]".format(self.mat_a.num_rows)
            file.Assignment("Value", "{}".format(first_operand))

            file.Emptyline()
            file.Pragma("unroll")
            with file.For("int n = 0; n < {}; ++n".format(self.mat_b.get_actual_num_cols())):
              second_operand = "ShrMatB[k + {} * n]".format(self.mat_b.get_actual_num_rows())
              file.Accumulate("Results[n]",
                              "Value * {}".format(second_operand))

          '''
          for k in range(self.mat_b.get_actual_num_rows()):
            first_operand = "MatA[threadIdx.x + {}]".format(k * self.mat_a.num_rows)
            file.Assignment("Value", first_operand)
            for n in range(self.mat_b.get_actual_num_cols()):
              second_operand = "ShrMatB[{}]".format(k + n * self.mat_b.get_actual_num_rows())
              file.Accumulate("Results[{}]".format(n),
                              "Value * {}".format(second_operand))
          '''

          # write results back to memory
          file.Emptyline()
          file.Pragma("unroll")
          with file.For("int n = 0; n < {}; ++n".format(self.mat_b.get_actual_num_cols())):
            rhs = "MatC[threadIdx.x + {} * n]".format(self.mat_c.num_rows)

            if self.alpha == 1.0:
              lhs = "Results[n]"
            else:
              lhs = "{} * Results[n]".format("{}{}".format(self.alpha, 'f' if self.precision == "float" else ''))

            if self.beta != 0.0:
              if self.beta == 1.0:
                lhs += " + {}".format(rhs)
              else:
                lhs += " + {} * {}".format("{}{}".format(self.beta, 'f' if self.precision == "float" else ''),
                                           rhs)

            file.Assignment(rhs, lhs)
          '''
          file.Emptyline()
          for column in range(self.mat_c.get_actual_num_cols()):
            rhs = "MatC[threadIdx.x + {}]".format(column * self.mat_c.num_rows)

            if self.alpha == 1.0:
              lhs = "Results[{}]".format(column)
            else:
              lhs = "{} * Results[{}]".format("{}{}".format(self.alpha,
                                                            'f' if self.precision == "float" else ''),
                                              column)

            if self.beta != 0.0:
              if self.beta == 1.0:
                lhs += " + {}".format(rhs)
              else:
                lhs += " + {} * {}".format("{}{}".format(self.beta,
                                                         'f' if self.precision == "float" else ''),
                                           rhs)

            file.Assignment(rhs, lhs)
            '''
          """
      self._kernel = src.getvalue()

  def _generate_launcher(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      with file.Function(self.base_name, self._get_func_params()):
        """
        file.VariableDeclaration("dim3", self._get_block_dim_spec())
        file.VariableDeclaration("dim3", self._get_grid_dim_spec())
        '''
        file.Expression("unsigned ShrMemSize = {} * sizeof({})".format(self._get_total_shared_mem_size(),
                                                                       self.precision))
        '''
        krnl_launch_param = "<<<Grid,Block>>>"
        # krnl_launch_param = "<<<Grid,Block,ShrMemSize>>>"
        file.Expression("kernel_{}{}({})".format(self.base_name,
                                                 krnl_launch_param,
                                                 self._get_func_args()))
        file.Expression("CHECK_ERR")
        """
      self._launcher = src.getvalue()

  def _generate_header(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      file.FunctionDeclaration(self.base_name, self._get_func_params())
      self._header = src.getvalue()

  def _check(self):
    if self.mat_d.get_actual_num_rows() != self.mat_c.get_actual_num_rows():
      raise GenerationError("cannot generate a triad matrix multiplication "
                            "with given parameters. Matrix D and C do not match")

    if self.mat_c.get_actual_num_cols() != self.mat_a.get_actual_num_rows():
      raise GenerationError("cannot generate a triad matrix multiplication "
                            "with given parameters. Matrix C and A do not match")

    if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
      raise GenerationError("cannot generate a triad matrix multiplication "
                            "with given parameters. Matrix A and B do not match")

  def _analyze(self):
    """
    # Adjust analysis to multiple of 32
    self.num_active_threads = self.mat_a.get_actual_num_rows()

    # num_active_threads = (self.mat_a.get_actual_num_rows() / self.arch.vec_unit_length) * self.arch.vec_unit_length
    num_active_threads = self.mat_a.get_actual_num_rows()
    shr_mem_bytes = self.mat_b.get_actual_volume() * Generator.PRECISION_TO_BYTES[self.precision]

    mults_wrt_shr_mem = self.arch.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = self.arch.max_reg_per_block / (num_active_threads * self.max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))

    # mults_wrp_threads = self.arch.max_num_threads / num_active_threads

    # TODO: must be 4
    self.num_mult_per_block = int(mults_per_sm / self.arch.max_block_per_sm)
    """

  """
  def _get_func_params(self):
    return "{}, {}, {}, unsigned {}".format(self._build_param(self.mat_a),
                                            self._build_param(self.mat_b),
                                            self._build_param(self.mat_c, mutalble=True),
                                            Generator.NUM_ELEMENTS_STR)
  """

  """
  def _get_func_args(self):
    return "{}, {}, {}, {}".format(self.mat_d.name,
                                   self.mat_a.name,
                                   self.mat_b.name,
                                   self.mat_c.name,
                                   Generator.NUM_ELEMENTS_STR)
  """

  def _get_total_shared_mem_size(self):
    return self.shr_mem_size_per_mult * self.num_mult_per_block

  def _get_global_matrix_ptr(self, matrix):
    """
    if matrix.addressing == "strided":
      main_offset = "{} * {}".format(Generator.TEAM_INDEX_STR, matrix.get_real_volume())
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {}]".format(matrix.name,
                                   main_offset,
                                   sub_offset)

    elif matrix.addressing == "pointer_based":
      main_offset = Generator.TEAM_INDEX_STR
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{}][{}]".format(matrix.name,
                                  main_offset,
                                  sub_offset)

    else:
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{}]".format(matrix.name, sub_offset)
    """


  def _generate_base_name(self):
    dim1 = "l{}_{}".format(self.mat_c.num_rows, self.mat_c.get_actual_num_rows())
    dim2 = "m{}_{}".format(self.mat_a.num_rows, self.mat_a.get_actual_num_rows())
    dim3 = "n{}_{}".format(self.mat_b.num_cols, self.mat_b.get_actual_num_cols())
    dim4 = "k{}_{}".format(self.mat_a.num_cols, self.mat_a.get_actual_num_cols())
    dims = "{}_{}_{}_{}".format(dim1, dim2, dim3, dim4)

    addressing = "{}{}{}{}".format(self.mat_c.addressing[0],
                                   self.mat_a.addressing[0],
                                   self.mat_b.addressing[0],
                                   self.mat_d.addressing[0])

    return "gemm_{}_{}".format(dims, addressing)
