from . import constructs
from io import StringIO
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator
from .abstract_generator import AbstractGenerator as Generator
from .loaders import shm_mem_factory, StubLoader
from gemmforge.vm import VM
from .thread_policies import TheadPolicyFactory
import math
import hashlib


class GemmGenerator(GemmLikeGenerator):
  """ Generates GEMM GPU kernels: C = alpha * A * B + beta * C
  """
  
  def __init__(self, vm: VM):
    super(GemmGenerator, self).__init__(vm)
    self.mat_a = None
    self.mat_b = None
    self.mat_c = None
    self.mat_a_loader = None
    self.mat_b_loader = None
    # For better readability for the remaining code
    self.team_index_str = self._lexic.batch_indexer_gemm()
    self.name_threadIdx_y = self._lexic.thread_idx_y
    self.name_threadIdx_x = self._lexic.thread_idx_x
  
  def set(self, mat_a, mat_b, mat_c, alpha, beta, base_name=None):
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
    self._is_set = True
  
  def generate(self):
    self._check_if_set()
    
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
      
      with self._lexic.kernel_definition(file,
                                         kernel_bounds,
                                         self.base_name,
                                         self._get_func_params(),
                                         self._precision,
                                         self._get_total_shared_mem_size()):
        
        file.VariableDeclaration("int", "batchId", self.team_index_str)
        
        with file.If("{} < {}".format("batchId", Generator.NUM_ELEMENTS_STR)):
          
          # declare ptrs for correct matrices
          file.VariableDeclaration(f'const {self._precision}*',
                                   glob_symbols[self.mat_a.name],
                                   self._get_global_matrix_ptr(self.mat_a))
          
          file.VariableDeclaration(f'const {self._precision}*',
                                   glob_symbols[self.mat_b.name],
                                   self._get_global_matrix_ptr(self.mat_b))
          
          file.VariableDeclaration(f'{self._precision}*',
                                   glob_symbols[self.mat_c.name],
                                   self._get_global_matrix_ptr(self.mat_c))
          
          # declare shared memory per kernel
          mem = self._lexic.declare_shared_memory_inline('Scratch',
                                                         self._precision,
                                                         self._get_total_shared_mem_size())
          if mem is not None:
            file.Expression(mem)
          
          # find address of matrix B within block shared memory
          shr_mem_address = f'&Scratch[{self.name_threadIdx_y} * {self.shr_mem_size_per_mult}]'
          file.VariableDeclaration("{}*".format(self._precision),
                                   self.mat_b_loader.get_output_symbol(),
                                   shr_mem_address)
          
          if self.mat_a.transpose:
            # find address of matrix A within block shared memory
            shr_mem_offset = self.mat_b_loader.compute_shared_mem_size()
            shr_mem_address = "&Scratch[{} * {} + {}]".format(self.name_threadIdx_y,
                                                              self.shr_mem_size_per_mult,
                                                              shr_mem_offset)
            file.VariableDeclaration("{}*".format(self._precision),
                                     self.mat_a_loader.get_output_symbol(),
                                     shr_mem_address)
          
          # load matrices into shared memory
          self.mat_b_loader.generate_scr(file, glob_symbols[self.mat_b.name])
          self.mat_a_loader.generate_scr(file, glob_symbols[self.mat_a.name])
          file.Expression(self._lexic.sync_threads())
          
          # set up current compute symbols within the rest of the scope
          current_symbols[self.mat_b.name] = self.mat_b_loader.get_output_symbol()
          current_symbols[self.mat_a.name] = self.mat_a_loader.get_output_symbol()
          file.Emptyline()
          
          with file.If(f'{self.name_threadIdx_x} < {self.num_compute_threads}'):
            # allocate a buffer for each cuda thread to hold computed results
            file.Emptyline()
            zero_fp_value = "0.0{}".format('f' if self._precision == 'float' else '')
            file.ArrayDeclaration(self._precision,
                                  'Results',
                                  [zero_fp_value] * self.mat_c.get_actual_num_cols())
            
            file.VariableDeclaration(self._precision, 'Value')
            
            # perform matrix multiplication
            # m, n, k - according to the BLAS documentation. Read BLAS spec.
            if self.mat_a.transpose:
              contraction_length = self.mat_a.get_actual_num_rows()
            else:
              contraction_length = self.mat_a.get_actual_num_cols()
            
            file.Emptyline()
            with file.For(f'int k = 0; k < {contraction_length}; ++k'):
              first_operand = "{}[{} + {} * k]".format(current_symbols[self.mat_a.name],
                                                       self.name_threadIdx_x,
                                                       self.mat_a_loader.get_lid_dim())
              file.Assignment('Value', f'{first_operand}')
              file.Emptyline()
              file.Pragma('unroll')
              with file.For(f'int n = 0; n < {self.mat_c.get_actual_num_cols()}; ++n'):
                if self.mat_b.transpose:
                  second_operand = "{}[n + {} * k]".format(current_symbols[self.mat_b.name],
                                                           self.mat_b_loader.get_lid_dim())
                else:
                  second_operand = "{}[k + {} * n]".format(current_symbols[self.mat_b.name],
                                                           self.mat_b_loader.get_lid_dim())
                
                file.Accumulate("Results[n]",
                                f'Value * {second_operand}')
            
            # write results back to memory
            file.Emptyline()
            file.Pragma("unroll")
            with file.For(f'int n = 0; n < {self.mat_c.get_actual_num_cols()}; ++n'):
              rhs = "{}[{} + {} * n]".format(glob_symbols[self.mat_c.name],
                                             self.name_threadIdx_x,
                                             self.mat_c.num_rows)
              
              if self.alpha == 1.0:
                lhs = 'Results[n]'
              else:
                if self._precision == 'float' and isinstance(self.alpha, float):
                  lhs = f'{self.alpha}f * Results[n]'
                else:
                  lhs = f'{self.alpha} * Results[n]'
              
              if self.beta != 0.0:
                if self.beta == 1.0:
                  lhs += f' + {rhs}'
                else:
                  lhs += " + {} * {}".format(
                    "{}{}".format(self.beta, 'f' if self._precision == "float" else ''),
                    rhs)
              
              file.Assignment(rhs, lhs)
      
      self._kernel = src.getvalue()
  
  def _generate_launcher(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      with file.Function(self.base_name, self._get_launcher_params()):
        file.VariableDeclaration(self._lexic.kernel_range_object(), self._get_block_dim_spec())
        file.VariableDeclaration(self._lexic.kernel_range_object(), self._get_grid_dim_spec())
        
        self._lexic.get_stream_via_pointer(file, 'stream', Generator.STREAM_PTR_STR)
        file.Expression(self._lexic.get_launch_code(self.base_name,
                                                    'Grid',
                                                    'Block',
                                                    'stream',
                                                    self._get_func_args()))
        err = self._lexic.check_error()
        if err is not None:
          file.Expression(err)
      
      self._launcher = src.getvalue()
  
  def _generate_header(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      file.FunctionDeclaration(self.base_name, self._get_launcher_params(with_defaults=True))
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
      matrices = {'A': self.mat_a, 'B': self.mat_b, 'C': self.mat_c}
      for name in matrices:
        print(f'Matrix {name}:')
        print(matrices[name])
        print("=" * 80)
      
      raise error
  
  def _analyze(self):
    if self.mat_a.transpose:
      lid_dim_length = self.mat_a.get_actual_num_cols()
    else:
      lid_dim_length = self.mat_a.get_actual_num_rows()
    
    num_vector_units_required = math.ceil(lid_dim_length / self._hw_descr.vec_unit_length)
    self.num_compute_threads = lid_dim_length
    self.num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length
    
    if self.mat_a.transpose:
      self.mat_a_loader = shm_mem_factory(vm=self._vm,
                                          matrix=self.mat_a,
                                          num_active_threads=self.num_active_threads,
                                          load_and_transpose=True)
    
    else:
      self.mat_a_loader = StubLoader(self._vm, self.mat_a, self.num_active_threads)
    
    self.mat_b_loader = shm_mem_factory(vm=self._vm,
                                        matrix=self.mat_b,
                                        num_active_threads=self.num_active_threads,
                                        load_and_transpose=False)
    
    self.shr_mem_size_per_mult = \
      self.mat_a_loader.compute_shared_mem_size() + self.mat_b_loader.compute_shared_mem_size()
    
    thread_policy = TheadPolicyFactory.get_gemm_policy(vm=self._vm,
                                                       reals_per_op=self.shr_mem_size_per_mult,
                                                       num_threads=self.num_active_threads,
                                                       op1=self.mat_a,
                                                       op2=self.mat_b,
                                                       res=self.mat_c)
    
    self.num_mult_per_block = thread_policy.get_num_ops_per_block()
  
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
    prefix = 's' if self._precision == "float" else "d"
    
    gemm_dims = f'm{self.mat_a.get_actual_num_rows()}'
    gemm_dims += f'_n{self.mat_b.get_actual_num_cols()}'
    gemm_dims += f'_k{self.mat_a.get_actual_num_cols()}'
    
    ldas = f'lda{self.mat_a.num_rows}_ldb{self.mat_b.num_rows}_ldc{self.mat_c.num_rows}'
    consts = f'alpha_{int(self.alpha)}_beta_{int(self.beta)}'
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
      return f'{self._precision} {self.alpha}, {base_params}'
  
  def _get_launcher_params(self, with_defaults=False):
    base_params = super(GemmGenerator, self)._get_launcher_params(with_defaults)
    if isinstance(self.alpha, float):
      return base_params
    else:
      return f'{self._precision} {self.alpha}, {base_params}'
  
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
      main_offset = "{} * {}".format("batchId", matrix.get_real_volume())
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {} + {}]".format(matrix.name,
                                        main_offset,
                                        sub_offset,
                                        extra_offset_symbol)
    
    elif matrix.addressing == "pointer_based":
      main_offset = "batchId"
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{}][{} + {}]".format(matrix.name,
                                       main_offset,
                                       sub_offset,
                                       extra_offset_symbol)
    
    else:
      sub_offset = matrix.get_offset_to_first_element()
      return "&{}[{} + {}]".format(matrix.name, sub_offset, extra_offset_symbol)
