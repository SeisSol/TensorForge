from . import constructs
from io import StringIO
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator
from .basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from .symbol_table import Symbol, SymbolType
from .abstract_generator import AbstractGenerator as Generator
from .instructions.builders import GetElementPtrBuilder, RegistersAllocBuilder, ShrMemAllocBuilder
from .instructions.builders import GemmBuilder
from .instructions import StoreRegToGlb
from .vm import VM
from .thread_policies import TheadPolicyFactory
import math
import hashlib


class GemmGenerator(GemmLikeGenerator):
  """ Generates GEMM GPU kernels: C = alpha * A * B + beta * C
  """
  
  def __init__(self, vm: VM):
    super(GemmGenerator, self).__init__(vm)
    self._trans_a = None
    self._trans_b = None
    self._mat_a = None
    self._mat_b = None
    self._mat_c = None
    
    self._reg_array_obj = None
    self._shr_mem_obj = None
    self._shr_mem_loads = []
  
  def set(self, trans_a, trans_b, mat_a, mat_b, mat_c, alpha, beta, base_name=None):
    self._instructions = []
    
    self._mat_a = mat_a
    self._trans_a = trans_a
    self._mat_a.set_name('A')
    self._mat_a.set_data_flow_direction(DataFlowDirection.SOURCE)
    
    self._mat_b = mat_b
    self._trans_b = trans_b
    self._mat_b.set_name('B')
    self._mat_b.set_data_flow_direction(DataFlowDirection.SOURCE)
    
    self._mat_c = mat_c
    self._mat_c.set_name('C')
    self._mat_c.set_data_flow_direction(DataFlowDirection.SINK)
    self._matrices = [self._mat_a, self._mat_b, self._mat_c]
    
    self._alpha = alpha
    self._beta = beta
    
    self._base_name = base_name if base_name is not None else self._generate_base_name()
    self._is_set = True
  
  def generate(self):
    self._check_if_set()
    
    self._check()
    self._deduce_num_threads()
    self._populate_global_scope()
    self._emit_instructions()
    self._analyze()
    
    self._generate_kernel()
    self._generate_header()
    self._generate_launcher()
  
  def _generate_kernel(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      
      max_num_threads_per_block = self._num_active_threads * self._num_ops_per_block
      kernel_bounds = [max_num_threads_per_block]
      team_index_str = self._lexic.batch_indexer_gemm()

      with self._lexic.kernel_definition(file,
                                         kernel_bounds,
                                         self._base_name,
                                         self._get_func_params(),
                                         self._precision,
                                         self._shr_mem_obj.get_total_size()):
        
        file(f'int {GeneralLexicon.BATCH_ID} = {team_index_str};')
        with file.If(f'{GeneralLexicon.BATCH_ID} < {GeneralLexicon.NUM_ELEMENTS}'):
  
          for instr in self._instructions:
            if instr.is_ready():
              instr.gen_code(file)
            else:
              raise GenerationError("gemm_generator: requested instr is not ready")
      
      self._kernel = src.getvalue()
  
  def _generate_launcher(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      with file.Function(self._base_name, self._get_launcher_params()):
        file(f'{self._lexic.kernel_range_object()} {self._get_block_dim_spec()};')
        file(f'{self._lexic.kernel_range_object()} {self._get_grid_dim_spec()};')
        
        self._lexic.get_stream_via_pointer(file, 'stream', GeneralLexicon.STREAM_PTR_STR)
        file.Expression(self._lexic.get_launch_code(self._base_name,
                                                    'grid',
                                                    'block',
                                                    'stream',
                                                    self._get_func_args()))
        err = self._lexic.check_error()
        if err is not None:
          file.Expression(err)
      
      self._launcher = src.getvalue()
  
  def _generate_header(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      file.FunctionDeclaration(self._base_name, self._get_launcher_params(with_defaults=True))
      content = src.getvalue()
    self._header = content
  
  def _check(self):
    try:
      
      # check whether C and A match each other
      if self._trans_a:
        if self._mat_c.get_actual_num_rows() != self._mat_a.get_actual_num_cols():
          raise GenerationError('Cannot generate a matrix multiplication '
                                'with given parameters. Matrix C and A (Trans) do not match')
      else:
        if self._mat_c.get_actual_num_rows() != self._mat_a.get_actual_num_rows():
          raise GenerationError('Cannot generate a matrix multiplication '
                                'with given parameters. Matrix C and A (NoTrans) do not match')
      
      # check whether C and B match each other
      if self._trans_b:
        if self._mat_c.get_actual_num_cols() != self._mat_b.get_actual_num_rows():
          raise GenerationError('Cannot generate a matrix multiplication '
                                'with given parameters. Matrix C and B (Trans) do not match')
      else:
        if self._mat_c.get_actual_num_cols() != self._mat_b.get_actual_num_cols():
          raise GenerationError('Cannot generate a matrix multiplication '
                                'with given parameters. Matrix C and B (NoTrans) do not match')
      
      # check whether A and B match each other
      if self._trans_a:
        if self._trans_b:
          if self._mat_a.get_actual_num_rows() != self._mat_b.get_actual_num_cols():
            raise GenerationError('Cannot generate a matrix multiplication with given parameters. '
                                  'Matrix A (Trans) and B (Trans) do not match')
        else:
          if self._mat_a.get_actual_num_rows() != self._mat_b.get_actual_num_rows():
            raise GenerationError('Cannot generate a matrix multiplication with given parameters. '
                                  'Matrix A (Trans) and B (NoTrans) do not match')
      else:
        if self._trans_b:
          if self._mat_a.get_actual_num_cols() != self._mat_b.get_actual_num_cols():
            raise GenerationError('Cannot generate a matrix multiplication with given parameters. '
                                  'Matrix A (NoTrans) and B (Trans) do not match')
        else:
          if self._mat_a.get_actual_num_cols() != self._mat_b.get_actual_num_rows():
            raise GenerationError('Cannot generate a matrix multiplication with given parameters. '
                                  'Matrix A (NoTrans) and B (NoTrans) do not match')
    
    except GenerationError as error:
      matrices = {'A': self._mat_a, 'B': self._mat_b, 'C': self._mat_c}
      for name in matrices:
        print(f'matrix {name}:')
        print(matrices[name])
        print("=" * 80)
      
      raise error

  def _deduce_num_threads(self):
    if self._trans_a:
      lead_dim_length = self._mat_a.get_actual_num_cols()
    else:
      lead_dim_length = self._mat_a.get_actual_num_rows()
  
    num_vector_units_required = math.ceil(lead_dim_length / self._hw_descr.vec_unit_length)
    self._num_compute_threads = lead_dim_length
    self._num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length

  def _populate_global_scope(self):
    for matrix in self._matrices:
      self._symbol_table.add_symbol(Symbol(obj=matrix,
                                    name=matrix.name,
                                    stype=SymbolType.Batch))
    self._symbol_table.add_scope()

  def _emit_instructions(self):
    # extract matrices from batches
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())
    
    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_cols(), 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()
    
    # create shared mem
    builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
    builder.build(size=None)
    self._instructions.extend(builder.get_instructions())
    self._shr_mem_obj = builder.get_resultant_obj()

    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = GemmBuilder(self._vm,
                          self._symbol_table,
                          self._reg_array_obj,
                          self._shr_mem_obj,
                          self._num_active_threads)
    
    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj])

    self._shr_mem_loads = builder.get_srh_mem_loads()
    
    self._instructions.extend(builder.get_instructions())
    
    store = StoreRegToGlb(self._vm,
                          self._symbol_table[self._mat_c],
                          self._symbol_table[self._reg_array_obj],
                          self._alpha,
                          self._beta,
                          self._num_compute_threads)
    self._instructions.append(store)

  def _analyze(self):
    # compute total required shr. mem
    shr_mem_counter = 0
    for instr in self._shr_mem_loads:
      instr.set_shr_mem_offset(shr_mem_counter)
      shr_mem_counter += instr.compute_shared_mem_size()
      
    self._shr_mem_obj.set_size_per_mult(shr_mem_counter)
    
    # compute num matrix multiplications per block
    thread_policy = TheadPolicyFactory.get_gemm_policy(vm=self._vm,
                                                       reals_per_op=shr_mem_counter,
                                                       num_threads=self._num_active_threads,
                                                       op1=self._mat_a,
                                                       op2=self._mat_b,
                                                       res=self._mat_c)
    
    self._num_ops_per_block = thread_policy.get_num_ops_per_block()
    self._shr_mem_obj.set_mults_per_block(self._num_ops_per_block)
  
  def _generate_base_name(self):
    if self._trans_a:
      dim1 = f'm{self._mat_a.get_actual_num_cols()}_{self._mat_a.num_rows}'
      dim3 = f'k{self._mat_a.get_actual_num_rows()}'
    else:
      dim1 = f'm{self._mat_a.get_actual_num_rows()}_{self._mat_a.num_rows}'
      dim3 = f'k{self._mat_a.get_actual_num_cols()}'
    
    if self._trans_b:
      dim2 = f'n{self._mat_b.get_actual_num_rows()}_{self._mat_b.num_rows}'
    else:
      dim2 = f'n{self._mat_b.get_actual_num_cols()}_{self._mat_b.num_rows}'
    
    dims = f'{dim1}_{dim2}_{dim3}'

    addresses = f'{self._mat_a.addressing[0]}{self._mat_b.addressing[0]}{self._mat_c.addressing[0]}'
    traspose = f'{"T" if self._trans_a else "NT"}_{"T" if self._trans_b else "NT"}'
    constants = f'{self._alpha}_{self._beta}'
    
    result = hashlib.md5(('{}_{}{}{}'.format(constants,
                                             self._mat_a.__str__(),
                                             self._mat_b.__str__(),
                                             self._mat_c.__str__()).encode()))
    md5encoding = result.hexdigest()
    prefix = 's' if self._precision == "float" else "d"
    
    gemm_dims = f'm{self._mat_a.get_actual_num_rows()}'
    gemm_dims += f'_n{self._mat_b.get_actual_num_cols()}'
    gemm_dims += f'_k{self._mat_a.get_actual_num_cols()}'
    
    ldas = f'lda{self._mat_a.num_rows}_ldb{self._mat_b.num_rows}_ldc{self._mat_c.num_rows}'
    consts = f'alpha_{int(self._alpha)}_beta_{int(self._beta)}'
    return '{0}gemm_{1}_{2}_{3}_{4}_{5}_{6}'.format(prefix,
                                                    traspose,
                                                    gemm_dims,
                                                    ldas,
                                                    consts,
                                                    addresses,
                                                    md5encoding[:Generator.ENCODING_LENGTH])
  
  def _get_func_params(self):
    base_params = super(GemmGenerator, self)._get_func_params()
    if isinstance(self._alpha, float):
      return base_params
    else:
      return f'{self._precision} {self._alpha}, {base_params}'
  
  def _get_launcher_params(self, with_defaults=False):
    base_params = super(GemmGenerator, self)._get_launcher_params(with_defaults)
    if isinstance(self._alpha, float):
      return base_params
    else:
      return f'{self._precision} {self._alpha}, {base_params}'
  
  def _get_func_args(self):
    base_args = super(GemmGenerator, self)._get_func_args()
    if isinstance(self._alpha, float):
      return base_args
    else:
      return f'{self._alpha}, {base_args}'
  
  def _get_block_dim_spec(self):
    super(GemmGenerator, self)._get_block_dim_spec()
    return f'block({self._num_active_threads}, {self._num_ops_per_block}, 1)'
  
  def _get_grid_dim_spec(self):
    super(GemmGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(GeneralLexicon.NUM_ELEMENTS,
                                                self._num_ops_per_block)
    return f'grid({num_blocks}, 1, 1)'
