from . import constructs
from io import StringIO
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator
from .abstract_generator import AbstractGenerator as Generator
from .basic_types import DataFlowDirection, GeneralLexicon
from .symbol_table import SymbolType, Symbol
from .instructions.builders import GetElementPtrBuilder
from gemmforge.vm import VM
from .thread_policies import TheadPolicyFactory
import math
import hashlib


class CsaGenerator(GemmLikeGenerator):
  """ Copy-Add-Scale Generator: B = beta * B + alpha * A, where alpha is a real number
  and beta is either 1.0 or 0.0
  """

  def __init__(self, vm: VM):
    super(CsaGenerator, self).__init__(vm)
    self._vm = vm

  def set(self, mat_a, mat_b, alpha, beta, base_name=None):
    self._mat_a = mat_a
    self._mat_a.set_name('A')
    self._mat_a.set_data_flow_direction(DataFlowDirection.SOURCE)

    self._mat_b = mat_b
    self._mat_b.set_name('B')
    self._mat_b.set_data_flow_direction(DataFlowDirection.SINK)

    self._matrices = [self._mat_a, self._mat_b]

    self._alpha = alpha
    self._beta = beta

    self._base_name = base_name if base_name is not None else self._generate_base_name()
    self._is_set = True

  def generate(self):
    self._check_if_set()

    self._check()
    self._deduce_num_threads()
    self._populate_global_scope()
    self._analyze()

    self._generate_kernel()
    self._generate_header()
    self._generate_launcher()

  def _check(self):
    try:

      def is_inside(rst_range, term_range):
        return term_range[0] >= rst_range[0] and term_range[1] <= rst_range[1]

      def make_range(bbox, dim):
        return [bbox[dim], bbox[dim + 2]]

      for dim in range(2):
        if not is_inside(make_range(self._mat_b.bbox, dim), make_range(self._mat_a.bbox, dim)):
          raise GenerationError(f'Cannot generate a copy-add-scale op. '
                                'Data of mat. A (term) in not inside of mat. B (result)')

    except GenerationError as error:
      matrices = {'A': self._mat_a, 'B': self._mat_b}
      for name in matrices:
        print(f'matrix {name}:')
        print(matrices[name])
        print("=" * 80)
      raise error

  def _deduce_num_threads(self):
    lead_dim_length = self._mat_a.get_actual_num_rows()

    # we use active threads to add a single column
    num_vector_units_required = math.ceil(lead_dim_length / self._hw_descr.vec_unit_length)
    self._num_compute_threads = lead_dim_length
    self._num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length

  def _populate_global_scope(self):
    for matrix in self._matrices:
      self._symbol_table.add_symbol(Symbol(obj=matrix,
                                           name=matrix.name,
                                           stype=SymbolType.Batch))
    self._symbol_table.add_scope()

  def _analyze(self):
    thread_policy = TheadPolicyFactory.get_csa_policy(vm=self._vm,
                                                      num_threads=self._num_active_threads,
                                                      op1=self._mat_a,
                                                      op2=self._mat_b)

    self._num_ops_per_block = thread_policy.get_num_ops_per_block()

  def _generate_kernel(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    src = StringIO()
    with constructs.Cpp(src) as file:
      max_num_threads_per_block = self._num_active_threads * self._num_ops_per_block
      kernel_bounds = [max_num_threads_per_block]
      with self._lexic.kernel_definition(file, kernel_bounds, self._base_name,
                                         self._get_func_params()):

        with file.If(f'{self.get_element_size_guard(file)}'):
          with file.If(f'{self.get_flag_guard(file)}'):

            # fill matrix b with zeros if necessary
            with file.Scope():
              if self._beta == 0.0:
                self._fill_with_zeros(self._mat_b, file)

            # get pointer from batches to matrices
            for instr in self._instructions:
              if instr.is_ready():
                instr.gen_code(file)
              else:
                raise GenerationError('gemm_generator: requested instr is not ready')

            # generate the rest of the operion
            op1 = self._symbol_table[self._mat_a]
            op2 = self._symbol_table[self._mat_b]

            with file.If(f'{self._lexic.thread_idx_x} < {op1.data_view.rows}'):
              with file.For(f'int col = 0; col < {op1.data_view.columns}; ++col'):
                addr_op1 = f'{self._lexic.thread_idx_x} + col * {op1.data_view.lead_dim}'
                addr_op2 = f'{self._lexic.thread_idx_x} + col * {op2.data_view.lead_dim}'

                expr = f'{op2.name}[{addr_op2}]'
                expr += f' = scale * {op1.name}[{addr_op1}]'
                if self._beta == 0.0:
                  pass
                elif self._beta == 1.0:
                  expr += f' + {op2.name}[{addr_op2}]'
                elif self._beta == -1.0:
                  expr += f' - {op2.name}[{addr_op2}]'
                else:
                  expr += f' + {self._beta} * {op2.name}[{addr_op2}]'

                file(f'{expr};')

      self._kernel = src.getvalue()

  def _fill_with_zeros(self, matrix, file):

    # find a pointer to a matrix without an extra offset
    self._symbol_table.add_scope()
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    builder.build(src=self._symbol_table.from_global[matrix], include_extra_offset=False)
    for instr in builder.get_instructions():
      instr.gen_code(file)

    real_literal = self._vm.get_real_literal()
    dest = self._symbol_table[matrix]
    with file.For(f'int cols = 0; cols < {matrix.num_cols}; ++cols'):
      num_hops = int(dest.data_view.lead_dim / self._num_active_threads)
      if num_hops > 0:
        for counter in range(num_hops):
          offset = counter * self._num_active_threads
          addr = f'{self._lexic.thread_idx_x} + {offset} + cols * {dest.data_view.lead_dim}'
          file(f'{dest.name}[{addr}] = 0.0{real_literal};')
      if (dest.data_view.lead_dim % self._num_active_threads) != 0:
        residue = dest.data_view.lead_dim - num_hops * self._num_active_threads
        with file.If(f'{self._lexic.thread_idx_x} < {residue}'):
          finial_offset = num_hops * self._num_active_threads
          addr = f'{self._lexic.thread_idx_x} + {finial_offset} + cols * {dest.data_view.lead_dim}'
          file(f'{dest.name}[{addr}] = 0.0{real_literal};')

    self._symbol_table.pop_scope()

  def _generate_launcher(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      with file.Function(self._base_name, self._get_launcher_params()):
        file(f'{self._lexic.kernel_range_object()} {self._get_block_dim_spec()}')
        file(f'{self._lexic.kernel_range_object()} {self._get_grid_dim_spec()}')

        self._lexic.get_stream_via_pointer(file, 'streamPtr', GeneralLexicon.STREAM_PTR_STR)
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

  def _generate_base_name(self):
    dim1 = f'm{self._mat_a.get_actual_num_rows()}_{self._mat_b.num_rows}'
    dim2 = f'n{self._mat_a.get_actual_num_cols()}_{self._mat_b.num_rows}'
    dims = f'{dim1}_{dim2}'
    addressing = f'{self._mat_b.addressing[0]}{self._mat_a.addressing[0]}'

    constants = f'{self._alpha}_{self._beta}'
    result = hashlib.md5(f'{constants}_{self._mat_a.__str__()}{self._mat_b.__str__()}'.encode())
    md5encoding = result.hexdigest()

    prefix = 's' if self._precision == 'float' else 'd'
    return "{}copyAddScale_{}_{}_{}".format(prefix,
                                            dims,
                                            addressing,
                                            md5encoding[:Generator.ENCODING_LENGTH])

  def _get_func_params(self):
    return f'{self._precision} scale, {super(CsaGenerator, self)._get_func_params()}'

  def _get_launcher_params(self, with_defaults=False):
    return f'{self._precision} scale, {super(CsaGenerator, self)._get_launcher_params(with_defaults)}'

  def _get_func_args(self):
    return f'scale, {super(CsaGenerator, self)._get_func_args()}'

  def _get_block_dim_spec(self):
    super(CsaGenerator, self)._get_block_dim_spec()
    return f'block({self._num_active_threads}, {self._num_ops_per_block}, 1);'

  def _get_grid_dim_spec(self):
    super(CsaGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(GeneralLexicon.NUM_ELEMENTS,
                                                self._num_ops_per_block)
    return f'grid({num_blocks}, 1, 1);'
