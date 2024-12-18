from typing import List, Union, Type
from copy import deepcopy
import hashlib
from tensorforge.generators.descriptions import OperationDescription, MultilinearDescr, ElementwiseDescr
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Addressing, GeneralLexicon, DataFlowDirection
from tensorforge.common.aux import get_extra_offset_name
from tensorforge.backend.data_types import ShrMemObject, RegMemObject
from tensorforge.backend.opt import OptimizationStage
from tensorforge.backend.scopes import Scopes
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.compute.elementwise import ElementwiseInstruction
from tensorforge.backend.instructions.builders.loader_builder import GlobalLoaderBuilder
from tensorforge.backend.instructions.builders.multilinear_builder import MultilinearBuilder
from tensorforge.backend.instructions.builders.ptr_manip_builder import GetElementPtrBuilder
from tensorforge.backend.instructions.builders.allocator_builder import ShrMemAllocBuilder
from tensorforge.backend.instructions.sync_block import SyncThreads, SyncBlock
from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import GenerationError

class AbstractThreadBlockPolicy:
  def __init__(self, context: Context, global_mem: int, mem_per_mult: int, num_threads: int):
    self._context: Context = context
    self._mem_per_mult: int = mem_per_mult
    self._global_mem: int = global_mem
    self._num_threads: int = num_threads

    vm = self._context.get_vm()
    self._max_blocks = vm.get_hw_descr().max_block_per_sm
    self._max_allowed_mem = vm.get_hw_descr().max_local_mem_size_per_block
    self._max_threads = vm.get_hw_descr().max_threads_per_block

  def get_num_mults_per_block(self):
    pass


class RegmaxBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self, context, global_mem, mem_size_per_mult, num_threads):
    super().__init__(context, global_mem, mem_size_per_mult, num_threads)

  def get_num_mults_per_block(self):
    return 1
    # the //2 is a heuristic
    # self._max_threads // self._num_threads // 2
    # max_thread_mults = 256 // self._num_threads
    if self._mem_per_mult == 0:
      return max_thread_mults
    else:
      max_mem_mults = (self._max_allowed_mem - self._global_mem * self._context.fp_type.size()) // (self._mem_per_mult * self._context.fp_type.size())
      return min(max_mem_mults, max_thread_mults)

class Generator:
  NAME_ENCODING_LENGTH = 10

  def __init__(self,
               gemm_list: List[OperationDescription],
               context: Context,
               thread_block_policy_type: Type[AbstractThreadBlockPolicy] = RegmaxBlockPolicy):
    self.descr_list: List[OperationDescription] = gemm_list
    self._context: Context = context
    self._thread_block_policy_type: Type[AbstractThreadBlockPolicy] = thread_block_policy_type
    self._base_kernel_name: Union[str, None] = None

    self._kernel = None
    self._launcher = None
    self._header = None

    self._matrix_list = None
    self._tmp_list = None
    self._scopes: Scopes = Scopes()
    self._is_registerd: bool = False

    self._num_threads: int = 0
    self._num_active_threads: int = 0
    self._accumulator_size: int = 0

    self._shr_mem_obj: Union[ShrMemObject, None] = None

    self._ir: List[AbstractInstruction] = []
    self._global_ir: List[AbstractInstruction] = []

    self._check_consistency_with_user_options()
    self._name_operands(self.descr_list)

    self._persistent_threading = False
    self._preload_globals = False

  def set_kernel_name(self, name):
    self._base_kernel_name = name

  def register(self):
    self._collect_tmp_matrices()
    self._populate_global_scope()
    if not self._base_kernel_name:
      self._generate_kernel_name()
    self._is_registerd = True

  def _set_threadconfig(self):
    for instr in self._global_ir:
      instr.set_threadconfig_pre(self._num_threads, self._shr_mem_obj.get_mults_per_block())

  def generate(self):
    if not self._is_registerd:
      self.register()

    self._deduce_num_threads()
    self._deduce_accumulator_size()
    self._emit_global_ir()
    self._emit_ir()
    opt = OptimizationStage(context=self._context,
                            shr_mem=self._shr_mem_obj,
                            instructions=self._ir,
                            num_threads=self._num_threads)
    opt.optimize()
    self._ir = opt.get_instructions()

    # add final sync for persistent threads
    if self._persistent_threading:
      self._ir += [SyncThreads(self._context, self._num_threads)]

    self._deduce_mults_per_block()
    self._set_threadconfig()

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

  def _generate_kernel(self):
    writer = Writer()
    with self._generate_kernel_proto(writer):
      self._write_kernel_meta_data(writer)

      vm = self._context.get_vm()
      mapped_keywords = vm.get_lexic().get_mapped_keywords()
      for kw in mapped_keywords:
        mapped_kw, real_kw, type = kw
        writer(f'const {type} {mapped_kw} = {real_kw};')

      for instruction in self._global_ir:
        if instruction.is_ready():
          instruction.gen_code(writer)
        else:
          raise GenerationError(f'instr is not ready to be generated: {instruction}')
      if not self._persistent_threading:
        writer(f'unsigned {GeneralLexicon.BATCH_ID_NAME} = {self._get_2d_block_id()};')
        with writer.If(f'{self._get_element_size_guard()}'):
          with writer.If(f'{self._get_flag_guard(writer)}'):
            for instruction in self._ir:
              if instruction.is_ready():
                instruction.gen_code(writer)
              else:
                raise GenerationError(f'instr is not ready to be generated: {instruction}')
      else:
        with writer.For(f'unsigned {GeneralLexicon.BATCH_ID_NAME} = {self._get_2d_block_id()}; {GeneralLexicon.BATCH_ID_NAME} < {GeneralLexicon.NUM_ELEMENTS}; {GeneralLexicon.BATCH_ID_NAME} += {vm.get_lexic().grid_dim_x} * {vm.get_lexic().block_dim_y}'):
          with writer.If(f'{self._get_flag_guard(writer)}'):
            for instruction in self._ir:
              if instruction.is_ready():
                instruction.gen_code(writer)
              else:
                raise GenerationError(f'instr is not ready to be generated: {instruction}')

    self._kernel = writer.get_src()

  def _generate_launcher(self):
    writer = Writer()
    proto = self._generate_launcher_proto(with_defaults=False)
    mults_per_block = self._shr_mem_obj.get_mults_per_block()
    lexic = self._context.get_vm().get_lexic()
    with writer.Block(f'{proto}'):
      kernel_name = f'kernel_{self._base_kernel_name}'

      shmemsize = f'{self._shr_mem_obj.get_total_size()} * sizeof({self._context.fp_as_str()})'

      writer(f'{lexic.kernel_range_object("block", f"{self._num_threads}, {mults_per_block}, 1")};')
      if not self._persistent_threading:
        num_blocks = f'({GeneralLexicon.NUM_ELEMENTS} + {mults_per_block} - 1) / {mults_per_block}'
      else:
        writer(f'{lexic.get_launch_size(kernel_name, "block", shmemsize)}')
        num_blocks = 'gridsize'
      writer(f'{lexic.kernel_range_object("grid", f"{num_blocks}, 1, 1")};')

      writer(lexic.set_shmem_size(kernel_name, shmemsize))

      lexic.get_stream_via_pointer(writer, 'stream', GeneralLexicon.STREAM_PTR_STR)

      args = self._generate_kernel_base_args()
      args = ', '.join(args)
      call_site = lexic.get_launch_code(func_name=kernel_name,
                                        grid='grid',
                                        block='block',
                                        stream='stream',
                                        func_params=args,
                                        shmem=shmemsize)#
      writer(f'{call_site};')
      writer('CHECK_ERR;')
    self._launcher = writer.get_src()

  def _generate_header(self):
    self._header = f'{self._generate_launcher_proto(with_defaults=True)};\n'

  def _deduce_num_threads(self):
    for descr in self.descr_list:
      num_threads, num_active_threads = descr.get_num_threads(self._context)

      self._num_threads = max(num_threads, self._num_threads)
      self._num_active_threads = max(num_active_threads, self._num_active_threads)

    compress = True
    for gemm_descr in self.descr_list:
      if isinstance(gemm_descr, ElementwiseDescr):
        compress = False
        break
    # if compress:
    #   self._num_threads = 32

  def _deduce_accumulator_size(self):
    for descr in self.descr_list:
      local_acc_size = descr.get_accumulator_size()
      self._accumulator_size = max(self._accumulator_size, local_acc_size)

  def _emit_global_ir(self):
    shmbuilder = ShrMemAllocBuilder(self._context, self._scopes)

    self._scopes.add_scope()
    # allocate shared memory
    shmbuilder.build(size=None)
    self._shr_mem_obj = shmbuilder.get_resultant_obj()
    self._global_ir.extend(shmbuilder.get_instructions())

    # load globals to shared memory (maybe)
    if self._preload_globals:
      builder = GetElementPtrBuilder(self._context, self._scopes)
      for symbol in self._scopes.get_global_scope().values():
        if symbol.obj.addressing == Addressing.SCALAR or (symbol.obj.addressing == Addressing.NONE and symbol.stype == SymbolType.Data):
          builder.build(symbol, symbol.obj.addressing == Addressing.PTR_BASED)
          self._global_ir.extend(builder.get_instructions())

      builder = GlobalLoaderBuilder(self._context, self._scopes, self._shr_mem_obj, self._num_threads)
      for symbol in self._scopes.get_global_scope().values():
        if symbol.obj.addressing == Addressing.NONE and symbol.stype != SymbolType.Data:
          builder.build(symbol, symbol.obj.addressing == Addressing.PTR_BASED)
          self._global_ir.extend(builder.get_instructions())
      
      self._global_ir.append(SyncBlock(self._context, self._num_threads))

  def _emit_ir(self):
    # find local data from batches
    builder = GetElementPtrBuilder(self._context, self._scopes)
    self._scopes.add_scope()
    for symbol in self._scopes.get_global_scope().values():
      if not self._preload_globals or not (symbol.obj.addressing == Addressing.NONE or symbol.obj.addressing == Addressing.SCALAR):
        builder.build(symbol, symbol.obj.addressing == Addressing.PTR_BASED)
        self._ir.extend(builder.get_instructions())

    self._scopes.add_scope()
    # generate GEMM and store operations
    builder = MultilinearBuilder(self._context,
                          self._scopes,
                          self._scopes.get_symbol(self._shr_mem_obj),
                          self._num_threads)
    # builder.build_prologue()

    for gemm_descr in self.descr_list:
      if isinstance(gemm_descr, MultilinearDescr):
        builder.build(ops=[SymbolView(self._scopes.get_symbol(op.tensor), op.bbox) for op in gemm_descr.ops],
                        dest_obj=gemm_descr.dest,
                        descr=gemm_descr)
        self._ir.extend(builder.get_instructions())
      if isinstance(gemm_descr, ElementwiseDescr):
        self._ir.append(ElementwiseInstruction(self._context, gemm_descr.oplist, self._scopes, False, self._num_threads))
    
    builder.build_epilogue()
    self._ir.extend(builder.get_instructions())

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._context,
                                            self._shr_mem_obj.get_global_size(),
                                            self._shr_mem_obj.get_size_per_mult(),
                                            self._num_threads)
    num_mults_per_block = policy.get_num_mults_per_block()
    self._shr_mem_obj.set_mults_per_block(num_mults_per_block)

  def get_kernel(self):
    return self._kernel

  def get_launcher(self):
    return self._launcher

  def get_header(self):
    return self._header

  def _check_consistency_with_user_options(self):
    user_options = self._context.get_user_options()
    for descr in self.descr_list:
      if not descr.is_strict_match() == user_options.exact_contraction_length:
        msg = 'gemm list is not consistent with user options. '
        msg += f'`strict_math` in gemm descr. set to {descr.is_strict_match()}, '
        msg += f'but `exact_contraction_length` is set to {user_options.exact_contraction_length}'
        raise RuntimeError(msg)

  def _name_operands(self, gemm_list: List[OperationDescription]):
    tmp_counter = 0
    op_counter = 0

    pre_matrix_list = set()
    pre_submatrix_list = set()
    for gemm in gemm_list:
      local_list = gemm.matrix_list()

      # gather all matrices
      for matrix in local_list:
        pre_matrix_list.add(matrix.tensor)
        pre_submatrix_list.add(matrix)
    self._matrix_list = list(pre_matrix_list)
    self._submatrix_list = list(pre_submatrix_list)

    for matrix in self._matrix_list:
      if matrix.is_tmp:
        matrix.name = f't{tmp_counter}'
        tmp_counter += 1
      else:
        matrix.name = f'm{op_counter}'
        op_counter += 1

  def _collect_tmp_matrices(self):
    self._tmp_list = []
    for matrix in self._matrix_list:
      if matrix.is_tmp and matrix not in self._tmp_list:
        self._tmp_list.append(matrix)

  def _populate_global_scope(self):
    """
    Add non-tmp matrices to the global scope
    :return:
    """
    for matrix in self._matrix_list:
      if matrix not in self._tmp_list:
        # temporary. For now, take only the selector matrices
        if matrix.has_values() and len(matrix.get_values()) < 16:
          stype = SymbolType.Data
        elif matrix.addressing == Addressing.SCALAR:
          stype = SymbolType.Scalar
        else:
          stype = SymbolType.Batch
        symbol = Symbol(obj=matrix,
                      name=matrix.name,
                      stype=stype)
        self._scopes.add_to_global(symbol)

  def _generate_kernel_name(self):
    global_symbols = self._scopes.get_global_scope().values()
    long_name = []
    for item in global_symbols:
      long_name.append(item.obj.gen_descr())

    for descr in self.descr_list:
      long_name.extend([
        str(descr)
      ])

    result = hashlib.md5(', '.join(long_name).encode())
    md5encoding = result.hexdigest()
    self._base_kernel_name = f'kernel_{md5encoding[:Generator.NAME_ENCODING_LENGTH]}'

  def get_base_name(self):
    return self._base_kernel_name

  def _get_scalar_name(self, scalar, default_name):
    scalar_type = type(scalar)
    is_pritable = scalar_type.__str__ is not object.__str__
    is_string = scalar_type == str
    return scalar if is_pritable or is_string else default_name

  def _write_kernel_meta_data(self, writer):
    writer('// meta data:')
    glb_matrices = self._scopes.get_global_scope().values()
    for matrix in glb_matrices:
      writer(f'// {matrix.obj.gen_descr()}')

    writer.new_line()
    for item in self.descr_list:
      writer(f'// {item}')
    writer.new_line()

  def _generate_base_params_list(self, symbol_list, with_types=True, with_defaults=False):
    params = []
    for symbol in symbol_list:
      datatype = self._context.fp_type if symbol.obj.datatype is None else symbol.obj.datatype
      if symbol.obj.addressing == Addressing.SCALAR:
        params.extend([f'{datatype} {symbol.name}' if with_types else f'{symbol.name}'])
      else:
        ptr_type = symbol.obj.addressing.to_pointer()
        const_modifier = 'const ' if symbol.obj.direction == DataFlowDirection.SOURCE else ''
        batch_type = f'{const_modifier}{datatype}{ptr_type}' if with_types else ''
        offset_type = 'unsigned' if with_types else ''
        params.extend([f'{batch_type} {symbol.name}'])
        if symbol.obj.addressing == Addressing.PTR_BASED:
          params.extend([f'{offset_type} {get_extra_offset_name(symbol)}'])

    batch_size_type = 'size_t' if with_types else ''
    params.append(f'{batch_size_type} {GeneralLexicon.NUM_ELEMENTS}')

    flags_type = 'unsigned*' if with_types else ''
    default_flags_value = '= nullptr' if with_defaults else ''
    params.append(f'{flags_type} {GeneralLexicon.FLAGS_NAME} {default_flags_value}')
    return params

  def _generate_kernel_base_args(self):
    global_symbols = self._scopes.get_global_scope().values()
    args = self._generate_base_params_list(global_symbols, with_types=False)
    return args

  def _generate_kernel_proto(self, writer):
    global_symbols = self._scopes.get_global_scope().values()

    params = self._generate_base_params_list(symbol_list=global_symbols, with_types=True)
    str_params = ', '.join(params)
    total_num_threads_per_block = self._num_threads * self._shr_mem_obj.get_mults_per_block()

    lexic = self._context.get_vm().get_lexic()

    launch_bounds = (total_num_threads_per_block,)

    return lexic.kernel_definition(writer, launch_bounds, self._base_kernel_name, str_params, self._context.fp_as_str(),
                                         self._shr_mem_obj.get_total_size(), global_symbols)

  def _generate_launcher_proto(self, with_defaults=True):
    global_symbols = self._scopes.get_global_scope().values()

    params = self._generate_base_params_list(symbol_list=global_symbols,
                                                  with_types=True,
                                                  with_defaults=with_defaults)

    default_value = ' = nullptr' if with_defaults else ''
    params.append(f'void* {GeneralLexicon.STREAM_PTR_STR}{default_value}')
    str_params = ', '.join(params)
    return f'void launcher_{self._base_kernel_name}({str_params})'

  def default_generate_call_site(self):
    if not self._is_registerd:
      raise RuntimeError('generator is not registered. Call register first.')
    symbols = deepcopy(list(self._scopes.get_global_scope().values()))
    for item in symbols:
      if item.obj.alias:
        item.name = item.obj.alias

    args = self._generate_base_params_list(symbol_list=symbols,
                                                with_types=False)

    args.append(f'{GeneralLexicon.FLAGS_NAME}')
    args.append(f'{GeneralLexicon.STREAM_PTR_STR}')
    str_args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({str_args});'

  def get_helper_headers(self):
    headerset = set()
    for irinst in self._global_ir:
      for header in irinst.get_headers():
        headerset.add(header)
    for irinst in self._ir:
      for header in irinst.get_headers():
        headerset.add(header)
    return list(headerset)

  def generate_call_site(self,
                         mat_name_map,
                         offset_name_map,
                         num_element,
                         flags=None,
                         stream=None):
    args = []

    # add tensors
    symbols = list(self._scopes.get_global_scope().values())
    for symbol in symbols:
      if symbol.obj.alias in mat_name_map:
        args.append(mat_name_map[symbol.obj.alias])
        if symbol.obj.addressing == Addressing.PTR_BASED:
          args.append(offset_name_map[symbol.obj.alias])

    # add num. elements
    args.append(num_element)

    # add flags
    if flags:
      args.append(flags)

    # add streams
    if stream:
      args.append(stream)

    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def _get_2d_block_id(self):
    lexic = self._context.get_vm().get_lexic()
    return f'{lexic.thread_idx_y} + {lexic.block_dim_y} * {lexic.block_idx_x}'

  def _get_element_size_guard(self):
    return f'{GeneralLexicon.BATCH_ID_NAME} < {GeneralLexicon.NUM_ELEMENTS}'

  def _get_flag_guard(self, writer):
    writer(f'bool allowed = true;')
    # with writer.If('{GeneralLexicon.FLAGS_NAME} != nullptr'):
    #   flag_value = f'static_cast<bool>({GeneralLexicon.FLAGS_NAME}[{GeneralLexicon.BATCH_ID_NAME}])'
    #   writer(f'allowed = {flag_value};')
    return 'allowed'
