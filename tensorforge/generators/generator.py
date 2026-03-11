from typing import List, Union, Type
from copy import deepcopy
import hashlib
from tensorforge.generators.descriptions import OperationDescription, MultilinearDescr, ElementwiseDescr, RegionDescription
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Addressing, GeneralLexicon, DataFlowDirection
from tensorforge.common.helper import get_extra_offset_name
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
from tensorforge.backend.instructions.sync_block import SyncThreads, SyncBlock, SyncGrid
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
    # the //2 is a heuristic
    # self._max_threads // self._num_threads // 2
    max_thread_mults = 256 // self._num_threads
    if self._mem_per_mult == 0:
      return max_thread_mults
    else:
      max_mem_mults = (self._max_allowed_mem - self._global_mem * self._context.fp_type.size()) // (self._mem_per_mult * self._context.fp_type.size())
      return min(max_mem_mults, max_thread_mults)

class Section:
  def __init__(self):
    self.ir: List[AbstractInstruction] = []
    self.global_ir: List[AbstractInstruction] = []
    self.shr_mem_obj: Union[ShrMemObject, None] = None
    self.scopes: Scopes = Scopes()
    self.barrier = False

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

    self._section: Section = Section()
    self._sections: List[Section] = []

    self._name_operands(self.descr_list)

    prefer_launchcontrol = context.get_vm().get_hw_descr().vendor == 'nvidia' and int(context.get_vm().get_hw_descr().model[3:]) >= 100
    prefer_persistent = context.get_vm().get_hw_descr().vendor in ['amd', 'nvidia'] and not prefer_launchcontrol
    prefer_preload = context.get_vm().get_hw_descr().vendor in ['amd'] and not prefer_launchcontrol

    self._persistent_threading = prefer_persistent
    self._preload_globals = prefer_preload

    self._clusterlaunchcontrol = prefer_launchcontrol

  def set_kernel_name(self, name):
    self._base_kernel_name = name

  def register(self):
    self._collect_tmp_matrices()
    self._populate_global_scope()

  def _set_threadconfig(self):
    for instr in self._section.global_ir:
      instr.set_threadconfig_pre(self._num_threads, self._section.shr_mem_obj.get_mults_per_block())

  def generate(self):
    self.register()

    self._deduce_num_threads()

    descrlist = []
    currlist = []
    barrier = []
    for descr in self.descr_list:
      if descr.barrier():
        # avoid empty sections
        if len(currlist) > 0:
          descrlist += [currlist]
          barrier += [descr.trueBarrier()]
        elif len(barrier) > 0:
          barrier[-1] = barrier[-1] or descr.trueBarrier()
        currlist = []
      else:
        currlist += [descr]
    if len(currlist) > 0:
      descrlist += [currlist]
      barrier += [False]

    for codesection, lastbarrier in zip(descrlist, barrier):
      scopecnt = self._scopes.get_num_scopes()
      self._scopes.add_scope()
      self._section = Section()

      self._emit_global_ir()
      self._emit_ir(codesection)
      opt = OptimizationStage(context=self._context,
                              shr_mem=self._section.shr_mem_obj,
                              instructions=self._section.ir,
                              num_threads=self._num_threads,
                              scopes = self._scopes)
      opt.optimize()
      self._section.ir = opt.get_instructions()
      self._section.global_ir += opt.get_global_instructions()

      # add final sync for persistent threads
      if self._persistent_threading or self._clusterlaunchcontrol:
        self._section.ir += [SyncThreads(self._context, self._num_threads)]

      self._deduce_mults_per_block()
      self._set_threadconfig()

      if lastbarrier:
        self._section.barrier = True

      while scopecnt < self._scopes.get_num_scopes():
        self._scopes.remove_scope()
      self._sections += [self._section]

    if not self._base_kernel_name:
      self._generate_kernel_name()

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

  def _generate_kernel(self):
    vm = self._context.get_vm()

    writer = Writer()
    with self._generate_kernel_proto(writer):
      self._write_kernel_meta_data(writer)

      for i,section in enumerate(self._sections):
        with writer.AnonymousScope():
          if self._context.get_vm().get_hw_descr().vendor == 'nvidia':
            writer(f'cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();')

          offset = []
          idx = i - 1
          for ssection in reversed(self._sections[:i]):
            if ssection.barrier:
              break
            offset += [f'{GeneralLexicon.NUM_ELEMENTS}{idx}']
            idx -= 1

          stride = f'({vm.get_lexic().grid_dim_x} * {vm.get_lexic().block_dim_y})'
          if len(offset) == 0:
            start = self._get_2d_block_id()
          else:
            start = f'({self._get_2d_block_id()} + {" + ".join(offset)}) % {stride}'

          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}_start = {start};')
          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}1 = {GeneralLexicon.BATCH_ID_NAME}_start < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}_start : 0;')
          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}2 = {GeneralLexicon.BATCH_ID_NAME}1 + {stride} < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}1 + {stride} : {GeneralLexicon.BATCH_ID_NAME}1;')

          for instruction in section.global_ir:
            if instruction.is_ready():
              instruction.gen_code(writer)
            else:
              raise GenerationError(f'instr is not ready to be generated: {instruction}')

          def generate_inner():
            with writer.If(f'{self._get_flag_guard(writer, i)}'):
              for instruction in section.ir:
                if instruction.is_ready():
                  instruction.gen_code(writer)
                else:
                  raise GenerationError(f'instr is not ready to be generated: {instruction}')

          if self._persistent_threading:
            # TODO: OMP target
            # TODO: maybe iterate over adjacent elements? (for indirect pointers)

            with writer.For(f'size_t {GeneralLexicon.BATCH_ID_NAME}0 = {start}; {GeneralLexicon.BATCH_ID_NAME}0 < {GeneralLexicon.NUM_ELEMENTS}{i}; {GeneralLexicon.BATCH_ID_NAME}0 += {stride}'):
              writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}1 = {GeneralLexicon.BATCH_ID_NAME}0 + {stride} < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}0 + {stride} : {GeneralLexicon.BATCH_ID_NAME}0;')
              writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}2 = {GeneralLexicon.BATCH_ID_NAME}1 + {stride} < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}1 + {stride} : {GeneralLexicon.BATCH_ID_NAME}1;')
              generate_inner()
          elif self._clusterlaunchcontrol:
            writer(f'__shared__ tensorforge::ClusterLaunchCtrl launchctrl;')
            writer(f'int phase = 0;')
            writer(f'launchctrl.init();')
            writer(f'size_t {GeneralLexicon.BATCH_ID_NAME}0 = {self._get_2d_block_id()};')
            with writer.While(f'true'):
              writer('launchctrl.setupNext();')
              with writer.If(f'{self._get_element_size_guard(i)}'):
                generate_inner()
              writer('const auto nextBlock = launchctrl.queryNext(phase);')
              with writer.If('!nextBlock.has_value()'):
                writer('break;')
              writer(f'{GeneralLexicon.BATCH_ID_NAME}0 = {self._get_2d_block_id("nextBlock.value()")};')
          else:
            writer(f'const size_t {GeneralLexicon.BATCH_ID_NAME}0 = {self._get_2d_block_id()};')
            with writer.If(f'{self._get_element_size_guard(i)}'):
              generate_inner()

    self._kernel = writer.get_src()

  def _generate_launcher(self):
    writer = Writer()
    proto = self._generate_launcher_proto(with_defaults=False)
    mults_per_block = self._section.shr_mem_obj.get_mults_per_block()
    lexic = self._context.get_vm().get_lexic()
    with writer.Block(f'{proto}'):
      kernel_name = f'kernel_{self._base_kernel_name}'

      shmemsize = f'{self._section.shr_mem_obj.get_total_size()} * sizeof({self._context.fp_as_str()})'

      # TODO: allow multi-kernel approach instead
      coop = any(section.barrier for section in self._sections)

      writer(f'{lexic.kernel_range_object("block", f"{self._num_threads}, {mults_per_block}, 1")};')
      if not self._persistent_threading:
        assert not coop
        num_blocks = f'({GeneralLexicon.NUM_ELEMENTS}0 + {mults_per_block} - 1) / {mults_per_block}'
      else:
        writer(f'{lexic.get_launch_size(kernel_name, "block", shmemsize)}')
        if coop:
          num_blocks = 'gridsize'
        else:
          num_blocks = f'std::min(gridsize, {GeneralLexicon.NUM_ELEMENTS}0)'
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
                                        shmem=shmemsize,
                                        coop=coop)
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
    if compress:
      self._num_threads = min(32, self._num_threads)

  def _emit_global_ir(self):
    nonfirst_block = len(self._sections) > 0
    last_barrier = len(self._sections) > 0 and self._sections[-1].barrier

    shmbuilder = ShrMemAllocBuilder(self._context, self._scopes)

    self._scopes.add_scope()
    # allocate shared memory
    shmbuilder.build(size=None)
    self._section.shr_mem_obj = shmbuilder.get_resultant_obj()
    self._section.global_ir.extend(shmbuilder.get_instructions())

    # load globals to shared memory (if requested)
    if self._preload_globals:
      load_ir = []
      shmem_load = 0

      if nonfirst_block:
        load_ir.append(SyncBlock(self._context))

      self._scopes.add_scope()

      builder = GlobalLoaderBuilder(self._context, self._scopes, self._section.shr_mem_obj, self._num_threads)
      for symbol in self._scopes.get_global_scope().values():
        if symbol.obj.addressing == Addressing.NONE and symbol.stype != SymbolType.Data:
          shmem_load += builder.build(symbol)
          load_ir.extend(builder.get_instructions())

      vm = self._context.get_vm()
      shmem_cap = vm.get_hw_descr().max_local_mem_size_per_block

      if shmem_load < shmem_cap:
        self._section.global_ir += load_ir
        if last_barrier:
          self._section.global_ir.append(SyncGrid(self._context))
        else:
          self._section.global_ir.append(SyncBlock(self._context))
        return True
      else:
        # make sure to clean up all new symbols that didn't get added
        self._scopes.remove_scope()
        self._preload_globals = False

    builder = GetElementPtrBuilder(self._context, self._scopes)
    for symbol in self._scopes.get_global_scope().values():
      if symbol.obj.addressing == Addressing.SCALAR or (symbol.obj.addressing == Addressing.NONE and (symbol.stype == SymbolType.Data or not self._preload_globals)):
        builder.build(symbol)
        self._section.global_ir.extend(builder.get_instructions())

    # pipelines
    for symbol in self._scopes.get_global_scope().values():
      if symbol.obj.addressing in [Addressing.STRIDED, Addressing.PTR_BASED]:
        pass

    if not self._preload_globals:
      if last_barrier:
        self._section.global_ir.append(SyncGrid(self._context))
      elif nonfirst_block:
        self._section.global_ir.append(SyncBlock(self._context))

    return False

  def _emit_ir(self, descr_list):
    # find local data from batches
    builder = GetElementPtrBuilder(self._context, self._scopes)
    self._scopes.add_scope()
    for symbol in self._scopes.get_global_scope().values():
      firstptr = symbol.obj.addressing == Addressing.SCALAR or symbol.obj.addressing == Addressing.NONE
      if not firstptr:
        builder.build(symbol)
        self._section.ir.extend(builder.get_instructions())

    self._scopes.add_scope()
    # generate GEMM and store operations
    builder = MultilinearBuilder(self._context,
                          self._scopes,
                          self._scopes.get_symbol(self._section.shr_mem_obj),
                          self._num_threads)
    # builder.build_prologue()

    def get_symbol_view(op):
      symbol = self._scopes.get_symbol(op.tensor)
      return SymbolView(symbol, op.bbox, op.offset)

    for gemm_descr in descr_list:
      if isinstance(gemm_descr, MultilinearDescr):
        builder.build(ops=[get_symbol_view(op) for op in gemm_descr.ops],
                        dest_obj=gemm_descr.dest,
                        descr=gemm_descr)
        self._section.ir.extend(builder.get_instructions())
      if isinstance(gemm_descr, ElementwiseDescr):
        self._section.ir.append(ElementwiseInstruction(self._context, gemm_descr.oplist, self._scopes, False, self._num_threads))

    builder.build_epilogue()
    self._section.ir.extend(builder.get_instructions())

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._context,
                                            self._section.shr_mem_obj.get_global_size(),
                                            self._section.shr_mem_obj.get_size_per_mult(),
                                            self._num_threads)
    num_mults_per_block = policy.get_num_mults_per_block()
    self._section.shr_mem_obj.set_mults_per_block(num_mults_per_block)

  def get_kernel(self):
    return self._kernel

  def get_launcher(self):
    return self._launcher

  def get_header(self):
    return self._header

  def _name_operands(self, gemm_list: List[OperationDescription]):
    tmp_counter = 0
    op_counter = 0

    pre_matrix_list = {}
    for gemm in gemm_list:
      local_list = gemm.matrix_list()

      # gather all matrices
      for matrix in local_list:
        # dict preserves ordering starting with 3.7
        pre_matrix_list[matrix.tensor] = None

    self._matrix_list = list(pre_matrix_list.keys())

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
        if matrix.has_values() and len(matrix.get_values()) < 16 and False:
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

    # needed for type differences (but same names)
    global_symbols = self._scopes.get_global_scope().values()
    params = self._generate_base_params_list(symbol_list=global_symbols, with_types=True)
    long_name.extend(params)

    sha = hashlib.new('md5', usedforsecurity=False)
    sha.update(', '.join(long_name).encode())
    md5encoding = sha.hexdigest()
    self._base_kernel_name = f'kernel_{md5encoding[:Generator.NAME_ENCODING_LENGTH]}'

  def get_base_name(self):
    return self._base_kernel_name

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
        if not symbol.stype == SymbolType.Data:
          params.extend([f'{datatype} {symbol.name}' if with_types else f'{symbol.name}'])
      else:
        ptr_type = symbol.obj.addressing.to_pointer()
        const_modifier = 'const ' if symbol.obj.direction == DataFlowDirection.SOURCE else ''
        batch_type = f'{const_modifier}{datatype}{ptr_type}' if with_types else ''
        offset_type = 'unsigned' if with_types else ''
        params.extend([f'{batch_type} {symbol.name}'])
        if symbol.obj.addressing != Addressing.NONE:
          params.extend([f'{offset_type} {get_extra_offset_name(symbol)}'])

    batch_size_type = 'size_t' if with_types else ''

    for i, section in enumerate(self._sections):
      params.append(f'{batch_size_type} {GeneralLexicon.NUM_ELEMENTS}{i}')

    flags_type = 'unsigned*' if with_types else ''
    default_flags_value = '= nullptr' if with_defaults else ''

    for i, section in enumerate(self._sections):
      params.append(f'{flags_type} {GeneralLexicon.FLAGS_NAME}{i} {default_flags_value}')

    return params

  def _generate_kernel_base_args(self):
    global_symbols = self._scopes.get_global_scope().values()
    args = self._generate_base_params_list(global_symbols, with_types=False)
    return args

  def _generate_kernel_proto(self, writer):
    global_symbols = self._scopes.get_global_scope().values()

    params = self._generate_base_params_list(symbol_list=global_symbols, with_types=True)
    str_params = ', '.join(params)

    mults_per_block = min(section.shr_mem_obj.get_mults_per_block() for section in self._sections)
    shr_total_size = max(section.shr_mem_obj.get_total_size() for section in self._sections)

    total_num_threads_per_block = self._num_threads * mults_per_block

    lexic = self._context.get_vm().get_lexic()

    launch_bounds = (total_num_threads_per_block,)

    return lexic.kernel_definition(writer, launch_bounds, self._base_kernel_name, str_params, self._context.fp_as_str(),
                                         shr_total_size, global_symbols)

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
    for section in self._sections:
      for irinst in section.global_ir:
        for header in irinst.get_headers():
          headerset.add(header)
      for irinst in section.ir:
        for header in irinst.get_headers():
          headerset.add(header)
    return list(headerset)

  def generate_call_site(self,
                         mat_name_map,
                         offset_name_map):
    args = []

    # add tensors
    symbols = list(self._scopes.get_global_scope().values())
    for symbol in symbols:
      if symbol.obj.alias in mat_name_map:
        args.append(mat_name_map[symbol.obj.alias])
        if symbol.obj.addressing not in [Addressing.SCALAR, Addressing.NONE]:
          args.append(offset_name_map[symbol.obj.alias])

    flags = []
    for desc in self.descr_list:
      if isinstance(desc, RegionDescription):
        args.append(f'{desc.name}.numElements')
        flags.append(f'{desc.name}.flags')
    if len(flags) == 0:
      args.append(f'numElements')
      flags.append(f'flags')

    args += flags

    args.append('streamPtr')

    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def _get_2d_block_id(self, block=None):
    lexic = self._context.get_vm().get_lexic()
    if block is None:
      block = lexic.block_idx_x
    return f'{lexic.thread_idx_y} + {lexic.block_dim_y} * ({block})'

  def _get_element_size_guard(self, i):
    return f'{GeneralLexicon.BATCH_ID_NAME}0 < {GeneralLexicon.NUM_ELEMENTS}{i}'

  def _get_flag_guard(self, writer, i):
    if False:
      writer(f'bool allowed = true;')
      with writer.If(f'{GeneralLexicon.FLAGS_NAME}{i} != nullptr'):
        writer(f'allowed = static_cast<bool>({GeneralLexicon.FLAGS_NAME}{i}[{GeneralLexicon.BATCH_ID_NAME}0]);')
      return 'allowed'
    else:
      return 'true'
