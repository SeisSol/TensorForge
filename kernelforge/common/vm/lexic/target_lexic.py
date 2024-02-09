from kernelforge.common.basic_types import GeneralLexicon
from .lexic import Lexic


class TargetLexic(Lexic):
  def __init__(self, backend, underlying_hardware):
    super().__init__(underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "ty"
    self.thread_idx_x = "tx"
    self.thread_idx_z = "tz"
    self.block_idx_x = "bx"
    self.block_idx_z = "bz"
    self.block_dim_y = "tY"
    self.block_dim_z = "tZ"
    self.stream_type = "int"
    self.restrict_kw = "__restrict"

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"{func_name}({stream}, {grid}[0], {block}[0], {block}[1], {func_params})"

  def declare_shared_memory_inline(self, name, precision, size, alignment):
    return ""

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None):
    bounds = "*".join(str(kb) for kb in kernel_bounds)
    stream_type = self.stream_type
    class TargetContext:
      def __init__(self):
        self.function = file.Function(f'kernel_{base_name}', f'{stream_type}* streamobj, int bX, int tX, int tY, {bounds}')
        self.blockloop = file.For('int bx = 0; bx < bX; ++bx')
        self.teamloop1 = file.For('int ty = 0; ty < tY; ++ty')
        self.teamloop2 = file.For('int tx = 0; tx < tX; ++tx')
      def __enter__(self):
        self.function.__enter__()
        file(f'#pragma omp target teams distribute nowait depend(inout: streamobj[0]) is_device_ptr({", ".join(symbol.name for symbol in global_symbols)}) thread_limit({bounds})')
        self.blockloop.__enter__()
        file(f'{precision} {GeneralLexicon.TOTAL_SHR_MEM} [{total_shared_mem_size}];')
        file(f'#pragma omp parallel for collapse(2) schedule(static, 1)')
        self.teamloop1.__enter__()
        self.teamloop2.__enter__()
      def __exit__(self, type, value, traceback):
        self.teamloop2.__exit__(type, value, traceback)
        self.teamloop1.__exit__(type, value, traceback)
        self.blockloop.__exit__(type, value, traceback)
        self.function.__exit__(type, value, traceback)
    
    backend = self._backend
    class TargetContext:
      def __init__(self):
        self.function = file.Function(f'kernel_{base_name}', f'{stream_type}* streamobj, int bX, int tX, int tY, {params}')
        self.blockloop = file.Block('')
        self.threadblock = file.Block('')
      def __enter__(self):
        self.function.__enter__()
        if backend == 'targetdart':
          device = 'device(TARGETDART_DEVICE(0))'
        else:
          device = ''
        file(f'#pragma omp target teams nowait num_teams(bX) depend(inout: streamobj[0]) is_device_ptr({", ".join(symbol.name for symbol in global_symbols)}) thread_limit({bounds}) {device}')
        self.blockloop.__enter__()
        file(f'{precision} {GeneralLexicon.TOTAL_SHR_MEM}[{total_shared_mem_size}];')
        file(f'#pragma omp parallel num_threads({bounds})')
        self.threadblock.__enter__()
        file(f'int bx = omp_get_team_num();')
        file(f'int ty = omp_get_thread_num() / tX;')
        file(f'int tx = omp_get_thread_num() % tX;')
      def __exit__(self, type, value, traceback):
        self.threadblock.__exit__(type, value, traceback)
        self.blockloop.__exit__(type, value, traceback)
        self.function.__exit__(type, value, traceback)

    return TargetContext()

  def sync_threads(self):
    return "#pragma omp barrier //"

  def sync_vec_unit(self):
    return "#pragma omp barrier //"

  def get_sub_group_id(self, sub_group_size):
    return f'{self.thread_idx_x} % {sub_group_size}'

  def active_sub_group_mask(self):
    return f'NOTIMPLEMENTED'

  def broadcast_sync(self, variable, lane, mask):
    return f'NOTIMPLEMENTED'

  def kernel_range_object(self, name, values):
    return f"size_t {name}[3] = {{ {values} }}"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    with file.If(f"{pointer_name} == nullptr"):
      file.Expression("throw std::invalid_argument(\"stream may not be null!\")")

    stream_obj = f'static_cast<{self.stream_type} *>({pointer_name})'
    file(f'{self.stream_type} *stream = {stream_obj};')

  def check_error(self):
    return None

  def batch_indexer_gemm(self):
    return self.get_tid_counter(self.thread_idx_y, self.block_dim_y, self.block_idx_z)

  def batch_indexer_csa(self):
    return self.get_tid_counter(self.thread_idx_z, self.block_dim_z, self.block_idx_z)

  def batch_indexer_init(self):
    return self.get_tid_counter(self.thread_idx_z, self.block_dim_z, self.block_idx_z)

  def get_headers(self):
    return ['cstdlib', 'stdexcept', 'omp.h']
