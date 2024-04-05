from kernelforge.common.basic_types import GeneralLexicon, Addressing, DataFlowDirection
from .lexic import Lexic, Operation

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
    self.grid_dim_x = "omp_get_num_teams()"
    self.stream_type = "int"
    self.restrict_kw = "__restrict"

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"{func_name}({stream}, {grid}[0], {block}[0], {block}[1], {func_params})"

  def declare_shared_memory_inline(self, name, precision, size, alignment):
    return ""

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None):
    bounds = "*".join(str(kb) for kb in kernel_bounds)
    stream_type = self.stream_type
    class TargetContextCpu:
      def __init__(self):
        self.function = file.Function(f'kernel_{base_name}', f'{stream_type}* streamobj, int bX, int tX, int tY, {bounds}')
        self.blockloop = file.For('int bx = 0; bx < bX; ++bx')
        self.teamloop1 = file.For('int ty = 0; ty < tY; ++ty')
        self.teamloop2 = file.For('int tx = 0; tx < tX; ++tx')
      def __enter__(self):
        self.function.__enter__()
        file(f'#pragma omp parallel for nowait depend(inout: streamobj[0])')
        self.blockloop.__enter__()
        file(f'{precision} {GeneralLexicon.TOTAL_SHR_MEM} [{total_shared_mem_size}];')
        file(f'#pragma omp simd collapse(2)')
        self.teamloop1.__enter__()
        self.teamloop2.__enter__()
      def __exit__(self, type, value, traceback):
        self.teamloop2.__exit__(type, value, traceback)
        self.teamloop1.__exit__(type, value, traceback)
        self.blockloop.__exit__(type, value, traceback)
        self.function.__exit__(type, value, traceback)
    
    class TargetContext:
      def __init__(self):
        self.function = file.Function(f'kernel_{base_name}', f'{stream_type}* streamobj, int bX, int tX, int tY, {bounds}')
        self.blockloop = file.For('int bx = 0; bx < bX; ++bx')
        self.teamloop1 = file.For('int ty = 0; ty < tY; ++ty')
        self.teamloop2 = file.For('int tx = 0; tx < tX; ++tx')
      def __enter__(self):
        self.function.__enter__()
        if backend == 'targetdart':
          device = 'device(TARGETDART_DEVICE(0))'
        else:
          device = ''
        file(f'#pragma omp target teams distribute nowait depend(inout: streamobj[0]) is_device_ptr({", ".join(symbol.name for symbol in global_symbols)}) thread_limit({bounds}) {device}')
        self.blockloop.__enter__()
        file(f'{precision} {GeneralLexicon.TOTAL_SHR_MEM} [{total_shared_mem_size}];')
        file(f'#pragma omp parallel for collapse(2)')
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
        self.blockloop = file.Scope()
        self.threadblock = file.Scope()
      def __enter__(self):
        self.function.__enter__()
        if backend == 'targetdart':
          batched_symbols_inout = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.PTR_BASED and symbol.obj.direction == DataFlowDirection.SOURCESINK]
          batched_symbols_in = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.PTR_BASED and symbol.obj.direction == DataFlowDirection.SOURCE]
          batched_symbols_out = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.PTR_BASED and symbol.obj.direction == DataFlowDirection.SINK]
          strided_symbols_inout = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.STRIDED and symbol.obj.direction == DataFlowDirection.SOURCESINK]
          strided_symbols_in = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.STRIDED and symbol.obj.direction == DataFlowDirection.SOURCE]
          strided_symbols_out = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.STRIDED and symbol.obj.direction == DataFlowDirection.SINK]
          constant_symbols = [symbol for symbol in global_symbols if symbol.obj.addressing == Addressing.NONE]

          device = 'device(TARGETDART_DEVICE(0))'
          deviceAny = 'device(TARGETDART_ANY)'
          for symbol in batched_symbols_in:
            file(f'static unordered_map<const {precision}**, {precision}(*)[{symbol.obj.get_real_volume()}]> {symbol.name}_datamap;')
            file(f'auto* {symbol.name}_ptr = {symbol.name}_datamap[{symbol.name}];')
            with file.If(f'{symbol.name}_ptr == nullptr'):
              file(f'{symbol.name}_ptr = reinterpret_cast<decltype({symbol.name}_ptr)>(std::malloc(sizeof({precision}[{symbol.obj.get_real_volume()}]) * bX));')
          for symbol in batched_symbols_out + batched_symbols_inout:
            file(f'static unordered_map<{precision}**, {precision}(*)[{symbol.obj.get_real_volume()}]> {symbol.name}_datamap;')
            file(f'auto* {symbol.name}_ptr = {symbol.name}_datamap[{symbol.name}];')
            with file.If(f'{symbol.name}_ptr == nullptr'):
              file(f'{symbol.name}_ptr = reinterpret_cast<decltype({symbol.name}_ptr)>(std::malloc(sizeof({precision}[{symbol.obj.get_real_volume()}]) * bX));')
          if len(batched_symbols_in + batched_symbols_inout) > 0:
            file(f'#pragma omp target teams nowait num_teams(bX) depend(inout: streamobj[0]) map(from: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_in + batched_symbols_inout)}) is_device_ptr({", ".join(symbol.name for symbol in global_symbols)}) {device}')
            with file.Scope():
              file('#pragma omp parallel')
              with file.Scope():
                for symbol in batched_symbols_in + batched_symbols_inout:
                  file('#pragma omp for nowait')
                  with file.For(f'int i = 0; i < {symbol.obj.get_real_volume()}; ++i'):
                    file(f'{symbol.name}_ptr[omp_get_team_num()][i] = {symbol.name}[omp_get_team_num()][i];')
          if len(batched_symbols_out + batched_symbols_inout) > 0:
            def epilogue():
              file(f'#pragma omp target teams nowait num_teams(bX) depend(inout: streamobj[0]) map(to: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_out + batched_symbols_inout)}) is_device_ptr({", ".join(symbol.name for symbol in global_symbols)}) {device}')
              with file.Scope():
                file('#pragma omp parallel')
                with file.Scope():
                  for symbol in batched_symbols_out + batched_symbols_inout:
                    file('#pragma omp for nowait')
                    with file.For(f'int i = 0; i < {symbol.obj.get_real_volume()}; ++i'):
                      file(f'{symbol.name}[omp_get_team_num()][i] = {symbol.name}_ptr[omp_get_team_num()][i];')
            self.epilogue = epilogue
          
          batched_symbols_out_str = f'map(from: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_out)})' if len(batched_symbols_out) > 0 else ''
          batched_symbols_in_str = f'map(to: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_in)})' if len(batched_symbols_in) > 0 else ''
          batched_symbols_inout_str = f'map(tofrom: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_inout)})' if len(batched_symbols_inout) > 0 else ''
          strided_symbols_out_str = f'map(from: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}*bX]" for symbol in strided_symbols_out)})' if len(strided_symbols_out) > 0 else ''
          strided_symbols_in_str = f'map(to: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}*bX]" for symbol in strided_symbols_in)})' if len(strided_symbols_in) > 0 else ''
          strided_symbols_inout_str = f'map(tofrom: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}*bX]" for symbol in strided_symbols_inout)})' if len(strided_symbols_inout) > 0 else ''
          constant_symbols_str = f'map(to: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}]" for symbol in constant_symbols)})' if len(constant_symbols) > 0 else ''
          file(f'#pragma omp target teams nowait num_teams(bX) depend(inout: streamobj[0]) {constant_symbols_str} {strided_symbols_in_str} {strided_symbols_out_str} {strided_symbols_inout_str} {batched_symbols_in_str} {batched_symbols_out_str} {batched_symbols_inout_str} thread_limit({bounds}) {deviceAny}')
        else:
          device = ''
          file(f'#pragma omp target teams nowait num_teams(bX) depend(inout: streamobj[0]) is_device_ptr({", ".join(symbol.name for symbol in global_symbols)}) thread_limit({bounds})')
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
        if backend == 'targetdart':
          self.epilogue()
        self.function.__exit__(type, value, traceback)

    return TargetContext()

  def sync_threads(self):
    return "#pragma omp barrier //"

  def sync_vec_unit(self):
    return "#pragma omp barrier //"

  def get_sub_group_id(self, sub_group_size):
    return f'{self.thread_idx_x} % {sub_group_size}'

  def active_sub_group_mask(self):
    return f''

  def broadcast_sync(self, variable, lane, mask):
    return 'NOTSUPPORTED' #f'__builtin_shufflevector()'

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
    return ['cstdlib', 'stdexcept', 'omp.h', 'cmath']

  def get_fptype(self, fptype, length=1):
    return f'__attribute__ ((vector_size (sizeof({fptype}) * {length}))) {fptype}'

  def get_operation(self, op: Operation, fptype, value1, value2):
    if op == Operation.COPY:
      return value1
    elif op == Operation.ADD:
      return f'({value1} + {value2})'
    elif op == Operation.SUB:
      return f'({value1} - {value2})'
    elif op == Operation.MUL:
      return f'({value1} * {value2})'
    elif op == Operation.DIV:
      return f'({value1} / {value2})'
    elif op == Operation.MIN:
      return f'std::min({value1}, {value2})'
    elif op == Operation.MAX:
      return f'std::max({value1}, {value2})'
    elif op == Operation.EXP:
      return f'std::exp({value1})' # has __expf
    elif op == Operation.LOG:
      return f'std::log({value1})' # has __logf
    elif op == Operation.SQRT:
      return f'std::sqrt({value1})'
    elif op == Operation.SIN:
      return f'std::sin({value1})' # has __sinf
    elif op == Operation.COS:
      return f'std::cos({value1})' # has __cosf
    elif op == Operation.TAN:
      return f'std::tan({value1})' # has __tanf
    elif op == Operation.ASIN:
      return f'std::asin({value1})'
    elif op == Operation.ACOS:
      return f'std::acos({value1})'
    elif op == Operation.ATAN:
      return f'std::atan({value1})'
