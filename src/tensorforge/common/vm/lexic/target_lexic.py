# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.basic_types import GeneralLexicon, Addressing, DataFlowDirection
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
    self.block_dim_x = "tX"
    self.block_dim_y = "tY"
    self.block_dim_z = "tZ"
    self.grid_dim_x = "omp_get_num_teams()"
    self.stream_type = "int"
    self.restrict_kw = "__restrict"

  def get_launch_size(self, func_name, block, shmem, resident=False):
    return ''

  def set_shmem_size(self, func_name, shmem):
    return ''

  def get_launch_code(self, func_name, grid, block, stream, func_params, shmem, coop):
    return f"{func_name}({stream}, {grid}[0], {block}[0], {block}[1], {func_params})"

  def declare_shared_memory(self, name, precision, size=None):
    return ""

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None, lanes=None):
    bounds = "*".join(str(kb) for kb in kernel_bounds)
    stream_type = self.stream_type
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
            file(f'static std::unordered_map<const {precision}**, {precision}(*)[{symbol.obj.get_real_volume()}]> {symbol.name}_datamap;')
            file(f'auto* {symbol.name}_ptr = {symbol.name}_datamap[{symbol.name}];')
            with file.If(f'{symbol.name}_ptr == nullptr'):
              file(f'{symbol.name}_ptr = reinterpret_cast<decltype({symbol.name}_ptr)>(std::malloc(sizeof({precision}[{symbol.obj.get_real_volume()}]) * bX));')
          for symbol in batched_symbols_out + batched_symbols_inout:
            file(f'static std::unordered_map<{precision}**, {precision}(*)[{symbol.obj.get_real_volume()}]> {symbol.name}_datamap;')
            file(f'auto* {symbol.name}_ptr = {symbol.name}_datamap[{symbol.name}];')
            with file.If(f'{symbol.name}_ptr == nullptr'):
              file(f'{symbol.name}_ptr = reinterpret_cast<decltype({symbol.name}_ptr)>(std::malloc(sizeof({precision}[{symbol.obj.get_real_volume()}]) * bX));')
          if len(batched_symbols_in + batched_symbols_inout) > 0:
            file(f'#pragma omp target nowait depend(inout: streamobj[0]) map(to:bX) map(from: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_in + batched_symbols_inout)}) is_device_ptr({", ".join(symbol.name for symbol in batched_symbols_in + batched_symbols_inout)}) {device}')
            with file.Scope():
              for symbol in batched_symbols_in + batched_symbols_inout:
                file('#pragma omp loop collapse(2)')
                with file.For('int j = 0; j < bX; ++j'):
                  with file.For(f'int i = 0; i < {symbol.obj.get_real_volume()}; ++i'):
                    file(f'{symbol.name}_ptr[j][i] = {symbol.name}[j][i];')
          if len(batched_symbols_out + batched_symbols_inout) > 0:
            def epilogue():
              file(f'#pragma omp target nowait depend(inout: streamobj[0]) map(to:bX) map(to: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_out + batched_symbols_inout)}) is_device_ptr({", ".join(symbol.name for symbol in batched_symbols_out + batched_symbols_inout)}) {device}')
              with file.Scope():
                for symbol in batched_symbols_out + batched_symbols_inout:
                  file('#pragma omp loop collapse(2)')
                  with file.For('int j = 0; j < bX; ++j'):
                    with file.For(f'int i = 0; i < {symbol.obj.get_real_volume()}; ++i'):
                      file(f'{symbol.name}[j][i] = {symbol.name}_ptr[j][i];')
            self.epilogue = epilogue
          else:
            self.epilogue = lambda: None

          batched_symbols_out_str = f'map(from: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_out)})' if len(batched_symbols_out) > 0 else ''
          batched_symbols_in_str = f'map(to: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_in)})' if len(batched_symbols_in) > 0 else ''
          batched_symbols_inout_str = f'map(tofrom: {", ".join(f"{symbol.name}_ptr[0:bX]" for symbol in batched_symbols_inout)})' if len(batched_symbols_inout) > 0 else ''
          strided_symbols_out_str = f'map(from: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}*bX]" for symbol in strided_symbols_out)})' if len(strided_symbols_out) > 0 else ''
          strided_symbols_in_str = f'map(to: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}*bX]" for symbol in strided_symbols_in)})' if len(strided_symbols_in) > 0 else ''
          strided_symbols_inout_str = f'map(tofrom: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}*bX]" for symbol in strided_symbols_inout)})' if len(strided_symbols_inout) > 0 else ''
          constant_symbols_str = f'map(to: {", ".join(f"{symbol.name}[0:{symbol.obj.get_real_volume()}]" for symbol in constant_symbols)})' if len(constant_symbols) > 0 else ''
          # TODO: map offsets
          file(f'#pragma omp target teams nowait num_teams(bX) map(to:tX) depend(inout: streamobj[0]) {constant_symbols_str} {strided_symbols_in_str} {strided_symbols_out_str} {strided_symbols_inout_str} {batched_symbols_in_str} {batched_symbols_out_str} {batched_symbols_inout_str} thread_limit({bounds}) {deviceAny}')
        else:
          device = ''
          file(f'#pragma omp target teams nowait num_teams(bX) depend(inout: streamobj[0]) is_device_ptr({", ".join(symbol.name for symbol in global_symbols if symbol.obj.addressing != Addressing.SCALAR)}) thread_limit({bounds})')
        self.blockloop.__enter__()
        file(f'{precision} {GeneralLexicon.TOTAL_SHR_MEM}[{total_shared_mem_size}];')
        file(f'#pragma omp parallel num_threads({bounds})')
        self.threadblock.__enter__()
        file('int bx = omp_get_team_num();')
        file('int ty = omp_get_thread_num() / tX;')
        file('int tx = omp_get_thread_num() % tX;')
      def __exit__(self, type, value, traceback):
        self.threadblock.__exit__(type, value, traceback)
        self.blockloop.__exit__(type, value, traceback)
        if backend == 'targetdart':
          self.epilogue()
        self.function.__exit__(type, value, traceback)

    return TargetContext()

  def sync_block(self):
    return "#pragma omp barrier"

  def sync_simd(self):
    return "#pragma omp barrier"

  def sync_grid(self):
    return ""

  def active_sub_group_mask(self):
    return ''

  def broadcast(self, variable, lane, block=None, subblock=None):
    return 'NOTSUPPORTED' #f'__builtin_shufflevector()'

  def kernel_range_object(self, name, values):
    return f"size_t {name}[3] = {{ {values} }}"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    with file.If(f"{pointer_name} == nullptr"):
      file.Expression("throw std::invalid_argument(\"stream may not be null!\")")

    stream_obj = f'static_cast<{self.stream_type} *>({pointer_name})'
    file(f'{self.stream_type} *stream = {stream_obj};')

  def batch_source(self, name):
    # targetDART reads a pointer-based operand through the device copy the
    # kernel prologue makes of it (`kernel_definition`).
    if self._backend == 'targetdart':
      return f'{name}_ptr'
    return name

  def get_headers(self):
    headers = ['cstdlib', 'stdexcept', 'omp.h', 'cmath']
    if self._backend == 'targetdart':
      headers += ['unordered_map']
    return headers

  def get_fptype(self, fptype, length=1, relaxed=False):
    align = f', aligned (sizeof({fptype}))' if relaxed else ''
    return (f'__attribute__ ((vector_size (sizeof({fptype}) * {length})'
            f'{align})) {fptype}')

  #: The C++ standard library's `<cmath>`.
  MATH = {
    Operation.ABS: 'std::fabs{f}({0})',
    Operation.MIN: 'std::min({t}({0}), {t}({1}))',
    Operation.MAX: 'std::max({t}({0}), {t}({1}))',
    Operation.POW: 'std::pow({0}, {1})',
    **{op: f'std::{name}({{0}})' for op, name in (
      (Operation.EXP, 'exp'), (Operation.LOG, 'log'),
      (Operation.EXPM1, 'expm1'), (Operation.LOG1P, 'log1p'),
      (Operation.SQRT, 'sqrt'), (Operation.CBRT, 'cbrt'),
      (Operation.SIN, 'sin'), (Operation.COS, 'cos'), (Operation.TAN, 'tan'),
      (Operation.ASIN, 'asin'), (Operation.ACOS, 'acos'),
      (Operation.ATAN, 'atan'), (Operation.SINH, 'sinh'),
      (Operation.COSH, 'cosh'), (Operation.TANH, 'tanh'),
      (Operation.ASINH, 'asinh'), (Operation.ACOSH, 'acosh'),
      (Operation.ATANH, 'atanh'))},
  }
