from .lexic import Lexic, Operation
from kernelforge.common.basic_types import FloatingPointType

class CudaLexic(Lexic):

  def __init__(self, underlying_hardware):
    super().__init__(underlying_hardware)
    self.thread_idx_y = "threadIdx.y"
    self.thread_idx_x = "threadIdx.x"
    self.thread_idx_z = "threadIdx.z"
    self.block_idx_x = "blockIdx.x"
    self.block_dim_y = "blockDim.y"
    self.block_dim_z = "blockDim.z"
    self.stream_type = "cudaStream_t"
    self.restrict_kw = "__restrict__"

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return "{}<<<{},{},0,{}>>>({})".format(func_name, grid, block, stream, func_params)

  def declare_shared_memory_inline(self, name, precision, size, alignment):
    return f"__shared__  __align__({alignment}) {precision} {name}[{size}]"
  
  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    params = [str(item) for item in [total_num_threads_per_block, min_blocks_per_mp] if item]
    return f'__launch_bounds__({", ".join(params)})'

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None,
                        total_shared_mem_size=None, global_symbols=None):
    return file.CudaKernel(base_name, params, kernel_bounds)

  def sync_threads(self):
    return "__syncthreads()"

  def sync_vec_unit(self):
    return "__syncwarp()"

  def get_sub_group_id(self, sub_group_size):
    return f'{self.thread_idx_x} % {sub_group_size}'

  def active_sub_group_mask(self):
    return "__activemask()"

  def broadcast_sync(self, variable, lane, mask):
    return f'__shfl_sync({mask}, {variable}, {lane})'

  def kernel_range_object(self, name, values):
    return f"dim3 {name} ({values})"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    if_stream_exists = f'({pointer_name} != nullptr)'
    stream_obj = f'static_cast<{self.stream_type}>({pointer_name})'
    file(f'{self.stream_type} stream = {if_stream_exists} ? {stream_obj} : 0;')

  def check_error(self):
    return "CHECK_ERR"

  def batch_indexer_gemm(self):
    return self.get_tid_counter(self.thread_idx_y, self.block_dim_y, self.block_idx_x)

  def batch_indexer_csa(self):
    return self.get_tid_counter(self.thread_idx_z, self.block_dim_z, self.block_idx_x)

  def batch_indexer_init(self):
    return self.get_tid_counter(self.thread_idx_y, self.block_dim_y, self.block_idx_x)

  def get_headers(self):
    return []
  
  def get_fptype(self, fptype, length=1):
    suffix = f'{length}' if length > 1 else ''
    return f'{fptype}{suffix}'

  def get_operation(self, op: Operation, fptype, value1, value2):
    fpsuffix = 'f' if fptype == FloatingPointType.FLOAT else ''
    fpprefix = 'f' if fptype == FloatingPointType.FLOAT else 'd'
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
      return f'fmin{fpsuffix}({value1}, {value2})'
    elif op == Operation.MAX:
      return f'fmax{fpsuffix}({value1}, {value2})'
    elif op == Operation.EXP:
      return f'exp{fpsuffix}({value1})' # has __expf
    elif op == Operation.LOG:
      return f'log{fpsuffix}({value1})' # has __logf
    elif op == Operation.SQRT:
      # return f'__{fpprefix}sqrt_rn({value1})'
      return f'sqrt{fpsuffix}({value1})'
    elif op == Operation.SIN:
      return f'sin{fpsuffix}({value1})' # has __sinf
    elif op == Operation.COS:
      return f'cos{fpsuffix}({value1})' # has __cosf
    elif op == Operation.TAN:
      return f'tan{fpsuffix}({value1})' # has __tanf
    elif op == Operation.ASIN:
      return f'asin{fpsuffix}({value1})'
    elif op == Operation.ACOS:
      return f'acos{fpsuffix}({value1})'
    elif op == Operation.ATAN:
      return f'atan{fpsuffix}({value1})'
