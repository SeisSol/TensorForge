from kernelforge.common.basic_types import GeneralLexicon
from .lexic import Lexic, Operation


class SyclLexic(Lexic):
  def __init__(self, backend, underlying_hardware):
    super().__init__(underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "item.get_local_id(1)"
    self.thread_idx_x = "item.get_local_id(0)"
    self.thread_idx_z = "item.get_local_id(2)"
    self.block_idx_x = "item.get_group().get_group_id(0)"
    self.block_idx_z = "item.get_group().get_group_id(2)"
    self.block_dim_y = "item.get_group().get_local_range(1)"
    self.block_dim_z = "item.get_group().get_local_range(2)"
    self.grid_dim_x = "item.get_global_range(0)"
    self.stream_type = "cl::sycl::queue"
    self.restrict_kw = "__restrict__"

  def get_launch_size(self, func_name, block):
    return f"""static int gridsize = -1;
    if (gridsize <= 0) {{
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gridsize, {func_name}, {block}.x * {block}.y * {block}.z, 0);
    }}
    """

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"{func_name}({stream}, {grid}, {block}, {func_params})"

  def declare_shared_memory_inline(self, name, precision, size, alignment):
    return None

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None):
    if total_shared_mem_size is not None and precision is not None:
      if self._backend == 'acpp':
        localmem = f'cl::sycl::accessor<{precision}, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>'
      else:
        localmem = f'cl::sycl::local_accessor<{precision}, 1>'

      localmem += f' {GeneralLexicon.TOTAL_SHR_MEM} ({total_shared_mem_size}, cgh);'
    else:
      localmem = None
    
    if self._underlying_hardware['name'] == 'intel':
      add_items = '[[intel::reqd_sub_group_size(16)]] [[intel::kernel_args_restrict]]'
    else:
      add_items = ''

    l1 = f"inline void kernel_{base_name}(cl::sycl::queue *stream, cl::sycl::range<3> group_count, cl::sycl::range<3> group_size, {params})"
    l2 = f"stream->submit([&](cl::sycl::handler &cgh)"
    l3 = f"cgh.parallel_for(cl::sycl::nd_range<3>{{{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}}, group_size}}, [=](cl::sycl::nd_item<3> item) {add_items}"

    if localmem is None:
      return MultiBlock(self, [l1, l2, l3], ["", ");", ");"])
    else:
      return MultiBlock(self, [l1, l2, localmem, l3], ["", ");", "", ");"])

  def sync_threads(self):
    return "item.barrier()"

  def sync_vec_unit(self):
    return "item.barrier()"

  def get_sub_group_id(self, sub_group_size):
    return f'item.get_sub_group().get_local_id()[0]'

  def active_sub_group_mask(self):
    return f'item.get_sub_group()'

  def broadcast_sync(self, variable, lane, mask):
    return f'group_broadcast({mask}, {variable}, {lane})'

  def kernel_range_object(self, name, values):
    return f"cl::sycl::range<3> {name} ({values})"

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
    return ['CL/sycl.hpp']
  
  def get_fptype(self, fptype, length=1):
    return f'sycl::vec<{fptype}, {length}>'

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
      return f'sycl::min({value1}, {value2})'
    elif op == Operation.MAX:
      return f'sycl::max({value1}, {value2})'
    elif op == Operation.EXP:
      return f'sycl::exp({value1})' # has __expf
    elif op == Operation.LOG:
      return f'sycl::log({value1})' # has __logf
    elif op == Operation.SQRT:
      return f'sycl::sqrt({value1})'
    elif op == Operation.SIN:
      return f'sycl::sin({value1})' # has __sinf
    elif op == Operation.COS:
      return f'sycl::cos({value1})' # has __cosf
    elif op == Operation.TAN:
      return f'sycl::tan({value1})' # has __tanf
    elif op == Operation.ASIN:
      return f'sycl::asin({value1})'
    elif op == Operation.ACOS:
      return f'sycl::acos({value1})'
    elif op == Operation.ATAN:
      return f'sycl::atan({value1})'
