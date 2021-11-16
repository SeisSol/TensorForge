from gemmforge.basic_types import GeneralLexicon
from .lexic import Lexic


class SyclLexic(Lexic):
  def __init__(self, underlying_hardware):
    super().__init__(underlying_hardware)
    self.thread_idx_y = "item.get_local_id(1)"
    self.thread_idx_x = "item.get_local_id(0)"
    self.thread_idx_z = "item.get_local_id(2)"
    self.block_idx_x = "item.get_group().get_id(0)"
    self.block_idx_z = "item.get_group().get_id(2)"
    self.block_dim_y = "item.get_group().get_local_range(1)"
    self.block_dim_z = "item.get_group().get_local_range(2)"
    self.stream_name = "cl::sycl::queue"

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"kernel_{func_name}({stream}, {grid}, {block}, {func_params})"

  def declare_shared_memory_inline(self, name, precision, size, alignment):
    return None

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None):
    if total_shared_mem_size is not None and precision is not None:
      localmem = f'cl::sycl::accessor<{precision}, 1,'
      localmem += ' cl::sycl::access::mode::read_write,'
      localmem += ' cl::sycl::access::target::local> '
      localmem += f'{GeneralLexicon.TOTAL_SHR_MEM} ({total_shared_mem_size}, cgh);'
    else:
      localmem = None

    return file.SyclKernel(base_name, params, kernel_bounds, localmem)

  def sync_threads(self):
    return "item.barrier()"

  def sync_vec_unit(self):
    return "item.barrier()"

  def kernel_range_object(self):
    return "cl::sycl::range<3>"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    with file.If(f"{pointer_name} == nullptr"):
      file.Expression("throw std::invalid_argument(\"stream may not be null!\")")

    stream_obj = f'static_cast<{self.stream_name} *>({pointer_name})'
    file(f'{self.stream_name} *stream = {stream_obj};')

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
