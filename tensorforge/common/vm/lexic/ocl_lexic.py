from .lexic import Lexic

class OpenCLLexic(Lexic):
  def __init__(self, backend, underlying_hardware):
    super().__init__(underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "get_local_id(1)"
    self.thread_idx_x = "get_local_id(0)"
    self.thread_idx_z = "get_local_id(2)"
    self.block_idx_x = "get_group_id(0)"
    self.block_idx_z = "get_group_id(2)"
    self.block_dim_y = "get_local_size(1)"
    self.block_dim_z = "get_local_size(2)"
    self.grid_dim_x = "get_num_groups(0)"
    self.stream_type = "clqueue_t"
    self.restrict_kw = "__restrict__"

  def multifile(self):
    return True

  def get_launch_size(self, func_name, block):
    return None

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"{func_name}({stream}, {grid}, {block}, {func_params})"

  def declare_shared_memory_inline(self, name, precision, size, alignment):
    return None

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None):
    pass
  #__kernel __attribute__(( reqd_work_group_size({kernel_bounds}) ))

  def sync_block(self):
    return "item.barrier()"

  def sync_simd(self):
    return "item.barrier()"

  def get_sub_group_id(self, sub_group_size):
    return f'item.get_sub_group().get_local_id()[0]'

  def active_sub_group_mask(self):
    return f'item.get_sub_group()'

  def broadcast_sync(self, variable, lane, mask):
    return f'group_broadcast({mask}, {variable}, {lane})'

  def kernel_range_object(self, name, values):
    return f"cl::sycl::range<3> {name} ({values})"

  def get_fptype(self, fptype, length=1):
    return f'sycl::vec<{fptype}, {length}>'

# TODO: nontemporal load/store
