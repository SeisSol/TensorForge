from . import CudaLexic
from .lexic import Lexic


class HipLexic(CudaLexic):
  def __init__(self, underlying_hardware):
    super().__init__(underlying_hardware)
    self.thread_idx_y = "hipThreadIdx_y"
    self.thread_idx_x = "hipThreadIdx_x"
    self.thread_idx_z = "hipThreadIdx_z"
    self.block_idx_x = "hipBlockIdx_x"
    self.block_dim_y = "hipBlockDim_y"
    self.block_dim_z = "hipBlockDim_z"
    self.stream_type = "hipStream_t"

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"hipLaunchKernelGGL({func_name}, {grid}, {block}, 0, {stream}, {func_params})"

  def sync_vec_unit(self):
    # RoCM (AMD) currently doesn't support __syncwarp
    return "__syncthreads()"

  def get_sub_group_id(self, sub_group_size):
    return f'{self.thread_idx_x} % {sub_group_size}'

  def active_sub_group_mask(self):
    return None

  def broadcast_sync(self, variable, lane, mask):
    return f'__shfl({variable}, {lane})'

  def get_headers(self):
    return ["hip/hip_runtime.h"]

  def glb_store(self, lhs, rhs, nontemporal=False):
    if nontemporal and False: # TODO: re-enable once tested
      return f'__builtin_nontemporal_store({rhs}, &{lhs});'
    else:
      return f'{lhs} = {rhs};'
  
  def glb_load(self, lhs, rhs, nontemporal=False):
    if nontemporal and False: # TODO: re-enable once tested
      return f'{lhs} = __builtin_nontemporal_load(&{rhs});'
    else:
      return f'{lhs} = {rhs};'
