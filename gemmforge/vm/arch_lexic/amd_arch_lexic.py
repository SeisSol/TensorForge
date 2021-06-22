from . import NvidiaArchLexic
from .abstract_arch_lexic import AbstractArchLexic


class AmdArchLexic(NvidiaArchLexic):
  def __init__(self):
    AbstractArchLexic.__init__(self)
    self.thread_idx_y = "hipThreadIdx_y"
    self.thread_idx_x = "hipThreadIdx_x"
    self.thread_idx_z = "hipThreadIdx_z"
    self.block_idx_x = "hipBlockIdx_x"
    self.block_dim_y = "hipBlockDim_y"
    self.block_dim_z = "hipBlockDim_z"
    self.stream_name = "hipStream_t"

  def get_launch_code(self, func_name, grid, block, stream, func_params):
    return f"hipLaunchKernelGGL(kernel_{func_name}, {grid}, {block}, 0, {stream}, {func_params})"

  def sync_vec_unit(self):
    # RoCM (AMD) currently doesn't support __syncwarp
    return "__syncthreads()"
