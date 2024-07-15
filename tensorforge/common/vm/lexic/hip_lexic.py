from . import CudaLexic


class HipLexic(CudaLexic):
  def __init__(self, backend, underlying_hardware):
    super().__init__(backend, underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "hipThreadIdx_y"
    self.thread_idx_x = "hipThreadIdx_x"
    self.thread_idx_z = "hipThreadIdx_z"
    self.block_idx_x = "hipBlockIdx_x"
    self.block_dim_x = "hipBlockDim_x"
    self.block_dim_y = "hipBlockDim_y"
    self.block_dim_z = "hipBlockDim_z"
    self.grid_dim_x = "gridDim.x"
    self.stream_type = "hipStream_t"

  def get_launch_size(self, func_name, block, shmem):
    return f"""static int gridsize = -1;
    if (gridsize <= 0) {{
      int device, smCount, blocksPerSM;
      hipGetDevice(&device);
      CHECK_ERR;
      hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device);
      CHECK_ERR;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, {func_name}, {block}.x * {block}.y * {block}.z, {shmem});
      CHECK_ERR;
      if (blocksPerSM > 0) {{
        gridsize = smCount * blocksPerSM;
      }}
      else {{
        gridsize = smCount;
      }}
    }}
    """

  def set_shmem_size(self, func_name, shmem):
    return f"""static bool shmemsizeset = false;
    if (!shmemsizeset) {{
      hipFuncSetAttribute({func_name}, hipFuncAttributeMaxDynamicSharedMemorySize, {shmem});
      CHECK_ERR;
      shmemsizeset = true;
    }}
    """

  def get_launch_code(self, func_name, grid, block, stream, func_params, shmem):
    return f"hipLaunchKernelGGL({func_name}, {grid}, {block}, {shmem}, {stream}, {func_params})"

  def sync_simd(self):
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
  
  def get_fptype(self, fptype, length=1):
    return f'HIP_vector_type<{fptype}, {length}>'

  def glb_store(self, lhs, rhs, nontemporal=False):
    if nontemporal and self._underlying_hardware == 'amd':
      return f'__builtin_nontemporal_store({rhs}, &{lhs});'
    else:
      return f'{lhs} = {rhs};'
  
  def glb_load(self, lhs, rhs, nontemporal=False):
    if nontemporal and self._underlying_hardware == 'amd':
      return f'{lhs} = __builtin_nontemporal_load(&{rhs});'
    else:
      return f'{lhs} = {rhs};'
