# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from . import CudaLexic
from tensorforge.common.basic_types import Datatype


class HipLexic(CudaLexic):
  def __init__(self, backend, underlying_hardware):
    super().__init__(backend, underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "threadIdx.y"
    self.thread_idx_x = "threadIdx.x"
    self.thread_idx_z = "threadIdx.z"
    self.block_idx_x = "blockIdx.x"
    self.block_dim_x = "blockDim.x"
    self.block_dim_y = "blockDim.y"
    self.block_dim_z = "blockDim.z"
    self.grid_dim_x = "gridDim.x"
    self.stream_type = "hipStream_t"

  def multifile(self):
    return False

  def get_launch_size(self, func_name, block, shmem):
    return f"""static std::size_t gridsize = 0;
    if (gridsize == 0) {{
      int device, smCount, blocksPerSM;
      CHECK_RES(hipGetDevice(&device));
      CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
      CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, {func_name}, {block}.x * {block}.y * {block}.z, {shmem}));
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
      CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&{func_name}), hipFuncAttributeMaxDynamicSharedMemorySize, {shmem}));
      CHECK_ERR;
      shmemsizeset = true;
    }}
    """

  def get_launch_code(self, func_name, grid, block, stream, func_params, shmem, coop):
    return f"hipLaunchKernelGGL({func_name}, {grid}, {block}, {shmem}, {stream}, {func_params})"

  def sync_simd(self):
    # noop
    return ""

  def get_sub_group_id(self, sub_group_size):
    return f'{self.thread_idx_x} % {sub_group_size}'

  def active_sub_group_mask(self):
    return None

  def broadcast(self, variable, lane, block=None, subblock=None):
    if block is None:
      return f'tensorforge::readlane({variable}, {lane})'
    else:
      if subblock is None:
        subblock = 1
      return f'tensorforge::broadcast<{block}, {subblock}, {lane}>({variable})'

  def get_headers(self):
    return ["hip/hip_runtime.h", "tensorforge_device/hip.h"]

  # CDNA has no __pipeline_*; the equivalent is a direct global->LDS load
  # plus an explicit vmcnt wait.  gfx90a/gfx94x accept 1, 2 and 4 bytes per
  # lane, gfx950 additionally 12 and 16.
  def copy_async_sizes(self):
    if self._underlying_hardware != 'amd':
      return super().copy_async_sizes()
    return (1, 2, 4)

  def copy_async(self, dst, src, nbytes):
    if self._underlying_hardware != 'amd':
      return super().copy_async(dst, src, nbytes)

    # TODO: use address space templates from tensorforge_device/hip.h
    return (f'__builtin_amdgcn_global_load_lds('
            f'(const __attribute__((address_space(1))) uint32_t*)({src}), '
            f'(__attribute__((address_space(3))) uint32_t*)({dst}), '
            f'{nbytes}, 0, 0);')

  def commit_async(self):
    if self._underlying_hardware != 'amd':
      return super().commit_async()
    return ''

  def wait_async(self, prior):
    if self._underlying_hardware != 'amd':
      return super().wait_async(prior)
    # Let the assembler encode vmcnt: the encoding is split across
    # non-contiguous bits on gfx9 and moved again on gfx10+, and
    # __builtin_amdgcn_s_waitcnt takes the *encoded* immediate.
    # TODO: remove again (and rely implicitly), replace by async waits (cf. latest LLVM)
    return f'__asm__ volatile("s_waitcnt vmcnt({prior})" ::: "memory");'

  def wait_async_regs(self, prior):
    # Same counter as the LDS path: on CDNA every vector memory operation,
    # global->VGPR included, decrements vmcnt.
    if self._underlying_hardware != 'amd':
      return super().wait_async_regs(prior)
    return self.wait_async(prior)

  def get_fptype(self, fptype, length=1):
    return f'tensorforge::VectorT<{fptype}, {length}>'

  def glb_store(self, lhs, rhs, nontemporal=False):
    if nontemporal and self._underlying_hardware == 'amd':
      return f'__builtin_nontemporal_store({rhs}, &{lhs});'
    else:
      return f'{lhs} = {rhs};'

  def glb_load(self, rhs, nontemporal=False):
    if nontemporal and self._underlying_hardware == 'amd':
      return f'__builtin_nontemporal_load(&{rhs})'
    else:
      return f'{rhs}'

  def atomic_store(self, access, variable, op, datatype):
    # those sometimes are faster
    if datatype == Datatype.F32:
      return f'__builtin_amdgcn_global_atomic_fadd_f32(&{access}, {variable});'
    if datatype == Datatype.F64:
      return f'__builtin_amdgcn_global_atomic_fadd_f64(&{access}, {variable});'
    return f'atomicAdd(&{access}, {variable});'

  def has_atomic_store(self, op, datatype):
    return True
