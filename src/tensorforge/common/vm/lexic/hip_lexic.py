# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from . import CudaLexic


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
    return None

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
    # Nothing.  `SIInsertWaitcnts` places this wait itself, and places it
    # better than we can.
    #
    # `copy_async` lowers to `llvm.amdgcn.global.load.lds`, which the pass
    # recognises as an LDS DMA: it tracks which LDS buffer each one writes
    # and emits the smallest count before the `ds_read` that needs it --
    # `vmcnt(2)` then `vmcnt(0)` for two distinct arrays, not `vmcnt(0)`
    # twice.  Its alias tracking has a fixed number of slots and falls back
    # to `vmcnt(0)` once they run out, so on a body with many buffers this is
    # conservative; the lever for that is `sched_group_barrier`, not a wait
    # written by hand.
    #
    # Writing one by hand was worse than redundant.  Inline asm with a
    # `"memory"` clobber is opaque to the very pass that would have computed
    # the count, and to the scheduler that decides the issue order the count
    # is derived from -- so it degraded the result it was meant to control.
    #
    # And it was wrong ahead of gfx12.  `vmcnt` is deprecated there: the
    # counter is split into loadcnt, storecnt, dscnt, kmcnt, samplecnt,
    # bvhcnt and expcnt, and gfx1250 adds asynccnt and tensorcnt for exactly
    # this class of transfer, reachable through `s_wait_asynccnt` rather than
    # through an encoded `s_waitcnt` immediate.  An instruction spelled here
    # would have to be respelled per target; a count left in the IR does not.
    #
    # The `prior` the IR derives stays: it is what a future emitter needs to
    # pick `s_wait_asynccnt` on gfx125x, and what `verify` checks the token
    # pairing against.  It is information, not an instruction.
    return ''

  def wait_async_regs(self, prior):
    # Also nothing, and here it never needed saying at all: a global load
    # into a VGPR has a register dependency, which is the thing
    # `SIInsertWaitcnts` was built to see.  It waits before the first use of
    # the destination register and not one instruction earlier.
    if self._underlying_hardware != 'amd':
      return super().wait_async_regs(prior)
    return self.wait_async(prior)

  def get_fptype(self, fptype, length=1, relaxed=False):
    kind = 'VectorRelaxedT' if relaxed else 'VectorT'
    return f'tensorforge::{kind}<{fptype}, {length}>'

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

  def atomic_store(self, ctx, access, variable, op, datatype):
    """The intrinsic where this target has one, `__hip_atomic_fetch_add` else.

    The intrinsic was emitted unconditionally, on every target and for both
    types.  `__builtin_amdgcn_global_atomic_fadd_f32` is gated on
    `atomic-fadd-rtn-insts` and `..._f64` on `gfx90a-insts`, so that was a
    compile error on gfx900, gfx906, gfx908, gfx1010 and gfx1030 -- and on
    gfx1250 and gfx1251 for f64, which have the instruction under a different
    feature and not the builtin.  `atomics.amd_add_builtin` asks LLVM's own
    table instead of the vendor string.

    Where there is no builtin the fallback is not a retreat to a
    compare-and-swap loop: agent scope and relaxed ordering are what let the
    backend select `global_atomic_add_*`, and it does so given an assurance
    that the pointer is not fine-grained (`unsafe_fp_atomics_required` names
    the targets where that assurance has to come from the build).  Relaxed
    because an add carries no ordering an accumulation depends on, and agent
    rather than system because system scope is what forces the loop.

    And it is reached at all only when `_underlying_hardware` is AMD.  HIP
    compiles for NVIDIA as well, where every name here is undeclared -- the
    same condition `glb_store` has always had and this never did.
    """
    if self._underlying_hardware != 'amd':
      return super().atomic_store(ctx, access, variable, op, datatype)
    from tensorforge.backend import atomics
    builtin = atomics.amd_add_builtin(ctx, datatype)
    if builtin is not None:
      return f'{builtin}(&{access}, {variable});'
    return (f'__hip_atomic_fetch_add(&{access}, {variable}, '
            f'__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);')
