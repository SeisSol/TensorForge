# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from . import CudaLexic


def _gfx_level(model):
  """`gfx1030` -> 0x1030, and None for anything that is not a gfx model.

  Hexadecimal, the way `amd/arch.py` reads the same string: the letters in
  gfx90a are digits of the number, and a decimal read would both fail on them
  and sort gfx940 above gfx1030. None rather than 0 keeps "this is not an AMD
  part" apart from "it is an early one", so a `sm_90` model does not answer an
  AMD question by comparing low.
  """
  text = str(model)
  if not text.startswith('gfx'):
    return None
  try:
    return int(text[3:], base=16)
  except ValueError:
    return None


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

  def storage_class(self, space):
    # Nothing, and not by omission: HIP kernel arguments already live in the
    # constant address space, and a uniform index into one is a scalar load.
    # Stated rather than inherited, since this shares its base with the backend
    # that does need the annotation.
    return ''

  #: `hip.h`'s names for the address spaces clang numbers.
  MEMSPACE = {'GLOBAL': 'tensorforge::GlobalMemspace',
              'CONSTANT': 'tensorforge::ConstantMemspace',
              'PARAM': 'tensorforge::ConstantMemspace'}

  def pointer_type(self, elem, space=None, readonly=False, restrict=False,
                   const=False):
    """The space-qualified pointer, where `hip.h` has a name for the space.

    A named space is worth spelling here and nowhere else: the attribute sits
    on the *pointee*, so `SpacePtr<T, S>` and `T*` are different types and the
    conversion between them runs one way only -- a space-qualified pointer
    converts to a generic one implicitly, and back only through a cast.  Which
    is why this used to be a cast on the right of one binding with `auto` on
    the left: with the type unsayable, the only way to have it was to never
    name it.  A pass declaring a copy of that value from its type got the
    generic pointer and lost the space silently.

    Restrict is fused rather than appended.  The alias carries both, and
    `SpacePtr<T, S> __restrict` is not the same declaration -- the attribute
    would apply to the alias rather than through it.

    Shared and register are left generic.  LDS pointers are produced by the
    arena binding, which spells its own declarator, and giving them a space
    here would make every window incompatible with the arena it is a window
    into.
    """
    name = self.MEMSPACE.get(getattr(space, 'name', None))
    if name is None:
      return super().pointer_type(elem, space, readonly, restrict, const)
    ro = 'const ' if readonly else ''
    alias = 'SpacePtrRestrict' if restrict else 'SpacePtr'
    # `SpacePtrRestrict<T, S> const` is `T *__restrict const`, which is the
    # declaration the generic spelling writes as `T *const __restrict__`.  The
    # alias ends in the pointer, so the qualifier goes after it and means the
    # same thing.
    tail = ' const' if const else ''
    return f'tensorforge::{alias}<{ro}{elem}, {name}>{tail}'

  def multifile(self):
    return False

  def get_launch_size(self, func_name, block, shmem, resident=False):
    # ROCm up to 7.2 sizes a block's LDS against one CU's 64 KB even in WGP
    # mode, where a "multiprocessor" is a WGP of two CUs and 128 KB -- the
    # waves and SIMDs it does count per WGP, so only the LDS bound is halved.
    # An LDS-bound kernel therefore got half the grid on RDNA: `local_flux`
    # with its 51 KB preload one block per WGP where two fit, 36 % slower on
    # gfx1150.  Fixed upstream (`localMemSizePerCU_ * (isWGPMode_ ? 2 : 1)` in
    # clr's `hip_platform.cpp`), so this only ever raises the runtime's answer
    # and lets a corrected runtime stand.  gfx9 has no WGPs and is untouched.
    #
    # Not for a cooperative launch: there every block has to be resident at
    # once, and the runtime's count is the one it checks the launch against.
    wgp = "" if resident else f"""
      int gfxMajor = 0;
      CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
      if (gfxMajor >= 10 && ({shmem}) > 0) {{
        int ldsPerMP = 0, blocksNoLds = 0;
        CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, {func_name}, {block}.x * {block}.y * {block}.z, 0));
        const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / ({shmem}));
        blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
      }}"""
    return f"""static std::size_t gridsize = 0;
    if (gridsize == 0) {{
      int device, smCount, blocksPerSM;
      CHECK_RES(hipGetDevice(&device));
      CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
      CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, {func_name}, {block}.x * {block}.y * {block}.z, {shmem}));
      CHECK_ERR;{wgp}
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
    if coop:
      return f"""
  auto args = tensorforge::argsPtrs({func_params});
  hipLaunchCooperativeKernel({func_name}, {grid}, {block}, args.data(), {shmem}, {stream});
"""
    return f"hipLaunchKernelGGL({func_name}, {grid}, {block}, {shmem}, {stream}, {func_params})"

  def sync_simd(self):
    return None

  def has_sync_mult(self, num_threads: int, hw) -> bool:
    """Within a wave, free; across waves, gfx12.5 and not before.

    A multiplication inside a wave needs no instruction at all -- the lanes of
    a wave are in lockstep, so the rendezvous has already happened -- which is
    the same reason `sync_simd` is None.  Saying True for it is not a shortcut:
    the barrier is genuinely narrower than a block and genuinely costs nothing.

    Above a wave it needs a barrier object of its own.  GFX12 splits `s_barrier`
    into signal and wait but leaves only the workgroup barrier visible to the
    shader; the objects a shader may assign, 1 through 16, arrive at GFX12.5.
    So the width has to be whole waves and the target has to be gfx1250 or
    later.
    """
    wave = hw.vec_unit_length
    if num_threads <= wave:
      return wave % num_threads == 0
    # Above a wave the answer is False until the prologue exists: the objects
    # need `s_barrier_init` with the expected count, a workgroup barrier so
    # that nobody joins before the init lands, and one `s_barrier_join` per
    # wave, all before the first use.  `_named_barriers` says which targets
    # could carry it.
    return False

  @staticmethod
  def _named_barriers(hw) -> bool:
    model = str(hw.model)
    if not model.startswith('gfx'):
      return False
    try:
      return int(model[3:], base=16) >= 0x1250
    except ValueError:
      return False

  def sync_mult(self, num_threads: int, hw):
    if num_threads <= hw.vec_unit_length:
      return None
    # Reached only once `has_sync_mult` agrees above a wave, which needs the
    # prologue.  The expected count would be in *waves* and not threads:
    # `s_barrier` synchronises at wavefront granularity, so what the object
    # counts is how many waves have to arrive.
    return self.sync_block()

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
    return ["hip/hip_runtime.h", "hip/hip_cooperative_groups.h", "tensorforge_device/hip.h"]

  def loop_body_fence(self):
    if self._underlying_hardware != 'amd':
      return super().loop_body_fence()
    # LLVM hoists the loads of the preloaded operators out of an unguarded
    # batch loop: on `local_flux`, 112 values from LDS held in registers across
    # the loop -- 256 VGPRs and half the occupancy on gfx942, 232 bytes of
    # scratch on gfx1250, and +90 % runtime on gfx1150 with the transfers
    # wrapped.  NVIDIA's compiler does not do it (150 registers either way).
    #
    # A scheduling barrier stops it, and not by accident: the intrinsic is
    # `IntrHasSideEffects`, which it has to be so that nothing deletes or moves
    # it, and a side effect is an unknown memory effect to LICM.  `sched_barrier`
    # rather than an asm memory clobber, because the clobber is opaque to
    # `SIInsertWaitcnts` and to the scheduler as well (see `wait_async`); the
    # barrier is what the backend knows how to schedule around.  Mask 0 keeps
    # the scheduler from moving anything across it, which at the head of the
    # body is only the head itself.
    return '__builtin_amdgcn_sched_barrier(0);'

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

  def has_prefetch(self, hw):
    """gfx12 and up, where `hasPrefetch` in LLVM's subtarget is `GFX12Insts`.

    Not a question about whether the call compiles: `__builtin_prefetch` is
    accepted at every AMD target and simply selects no instruction below
    gfx12. So a False here does not avert an error, it avoids carrying a
    statement that reaches nothing -- and it is what makes the emitter say
    once, per body, that this part has no prefetch to give.

    HIP compiles for NVIDIA as well, where the builtin would be an NVPTX
    question and not this one; the same condition `glb_store` and
    `atomic_store` carry.
    """
    if self._underlying_hardware != 'amd':
      return False
    arch = _gfx_level(getattr(hw, 'model', None))
    return arch is not None and arch >= 0x1200

  def prefetch(self, address, *, datatype, elems=1, level='l2'):
    """`__builtin_prefetch`, which has no cache level to be given.

    The builtin takes (address, rw, locality) and always asks for the data
    cache. On AMDGPU the locality is what becomes a memory *scope* -- 0 is
    SCOPE_SYS, 1 SCOPE_DEV, 2 and 3 SCOPE_SE -- so there is no L1/L2 choice
    to make here and the requested level is honoured by being ignored.
    SCOPE_CU, which is what a locality argument would have to reach for the
    nearest cache, is not generated at all: it is unsafe on an address that
    does not resolve, and a prefetch that faults is worse than one that
    misses.

    The builtin rather than the intrinsic, because which instruction it means
    follows the target: `s_prefetch_data` from gfx12, and the VMEM
    `global_prefetch` into GL2 on gfx1250, where that exists. Naming one of
    them here would fix the other in place.
    """
    if self._underlying_hardware != 'amd':
      # Unreachable through `has_prefetch`, and spelled out rather than
      # inherited: this class extends the CUDA one, so a `super()` call here
      # would hand a HIP-on-NVIDIA build the PTX helper from `cuda.h`, which
      # that translation unit does not include.
      return None
    return f'__builtin_prefetch({address}, 0, 3);'

  def vector_fma(self, a, b, c):
    # GNU vectors: the infix form contracts, into packed FMAs where the
    # target has them.  `hip.h` declares no `tensorforge::fma` to call.
    return None

  def get_fptype(self, fptype, length=1, relaxed=False):
    kind = 'VectorRelaxedT' if relaxed else 'VectorT'
    return f'tensorforge::{kind}<{fptype}, {length}>'

  def has_nontemporal(self, datatype, length=1):
    """The hardware, not the type: the builtins are generic.

    `__builtin_nontemporal_load` and `_store` take any scalar or vector
    operand, so unlike the CUDA pair there is no overload set to fall outside
    of and no type this has to turn away -- `__float128` and a `VectorT` are
    both accepted.

    HIP compiles for NVIDIA as well, where neither builtin is declared; that
    is the condition `glb_store` and `glb_load` have always carried, stated
    here once so `atomic_store` and these two answer it the same way.
    """
    return self._underlying_hardware == 'amd'

  def glb_store(self, lhs, rhs, *, datatype, length=1, nontemporal=False):
    if nontemporal and self.has_nontemporal(datatype, length):
      return f'__builtin_nontemporal_store({rhs}, &{lhs});'
    else:
      return f'{lhs} = {rhs};'

  def glb_load(self, rhs, *, datatype, length=1, nontemporal=False):
    if nontemporal and self.has_nontemporal(datatype, length):
      return f'__builtin_nontemporal_load(&{rhs})'
    else:
      return f'{rhs}'

  def atomic_store(self, ctx, access, variable, op, datatype, length=1):
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
      return super().atomic_store(ctx, access, variable, op, datatype, length)
    # No packed FP32 or FP64 add exists here, so `has_atomic_store` refuses
    # every width but one and this never sees a vector.  Asserted rather than
    # assumed: the builtin below takes a scalar and would take a GNU vector
    # by silent truncation of nothing -- it simply would not compile, which
    # is a worse way to learn it than this.
    assert length == 1, f'no packed atomic add on AMD (length {length})'
    from tensorforge.backend import atomics
    builtin = atomics.amd_add_builtin(ctx, datatype)
    if builtin is not None:
      return f'{builtin}(&{access}, {variable});'
    return (f'__hip_atomic_fetch_add(&{access}, {variable}, '
            f'__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);')
