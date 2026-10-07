# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from .lexic import Lexic, Operation
from tensorforge.common.basic_types import Datatype
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.backend.writer import MultiBlock

class CudaLexic(Lexic):

  def __init__(self, backend, underlying_hardware):
    super().__init__(underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "threadIdx.y"
    self.thread_idx_x = "threadIdx.x"
    self.thread_idx_z = "threadIdx.z"
    self.block_idx_x = "blockIdx.x"
    self.block_dim_x = "blockDim.x"
    self.block_dim_y = "blockDim.y"
    self.block_dim_z = "blockDim.z"
    self.grid_dim_x = "gridDim.x"
    self.stream_type = "cudaStream_t"
    self.restrict_kw = "__restrict__"
    # sm_70 and up; below it the annotation does not exist and the parameter
    # is copied per thread, which is the behavior without it anyway.

  def get_launch_size(self, func_name, block, shmem, resident=False):
    return f"""static std::size_t gridsize = 0;
    if (gridsize == 0) {{
      int device, smCount, blocksPerSM;
      cudaGetDevice(&device);
      CHECK_ERR;
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
      CHECK_ERR;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, {func_name}, {block}.x * {block}.y * {block}.z, {shmem});
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
      cudaFuncSetAttribute({func_name}, cudaFuncAttributeMaxDynamicSharedMemorySize, {shmem});
      CHECK_ERR;
      shmemsizeset = true;
    }}
    """

  def get_launch_code(self, func_name, grid, block, stream, func_params, shmem, coop):
    if coop:
      return f"""
  auto args = tensorforge::argsPtrs({func_params});
  cudaLaunchCooperativeKernel({func_name}, {grid}, {block}, args.data(), {shmem}, {stream});
"""
    else:
      return f"{func_name}<<<{grid},{block},{shmem},{stream}>>>({func_params})"

  def declare_shared_memory(self, name, precision, size=None):
    return f'auto* {name} = reinterpret_cast<{precision}*>({GeneralLexicon.TOTAL_SHR_MEM}Ptr)'

  def storage_class(self, space):
    # The one space CUDA needs told: without it a by-value parameter whose
    # address is taken, or which a runtime value indexes, is copied to `.local`
    # per thread -- which is the whole cost `TableForm.PARAM` exists to avoid.
    if getattr(space, 'name', None) == 'PARAM':
      return '__grid_constant__'
    return ''

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None,
                        total_shared_mem_size=None, global_symbols=None,
                        lanes=None):
    args = [str(item) for item in kernel_bounds]
    bounds = f"\n__launch_bounds__({', '.join(args)})\n"
    header = f'__global__ void {bounds} kernel_{base_name}({params})'
    shmdef = f'extern __shared__ char {GeneralLexicon.TOTAL_SHR_MEM}Ptr[];\n'
    return MultiBlock(file, [header, shmdef])

  def sync_block(self):
    return "__syncthreads();"

  def sync_simd(self):
    return "__syncwarp();"

  def sync_grid(self):
    return "cooperative_groups::this_grid().sync();"

  #: Sixteen barrier resources per CTA, 0 through 15.  `__syncthreads()` is
  #: barrier 0, so a multiplication may take one of the remaining fifteen.
  NAMED_BARRIERS = 16

  def sync_mult(self, num_threads: int, hw):
    wave = hw.vec_unit_length
    if num_threads < wave:
      # The lanes of this multiplication and no others.  A full mask here
      # would wait for the neighboring multiplications in the same warp,
      # which are free to run the body a different number of times.
      mults = wave // num_threads
      mask = ((1 << num_threads) - 1)
      return (f'__syncwarp(0x{mask:08x}u << '
              f'({self.thread_idx_y} % {mults} * {num_threads}));')
    # `threadIdx.y + 1`, because barrier 0 is the one `__syncthreads()` takes.
    # Two multiplications sharing an id rendezvous with each other, which is a
    # deadlock the moment they run the body a different number of times -- so
    # the id has to be per multiplication, and `mults_per_block` is capped to
    # the resources by the thread-block policy.
    return (f'asm volatile("barrier.sync %0, %1;" :: '
            f'"r"({self.thread_idx_y} + 1), "r"({num_threads}) : "memory");')

  def active_sub_group_mask(self):
    return "__activemask()"

  def broadcast(self, variable, lane, block=None, subblock=1):
    if block is None or block == 32:
      return f'tensorforge::readlane({variable}, {lane})'
    else:
      if subblock is None:
        subblock = 1
      return f'tensorforge::broadcast<{block}, {subblock}, {lane}>({variable})'

  def kernel_range_object(self, name, values):
    return f"dim3 {name} ({values})"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    if_stream_exists = f'({pointer_name} != nullptr)'
    stream_obj = f'static_cast<{self.stream_type}>({pointer_name})'
    file(f'{self.stream_type} stream = {if_stream_exists} ? {stream_obj} : 0;')

  def get_headers(self):
    return ["tensorforge_device/cuda.h", "cuda_pipeline.h"]

  def copy_async(self, dst, src, nbytes, zfill: int = 0):
    # The size is not only a count here.  Measured from what this toolchain
    # emits for sm_120: sixteen bytes lower to `cp.async.cg` and
    # `LDGSTS.E.BYPASS.128`, four and eight to `cp.async.ca`, which fills L1 on
    # the way -- so a staging transfer that ends in narrower accesses evicts
    # the working set of every other read in the kernel.
    #
    # `zfill` is the src-size operand: the copy moves `nbytes - zfill` from
    # global memory and zeroes the rest, and it stays on the bypass
    # (`LDGSTS.E.BYPASS.128.ZFILL`).  That is what lets a run whose length is
    # not a whole number of accesses be covered at sixteen anyway.
    if zfill:
      return f'__pipeline_memcpy_async({dst}, {src}, {nbytes}, {zfill});'
    return f'__pipeline_memcpy_async({dst}, {src}, {nbytes});'

  def commit_async(self):
    return '__pipeline_commit();'

  def wait_async(self, prior):
    return f'__pipeline_wait_prior({prior});'

  def prefetch(self, address, *, datatype, elems=1, level='l2'):
    """The one target where the cache level is part of the instruction.

    `elems` is not: `prefetch.global` names an address and brings in the line
    around it, with no count to widen that. A run longer than a line is
    therefore several statements, which is the caller's business since only
    it knows the line width.
    """
    fn = 'prefetchL1' if str(level).lower() == 'l1' else 'prefetchL2'
    return f'tensorforge::{fn}({address});'

  def get_fptype(self, fptype, length=1, relaxed=False):
    if length == 1:
      return f'{fptype}'
    # Not `float2`/`float4`.  Those are CUDA's own structs: they have no
    # arithmetic operators, so an elementwise op on one does not compile, and
    # they are unrelated to `VectorRelaxedT`, so the two spellings cannot be
    # assigned to each other -- a staging transfer would load a `float4` and
    # try to store it through a relaxed pointer of a different type.  `hip.h`
    # and `cuda.h` declare the same pair, and both are usable as values and as
    # cast targets.
    kind = 'VectorRelaxedT' if relaxed else 'VectorT'
    return f'tensorforge::{kind}<{fptype}, {length}>'

  def vector_fma(self, a, b, c):
    # `cuda.h`: pairwise `__ffma2_rn` where there is a paired FMA, the plain
    # contracted expression everywhere else.
    return f'tensorforge::fma({a}, {b}, {c})'

  #: The C math library, as CUDA declares it for device code (`sinf`,
  #: `sin`).  HIP declares the same names.
  MATH = {
    Operation.MIN: 'fmin{f}({0}, {1})',
    Operation.MAX: 'fmax{f}({0}, {1})',
    Operation.POW: 'pow{f}({0}, {1})',
    Operation.ABS: 'fabs{f}({0})',
    Operation.GAMMA: 'tgamma{f}({0})',
    **{op: name + '{f}({0})' for op, name in (
      (Operation.ERF, 'erf'), (Operation.EXP, 'exp'), (Operation.LOG, 'log'),
      (Operation.EXPM1, 'expm1'), (Operation.LOG1P, 'log1p'),
      (Operation.SQRT, 'sqrt'), (Operation.CBRT, 'cbrt'),
      (Operation.SIN, 'sin'), (Operation.COS, 'cos'), (Operation.TAN, 'tan'),
      (Operation.ASIN, 'asin'), (Operation.ACOS, 'acos'),
      (Operation.ATAN, 'atan'), (Operation.SINH, 'sinh'),
      (Operation.COSH, 'cosh'), (Operation.TANH, 'tanh'),
      (Operation.ASINH, 'asinh'), (Operation.ACOSH, 'acosh'),
      (Operation.ATANH, 'atanh'))},
  }

  #: C++ `tensorforge::Operation` members, by the `Operation` they lower from.
  REDUCTION_OPS = {
      Operation.ADD: 'Add',
      Operation.MUL: 'Mul',
      Operation.MIN: 'Min',
      Operation.MAX: 'Max',
      Operation.AND: 'And',
      Operation.OR: 'Or',
      Operation.XOR: 'Xor',
  }

  def reduction(self, variable, optype, fptype, block, subblock=1):
    """An all-reduce across `block` lanes, in groups of `subblock`.

    A call rather than a butterfly spelled out here, for the same reason
    `broadcast` is one: the exchange already exists in `tensorforge_device`,
    both backends define it under the same name, and `multilinear`'s
    lead-dimension fold wants the same thing.  A lexic that emitted the
    intrinsics directly would be a second copy of it -- and `HipLexic`
    inherits this method, so an intrinsic spelled here would reach HIP too.
    """
    if optype not in self.REDUCTION_OPS:
      raise NotImplementedError(f'reduction over {optype}')
    ctype = fptype.ctype()
    op = (f'tensorforge::ReductionOperation<{ctype}, '
          f'tensorforge::Operation::{self.REDUCTION_OPS[optype]}>')
    return (f'tensorforge::reduction<{op}, {block}, {subblock}, {ctype}>'
            f'({variable})')

  @staticmethod
  def _hint_kind(nontemporal):
    """`cs` where the access asked to stream (`Options.cache_hints`), `cg`
    for any other hint -- a bare `True` included.  `__ldcs`/`__stcs` are
    declared beside `__ldcg`/`__stcg`, over the same types."""
    return 'cs' if nontemporal == 'cs' else 'cg'

  def glb_store(self, lhs, rhs, *, datatype, length=1, nontemporal=False):
    if nontemporal:
      return f'__st{self._hint_kind(nontemporal)}(&{lhs}, {rhs});'
    else:
      return f'{lhs} = {rhs};'

  def glb_load(self, rhs, *, datatype, length=1, nontemporal=False):
    if nontemporal:
      return f'__ld{self._hint_kind(nontemporal)}(&{rhs})'
    else:
      return f'{rhs}'

  def atomic_store(self, ctx, access, variable, op, datatype, length=1):
    """`atomicAdd` with the result dropped, which is what makes it a `RED`.

    ptxas emits the reduction form -- fire-and-forget, no return path to
    scoreboard -- only when nothing reads the old value.  Binding the result
    to a name gets `ATOM` instead, at the same address and for nothing, so the
    call is a statement here rather than an expression a caller might keep.

    Device scope, not block: the reason an accumulation is atomic at all is
    that another block may be writing the same element.  `atomicAdd_block` is
    the cheaper spelling for a destination one block owns, and a destination
    one block owns does not need an atomic.

    A wide update goes through the vector overloads, which exist from sm_90
    for `float2` and `float4` and are global-memory only.  The cast is what
    the spelling costs: the value arrives as `tensorforge::VectorT<float, N>`,
    a struct of `cuda.h`, and `atomicAdd` is declared over CUDA's `floatN` --
    same size, same alignment, no implicit conversion between them.  Only
    reached when `Target.native_atomic` agreed for this width, so the overload
    it names exists.
    """
    if length == 1:
      return f'atomicAdd(&{access}, {variable});'
    vec = f'float{length}' if datatype == Datatype.F32 else f'{datatype}{length}'
    return (f'atomicAdd(reinterpret_cast<{vec}*>(&{access}), '
            f'*reinterpret_cast<const {vec}*>(&{variable}));')
