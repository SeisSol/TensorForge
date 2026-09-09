# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from .lexic import Lexic, Operation
from tensorforge.common.basic_types import Datatype
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.backend.writer import MultiBlock

#: The types `__ldcg` and `__stcg` are declared over, as this backend spells
#: them.
#:
#: A list and not a rule, because the underlying set is a list too -- CUDA
#: declares the pair one overload at a time -- and because the answer depends
#: on the *spelling* a datatype gets here, not on the datatype.  `tf32` is
#: `uint32_t` on this target, so it takes the `unsigned int` overload and
#: belongs here; on Intel the same member is a class type and would not.
#:
#: Absent, each for its own reason.  `F128` has no overload at any
#: architecture, which is the defect this set exists to stop.  `BOOL` has
#: none either, and `const bool*` converts to no other pointer type, so it
#: would fail the same way the day something loads one.  `F16` and `BF16` are
#: spelled `half` and `bfloat16`, which nothing in `include/` declares for
#: CUDA -- so a kernel carrying them fails earlier than this, and claiming an
#: overload for a type that has no declaration would be a guess.  When that
#: spelling arrives and resolves to `__half`/`__nv_bfloat16`, `cuda_fp16.hpp`
#: and `cuda_bf16.hpp` do declare the pair, and this is the line that changes.
_CACHE_HINT_TYPES = frozenset({
    Datatype.F32,
    Datatype.F64,
    Datatype.I8,
    Datatype.I16,
    Datatype.I32,
    Datatype.I64,
    Datatype.U32,
    Datatype.TF32,
})


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
    # is copied per thread, which is the behaviour without it anyway.


  def multifile(self):
    return False

  def get_launch_size(self, func_name, block, shmem):
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

  def declare_shared_memory(self, name, precision):
    return f'auto* {name} = reinterpret_cast<{precision}*>({GeneralLexicon.TOTAL_SHR_MEM}Ptr)'

  def get_launch_bounds(self, total_num_threads_per_block, min_blocks_per_mp=None):
    params = [str(item) for item in [total_num_threads_per_block, min_blocks_per_mp] if item]
    return f'__launch_bounds__({", ".join(params)})'

  def storage_class(self, space):
    # The one space CUDA needs told: without it a by-value parameter whose
    # address is taken, or which a runtime value indexes, is copied to `.local`
    # per thread -- which is the whole cost `TableForm.PARAM` exists to avoid.
    if getattr(space, 'name', None) == 'PARAM':
      return '__grid_constant__'
    return ''

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None,
                        total_shared_mem_size=None, global_symbols=None):
    #return file.CudaKernel(base_name, params, kernel_bounds)
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

  def has_sync_mult(self, num_threads: int, hw) -> bool:
    """Two spellings, and which one applies turns on the width.

    Below a wave the multiplication is a run of lanes inside one, and
    `__syncwarp` takes the mask of exactly those lanes.  Above it the
    multiplication is a whole number of waves and `barrier.sync id, count`
    meets exactly `count` threads -- but the count must be a multiple of the
    warp size, so a width that leaves a partial wave has no spelling here and
    falls back to the group.
    """
    wave = hw.vec_unit_length
    if num_threads < wave:
      return wave % num_threads == 0
    return num_threads % wave == 0

  def sync_mult(self, num_threads: int, hw):
    wave = hw.vec_unit_length
    if num_threads < wave:
      # The lanes of this multiplication and no others.  A full mask here
      # would wait for the neighbouring multiplications in the same warp,
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

  def get_sub_group_id(self, sub_group_size):
    return f'{self.thread_idx_x} % {sub_group_size}'

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

  def check_error(self):
    return "CHECK_ERR"

  def batch_indexer_gemm(self):
    return self.get_tid_counter(self.thread_idx_y, self.block_dim_y, self.block_idx_x)

  def batch_indexer_csa(self):
    return self.get_tid_counter(self.thread_idx_z, self.block_dim_z, self.block_idx_x)

  def batch_indexer_init(self):
    return self.get_tid_counter(self.thread_idx_y, self.block_dim_y, self.block_idx_x)

  def get_headers(self):
    return ["tensorforge_device/cuda.h", "cuda_pipeline.h"]

  # cp.async: sm_80 and newer.  Gating on the architecture happens in the
  # caller (pir.emit), which has the hardware descriptor; the lexic only
  # knows how the text looks.
  def copy_async_sizes(self):
    return (4, 8, 16)

  def copy_async(self, dst, src, nbytes):
    return f'__pipeline_memcpy_async({dst}, {src}, {nbytes});'

  def commit_async(self):
    return '__pipeline_commit();'

  def wait_async(self, prior):
    return f'__pipeline_wait_prior({prior});'

  def has_prefetch(self, hw):
    """`prefetch.global.L1` and `.L2`: PTX ISA 2.0, sm_50 and up.

    Everything this generator has a row for is above that, so the check is a
    statement of what the helpers in `cuda.h` are allowed to assume rather
    than a gate anything is expected to fail.  It is still asked, because the
    inline PTX there carries no architecture guard of its own -- a target
    below the line would reach `ptxas` and fail there, which is a worse place
    to learn it.
    """
    level = hw.sm_level() if hw is not None else None
    return level is not None and level >= 50

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
    # they are unrelated to `VectorRelaxedT`, so the two spellings could not
    # be assigned to each other -- a staging transfer would load a `float4`
    # and try to store it through a relaxed pointer of a different type.
    # `hip.h` and `cuda.h` now declare the same pair, and both are usable as
    # values and as cast targets.
    kind = 'VectorRelaxedT' if relaxed else 'VectorT'
    return f'tensorforge::{kind}<{fptype}, {length}>'

  def get_operation(self, op: Operation, fptype, value1, value2):
    fpsuffix = 'f' if fptype == Datatype.F32 else ''
    fpprefix = 'f' if fptype == Datatype.F32 else 'd'
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
    elif op == Operation.RCP:
      return f'(1 / {value1})'
    elif op == Operation.MIN:
      return f'fmin{fpsuffix}({value1}, {value2})'
    elif op == Operation.MAX:
      return f'fmax{fpsuffix}({value1}, {value2})'
    elif op == Operation.ABS:
      return f'fabs{fpsuffix}({value1})'
    elif op == Operation.NEG:
      return f'(-{value1})'
    elif op == Operation.GAMMA:
      return f'tgamma{fpsuffix}({value1})'
    elif op == Operation.ERF:
      return f'erf{fpsuffix}({value1})'
    elif op == Operation.EXP:
      return f'exp{fpsuffix}({value1})' # has __expf
    elif op == Operation.LOG:
      return f'log{fpsuffix}({value1})' # has __logf
    elif op == Operation.EXPM1:
      return f'expm1{fpsuffix}({value1})'
    elif op == Operation.LOGP1:
      return f'logp1{fpsuffix}({value1})'
    elif op == Operation.SQRT:
      # return f'__{fpprefix}sqrt_rn({value1})'
      return f'sqrt{fpsuffix}({value1})'
    elif op == Operation.CBRT:
      return f'cbrt{fpsuffix}({value1})'
    elif op == Operation.POW:
      return f'pow{fpsuffix}({value1}, {value2})'
    elif op == Operation.SIN:
      return f'sin{fpsuffix}({value1})' # has __sinf
    elif op == Operation.COS:
      return f'cos{fpsuffix}({value1})' # has __cosf
    elif op == Operation.TAN:
      return f'tan{fpsuffix}({value1})' # has __tanf
    elif op == Operation.ASIN:
      return f'asin{fpsuffix}({value1})'
    elif op == Operation.ACOS:
      return f'acos{fpsuffix}({value1})'
    elif op == Operation.ATAN:
      return f'atan{fpsuffix}({value1})'
    elif op == Operation.SINH:
      return f'sinh{fpsuffix}({value1})'
    elif op == Operation.COSH:
      return f'cosh{fpsuffix}({value1})'
    elif op == Operation.TANH:
      return f'tanh{fpsuffix}({value1})'
    elif op == Operation.ASINH:
      return f'asinh{fpsuffix}({value1})'
    elif op == Operation.ACOSH:
      return f'acosh{fpsuffix}({value1})'
    elif op == Operation.ATANH:
      return f'atanh{fpsuffix}({value1})'
    elif op == Operation.NOT and fptype == Datatype.BOOL:
      return f'(!{value1})'
    elif op == Operation.NOT and fptype != Datatype.BOOL:
      return f'(~{value1})'
    elif op == Operation.AND and fptype == Datatype.BOOL:
      return f'({value1} && {value2})'
    elif op == Operation.OR and fptype == Datatype.BOOL:
      return f'({value1} || {value2})'
    elif op == Operation.AND and fptype != Datatype.BOOL:
      return f'({value1} & {value2})'
    elif op == Operation.OR and fptype != Datatype.BOOL:
      return f'({value1} | {value2})'
    elif op == Operation.XOR:
      return f'({value1} ^ {value2})'
    elif op == Operation.LT:
      return f'({value1} < {value2})'
    elif op == Operation.LE:
      return f'({value1} <= {value2})'
    elif op == Operation.GT:
      return f'({value1} > {value2})'
    elif op == Operation.GE:
      return f'({value1} >= {value2})'
    elif op == Operation.EQ:
      return f'({value1} == {value2})'
    elif op == Operation.NEQ:
      return f'({value1} != {value2})'

    raise NotImplementedError(f'{op}')

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
    intrinsics directly would be a second copy of it.

    The previous body was unreachable and would not have worked if it had
    been: it returned `None` (the loop's last statement was a bare f-string),
    tested `blocks == [2, 4, 8, 16, 32]` for what is a single width, named
    `__and_sync`, which does not exist, and emitted CUDA intrinsics
    unconditionally -- and `HipLexic` inherits this method, so HIP would have
    got `__shfl_xor_sync` too.
    """
    if optype not in self.REDUCTION_OPS:
      raise NotImplementedError(f'reduction over {optype}')
    ctype = fptype.ctype()
    op = (f'tensorforge::ReductionOperation<{ctype}, '
          f'tensorforge::Operation::{self.REDUCTION_OPS[optype]}>')
    return (f'tensorforge::reduction<{op}, {block}, {subblock}, {ctype}>'
            f'({variable})')

  def has_nontemporal(self, datatype, length=1):
    """Whether `__ldcg`/`__stcg` are declared for this type.

    They are an overload set and not a generic: `sm_32_intrinsics.h` declares
    them over the built-in integer types, `float`, `double`, and CUDA's own
    vector structs.  A type outside it does not get a slower load, it gets
    `no instance of overloaded function "__ldcg" matches the argument list`
    -- at every architecture, since the overload set is a property of the
    header and not of the target.

    `length > 1` is refused twice over.  A wide value is spelled
    `tensorforge::VectorT<T, N>` here, a GNU vector, and the overloads are
    declared over `floatN` -- same size, same alignment, no conversion
    between them, which is the cast `atomic_store` has to write out.  And a
    value of vector type does not survive nvcc in device code at all, so
    there is nothing for the hint to be attached to.  Both are spellings the
    lexic would have to change; neither is answered by naming an intrinsic
    here.
    """
    return length == 1 and datatype in _CACHE_HINT_TYPES

  def glb_store(self, lhs, rhs, *, datatype, length=1, nontemporal=False):
    if nontemporal and self.has_nontemporal(datatype, length):
      return f'__stcg(&{lhs}, {rhs});'
    else:
      return f'{lhs} = {rhs};'

  def glb_load(self, rhs, *, datatype, length=1, nontemporal=False):
    if nontemporal and self.has_nontemporal(datatype, length):
      return f'__ldcg(&{rhs})'
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
    a GNU vector, and `atomicAdd` is declared over CUDA's `floatN` -- same
    size, same alignment, no implicit conversion between them.  Only reached
    when `has_atomic_store` agreed for this width, so the overload it names
    exists.
    """
    if length == 1:
      return f'atomicAdd(&{access}, {variable});'
    vec = f'float{length}' if datatype == Datatype.F32 else f'{datatype}{length}'
    return (f'atomicAdd(reinterpret_cast<{vec}*>(&{access}), '
            f'*reinterpret_cast<const {vec}*>(&{variable}));')
