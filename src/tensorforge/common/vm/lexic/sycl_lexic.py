# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.basic_types import GeneralLexicon
from .lexic import INFIX, Lexic, Operation
from tensorforge.backend.writer import MultiBlock
from tensorforge.common.basic_types import Datatype

def smallest_sub_group(sizes, lanes):
  """The smallest of `sizes` that `lanes` fit in and divide, so that a
  multiplication of that many lanes lies within one sub-group; None where
  none does."""
  for size in sizes:
    if lanes <= size and size % lanes == 0:
      return size
  return None


class SyclLexic(Lexic):
  def __init__(self, backend, underlying_hardware, explicit_simd=False,
               sub_groups=None):
    super().__init__(underlying_hardware)
    self._backend = backend
    # CUDA's x is SYCL's dimension 2, and y is 1: SYCL linearizes with the
    # *last* dimension fastest, and that is the one sub-groups are cut along.
    # With the lanes in dimension 0 a block of 32 x 8 would put two lanes of
    # each of eight multiplications into every sub-group, and a sub-group
    # broadcast would read another multiplication's lane.  The groups are
    # counted along the same dimension (`kernel_definition`), and their number
    # is the group range: `get_global_range` is work-items, and with it the
    # batch loop would step 32 times too far and every group would start at
    # group 0's element.
    self.thread_idx_y = "item.get_local_id(1)"
    self.thread_idx_x = "item.get_local_id(2)"
    self.thread_idx_z = "item.get_local_id(0)"
    self.block_idx_x = "item.get_group().get_group_id(2)"
    self.block_idx_z = "item.get_group().get_group_id(0)"
    self.block_dim_x = "item.get_group().get_local_range(2)"
    self.block_dim_y = "item.get_group().get_local_range(1)"
    self.block_dim_z = "item.get_group().get_local_range(0)"
    self.grid_dim_x = "item.get_group_range(2)"
    self.stream_type = "sycl::queue"
    self.restrict_kw = "__restrict__"

    # Which *lowering* the kernel body uses, not which hardware it runs on.
    #
    # A request the caller makes rather than something derived from the
    # target -- `intel and oneapi`, say: a derivation would make selecting a
    # target select a programming model as well.
    self._explicit_simd = explicit_simd
    # The sub-group sizes the kernel states, or None where the device picks
    # its own (`Target.pinned_sub_groups`).
    self._sub_groups = sub_groups

  def get_launch_size(self, func_name, block, shmem, resident=False):
    # Declares `gridsize`, which the grid right after this reads:
    # `std::min(gridsize, ...)`.  One work-group per compute unit, which is
    # what the CUDA/HIP launchers fall back to when their occupancy query
    # answers nothing; SYCL has no such query.  The queue is read from the
    # pointer here because the launcher binds `stream` only after this, and
    # not dereferenced when null -- the null check that follows is what
    # reports that.
    ptr = GeneralLexicon.STREAM_PTR_STR
    return (f"static std::size_t gridsize = 0;\n"
            f"if (gridsize == 0 && {ptr} != nullptr) {{\n"
            f"  gridsize = static_cast<{self.stream_type} *>({ptr})->get_device()"
            f".get_info<sycl::info::device::max_compute_units>();\n"
            f"}}")

  def set_shmem_size(self, func_name, shmem):
    return ''

  def get_launch_code(self, func_name, grid, block, stream, func_params, shmem, coop):
    return f"{func_name}({stream}, {grid}, {block}, {func_params})"

  def declare_shared_memory(self, name, precision, size=None):
    """Nothing under SPMD; the reserved SLM chunk under the explicit vector.

    A `local_accessor` is what SYCL offers and what the SPMD kernel takes, and
    it is declared in the handler rather than the kernel -- so there is no
    statement to return here at all.

    ESIMD cannot use it as one.  Local memory there is reached through the
    `slm_` accessors, which take a byte offset into a chunk reserved by
    `slm_init`, and an accessor would have to be carried to every access site
    as a second operand for no gain: the offsets are the same numbers either
    way.  So the arena becomes the chunk, and its base is offset zero.

    `slm_init` wants the size as a template argument, which is why this takes
    one.  The generator has it: the arena's declaration is built once the
    body's layout has fixed the launch (`Generator._with_arena`).
    """
    if not self._explicit_simd:
      return ""
    if size is None:
      raise ValueError('the explicit-vector arena is `slm_init<Bytes>()` and '
                       'needs its size at the declaration; the caller has it '
                       'and has to pass it')
    # The base only: `slm_init` is once per kernel, and a kernel of several
    # sections declares its arena once per section -- IGC refuses the second
    # call ("slm_init is called more than once").  `kernel_definition`
    # reserves the largest section's size at the top of the kernel instead.
    return f'tensorforge::SlmPtr<{precision}> {name} = tensorforge::SlmPtr<{precision}>(0)'

  def exchange_xor(self, variable, mask):
    """Under SPMD, `permute_group_by_xor` over the sub-group.  A mask below
    the multiplication's width keeps a lane inside its own multiplication,
    since `Target.exchange_reach` places one where its size divides the
    sub-group --
    so several multiplications in one sub-group each exchange among their own
    lanes, which `reduce_over_group` over the whole sub-group would not.
    Under ESIMD the lanes are elements of one vector and `reduction` answers
    directly."""
    if self._explicit_simd:
      return None
    return (f'sycl::permute_group_by_xor(item.get_sub_group(), {variable}, '
            f'{mask})')

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None, lanes=None):
    if self._explicit_simd:
      # The arena is reserved inside the kernel instead; see
      # `declare_shared_memory`.  Declaring an accessor as well would reserve
      # the space twice -- once by the accessor's range and once by
      # `slm_init` -- and the second is the one the accesses address.
      localmem = None
    elif total_shared_mem_size is not None and precision is not None:
      if self._backend == 'acpp':
        localmem = f'sycl::accessor<{precision}, 1, sycl::access::mode::read_write, sycl::access::target::local>'
      else:
        localmem = f'sycl::local_accessor<{precision}, 1>'

      localmem += f' {GeneralLexicon.TOTAL_SHR_MEM} ({total_shared_mem_size}, cgh);'
    else:
      localmem = None

    props = ''
    if self._underlying_hardware == 'intel' and self._backend == 'oneapi':
      if self._explicit_simd:
        add_items = '[[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]]'
        # The large register file as a kernel property, not as the attribute
        # `[[intel::grf_size(256)]]`, which DPC++ 2026 does not know: it warns
        # "unknown attribute ignored" and compiles for the small file, so a
        # kernel sized for 256 registers gets 128.
        props = ('sycl::ext::oneapi::experimental::properties{'
                 'sycl::ext::intel::experimental::grf_size<256>}, ')
      else:
        # The sub-group follows the multiplication.  The lane search puts 32
        # lanes on a multiplication -- the ceiling is deliberately not the
        # 16-wide vector unit (`lanes.deduce`) -- so at a fixed 16 a
        # multiplication would span two sub-groups, and a broadcast addressed
        # within one would read undefined lanes: on the CPU the kernel writes
        # nothing, or 1e27 (chain_five).  A kernel of no multiplication asks
        # for 16.
        sizes = self._sub_groups
        size = (smallest_sub_group(sizes, lanes) or
                next((s for s in sizes if s >= lanes),
                     sizes[-1])) if lanes else 16
        add_items = (f'[[intel::reqd_sub_group_size({size})]] '
                     f'[[intel::kernel_args_restrict]]')
    else:
      add_items = ''

    l1 = f"inline void kernel_{base_name}(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, {params})"
    l2 = "stream->submit([&](sycl::handler &cgh)"
    # `group_count` and `group_size` come in CUDA order (x, y, z) and go out in
    # SYCL order, x last; see the index spellings in `__init__`.
    l3 = (f"cgh.parallel_for(sycl::nd_range<3>{{{{group_count.get(2) * group_size.get(2), "
          f"group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}}, "
          f"{{group_size.get(2), group_size.get(1), group_size.get(0)}}}}, "
          f"{props}[=](sycl::nd_item<3> item) {add_items}")

    if self._explicit_simd and total_shared_mem_size and precision is not None:
      # Once, before any section binds a window into it; see
      # `declare_shared_memory`.  `total_shared_mem_size` is the largest
      # section's arena, which is what every section addresses.
      reserve = (f'tensorforge::slmReserve<{total_shared_mem_size} * '
                 f'sizeof({precision})>();')
      return MultiBlock(file, [l1, l2, l3, reserve], ["", ");", ");", ""])
    if localmem is None:
      return MultiBlock(file, [l1, l2, l3], ["", ");", ");"])
    else:
      return MultiBlock(file, [l1, l2, localmem, l3], ["", ");", "", ");"])

  def sync_block(self):
    return "item.barrier();"

  def sync_simd(self):
    if self._explicit_simd:
      # One work-item *is* the vector: there are no lanes to synchronize.
      return None
    # A sub-group barrier, not a work-group one.
    #
    # `SyncThreads.barrier_scope()` reports `SIMD` whenever the thread count
    # fits in a wave, and `verify()` admits the instruction on that basis --
    # a `BatchLoop` is only SIMD-uniform, so a GROUP barrier inside it is
    # rejected as a deadlock.  Emitting `item.barrier()` here would make the
    # code do exactly what the check has just forbidden: the scope would say
    # SIMD and the instruction would be work-group wide.  On CUDA and HIP the
    # two agree (`__syncwarp`, `s_waitcnt`), and only on SYCL is the wave
    # narrow enough (16 on PVC) for ordinary operator shapes to reach it.
    return "sycl::group_barrier(item.get_sub_group());"

  def sync_mult(self, num_threads: int, hw):
    if self._explicit_simd:
      return None
    if (self._sub_groups is not None
        and smallest_sub_group(self._sub_groups, num_threads) == num_threads):
      # The stated sub-group is this multiplication, so its barrier is exact.
      return 'sycl::group_barrier(item.get_sub_group());'
    return self.sync_block()

  def sync_grid(self):
    raise NotImplementedError() # TODO
    #return "item.barrier();"

  def active_sub_group_mask(self):
    return 'item.get_sub_group()'

  def broadcast(self, variable, lane, block=None, subblock=1):
    if self._explicit_simd:
      return f'{variable}.select<{block}, {subblock}>({lane})'
    else:
      # `lane` counts within the multiplication, and `group_broadcast` takes
      # one index for the whole sub-group.  The two agree only where the
      # sub-group *is* the multiplication, which is known where the kernel
      # states its size.  Everywhere else a sub-group may hold several
      # multiplications -- 16 lanes in a sub-group of 32, or of a size the
      # device picks (acpp, a plug-in) -- and each has to read its own lanes,
      # at its own base: an index per work-item, `select_from_group`.
      group = 'item.get_sub_group()'
      if block is None:
        return f'sycl::group_broadcast({group}, {variable}, {lane})'
      if self._sub_groups is not None:
        size = smallest_sub_group(self._sub_groups, block)
        if size is None:
          # `block` is the *image's* lane block, so this says the image is
          # spread over more lanes than a sub-group holds and no group can
          # read it whole.  The way across is not a wider broadcast -- there
          # is none -- but a replicated image: spread over a sub-group and
          # copied into each (`Temporaries._lead_axes`), which lands in the
          # `size == block` case below and needs nothing here.  So this stays
          # a refusal, and it names what to do instead.
          from tensorforge.common.exceptions import GenerationError
          raise GenerationError(
              f'a register image spread over {block} lanes lies in no '
              f'sub-group of {" or ".join(map(str, self._sub_groups))}, '
              f'so a broadcast cannot read it within one; stage it over a '
              f'sub-group and replicate it instead')
        if size == block:
          return f'sycl::group_broadcast({group}, {variable}, {lane})'
      base = f'({group}.get_local_linear_id() / {block}) * {block}'
      return f'sycl::select_from_group({group}, {variable}, {base} + ({lane}))'

  def kernel_range_object(self, name, values):
    return f"sycl::range<3> {name} ({values})"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    with file.If(f"{pointer_name} == nullptr"):
      file.Expression("throw std::invalid_argument(\"stream may not be null!\")")

    stream_obj = f'static_cast<{self.stream_type} *>({pointer_name})'
    file(f'{self.stream_type} *stream = {stream_obj};')

  def get_headers(self):
    # The explicit-SIMD lowering spells its body through `tensorforge::
    # intel_esimd`, which `isycl.h` defines; without it a compile stops at the
    # first vector.  Only there: the header includes the ESIMD extension,
    # which AdaptiveCpp does not have.
    if self._explicit_simd:
      return ['sycl/sycl.hpp', 'tensorforge_device/isycl.h']
    return ['sycl/sycl.hpp']

  def atomic_store(self, ctx, access, variable, op, datatype, length=1):
    """A relaxed, device-scope `atomic_ref` over the destination element.

    Relaxed because an add carries no ordering the accumulation depends on,
    and device rather than system scope for the same reason it is agent scope
    on AMD: system scope is what makes the backend give up on the instruction.

    Worth checking against the generated SPIR-V rather than against results
    alone the first time this runs: `fetch_add` on a floating-point type is
    expanded into a compare-and-swap loop where the backend is not told the
    native instruction may be used, and a CAS loop is correct -- it just
    undoes the entire reason for choosing an atomic.
    """
    return (f'sycl::atomic_ref<{datatype}, sycl::memory_order::relaxed, '
            f'sycl::memory_scope::device, '
            f'sycl::access::address_space::global_space>'
            f'({access}).fetch_add({variable});')

  def prefetch_runs(self, addresses, byte_counts, level='l2'):
    # A gather with a lane per line of each run (`prefetchRunsHinted` in
    # `isycl.h`); SPMD has no instruction that names several addresses.
    if not self._explicit_simd:
      return None
    fn = 'prefetchRunsL1' if str(level).lower() == 'l1' else 'prefetchRunsL2'
    lengths = ', '.join(str(int(b)) for b in byte_counts)
    return f'tensorforge::{fn}<{lengths}>({", ".join(addresses)});'

  def prefetch(self, address, *, datatype, elems=1, level='l2'):
    """Two spellings, and only one of them can carry a level.

    Under ESIMD the cache hints are mandatory -- a prefetch with an empty
    property list does not compile -- so the level maps onto a pair of them
    and `tensorforge::prefetchL1`/`prefetchL2` in `isycl.h` hold which pairs
    are legal. `elems` is not passed: the block form takes its extent as a
    template argument and one element is what a pointer chase wants.

    Under SPMD the level has nowhere to go. `sycl_ext_oneapi_prefetch` does
    carry one -- `cache_level::L1` and up, with a cooperative form besides --
    but it sits behind `SYCL_EXT_ONEAPI_PREFETCH`, and a preprocessor test is
    not something an expression can hold. It belongs beside the other backend
    helpers on the day the extension is depended on.

    `address_space_cast` rather than a `multi_ptr` built from the pointer: the
    raw-pointer constructor exists only for the deprecated `decorated::legacy`
    spelling, and tying generated kernels to an interface both implementations
    are moving off is not worth the shorter line.
    """
    if self._explicit_simd:
      fn = 'prefetchL1' if str(level).lower() == 'l1' else 'prefetchL2'
      if elems > 1:
        # The block form: `elems` elements from one address, one message.
        return f'tensorforge::{fn}<{int(elems)}>({address});'
      return f'tensorforge::{fn}({address});'
    return (f'sycl::address_space_cast<sycl::access::address_space::'
            f'global_space, sycl::access::decorated::no>({address})'
            f'.prefetch({elems});')

  def get_fptype(self, fptype, length=1, relaxed=False):
    # `sycl::vec` carries its own alignment and there is no relaxed spelling
    # for it, so the flag is accepted and ignored rather than silently
    # changing the type.  A relaxed access on this backend is therefore only
    # as legal as `sycl::vec` makes it, which is why `widths_for` has to keep
    # answering from a base that proves what it needs.
    return f'sycl::vec<{fptype}, {length}>'

  def pointer_type(self, elem, space=None, readonly=False, restrict=False,
                   const=False, depth=1):
    """`SlmPtr<T>` for shared memory under the explicit vector, else generic.

    The address space is in the type here for the same reason it is on AMD --
    it is a different space, and a pointer that forgets which one it is in
    reads the wrong memory -- but the shape of the answer differs. On AMD the
    space is an attribute on the pointee and the value is still an address;
    ESIMD has no address into SLM at all, so `SlmPtr` is an offset wearing
    enough of a pointer's interface (`+`, `[]`) for the generator's own
    address arithmetic to go through unchanged.

    `restrict` is dropped rather than fused: it is a promise about aliasing
    between pointers, and there are none here -- two `SlmPtr` are two integers
    and the accesses they name are ordinary SLM messages the compiler already
    orders. Saying `__restrict__` about a class type is not a weaker promise,
    it does not parse.
    """
    if (self._explicit_simd and getattr(space, 'name', None) == 'SHARED'
        and depth == 1):
      ro = 'const ' if readonly else ''
      tail = ' const' if const else ''
      return f'tensorforge::SlmPtr<{ro}{elem}>{tail}'
    return super().pointer_type(elem, space, readonly, restrict, const,
                                depth)

  def shared_pointer_type(self, elem, restrict=False):
    if self._explicit_simd:
      # `pointer_type` already answers for this space; the two must not drift.
      from tensorforge.backend.pir.core import MemSpace
      return self.pointer_type(elem, MemSpace.SHARED, restrict=restrict)
    return super().shared_pointer_type(elem, restrict)

  def shared_window_expr(self, arena, offset):
    if self._explicit_simd:
      # An offset into an offset.  `&arena[off]` would be the address of the
      # proxy `operator[]` returns, which is a temporary.
      return f'{arena} + ({offset})'
    return super().shared_window_expr(arena, offset)

  def shared_window_retype(self, window, elem):
    if self._explicit_simd:
      # An offset counts elements of its own type, so the same byte address
      # is another count; `slmCast` converts through the bytes.
      return f'tensorforge::slmCast<{elem}>({window})'
    return super().shared_window_retype(window, elem)

  def get_slm_load(self, elem, width, address):
    if not self._explicit_simd:
      return None
    return f'tensorforge::slmLoad<{elem}, {width}>({address})'

  def get_slm_store(self, elem, width, address, value):
    if not self._explicit_simd:
      return None
    return f'tensorforge::slmStore<{elem}, {width}>({address}, {value});'

  def get_simd(self, fptype, size):
    return f'tensorforge::intel_esimd::simd<{fptype}, {size}>'

  #: The ESIMD math intrinsics, by arity.
  #:
  #: Read off `sycl/ext/intel/esimd/math.hpp` rather than assumed from the
  #: `sycl::` names: the two namespaces do not have the same functions, and a
  #: `sycl::tanh` applied to a `simd<>` is not a slower tanh, it is a
  #: compile error -- or, where a conversion exists, one element broadcast.
  #: What is absent here is absent in the hardware library, so it is declined
  #: rather than substituted.
  _ESIMD_UNARY = {
    Operation.ABS: 'abs', Operation.SQRT: 'sqrt', Operation.RSQRT: 'rsqrt',
    Operation.EXP: 'exp', Operation.LOG: 'log',
    Operation.SIN: 'sin', Operation.COS: 'cos',
    Operation.RCP: 'inv', Operation.TRUNC: 'trunc',
  }
  _ESIMD_BINARY = {
    Operation.MIN: 'min', Operation.MAX: 'max', Operation.POW: 'pow',
  }
  #: What the library takes in float and half only, spelled for double.
  #: `exp` is composed (`tensorforge::expF64`) and a reciprocal is a division;
  #: handed a `simd<double, N>`, `intel_esimd::exp` and `inv` are compile
  #: errors (every F64 `damageStep`).  `tanh` is composed from `expF64` too.
  _ESIMD_F64 = {
    Operation.EXP: 'tensorforge::expF64({})', Operation.RCP: '(1.0 / {})',
    Operation.TANH: 'tensorforge::tanhF64({})',
  }
  #: What only the experimental ESIMD math has, float only, through a helper
  #: in `isycl.h` that takes a view as well (`tanhF32`).  A scalar
  #: `sycl::tanh` is refused outright in ESIMD code ("not supported in ESIMD
  #: context", oneAPI 2025.0).
  _ESIMD_F32 = {
    Operation.TANH: 'tensorforge::tanhF32({})',
  }
  _ESIMD_COMPARISONS = frozenset((Operation.LT, Operation.LE, Operation.GT,
                                  Operation.GE, Operation.EQ, Operation.NEQ))

  def _esimd_operation(self, op: Operation, fptype, value1, value2):
    ns = 'tensorforge::intel_esimd'
    if op == Operation.COPY:
      return value1
    if op == Operation.NEG:
      return f'(-{value1})'
    if fptype == Datatype.F64 and op in self._ESIMD_F64:
      return self._ESIMD_F64[op].format(value1)
    if fptype == Datatype.F32 and op in self._ESIMD_F32:
      return self._ESIMD_F32[op].format(value1)
    if op in self._ESIMD_UNARY:
      return f'{ns}::{self._ESIMD_UNARY[op]}({value1})'
    if op in self._ESIMD_BINARY:
      return f'{ns}::{self._ESIMD_BINARY[op]}({value1}, {value2})'
    if op in self._ESIMD_COMPARISONS and fptype != Datatype.BOOL:
      # A comparison of vectors is a `simd_mask`; asked for as a number --
      # the value a boolean tensor's register image holds, which is the
      # kernel's floating-point type -- it is 1 and 0 of that type.  The
      # mask has no `copy_to` into one (every SeisSol `damageStep`).
      return (f'tensorforge::asNumber<{fptype.ctype()}>('
              f'{value1} {INFIX[op]} {value2})')
    # The operators, which `simd<>` overloads.
    if op in INFIX:
      return f'({value1} {INFIX[op]} {value2})'
    # A logical operation typed as a floating-point number -- the value a
    # boolean tensor's register image holds -- takes its operands as masks,
    # whatever they arrive as, and gives 1 and 0 of that type.  `~` and `&`
    # are bitwise on a float, which is not an operation at all, and on masks
    # they give a mask that has no `copy_to` into the image.
    logical = fptype != Datatype.BOOL and fptype.ctype() in ('float', 'double')
    if op == Operation.NOT:
      if logical:
        return (f'tensorforge::asNumber<{fptype.ctype()}>('
                f'!tensorforge::asMask({value1}))')
      return f'(!{value1})' if fptype == Datatype.BOOL else f'(~{value1})'
    if op in (Operation.AND, Operation.OR):
      sym = {Operation.AND: '&', Operation.OR: '|'}[op]
      if logical:
        return (f'tensorforge::asNumber<{fptype.ctype()}>('
                f'tensorforge::asMask({value1}) {sym} '
                f'tensorforge::asMask({value2}))')
      if fptype == Datatype.BOOL:
        sym *= 2
      return f'({value1} {sym} {value2})'
    raise NotImplementedError(
      f'{op} has no ESIMD intrinsic. `sycl::{op.name.lower()}` is not a '
      f'substitute -- it does not accept a simd<> operand, and where a '
      f'conversion exists it would silently compute on one element. '
      f'Composing it from the intrinsics that do exist is a numerics '
      f'decision, not a spelling one.')

  #: C++ `tensorforge::Operation` members, by the `Operation` they lower from.
  #: The same table `CudaLexic` has, because it names the same device-side
  #: `ReductionOperation` specializations -- `base.h` defines them once for
  #: every backend.
  REDUCTION_OPS = {
      Operation.ADD: 'Add',
      Operation.MUL: 'Mul',
      Operation.MIN: 'Min',
      Operation.MAX: 'Max',
      Operation.AND: 'And',
      Operation.OR: 'Or',
      Operation.XOR: 'Xor',
  }

  #: The ESIMD spelling of each all-reduce, by the `Operation` it lowers from.
  #:
  #: `reduce` covers only `std::plus` and `std::multiplies` -- its other
  #: branches fall through to nothing -- and min/max have their own entry
  #: points.  Bitwise reductions have neither, which is why they are absent
  #: rather than spelled optimistically.
  _ESIMD_REDUCE = {
    Operation.ADD: 'reduce<{t}>({v}, std::plus<>())',
    Operation.MUL: 'reduce<{t}>({v}, std::multiplies<>())',
    Operation.MAX: 'hmax<{t}>({v})',
    Operation.MIN: 'hmin<{t}>({v})',
  }

  def reduction(self, variable, optype, fptype, block, subblock=1):
    """An all-reduce across `block` lanes, in groups of `subblock`.

    Under an explicit vector this is not a cross-lane construct at all: the
    lanes are elements of one work-item's register, so the reduction is an
    operation on a `simd<T, N>` and the "all" part is a broadcast back.

    Two shapes, and they differ in more than width.  `subblock == 1`
    collapses the lane axis and the intrinsics answer it in one call, giving a
    *scalar*.  `subblock > 1` keeps a group, so the answer is a vector and the
    butterfly has to be spelled out -- `segmentedReduction` in `isycl.h` does
    that with the two-dimensional region `select<Block/(2i), 2i, i, 1>`, which
    is exactly the pairing `shfl_xor(x, i)` performs.

    Nothing in the generator asks for the second shape today: `reduction.py`
    passes `subblock=1` at its only call site, and no descriptor produces a
    lead axis that carries two tensor dimensions.  It is implemented anyway
    because the primitive is what the *lexic* promises, and a promise with a
    hole in it is found by whoever first needs it.
    """
    if block == subblock:
      # Nothing to combine: each group is one lane wide already.
      return variable
    if not self._explicit_simd:
      raise NotImplementedError(
          f'{type(self).__name__} has no SPMD cross-lane reduction. '
          f'`sycl::reduce_over_group(item.get_sub_group(), ...)` answers this '
          f'for subblock == 1 and block == the sub-group size, but the lexic '
          f'cannot see the sub-group size to check the second condition, and '
          f'a reduction over the wrong width is wrong quietly.')
    if subblock != 1:
      # A group survives, so this is not a collapse and the intrinsics do not
      # answer it: `reduce` and `hmax` return one value for the whole vector.
      # `segmentedReduction` in `isycl.h` is the butterfly, and its result is
      # a vector -- unlike the collapse below, which yields a scalar.
      if optype not in self.REDUCTION_OPS:
        raise NotImplementedError(f'reduction over {optype}')
      ctype = fptype.ctype()
      op = (f'tensorforge::ReductionOperation<{ctype}, '
            f'tensorforge::Operation::{self.REDUCTION_OPS[optype]}>')
      return (f'tensorforge::segmentedReduction<{op}, {block}, {subblock}, '
              f'{ctype}>({variable})')
    if optype not in self._ESIMD_REDUCE:
      raise NotImplementedError(
          f'reduction over {optype} has no ESIMD entry point')
    ctype = fptype.ctype()
    call = self._ESIMD_REDUCE[optype].format(t=ctype, v=variable)
    # A scalar, not broadcast back.
    #
    # In SPMD the two are the same statement -- an all-reduce leaves every
    # thread holding a copy, and "the result" is that copy.  Here they are
    # not: the reduction *collapses* the lane axis, so its result is one
    # value, and a caller that stores it stores one element.  Broadcasting it
    # back into a vector would produce `glb_m1[k] = simd<float, 16>(...)`, a
    # sixteen-wide value assigned to a scalar destination.
    #
    # A caller that does want it in every lane spells that itself, and
    # `simd<T, N>(scalar)` is what it spells.
    return f'tensorforge::intel_esimd::{call}'

  def get_simd_mask(self, size):
    """`simd_mask<N>`: the type a comparison over a `simd<T, N>` produces.

    Its own family, not `simd<bool, N>` -- the hardware keeps masks in mask
    registers and the API follows, so a predicated operation takes one of
    these and nothing else converts to it.
    """
    return f'tensorforge::intel_esimd::simd_mask<{size}>'

  #: SYCL's math functions, under SPMD (`get_operation`).
  MATH = {
    Operation.ABS: 'sycl::fabs({0})',
    Operation.MIN: 'sycl::min({t}({0}), {t}({1}))',
    Operation.MAX: 'sycl::max({t}({0}), {t}({1}))',
    Operation.POW: 'sycl::pow({0}, {1})',
    **{op: f'sycl::{name}({{0}})' for op, name in (
      (Operation.EXP, 'exp'), (Operation.LOG, 'log'),
      (Operation.EXPM1, 'expm1'), (Operation.LOG1P, 'log1p'),
      (Operation.SQRT, 'sqrt'), (Operation.CBRT, 'cbrt'),
      (Operation.RSQRT, 'rsqrt'),
      (Operation.SIN, 'sin'), (Operation.COS, 'cos'), (Operation.TAN, 'tan'),
      (Operation.ASIN, 'asin'), (Operation.ACOS, 'acos'),
      (Operation.ATAN, 'atan'), (Operation.SINH, 'sinh'),
      (Operation.COSH, 'cosh'), (Operation.TANH, 'tanh'),
      (Operation.ASINH, 'asinh'), (Operation.ACOSH, 'acosh'),
      (Operation.ATANH, 'atanh'))},
  }

  def get_operation(self, op: Operation, fptype, value1, value2):
    if self._explicit_simd:
      return self._esimd_operation(op, fptype, value1, value2)
    return super().get_operation(op, fptype, value1, value2)
