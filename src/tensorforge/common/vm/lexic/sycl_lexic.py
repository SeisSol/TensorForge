# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.basic_types import GeneralLexicon
from .lexic import Lexic, Operation
from tensorforge.backend.writer import MultiBlock
from tensorforge.common.basic_types import Datatype

class SyclLexic(Lexic):
  def __init__(self, backend, underlying_hardware, explicit_simd=False):
    super().__init__(underlying_hardware)
    self._backend = backend
    self.thread_idx_y = "item.get_local_id(1)"
    self.thread_idx_x = "item.get_local_id(0)"
    self.thread_idx_z = "item.get_local_id(2)"
    self.block_idx_x = "item.get_group().get_group_id(0)"
    self.block_idx_z = "item.get_group().get_group_id(2)"
    self.block_dim_y = "item.get_group().get_local_range(1)"
    self.block_dim_z = "item.get_group().get_local_range(2)"
    self.grid_dim_x = "item.get_global_range(0)"
    self.stream_type = "sycl::queue"
    self.restrict_kw = "__restrict__"

    # Which *lowering* the kernel body uses, not which hardware it runs on.
    #
    # This used to be derived -- `intel and oneapi` -- and that derivation was
    # the defect: selecting a target implied selecting a programming model,
    # and the model it selected had no emitter behind it.  What it did have
    # was a set of branches in `symbol.py` that returned early instead of
    # emitting, so an Intel target silently produced a kernel with the
    # arithmetic missing.
    #
    # Now it is a request the caller makes, and the only thing it still
    # governs is the *spelling* the lexic hands out: the kernel attributes,
    # the broadcast, and the wave-level barrier.  Nothing outside this file
    # asks about it, which is the property that has to hold until an ESIMD
    # emitter exists to answer for the body as well.
    self.simd_mode = explicit_simd

  def multifile(self):
    return False

  def get_launch_size(self, func_name, block, shmem):
    # `shmem` was missing here while `generator.py` has passed three arguments
    # for as long as the persistent-launch path has existed, so every SYCL
    # target that reaches it died with a TypeError before emitting a line --
    # which is also why nothing noticed: the path is only taken for some
    # arch/occupancy combinations, and no SYCL target was in the snapshot
    # corpus to take it.
    return f"""""" # TODO: occupancy query via device info

  def set_shmem_size(self, func_name, shmem):
    return ''

  def get_launch_code(self, func_name, grid, block, stream, func_params, shmem, coop):
    return f"{func_name}({stream}, {grid}, {block}, {func_params})"

  def declare_shared_memory(self, name, precision):
    return ""

  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None, global_symbols=None):
    if total_shared_mem_size is not None and precision is not None:
      if self._backend == 'acpp':
        localmem = f'sycl::accessor<{precision}, 1, sycl::access::mode::read_write, sycl::access::target::local>'
      else:
        localmem = f'sycl::local_accessor<{precision}, 1>'

      localmem += f' {GeneralLexicon.TOTAL_SHR_MEM} ({total_shared_mem_size}, cgh);'
    else:
      localmem = None

    if self._underlying_hardware == 'intel' and self._backend == 'oneapi':
      if self.simd_mode:
        add_items = '[[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]]'
      else:
        add_items = '[[intel::reqd_sub_group_size(16)]] [[intel::kernel_args_restrict]]'
    else:
      add_items = ''

    l1 = f"inline void kernel_{base_name}(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, {params})"
    l2 = f"stream->submit([&](sycl::handler &cgh)"
    l3 = f"cgh.parallel_for(sycl::nd_range<3>{{{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}}, group_size}}, [=](sycl::nd_item<3> item) {add_items}"

    if localmem is None:
      return MultiBlock(file, [l1, l2, l3], ["", ");", ");"])
    else:
      return MultiBlock(file, [l1, l2, localmem, l3], ["", ");", "", ");"])

  def sync_block(self):
    return "item.barrier();"

  def sync_simd(self):
    if self.simd_mode:
      # One work-item *is* the vector: there are no lanes to synchronise.
      return None
    # A sub-group barrier, not a work-group one.
    #
    # `SyncThreads.barrier_scope()` reports `SIMD` whenever the thread count
    # fits in a wave, and `verify()` admits the instruction on that basis --
    # a `BatchLoop` is only SIMD-uniform, so a GROUP barrier inside it is
    # rejected as a deadlock.  Emitting `item.barrier()` here made the code
    # do exactly what the check had just forbidden: the scope said SIMD and
    # the instruction was work-group wide.  On CUDA and HIP the two agree
    # (`__syncwarp`, `s_waitcnt`); only SYCL had the claim and the code
    # disagreeing, and only on SYCL is the wave narrow enough (16 on PVC) for
    # ordinary operator shapes to reach it.
    return "sycl::group_barrier(item.get_sub_group());"

  def has_sync_mult(self, num_threads: int, hw) -> bool:
    """True under an explicit vector, False under SPMD, and the asymmetry is
    the point.

    Under ESIMD one work-item *is* the vector: a multiplication of any width is
    held by a single work-item, executed in order, with no second party to wait
    for.  So the rendezvous costs nothing and is exact.

    Under SPMD there is no spelling.  `sycl::group_barrier` takes a group
    object and the narrowest one available is the sub-group, which is all of
    it -- there is no mask, so a multiplication occupying part of a sub-group
    cannot be met on its own.  The named barriers Xe has are reachable from
    ESIMD (`named_barrier_signal` / `named_barrier_wait`) and exactly there
    they are not needed; from SPMD, where they would be, they are not exposed.
    So a narrow multiplication is met at its group instead, and the block is
    sized to hold one.
    """
    return bool(self.simd_mode)

  def sync_mult(self, num_threads: int, hw):
    if self.simd_mode:
      return None
    return self.sync_block()

  def sync_grid(self):
    raise NotImplementedError() # TODO
    #return "item.barrier();"

  def get_sub_group_id(self, sub_group_size):
    return f'item.get_sub_group().get_local_id()[0]'

  def active_sub_group_mask(self):
    return f'item.get_sub_group()'

  def broadcast(self, variable, lane, block=None, subblock=1):
    if self.simd_mode:
      return f'{variable}.select<{block}, {subblock}>({lane})'
    else:
      # `group_broadcast(-1, ...)` before: an unqualified name and `-1` where
      # a group object belongs.  A placeholder nothing had ever reached, which
      # is how it survived -- the Intel register path is the first caller.
      return f'sycl::group_broadcast(item.get_sub_group(), {variable}, {lane})'

  def kernel_range_object(self, name, values):
    return f"sycl::range<3> {name} ({values})"

  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    with file.If(f"{pointer_name} == nullptr"):
      file.Expression("throw std::invalid_argument(\"stream may not be null!\")")

    stream_obj = f'static_cast<{self.stream_type} *>({pointer_name})'
    file(f'{self.stream_type} *stream = {stream_obj};')

  def check_error(self):
    return None

  def get_headers(self):
    return ['sycl/sycl.hpp']

  def has_atomic_store(self, ctx, op, datatype, length=1):
    """No under the explicit-SIMD lowering, whatever the hardware can do.

    `atomic_ref` binds one reference to one element, and under ESIMD the value
    a store carries is a `simd<T, N>` with a mask beside it -- there is no
    scalar to bind.  The instruction is `esimd::atomic_update<atomic_op::fadd>`,
    which takes the vector and the `simd_mask` together and updates the whole
    of it with atomicity per element.  That is a better fit for what the store
    path wants than the SPMD spelling is, and it is a different emitter; until
    it exists, refusing is what keeps an ESIMD kernel from being handed an
    `atomic_ref<simd<float, 16>>` that does not compile.
    """
    return (not self.simd_mode
            and super().has_atomic_store(ctx, op, datatype, length))

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

  #: Intel parts whose LSC has the prefetch the ESIMD API lowers to.
  #:
  #: A list because the API's own documentation is one -- "DG2, PVC only" --
  #: and there is no feature query to derive it from. An earlier Xe part
  #: compiles the call and has no instruction under it.
  _ESIMD_PREFETCH_ARCHS = frozenset({'pvc', 'dg2'})

  def has_prefetch(self, hw):
    """Core SYCL 2020 under SPMD, and an LSC instruction under ESIMD.

    `multi_ptr::prefetch` is declared for every global-space pointer and an
    implementation with nothing behind it still has to accept the call, which
    is the right shape for a hint -- so the SPMD answer does not consult the
    hardware at all.

    The explicit-SIMD lowering is the opposite: `esimd::prefetch` is a thin
    wrapper over an LSC message that two Intel generations have, so the model
    decides. Both paths spell a single address; they differ in what has to be
    said about the cache, which `isycl.h` says once.
    """
    if self.simd_mode:
      return str(getattr(hw, 'model', '')) in self._ESIMD_PREFETCH_ARCHS
    return True

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
    if self.simd_mode:
      fn = 'prefetchL1' if str(level).lower() == 'l1' else 'prefetchL2'
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
  #: Spelled with C++ operators, which `simd<>` overloads.
  _ESIMD_INFIX = {
    Operation.ADD: '+', Operation.SUB: '-', Operation.MUL: '*',
    Operation.DIV: '/', Operation.XOR: '^',
    Operation.LT: '<', Operation.LE: '<=', Operation.GT: '>',
    Operation.GE: '>=', Operation.EQ: '==', Operation.NEQ: '!=',
  }

  def _esimd_operation(self, op: Operation, fptype, value1, value2):
    ns = 'tensorforge::intel_esimd'
    if op == Operation.COPY:
      return value1
    if op == Operation.NEG:
      return f'(-{value1})'
    if op in self._ESIMD_UNARY:
      return f'{ns}::{self._ESIMD_UNARY[op]}({value1})'
    if op in self._ESIMD_BINARY:
      return f'{ns}::{self._ESIMD_BINARY[op]}({value1}, {value2})'
    if op in self._ESIMD_INFIX:
      return f'({value1} {self._ESIMD_INFIX[op]} {value2})'
    if op == Operation.NOT:
      return f'(!{value1})' if fptype == Datatype.BOOL else f'(~{value1})'
    if op in (Operation.AND, Operation.OR):
      sym = {Operation.AND: '&', Operation.OR: '|'}[op]
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
  #: `ReductionOperation` specialisations -- `base.h` defines them once for
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
    if not self.simd_mode:
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
    # back into a vector produced `glb_m1[k] = simd<float, 16>(...)`, a
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

  def get_operation(self, op: Operation, fptype, value1, value2):
    if self.simd_mode:
      return self._esimd_operation(op, fptype, value1, value2)
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
    elif op == Operation.ABS:
      return f'sycl::fabs({value1})'
    elif op == Operation.MIN:
      return f'sycl::min({fptype}({value1}), {fptype}({value2}))'
    elif op == Operation.MAX:
      return f'sycl::max({fptype}({value1}), {fptype}({value2}))'
    elif op == Operation.POW:
      return f'sycl::pow({value1}, {value2})'
    elif op == Operation.ABS:
      return f'sycl::abs({value1})'
    elif op == Operation.NEG:
      return f'(-{value1})'
    elif op == Operation.EXP:
      return f'sycl::exp({value1})' # has __expf
    elif op == Operation.LOG:
      return f'sycl::log({value1})' # has __logf
    elif op == Operation.EXPM1:
      return f'sycl::expm1({value1})'
    elif op == Operation.LOGP1:
      return f'sycl::logp1({value1})'
    elif op == Operation.SQRT:
      return f'sycl::sqrt({value1})'
    elif op == Operation.CBRT:
      return f'sycl::cbrt({value1})'
    elif op == Operation.SIN:
      return f'sycl::sin({value1})' # has __sinf
    elif op == Operation.COS:
      return f'sycl::cos({value1})' # has __cosf
    elif op == Operation.TAN:
      return f'sycl::tan({value1})' # has __tanf
    elif op == Operation.ASIN:
      return f'sycl::asin({value1})'
    elif op == Operation.ACOS:
      return f'sycl::acos({value1})'
    elif op == Operation.ATAN:
      return f'sycl::atan({value1})'
    elif op == Operation.SINH:
      return f'sycl::sinh({value1})' # has __sinf
    elif op == Operation.COSH:
      return f'sycl::cosh({value1})' # has __cosf
    elif op == Operation.TANH:
      return f'sycl::tanh({value1})' # has __tanf
    elif op == Operation.ASINH:
      return f'sycl::asinh({value1})'
    elif op == Operation.ACOSH:
      return f'sycl::acosh({value1})'
    elif op == Operation.ATANH:
      return f'sycl::atanh({value1})'
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
