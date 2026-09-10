# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from enum import Enum
from tensorforge.common.operation import Operation

class Lexic(ABC):
  """
  You can use this abstract class to add a dictionary for any backend for variables like e.g.
  threadIdx.x for CUDA that are used by the generators and loaders
  """

  def __init__(self, underlying_hardware):
    self._underlying_hardware = underlying_hardware
    self.thread_idx_x = None
    self.thread_idx_y = None
    self.thread_idx_z = None
    self.block_dim_y = None
    self.block_dim_z = None
    self.block_idx_x = None
    self.stream_type = None
    self.restrict_kw = None
    self.simd_mode = False

  def storage_class(self, space) -> str:
    """How a declaration says where its object resides, where it has to.

    Empty for almost everything, and for two different reasons that are worth
    keeping apart.  Most spaces need no annotation on a declaration because the
    declaration's *form* already fixes them -- a kernel argument is a kernel
    argument.  `MemSpace.PARAM` is the one that does not: CUDA otherwise copies
    a by-value parameter into per-thread memory as soon as its address is taken
    or a runtime value indexes it, and `__grid_constant__` is how that copy is
    refused.  Backends whose kernel arguments already live in a broadcast space
    -- which is most of them -- need nothing and say so by returning nothing.

    A space and not a keyword the caller looks up, so that the one site asking
    the question asks it about residency rather than about a vendor.
    """
    return ''

  def pointer_type(self, elem: str, space=None, readonly: bool = False,
                   restrict: bool = False, const: bool = False) -> str:
    """How a pointer into `space` is spelled, for a declaration of one.

    Asked of the backend rather than assembled from `restrict_kw` at the call
    site, because on one of them the space and the restrict promise are not
    independently spellable: HIP puts the space in an attribute on the pointee
    and ships `SpacePtrRestrict` as the fused alias, so a caller concatenating
    two strings gets something that does not compile.  Both facts arrive here
    together and the backend decides.

    The default is the generic pointer, which is what every backend without
    address spaces in its type system wants and what the emitter spelled by
    hand before this existed.

    `readonly` is about the pointee and `const` about the pointer: a binding
    the pipelined form advances is `T *` and the ordinary one `T *const`, and
    both may point at something nothing writes.

    `space` is a `pir.MemSpace`, taken structurally so this module does not
    have to import the IR.
    """
    lhs = 'const ' if readonly else ''
    ptr = 'const' if const else ''
    qual = f' {self.restrict_kw}' if restrict and self.restrict_kw else ''
    return f'{lhs}{elem} *{ptr}{qual}'

  @abstractmethod
  def multifile(self):
    pass

  @abstractmethod
  def get_launch_code(self, func_name, grid, block, stream, func_params):
    pass

  @abstractmethod
  def set_shmem_size(self, func_name, shmem):
    pass

  @abstractmethod
  def declare_shared_memory(self, name, precision):
    pass

  @abstractmethod
  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None,
                        total_shared_mem_size=None, global_symbols=None):
    pass

  @abstractmethod
  def sync_block(self):
    pass

  @abstractmethod
  def sync_simd(self):
    pass

  def loop_body_fence(self) -> str:
    """A statement for the head of a loop body that loads cannot be hoisted
    across, or empty where the backend's compiler needs none.

    For the batch loop without a mask.  With one, the body sits under
    `if (allowed)`, and a load there runs only conditionally, so LICM may not
    speculate it into the preheader.  Without one the body runs every
    iteration and LICM is free to -- and it takes loads that are invariant by
    design, the operators staged into shared memory once per block, and holds
    them in registers across the whole loop.
    """
    return ''

  def has_sync_mult(self, num_threads: int, hw) -> bool:
    """Whether `sync_mult` rendezvouses fewer threads than the whole block.

    Asked before the thread-block policy sizes a block, because the answer
    decides how many multiplications may share one.  A target that says False
    gets one multiplication per block whenever a multiplication is wider than
    a wave: the block barrier is then the multiplication's own barrier, and
    the loop around it is block-uniform, so there is a legal spelling.  Say
    True without an implementation below and the policy packs several
    multiplications into a block that cannot separate them.

    `num_threads` is part of the question rather than decoration on it. Every
    sub-block rendezvous counts participants, and every one of them counts in
    a unit -- threads, waves, sub-groups -- that a width not divisible by the
    wave cannot express.  `hw` is passed rather than read off the lexic: the
    lexic is built from the vendor alone, while the answer turns on the
    architecture -- named barriers arrive at gfx12.5 and not at gfx12.
    """
    return False

  def sync_mult(self, num_threads: int, hw):
    """Rendezvous exactly the ``num_threads`` threads of one multiplication.

    Only reached where the multiplication is *wider* than a wave -- narrower
    than that and `SyncThreads` asks for `sync_simd` directly, because the
    threads are in lockstep anyway.  So the default is the honest one: a whole
    block, which over-synchronises but never deadlocks, and which is exact
    once `has_sync_mult` says False, because the policy then puts one
    multiplication in a block.

    The opportunity a vendor can take here is a sub-block rendezvous, which
    lets several wide multiplications share a block:

    * NVIDIA, `bar.sync id, count` -- meets exactly `count` threads at named
      barrier `id`.
    * AMD from gfx12.5, `s_barrier_init` / `s_barrier_join` on barrier objects
      1..16, whose expected count is in waves.
    * Intel under ESIMD, `named_barrier_signal` / `named_barrier_wait`, whose
      counts are in sub-groups.

    The catch is the same on all three: every multiplication in the block
    needs an `id` of its own, so it holds only while `mults_per_block` stays
    under the number of barrier resources.  Two multiplications sharing an
    `id` rendezvous with each other, which is a deadlock the moment they run
    the body a different number of times.
    """
    return self.sync_block()

  @abstractmethod
  def get_sub_group_id(self, sub_group_size):
    return None

  @abstractmethod
  def kernel_range_object(self, name, values):
    pass

  @abstractmethod
  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    pass

  @abstractmethod
  def check_error(self):
    pass

  @abstractmethod
  def get_headers(self):
    pass

  @abstractmethod
  def get_operation(self, op: Operation, value1, value2):
    pass

  def vector_fma(self, a, b, c):
    """`a * b + c` over the target's vector type, spelled as a call, or
    None where the infix form already says everything.

    It does for scalars everywhere and for GNU vectors, which contract it.
    A target whose vector type has no fused operation of its own -- CUDA's
    structs, whose paired FMA needs an intrinsic -- names its function here.
    """
    return None

  def reduction(self, variable, optype, fptype, block, subblock=1):
    """An all-reduce of `variable` across `block` lanes, in groups of
    `subblock`.

    Declared here so a backend that has no answer says so.  `CudaLexic`
    implements it and `HipLexic` inherits that; `SyclLexic` has `broadcast`
    but not this, so a cross-lane reduction reached it as
    `AttributeError: 'SyclLexic' object has no attribute 'reduction'` --- a
    missing attribute reads as a typo, and this is a missing feature.

    Not implemented for SYCL because the signature is the open question, not
    the body.  `sycl::reduce_over_group` takes a whole group and has no
    `subblock`, so it answers only for `subblock == 1` and
    `block == sub_group_size`; anything else is a hand-built exchange over
    `permute_group_by_xor`.  Under ESIMD the model is different again --- the
    vector is explicit and the reduction is an operation on `simd<T, N>`
    rather than a cross-lane construct.  Committing to a spelling before that
    is decided would fix the wrong shape in place.
    """
    raise NotImplementedError(
        f'{type(self).__name__} has no cross-lane reduction; see '
        f'Lexic.reduction')

  # --- global accesses ------------------------------------------------------
  # `datatype` is keyword-only and has no default, so a call site cannot omit
  # it and a positional `nontemporal` cannot land in it.  The hint is spelled
  # by an overload set on at least one target, which makes the type part of
  # the question rather than decoration on it: what an unanswered type buys is
  # not a plainer access but a kernel that does not compile.

  def has_nontemporal(self, datatype, length=1):
    """Whether this target spells a nontemporal access of `length` x `datatype`.

    The counterpart to `has_atomic_store`, asked for the same reason and
    answered on the same terms: what the caller needs to know is whether the
    hint *exists* for this type, not whether something could be written.  A
    target without one says False and the access is emitted plainly, which
    costs a cache policy and nothing else -- the hint is an optimisation over
    exactly that access.

    False here, because the base spelling below has no hint to give.
    """
    return False

  def glb_store(self, lhs, rhs, *, datatype, length=1, nontemporal=False):
    return f'{lhs} = {rhs};'

  def glb_load(self, rhs, *, datatype, length=1, nontemporal=False):
    return f'{rhs}'

  # --- atomic accumulation --------------------------------------------------
  # Declared here because the three implementations had drifted into three
  # different interfaces: `CudaLexic.atomic_store` took three arguments where
  # the one call site passes four, `SyclLexic` had neither method, and
  # `has_atomic_store` existed on `HipLexic` alone and returned True without
  # looking at anything.  Two of the three targets could not have reached this
  # path without a TypeError or an AttributeError, which is a fair description
  # of the state they were in.
  #
  # `ctx` is a parameter and not a field because the lexic is constructed with
  # the vendor alone (`vm.py` passes `descr.vendor`), while the answer here
  # turns on the architecture.  That is the whole of the bug this replaces:
  # a per-vendor answer to a per-architecture question.

  def has_atomic_store(self, ctx, op, datatype, length=1):
    """Whether an atomic update of `length` x `datatype` is one instruction.

    Not "can it be spelled": every target can spell it, and one that has no
    instruction gets a compare-and-swap loop -- slower than the
    read-modify-write the atomic was chosen to replace.  So the honest answer
    to give the placement policy is about the instruction, and a backend
    without one says False and is accumulated into normally.
    """
    from tensorforge.backend import atomics
    return op is None and atomics.native_add(ctx, datatype, length)

  def atomic_store(self, ctx, access, variable, op, datatype, length=1):
    """One atomic update, as a statement.

    Only reached when `has_atomic_store` agreed, so a backend overriding this
    does not have to answer for the cases that one turns away.
    """
    raise NotImplementedError(
        f'{type(self).__name__} has no atomic store; see Lexic.atomic_store')

  # --- asynchronous global -> shared copies --------------------------------
  # A backend without a hardware path returns None; the caller then emits a
  # synchronous fallback, so correctness never depends on these being present.
  # All three are *per thread*: every lane copies `nbytes` bytes.

  def copy_async_sizes(self):
    """Per-thread copy sizes in bytes the hardware path accepts."""
    return ()

  def copy_async(self, dst, src, nbytes):
    return None

  def commit_async(self):
    return None

  def wait_async(self, prior):
    """Wait until at most `prior` issued copies are still in flight."""
    return None

  def wait_async_regs(self, prior):
    """Same, for global -> register loads.

    Separate from `wait_async` because the two are not the same counter
    everywhere: AMD tracks both in `vmcnt`, while NVIDIA scoreboards register
    loads in hardware and needs no instruction at all (hence None).
    """
    return None

  # --- data prefetch --------------------------------------------------------
  # A hint and nothing else: it moves no value, releases no token, and the
  # emitter drops it on a target that has none.  So a backend with no answer
  # costs a cache policy and not a kernel, which is the same bargain
  # `has_nontemporal` makes.

  def has_prefetch(self, hw):
    """Whether this target reaches a data prefetch instruction at all.

    Takes the hardware descriptor for the reason `has_atomic_store` takes a
    context: the lexic is built from the vendor alone, and the answer here is
    not a per-vendor one.  On AMD it turns on the architecture -- there is no
    prefetch below gfx12 -- while a SYCL target answers from the library it
    compiles against and not from the part it runs on.

    False here, because the spelling below has nothing to give.
    """
    return False

  def prefetch(self, address, *, datatype, elems=1, level='l2'):
    """One prefetch, as a statement.  `address` is a pointer expression.

    `level` is `'l1'` or `'l2'`, and it is a request rather than an
    instruction: a target with one prefetch for both honours neither, and
    says so here instead of pretending the distinction survived.  `elems` is
    likewise honoured only where the instruction takes a count.

    Only reached when `has_prefetch` agreed, so an implementation does not
    have to answer for the targets that one turns away.
    """
    return None
