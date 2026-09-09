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
    #: How a by-value kernel parameter is kept out of per-thread memory when
    #: its address is taken or it is indexed by a runtime value.  Empty where
    #: the backend needs no annotation because its kernel arguments already
    #: live in a broadcast space -- which is most of them; CUDA is the one that
    #: otherwise copies such a parameter to `.local` per thread.
    self.grid_constant_kw = ''
    self.simd_mode = False

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

  def sync_mult(self, num_threads: int):
    """Rendezvous exactly the ``num_threads`` threads of one multiplication.

    Only reached where the multiplication is *wider* than a wave -- narrower
    than that and `SyncThreads` asks for `sync_simd` directly, because the
    threads are in lockstep anyway.  So the default is the honest one: a whole
    block, which over-synchronises but never deadlocks.

    The opportunity a vendor can take here is a sub-block rendezvous, and both
    of the ones that matter have one:

    * NVIDIA, `bar.sync id, count` -- meets exactly `count` threads at named
      barrier `id`.
    * AMD from gfx12 (checked: gfx1250 assembles `s_barrier_signal 1` /
      `s_barrier_wait 1`, gfx1150 has only the monolithic `s_barrier`).

    The catch is the same on both: every multiplication in the block needs an
    `id` of its own -- `threadIdx.y + 1`, since 0 is the one a full block
    barrier uses -- so it holds only while `mults_per_block` stays under the
    number of barrier resources, 15 usable of 16 on NVIDIA.  Two
    multiplications sharing an `id` rendezvous with each other, which is a
    deadlock the moment they run the body a different number of times.

    Not implemented for any target, and that is deliberate rather than
    pending: no lane configuration the generator produces today is wider than
    its wave.  `lanes.py` clamps `num_threads` to `vec_unit_length` for
    everything except an `ElementwiseDescr`, and no elementwise case in the
    corpus reaches the cap either.  An implementation would be untestable
    code; the hook exists so that widening a multiplication is a change in one
    place rather than a search.
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
