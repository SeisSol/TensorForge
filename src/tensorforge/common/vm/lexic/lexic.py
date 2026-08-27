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

  def glb_store(self, lhs, rhs, nontemporal=False):
    return f'{lhs} = {rhs};'

  def glb_load(self, rhs, nontemporal=False):
    return f'{rhs}'

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
