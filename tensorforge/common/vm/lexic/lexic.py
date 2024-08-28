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
  def declare_shared_memory_inline(self, name, precision, size, alignment):
    pass

  @abstractmethod
  def declare_shared_memory(self, name, precision):
    pass

  @abstractmethod
  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None,
                        total_shared_mem_size=None, global_symbols=None):
    pass

  def get_mapped_keywords(self):
    return []

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

  def glb_store(self, lhs, rhs, nontemporal=False):
    return f'{lhs} = {rhs};'
  
  def glb_load(self, lhs, rhs, nontemporal=False):
    return f'{lhs} = {rhs};'
