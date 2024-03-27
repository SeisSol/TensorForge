from abc import abstractmethod
from typing import Union
import enum
import math
from kernelforge.common.matrix.tensor import Tensor
from . import AbstractShrMemWrite, MemoryInstruction
from kernelforge.backend.symbol import SymbolType, Symbol, DataView
from kernelforge.common.exceptions import InternalError
from kernelforge.backend.writer import Writer
from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.context import Context
from typing import Union, List

# to find a number coprime to the number of shared memory banks
def _find_next_coprime(number, conumber):
  for i in range(number, number + conumber):
    if math.gcd(i, conumber) == 1:
      return i

class GlbToShrLoader(AbstractShrMemWrite):
  def __init__(self, **kwargs):
    super(GlbToShrLoader, self).__init__(kwargs['context'])
    self._dest = kwargs['dest']
    self._src = kwargs['src']
    self._shr_mem = kwargs['shr_mem']
    self._num_threads = kwargs['num_threads']
    self._permute: None = kwargs['permute']
    self._manual_unroll_threshold = 4

    self._check()
    self._lid_dim: Union[int, None] = None
    self._align_shm_volume: Union[int, None] = None
    self._tensor: Tensor = self._src.obj

    self._dest.add_user(self)
    self._src.add_user(self)
    self._shr_mem.add_user(self)
    self._is_ready: bool = False

    self._use_cuda_memcpy = False

    if self._permute is None:
      self._permute = [i for i in range(len(self._src.obj.shape))]
    
    self._needs_reorder = self._permute != [i for i in range(len(self._src.obj.shape))]

    self._get_bounding_box_dense()

  def _next_size(self, size):
    return _find_next_coprime(size, 32)

  def _get_bounding_box_dense(self):
    self._src.data_view = DataView(shape=self._tensor.shape,
                                   permute=None,
                                   bbox=self._tensor.get_bbox())

    src_real_shape = self._tensor.bbox.sizes()
    dst_bbox = BoundingBox([0] * len(self._tensor.shape), src_real_shape)
    dst_shape = []
    read_shape = []
    loop_indices = []
    offset = 0
    loadsize = 1
    need_transpose = True
    for i in range(len(src_real_shape)):
      offset += self._tensor.shape[i] - src_real_shape[i]
      if offset < self._num_threads:
        readshape = self._tensor.shape[i]
      else:
        readshape = src_real_shape[i]
        loop_indices += [i]
      if self._permute[i] == 0: # TODO: not ideal
        need_transpose = False
      if need_transpose:
        dstshape = self._next_size(readshape)
      else:
        dstshape = readshape
      dst_shape += [dstshape]
      read_shape += [readshape]
      if len(loop_indices) <= 1:
        loadsize *= readshape
    
    # cap the first loop index, we're still contiguous there
    if len(loop_indices) > 0:
      loop_indices = loop_indices[1:]

    self._dest.data_view = DataView(shape=dst_shape,
                                    permute=None,
                                    bbox=dst_bbox)

    self._loop_indices = loop_indices
    self._loadsize = loadsize
    self._shm_volume = 1
    for dsts in dst_shape:
      self._shm_volume *= dsts

  def gen_code_declare(self, writer: Writer) -> None:
    writer.new_line()
    lhs = f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    rhs = f'{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = &{rhs};')

  def gen_code_inner(self, writer: Writer) -> None:
    allow_nontemporal = len(self._src.get_user_list()) == 1

    src_offset = 0#self._src.data_view.get_offset()

    if self._needs_reorder:
      loops = [writer.For(f'int i{i} = 0; i{i} < {self._dest.data_view.shape[i]}; ++i{i}') for i in range(1, len(self._dest.data_view.shape))]
      index = [0] * len(self._dest.data_view.shape)
      for li in self._loop_indices:
        index[li] = f'i{li}'
      index[0] = self._vm.get_lexic().thread_idx_x

      for loop in loops:
        loop.__enter__()
      self._src.load(writer, self._context, 'value', index, allow_nontemporal)
      self._dest.store(writer, self._context, 'value', index, False)
      for loop in loops[::-1]:
        loop.__exit__(None, None, None)
    else:
      loops = [writer.For(f'int i{i} = 0; i{i} < {self._dest.data_view.shape[i]}; ++i{i}') for i in self._loop_indices]

      for loop in loops:
        writer.insert_pragma_unroll()
        loop.__enter__()
      
      index = [0] * len(self._dest.data_view.shape)
      for li in self._loop_indices:
        index[li] = f'i{li}'
      
      self._write_datatransfer(writer, 0, 0, index, self._loadsize, allow_nontemporal)

      for loop in loops[::-1]:
        loop.__exit__(None, None, None)
    
    #if False:
    #  writer('cooperative_groups::wait(cooperative_groups::this_thread_block());')

  def _write_datatransfer(self, writer, src_offset, dst_offset, index, length, nontemporal):
    if not self._use_cuda_memcpy:
      pos = 0
      for vecsize in [1]:
        if src_offset % vecsize == 0:
          num_hops = ((length - pos * self._num_threads) // (self._num_threads * vecsize)) * vecsize
          self._write_hop(writer, src_offset, dst_offset, index, pos, pos + num_hops, vecsize, nontemporal)
          pos += num_hops
      rest = length % self._num_threads
      if rest > 0:
        with writer.If(f'{self._vm.get_lexic().thread_idx_x} < {rest}'):
          self._write_hop(writer, src_offset, dst_offset, index, pos, pos+1, 1, nontemporal)
    else:
      dest_access_index = self._dest.access_address(self._context, index)
      src_access_index = self._src.access_address(self._context, index)
      writer(f'cooperative_groups::memcpy_async(cooperative_groups::this_thread_block(), &{self._dest.name}[{dst_offset} + {dest_access_index}], &{self._src.name}[{src_offset} + {src_access_index}], {length * self._dest.get_fptype(self._context).size()});')

  def _write_hop(self, writer, src_offset, dst_offset, index, start, end, increment, nontemporal):
    if end > start:
      if increment > 1:
        vectortype = self._vm.get_lexic().get_fptype(self._dest.get_fptype(self._context), increment)
        typeprefix = f'*({vectortype}*)&'
      else:
        typeprefix = ''
      if (end - start) / increment > self._manual_unroll_threshold:
        # load using a for-loop
        writer.insert_pragma_unroll()
        with writer.For(f'int i = {start}; i < {end}; i += {increment}'):
          contiguous_index = f'{increment} * {self._vm.get_lexic().thread_idx_x} + i * {self._num_threads}'
          dest_access_index = self._dest.access_address(self._context, index)
          src_access_index = self._src.access_address(self._context, index)
          lhs = f'{typeprefix}{self._dest.name}[{dst_offset} + {dest_access_index} + {contiguous_index}]'
          rhs = f'{typeprefix}{self._src.name}[{src_offset} + {src_access_index} + {contiguous_index}]'
          writer(self._context.get_vm().get_lexic().glb_load(lhs, rhs, nontemporal=nontemporal))
      else:
        # load using manual loop unrolling
        for counter in range(start, end, increment):
          contiguous_index = f'{increment} * {self._vm.get_lexic().thread_idx_x} + {counter * self._num_threads}'
          dest_access_index = self._dest.access_address(self._context, index)
          src_access_index = self._src.access_address(self._context, index)
          lhs = f'{typeprefix}{self._dest.name}[{dst_offset} + {dest_access_index} + {contiguous_index}]'
          rhs = f'{typeprefix}{self._src.name}[{src_offset} + {src_access_index} + {contiguous_index}]'
          writer(self._context.get_vm().get_lexic().glb_load(lhs, rhs, nontemporal=nontemporal))

  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  def get_permute(self) -> List[int]:
    return self._permute

  def _check(self) -> None:
    if self._src.stype != SymbolType.Global:
      raise InternalError('shr-load: `src` operand is not in global mem.')

    if not isinstance(self._src.obj, Tensor):
      raise InternalError(f'shr-load: `src` operand is not a tensor, instead: {self._src.obj}')

    if self._dest.stype != SymbolType.SharedMem:
      raise InternalError('shr-load: `dest` operand is not in shr. mem.')

    if not isinstance(self._dest.obj, Tensor):
      raise InternalError(f'shr-load: `dest` operand is not a tensor, instead: {self._dest.obj}')

  def get_headers(self) -> List[str]:
    if self._use_cuda_memcpy:
      return ['cooperative_groups.h', 'cooperative_groups/memcpy_async.h']
    else:
      return []

  def __str__(self):
    return f'{self._dest.name} = load{{g>s}}({self._src.name}[{", ".join(str(p) for p in self._permute)}])'

class GlbToRegLoader(MemoryInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(GlbToRegLoader, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in reg mem')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.Global:
      raise InternalError('store: operand `dest` is not in global memory.')

    if not isinstance(dest.obj, Tensor):
      raise InternalError('store: operand `dest` is not a matrix')

    src.add_user(self)
    dest.add_user(self)

    dest.data_view = DataView(shape=dest.obj.shape,
                              permute=None,
                              bbox=dest.obj.get_bbox())
    
    # if dest.data_view.get_dim_size(0) > src.data_view.get_dim_size(0):
    #   raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

    self._dest: Symbol = dest
    self._src: Symbol = src.clone()
    self._alpha = alpha
    self._beta = beta
    self._num_threads: int = num_threads
    self._is_ready: bool = True

  def gen_code(self, writer: Writer) -> None:
    writer.new_line()
    dest_view = self._dest.data_view

    allow_nontemporal = len(self._src.get_user_list()) == 1

    writer(f'// {self}')
    src_bbox = self._src.data_view.get_bbox()
    with writer.If(self.gen_range_mask_threads(begin=src_bbox.lower()[0], end=src_bbox.upper()[0])):
      loops = []
      indices = [self._vm.get_lexic().thread_idx_x]
      for i in range(1, src_bbox.rank()):
        writer.insert_pragma_unroll()
        loop = f'int i{i} = 0; i{i} < {dest_view.shape[i]}; ++i{i}'
        loops += [writer.For(loop)]
        loops[-1].__enter__()
        indices += [f'i{i}']

      self._src.load(writer, self._context, 'value', indices, False)
      self._dest.store(writer, self._context, 'value', indices, allow_nontemporal)
      
      for loop in reversed(loops):
        loop.__exit__(None, None, None)

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{g>r}}({self._src.name});'
