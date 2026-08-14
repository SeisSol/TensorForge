from typing import Union
import math
from tensorforge.common.matrix.tensor import Tensor
from . import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.symbol import Symbol, SymbolType, DataView, LeadIndex, write_loops, LeadLoop, Loop, add_offset
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.context import Context
from tensorforge.backend.data_types import RegMemObject
from typing import Union, List

# to find a number coprime to the number of shared memory banks
def _find_next_coprime(number, conumber):
  for i in range(number, number + conumber):
    if math.gcd(i, conumber) == 1:
      return i

class LoadInstruction:
  pass

class GlbToShrLoader(AbstractShrMemWrite, LoadInstruction):
  def __init__(self, **kwargs):
    super(GlbToShrLoader, self).__init__(kwargs['context'])
    # Kept so the transfer can be cloned.  Software pipelining peels a copy of
    # this load into the loop prologue with a different source pointer and
    # stage; re-invoking the constructor is the only way to get every derived
    # field (data_view, alignment, lid_dim, the user registrations) right --
    # copying the object would silently share them.
    self._ctor_kwargs = dict(kwargs)
    self._dest = kwargs['dest']
    self._src = kwargs['src']
    self._shr_mem = kwargs['shr_mem']
    self._num_threads = kwargs['num_threads']
    self._permute: None = kwargs['permute']
    self._manual_unroll_threshold = 4
    self._no_memcpy = kwargs['no_memcpy'] if 'no_memcpy' in kwargs else False

    if 'max_load_offset' in kwargs:
      self._max_load_offset = kwargs['max_load_offset']
    else:
      self._max_load_offset = self._num_threads

    if 'blockwide' in kwargs:
      self._blockwide = kwargs['blockwide']
    else:
      self._blockwide = False

    if 'alignment' in kwargs:
      self._alignment = kwargs['alignment']
    else:
      self._alignment = 1

    self._check()
    self._lid_dim: Union[int, None] = None
    self._align_shm_volume: Union[int, None] = None
    self._tensor: Tensor = self._src.obj

    self._dest.add_user(self)
    self._src.add_user(self)
    self._shr_mem.add_user(self)
    self._is_ready: bool = False

    self._use_cuda_memcpy = self._context.get_vm().get_hw_descr().vendor == 'nvidia' and not self._no_memcpy
    self._use_tma_memcpy = False

    if self._permute is None:
      self._permute = [i for i in range(len(self._src.obj.shape))]

    self._needs_reorder = self._permute != [i for i in range(len(self._src.obj.shape))]

    self._pipeline = 'pipeline'

    self._get_bounding_box_dense()

  def set_threadconfig_pre(self, num_threads, mults):
    if self._blockwide:
      self._num_threads = num_threads * mults

  def _next_size(self, size):
    return _find_next_coprime(size, self._context.get_vm().get_hw_descr().shmem_banks)

  def _linear_idx(self):
    lexic = self._context.get_vm().get_lexic()
    if self._blockwide:
      return f'({lexic.thread_idx_x} + {lexic.thread_idx_y} * {lexic.block_dim_x})'
    else:
      return f'{lexic.thread_idx_x}'

  def _get_bounding_box_dense(self):
    self._src.data_view = DataView(shape=self._tensor.get_actual_shape(),
                                   permute=None,
                                   bbox=self._tensor.get_bbox())

    src_real_shape = self._tensor.bbox.sizes()
    dst_bbox = self._tensor.get_bbox() # BoundingBox([0] * len(self._tensor.shape), src_real_shape)
    dst_shape = []
    read_shape = []
    loop_indices = []
    offset = 0
    loadsize = 1
    need_transpose = True

    if self._tensor.is_dense():

      # TODO: remove distinction between tensor shape and real shape
      for i in range(len(src_real_shape)):
        # offset += self._tensor.shape[i] - src_real_shape[i]
        if offset <= self._max_load_offset:
          readshape = src_real_shape[i] # self._tensor.shape[i]
        else:
          readshape = src_real_shape[i]
          loop_indices += [i]
        if self._permute[i] == 0: # TODO: not ideal
          need_transpose = False
        if need_transpose:
          dstshape = self._next_size(readshape)
        else:
          dstshape = readshape

        # TODO: move somewhere else?
        if i == 0:
          dstshape = ((dstshape + self._alignment - 1) // self._alignment) * self._alignment

        dst_shape += [dstshape]
        read_shape += [readshape]
        if len(loop_indices) <= 1:
          loadsize *= readshape

      # cap the first loop index, we're still contiguous there
      if len(loop_indices) > 0:
        loop_indices = loop_indices[1:]

      self._shm_volume = 1
      for dsts in dst_shape:
        self._shm_volume *= dsts
    else:
      loadsize = self._tensor.memory()
      self._shm_volume = loadsize
      read_shape = list(src_real_shape)
      dst_shape = list(src_real_shape)

    self._dest.data_view = DataView(shape=dst_shape,
                                    permute=None,
                                    bbox=dst_bbox)

    self._read_shape = read_shape
    self._dst_shape = dst_shape

    self._loop_indices = loop_indices
    self._loadsize = loadsize

  def gen_code_inner(self, writer: Writer) -> None:
    allow_nontemporal = len(self._src.get_user_list()) == 1


    if self._needs_reorder:
      src_bbox = self._src.data_view.get_bbox()
      loops = []
      loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0], self._num_threads, 1)]
      for i in range(1, src_bbox.rank()):
        loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

      def inner(indices):
        value = self._src.load(writer, self._context, None, indices, allow_nontemporal)
        self._dest.store(writer, self._context, value, indices, False,
                         base=self.write_base())

      write_loops(self._context, writer, loops, inner)
    else:
      if self._use_cuda_memcpy:
        writer(f'{self._pipeline}.producer_acquire();')

      loops = [writer.For(f'int32_t i{i} = 0; i{i} < {self._dest.data_view.shape[i]}; ++i{i}', True) for i in self._loop_indices]

      for loop in loops:
        loop.__enter__()

      index = list(self._dest.data_view.get_dim_offsets())
      for li in self._loop_indices:
        index[li] = f'i{li}'

      linscale = None
      if len(self._dst_shape) > 0 and self._dst_shape[0] != self._read_shape[0]:
        linscale = (self._read_shape[0], self._dst_shape[0])

      self._write_datatransfer(writer, 0, 0, index, self._loadsize, allow_nontemporal, linscale)

      for loop in loops[::-1]:
        loop.__exit__(None, None, None)

      if self._use_cuda_memcpy:
        writer(f'__syncwarp();')
        writer(f'{self._pipeline}.producer_commit();')
      if self._use_tma_memcpy:
        writer(f'__syncwarp();')
        writer(f'cuda::device::barrier_arrive_tx(mbarrier, 1, {self._loadsize});')

  def _write_datatransfer(self, writer, src_offset, dst_offset, index, length, nontemporal, linscale=None):
    pos = 0

    if self._use_cuda_memcpy and self._src.obj.alignment < 16:
      granularities = [1]
    else:
      granularities = [m for m in [4, 2, 1] if m * self._dest.get_fptype().size() <= 16]

    if self._use_tma_memcpy:
      dest_access_index = self._dest.access_address(self._context, index, writer)
      src_access_index = self._src.access_address(self._context, index, writer)
      writer(f'cuda::device::memcpy_async_tx(&{self.write_base()}[{dest_access_index}], &{self._src.name}[{src_access_index}], cuda::aligned_size_t<16>({length}), mbarrier);')
    else:
      for vecsize in granularities:
        if src_offset % vecsize == 0:
          num_hops = ((length - pos * self._num_threads) // (self._num_threads * vecsize)) * vecsize
          self._write_hop(writer, src_offset, dst_offset, index, pos, pos + num_hops, vecsize, nontemporal, linscale)
          pos += num_hops
      rest = length % self._num_threads
      if rest > 0:
        with writer.If(f'{self._linear_idx()} < {rest}'):
          self._write_hop(writer, src_offset, dst_offset, index, pos, pos+1, 1, nontemporal, linscale)

  def _write_hop(self, writer, src_offset, dst_offset, index, start, end, increment, nontemporal, linscale):
    if end > start:
      if increment > 1:
        vectortype = self._vm.get_lexic().get_fptype(self._dest.get_fptype(), increment)
        typeprefix = f'*({vectortype}*)&'
      else:
        typeprefix = ''

      if self._use_cuda_memcpy:
        elsize = self._dest.get_fptype().size() * increment
        def write_load(lhs, rhs):
          writer(f'cuda::memcpy_async(&{lhs}, &{rhs}, cuda::aligned_size_t<{elsize}>({elsize}), {self._pipeline});')
      else:
        def write_load(lhs, rhs):
          writer(self._context.get_vm().get_lexic().glb_load(lhs, rhs, nontemporal=nontemporal))

      if linscale is None:
        indexwrapper = lambda x: x
      else:
        indexwrapper = lambda x: f'((({x}) / {linscale[0]}) * {linscale[1]} + (({x}) % {linscale[0]}))'

      if (end - start) / increment > self._manual_unroll_threshold:
        # load using a for-loop
        with writer.For(f'int32_t i = {start}; i < {end}; i += {increment}', True):
          contiguous_index = indexwrapper(f'{increment} * {self._linear_idx()} + i * {self._num_threads}')
          dest_access_index = self._dest.access_address(self._context, index, writer)
          src_access_index = self._src.access_address(self._context, index, writer)
          lhs = f'{typeprefix}{self.write_base()}[{dst_offset} + {dest_access_index} + {contiguous_index}]'
          rhs = f'{typeprefix}{self._src.name}[{src_offset} + {src_access_index} + {contiguous_index}]'
          write_load(lhs, rhs)
      else:
        # load using manual loop unrolling
        for counter in range(start, end, increment):
          contiguous_index = indexwrapper(f'{increment} * {self._linear_idx()} + {counter * self._num_threads}')
          dest_access_index = self._dest.access_address(self._context, index, writer)
          src_access_index = self._src.access_address(self._context, index, writer)
          lhs = f'{typeprefix}{self.write_base()}[{dst_offset} + {dest_access_index} + {contiguous_index}]'
          rhs = f'{typeprefix}{self._src.name}[{src_offset} + {src_access_index} + {contiguous_index}]'
          write_load(lhs, rhs)

  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  def get_permute(self) -> List[int]:
    return self._permute

  def _check(self) -> None:
    #if self._src.stype != SymbolType.Global:
    #  raise InternalError('shr-load: `src` operand is not in global mem.')

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

  def clone(self, **overrides) -> 'GlbToShrLoader':
    """A fresh transfer with the same configuration, minus the overrides.

    Software pipelining peels a copy of this load into the loop prologue with a
    different source pointer and stage.  Re-invoking the constructor is the only
    way to get every derived field right -- data_view, alignment, lid_dim and
    the user registrations are all computed there, and copying the object would
    silently share them.
    """
    kwargs = dict(self._ctor_kwargs)
    kwargs.update(overrides)
    return type(self)(**kwargs)

  def __str__(self):
    return f'{self._dest.name} = load{{g>s}}({self._src.name}[{", ".join(str(p) for p in self._permute)}])'

class ShrToShrLoader(GlbToShrLoader):
  def __init__(self, **kwargs):
    super().__init__(no_memcpy=True, **kwargs)

  def __str__(self):
    return f'{self._dest.name} = load{{s>s}}({self._src.name}[{", ".join(str(p) for p in self._permute)}])'

  def set_shr_mem_offset(self, offset: int, first: bool, global_offset: bool) -> None:
    # TODO: refactor users instead
    super().set_shr_mem_offset(offset, True, global_offset)

class GlbToRegLoader(MemoryInstruction, LoadInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               num_threads: int,
               linearize: bool,
               src_bbox=None,
               src_offset=None):
    super(GlbToRegLoader, self).__init__(context)

    if dest.stype != SymbolType.Register:
      raise InternalError('store: operand `dest` is not in reg mem')

    if not isinstance(dest.obj, RegMemObject):
      raise InternalError(f'store: operand `dest` is registers, instead: {type(dest.obj)}')

    if src.stype != SymbolType.Global:
      raise InternalError('store: operand `src` is not in global memory.')

    if not isinstance(src.obj, Tensor):
      raise InternalError('store: operand `src` is not a matrix')

    src.add_user(self)
    dest.add_user(self)

    # `src_bbox` is the region to load, in *logical* coordinates; `src_offset`
    # is the logical->storage shift of the operand this load stages.  Registers
    # dispatch on `isinstance(index, LeadIndex)` (Symbol.build_address), so an
    # offset cannot ride along as a VarOffset the way it can for a global or
    # shared-memory operand --- it would land in the non-lead branch and pick up
    # both the wrong divisor and the wrong stride.  It is therefore consumed
    # here: read at `x + offset`, write at `x`, and the register image is in
    # logical coordinates from then on.
    self._bbox = src_bbox if src_bbox is not None else src.obj.get_bbox()
    self._offset = list(src_offset) if src_offset is not None else [0] * self._bbox.rank()

    dest.data_view = DataView(shape=src.obj.shape,
                              permute=None,
                              bbox=self._bbox)

    # if dest.data_view.get_dim_size(0) > src.data_view.get_dim_size(0):
    #   raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._num_threads: int = num_threads
    self._is_ready: bool = True
    self._linearize = linearize

  def gen_code_inner(self, writer: Writer) -> None:
    writer.new_line()
    dest_view = self._dest.data_view

    allow_nontemporal = len(self._src.get_user_list()) == 1

    src_bbox = self._bbox

    if self._linearize:
      # a flat run over spp.count_nz() cannot express a sub-slice
      assert all(o == 0 for o in self._offset), \
          (f'{self._src.name}: linearized register load cannot apply slicing '
           f'offset {self._offset}')
      # TODO: box better?
      total_size = self._src.obj.spp.count_nz()

      start = 0
      for g in [1]: #[4, 2, 1]:
        granularity = self._num_threads * g
        for i in range(start, total_size, granularity):
          self._src.load_linear(writer, self._context, f'v{i}', i, g)
          self._dest.store_linear(writer, self._context, f'v{i}', i, g,
                                  base=self.write_base())

        start = (total_size // granularity) * granularity

    elif self._context.get_vm().get_hw_descr().vendor in ['amd'] and False:

      # float4 load

      # for now: use  0 1 2 3, transpose4x4

      # TODO: sort into 4x4x4 blocks

      lead_size = src_bbox.size(0)
      lead_count = (lead_size + self._num_threads - 1) // self._num_threads

      total_count = lead_count
      for dim in src_bbox.sizes()[1:]:
        total_count *= dim

      start = 0

      prec = 'float'

      for g in [4, 2, 1]: # [4, 3, 2, 1]
        # 4x4
        # writer(f'const auto f{g}idx = (threadIdx.x % {g}) * {self._num_threads} + (threadIdx.x / {g}) * {g};')
        total_count_g = (total_count // g) * g

        if start != total_count_g:
          writer(f'const auto f{g}idx = ((threadIdx.x / {16 // g}) % {g}) * {self._num_threads} + (threadIdx.x % {16 // g}) * {g} + (threadIdx.x / 16) * 16;')

        for i in range(start, total_count_g, g):
          sidx = i // lead_count
          ridx = i % lead_count
          index = sidx * lead_size + ridx * self._num_threads
          writer(f'const auto v{i} = __builtin_nontemporal_load((tensorforge::VectorT<{prec}, {g}>*)&{self._src.name}[{index} + f{g}idx]);')

          args2 = ', '.join(f'v{i}[{k}]' for k in range(g))

          for k in range(g):
            writer(f'{prec} v{i}w{k} = 0;')

          args1 = ', '.join(f'v{i}w{k}' for k in range(g))

          if g == 4:
            writer(f'tensorforge::transpose16x4({args1}, {args2});')
          if g == 2:
            writer(f'tensorforge::transpose16x2({args1}, {args2});')
          if g == 1:
            writer(f'{args1} = {args2};')

          # TODO: generalize
          for k in range(g):
            writer(f'{self._dest.name}[{i + k}] = v{i}w{k};')

        start = total_count_g

    else:
      # The lane axis is whichever dimension the destination declares, not
      # dimension 0: a transposed operand carries the lead index elsewhere, and
      # writing the image with the lane on dimension 0 while every reader
      # addresses it through `lead_dims` puts the two out of step.
      lead_pos = self._dest.lead_dims[0]
      loops = []
      for i in range(src_bbox.rank()):
        if i == lead_pos:
          loops += [LeadLoop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i],
                             self._num_threads, 1)]
        else:
          loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

      def inner(indices):
        # logical index in on the register side, storage index out on the
        # global side --- add_offset folds the (usual) zero away
        value = self._src.load(writer, self._context, None,
                       [add_offset(x, self._offset[i])
                        for i, x in enumerate(indices)], allow_nontemporal)
        self._dest.store(writer, self._context, value, indices, False,
                         base=self.write_base())

      write_loops(self._context, writer, loops, inner)

  def __str__(self) -> str:
    return f'{self._dest.name} = load{{g>r}}({self._src.name});'

class LoadWait(MemoryInstruction, LoadInstruction):
  def __init__(self, instr):
    super(LoadWait, self).__init__(instr._context)
    self._instr = instr
    self._is_ready = True

  # A wait completes the awaited transfer, so from a data-flow point of view
  # it *is* the write.  Consumers must therefore be ordered after the wait,
  # not after the issuing load.  Once async/wait carries a real token this
  # becomes a use of that token instead.
  def awaited(self):
    return self._instr

  def defs(self):
    return self._instr.defs()

  def uses(self):
    # ...but the destination buffer is occupied from the moment the copy is
    # *issued*: the DMA writes into it while it is in flight.  Reporting the
    # def alone makes this instruction kill the issuing loader's live range,
    # leaving a hole over [issue, wait) in which the region allocator happily
    # hands the very same offset to another buffer --- which then overwrites
    # the transfer as it lands.  Naming it here as well re-establishes
    # liveness backwards past the wait, so the range runs from the issue to
    # the last consumer, while `defs` keeps ordering consumers after us.
    return self._instr.defs()

  def gen_code_inner(self, writer: Writer) -> None:
    if isinstance(self._instr, GlbToShrLoader):
      if self._instr._use_cuda_memcpy:
        writer(f'{self._instr._pipeline}.consumer_wait();')
        writer(f'{self._instr._pipeline}.consumer_release();')

  def __str__(self) -> str:
    return f'wait({self._instr});'
