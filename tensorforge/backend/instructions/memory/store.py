from typing import Union
from tensorforge.common.context import Context
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.backend.data_types import RegMemObject
from tensorforge.backend.symbol import Symbol, SymbolType, DataView, LeadIndex, write_loops, LeadLoop, Loop, Immediate, add_offset
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from . import AbstractShrMemWrite, MemoryInstruction
from ..abstract_instruction import AbstractInstruction


class StoreRegToReg(MemoryInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               num_threads: int):
    super(StoreRegToReg, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in registers')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is not registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.Register:
      raise InternalError('store: operand `dest` is not a register.')

    if not isinstance(dest.obj, RegMemObject):
      raise InternalError(f'store: operand `dest` is not a matrix, instead: {type(dest.obj)}')

    src.add_user(self)
    dest.add_user(self)

    self._is_ready = True

    bbox = src.data_view.get_bbox()
    bbox = BoundingBox([0] * bbox.rank(), bbox.sizes())
    dest.data_view = DataView(bbox.sizes(),
                              permute=None,
                              bbox=bbox)

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._num_threads: int = num_threads
    view: DataView = self._dest.data_view

  def gen_code_inner(self, writer: Writer) -> None:
    dest_view = self._dest.data_view
    src_bbox = self._src.data_view.get_bbox()

    loops = []
    loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0], self._num_threads, 1)]
    for i in range(1, src_bbox.rank()):
      loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

    def inner(indices):
      self._src.load(writer, self._context, 'value', indices, False)
      self._dest.store(writer, self._context, 'value', indices, False)

    write_loops(self._context, writer, loops, inner)

  def get_dest(self) -> Symbol:
    return self._dest

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{r>r}}({self._src.name});'

class StoreRegToShr(AbstractShrMemWrite):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               shr_mem: Symbol,
               num_threads: int,
               dest_bbox=None,
               dest_offset=None):
    super(StoreRegToShr, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in registers')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is not registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.SharedMem:
      raise InternalError('store: operand `dest` is not in shared mem.')

    if not isinstance(dest.obj, Tensor):
      raise InternalError(f'store: operand `dest` is not a matrix, instead: {type(dest.obj)}')

    src.add_user(self)
    dest.add_user(self)
    shr_mem.add_user(self)

    # `dest_bbox` is the whole buffer, `dest_offset` where this store lands in
    # it.  The two differ as soon as a tensor is assembled from several writes:
    # sizing the view to `src` alone would give every slice its own idea of how
    # big the shared buffer is, and the last one written would win.
    self._dest_offset = (list(dest_offset) if dest_offset is not None
                         else [0] * src.data_view.get_bbox().rank())
    buffer_bbox = dest_bbox if dest_bbox is not None else src.data_view.get_bbox()
    dest.data_view = DataView(buffer_bbox.sizes(),
                              permute=None,
                              bbox=buffer_bbox)

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._shr_mem: Symbol = shr_mem
    self._num_threads: int = num_threads
    self._shr_mem_offset: Union[int, None] = None
    view: DataView = self._dest.data_view
    self._shm_volume: int = view.get_volume()

  def gen_code_inner(self, writer: Writer) -> None:
    dest_view = self._dest.data_view
    src_bbox = self._src.data_view.get_bbox()

    loops = []
    loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0], self._num_threads, 1)]
    for i in range(1, src_bbox.rank()):
      loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

    def inner(indices):
      self._src.load(writer, self._context, 'value', indices, False)
      self._dest.store(writer, self._context, 'value',
                       [add_offset(x, self._dest_offset[i])
                        for i, x in enumerate(indices)], False)

    write_loops(self._context, writer, loops, inner)

  def get_dest(self) -> Symbol:
    return self._dest

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{r>s}}({self._shr_mem.name}, {self._src.name});'


class StoreRegToGlb(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               num_threads: int,
               atomic,
               dest_offset=None,
               dest_bbox=None):
    super(StoreRegToGlb, self).__init__(context)

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

    dest.data_view = DataView(shape=dest.obj.get_actual_shape(),
                              permute=None,
                              bbox=dest.obj.get_bbox())

    #if dest.data_view.get_dim_size(0) < src.data_view.get_dim_size(0):
    #  raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._num_threads: int = num_threads
    self._is_ready: bool = True
    self._atomic = atomic
    # logical->storage shift of the destination slice.  `src` (registers) is
    # indexed logically, `dest` (global) in storage coordinates, so the shift
    # enters both the loop bounds --- which mix the two bboxes --- and the
    # store itself.
    self._dest_offset = (list(dest_offset) if dest_offset is not None
                         else [0] * dest.data_view.rank())
    # What this store promises to define, in the same logical coordinates as
    # `src`.  `_analyze` intersects the accumulator's range down to what the
    # operands support, so it can end up narrower than the descriptor declared
    # and the difference has to be zero-filled.  The *tensor's* box is not that
    # promise: a sliced write declares a small part of it, and the rest belongs
    # to other descriptors or to whatever the caller put there.
    self._promise = dest_bbox

  def gen_ir(self, writer: Writer) -> None:
    writer.new_line()
    dest_view = self._dest.data_view

    allow_nontemporal = len(self._src.get_user_list()) == 1 # self._src.get_last_user() is self

    writer(f'// {self}')
    src_bbox = self._src.data_view.get_bbox()
    if self._promise is not None:
      dest_bbox = self._promise
    else:
      # pull the destination's storage bbox back into logical coordinates so
      # the union below compares like with like
      raw_dest_bbox = self._dest.data_view.get_bbox()
      dest_bbox = BoundingBox(
          [l - o for l, o in zip(raw_dest_bbox.lower(), self._dest_offset)],
          [u - o for u, o in zip(raw_dest_bbox.upper(), self._dest_offset)])
    with writer.Scope():
      manual = [False]
      loops = []
      loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0], self._num_threads, 1)]
      for i in range(1, src_bbox.rank()):
        unroll = (src_bbox.lower()[i], src_bbox.upper()[i]) != (dest_bbox.lower()[i], dest_bbox.upper()[i])
        lower = min(src_bbox.lower()[i], dest_bbox.lower()[i])
        upper = max(src_bbox.upper()[i], dest_bbox.upper()[i])
        loops += [Loop(f'i{i}', lower, upper, 1, unroll)]
        manual += [unroll]

      def inner(indices):
        needsLoad = all(not isinstance(index, Immediate) or (src_bbox.lower()[i] <= index._value and src_bbox.upper()[i] > index._value) for i,index in enumerate(indices))
        dest_indices = [add_offset(x, self._dest_offset[i])
                        for i, x in enumerate(indices)]
        if needsLoad:
          self._src.load(writer, self._context, 'value', indices, False)
          self._dest.store(writer, self._context, 'value', dest_indices, allow_nontemporal, self._atomic)
        else:
          self._dest.store(writer, self._context, '0', dest_indices, allow_nontemporal, self._atomic)

      if not any(manual) and self._context.get_vm().get_hw_descr().vendor in ['amd'] and False:
        pass
      else:
        write_loops(self._context, writer, loops, inner)

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{r>g}}({self._src.name});'

def round_up_to_nearest_vec_length(n, vec_length):
    return math.ceil(n / vec_length) * vec_length

class StoreShrMemToGlb(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               num_threads: int):
    super(StoreShrMemToGlb, self).__init__(context)

    #if src.stype != SymbolType.SharedMem:
    #  raise InternalError('store: operand `src` is not in shr mem.')

    #if dest.stype != SymbolType.Global:
    #  raise InternalError('store: operand `dest` is not in glb mem.')

    self._dest = dest
    self._src = src
    self._num_threads = num_active_threads
    self._is_ready = True

    src.add_user(self)
    dest.add_user(self)

  # NOTE: LivenessAnalysis._check_store already called get_src() on this
  # class, which never had it -- latent AttributeError, dormant only because
  # nothing constructs StoreShrMemToGlb today.
  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  def gen_ir(self, writer):
    dest_matrix = self._dest.obj

    dest_name = self._dest.name
    src_name = self._src.name
    vec_unit_length = self._vm._hw_descr.vec_unit_length

    thread_idx_x = self._vm.get_lexic().thread_idx_x
    num_hops = int(dest_matrix.get_actual_num_rows() / self._num_threads)

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view

    # TODO: float4 storage

    writer('// {self}')

    writer.Pragma("unroll")
    with writer.For(f'int32_t k = 0; k < {dest_data_view.columns}; ++k'):
      num_hops = int(dest_data_view.lead_dim / self._num_threads)
      if num_hops > 0:
        writer.Pragma("unroll")
        with writer.For(f'int32_t counter = 0; counter < {num_hops}; ++counter'):
          shr_mem_addr = f'{thread_idx_x}'
          shr_mem_addr += f' + counter * {self._num_threads} + k * {dest_data_view.lead_dim}'

          glb_mem_addr = f'{thread_idx_x}'
          glb_mem_addr += f' + counter * {self._num_threads} + k * {self._src.obj.num_rows}'

          lhs = "{}[{}]".format(dest_name, glb_mem_addr)
          rhs = "{}[{}]".format(src_name,  shr_mem_addr)
          writer(self._vm.get_lexic().glb_store(lhs, rhs, self._src.get_last_user() is self))

      # the last hop to fill shared mem with data
      if (dest_data_view.lead_dim % self._num_threads) != 0:
        residue = dest_data_view.lead_dim - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          finial_offset = num_hops * self._num_threads
          shr_mem_addr = f'{thread_idx_x} + {finial_offset} + k * {dest_data_view.lead_dim}'
          glb_mem_addr = f'{thread_idx_x} + {finial_offset} + k * {src_data_view.lead_dim}'

          lhs = "{}[{}]".format(dest_name, glb_mem_addr)
          rhs = "{}[{}]".format(src_name,  shr_mem_addr)
          writer(self._vm.get_lexic().glb_store(lhs, rhs, self._src.get_last_user() is self))

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{s>g}}({self._src.name});'
