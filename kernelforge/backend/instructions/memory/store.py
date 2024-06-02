from typing import Union
from kernelforge.common.context import Context
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.backend.data_types import RegMemObject
from kernelforge.backend.symbol import Symbol, SymbolType, DataView
from kernelforge.common.exceptions import InternalError
from kernelforge.backend.writer import Writer
from . import AbstractShrMemWrite, MemoryInstruction
from ..abstract_instruction import AbstractInstruction
from kernelforge.common.basic_types import FloatingPointType


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
    self._src: Symbol = src.clone()
    self._num_threads: int = num_threads
    view: DataView = self._dest.data_view

  def gen_code_declare(self, writer: Writer):
    pass

  def gen_code_inner(self, writer: Writer) -> None:
    dest_view = self._dest.data_view
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
      self._dest.store(writer, self._context, 'value', indices, False)
      
      for loop in reversed(loops):
        loop.__exit__(None, None, None)

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
               num_threads: int):
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

    # bbox = dest.obj.get_bbox()
    # bbox = BoundingBox([0] * bbox.rank(), bbox.sizes())
    dest.data_view = DataView(src.data_view.get_bbox().sizes(),
                              permute=None,
                              bbox=src.data_view.get_bbox())

    self._dest: Symbol = dest
    self._src: Symbol = src.clone()
    self._shr_mem: Symbol = shr_mem
    self._num_threads: int = num_threads
    self._shr_mem_offset: Union[int, None] = None
    view: DataView = self._dest.data_view
    self._shm_volume: int = view.get_volume()

  def gen_code_declare(self, writer: Writer):
    lhs = f'{self._fp_as_str}* {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    rhs = f'&{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = {rhs};')

  def gen_code_inner(self, writer: Writer) -> None:
    dest_view = self._dest.data_view
    src_bbox = self._src.data_view.get_bbox()

    with writer.If(self.gen_range_mask_threads(begin=src_bbox.lower()[0], end=src_bbox.upper()[0])):
      loops = []
      indicesSrc = [self._vm.get_lexic().thread_idx_x]
      indicesDest = [self._vm.get_lexic().thread_idx_x]# [f'({self._vm.get_lexic().thread_idx_x} - {src_bbox.lower()[0]})']
      for i in range(1, src_bbox.rank()):
        writer.insert_pragma_unroll()
        loop = f'int i{i} = 0; i{i} < {dest_view.shape[i]}; ++i{i}'
        loops += [writer.For(loop)]
        loops[-1].__enter__()
        indicesSrc += [f'i{i}']
        indicesDest += [f'i{i}']

      self._src.load(writer, self._context, 'value', indicesSrc, False)
      self._dest.store(writer, self._context, 'value', indicesDest, False)
      
      for loop in reversed(loops):
        loop.__exit__(None, None, None)

  def get_dest(self) -> Symbol:
    return self._dest

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{r>s}}({self._shr_mem.name}, {self._src.name});'


class StoreRegToGlb(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
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

    dest.data_view = DataView(shape=dest.obj.shape,
                              permute=None,
                              bbox=dest.obj.get_bbox())
    
    #if dest.data_view.get_dim_size(0) < src.data_view.get_dim_size(0):
    #  raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

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
    return f'{self._dest.name} = store{{r>g}}({self._src.name});'

def round_up_to_nearest_vec_length(n, vec_length):
    return math.ceil(n / vec_length) * vec_length

class StoreShrMemToGlb(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               alpha: float,
               beta: float,
               num_compute_threads: int,
               num_active_threads: int):
    super(StoreShrMemToGlb, self).__init__(context)

    #if src.stype != SymbolType.SharedMem:
    #  raise InternalError('store: operand `src` is not in shr mem.')

    #if dest.stype != SymbolType.Global:
    #  raise InternalError('store: operand `dest` is not in glb mem.')

    self._dest = dest
    self._src = src
    self._alpha = alpha
    self._beta = beta
    self._num_threads = num_active_threads
    self._is_ready = True

    src.add_user(self)
    dest.add_user(self)

  def gen_code(self, writer):
    dest_matrix = self._dest.obj
    
    dest_name = self._dest.name
    src_name = self._src.name
    precision = self._vm.fp_as_str()
    vec_unit_length = self._vm._hw_descr.vec_unit_length

    thread_idx_x = self._vm.get_lexic().thread_idx_x
    num_hops = int(dest_matrix.get_actual_num_rows() / self._num_threads)

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view

    # TODO: float4 storage

    writer('// {self}')

    writer.Pragma("unroll")
    with writer.For(f'int k = 0; k < {dest_data_view.columns}; ++k'):
      num_hops = int(dest_data_view.lead_dim / self._num_threads)
      if num_hops > 0:
        writer.Pragma("unroll")
        with writer.For(f'int counter = 0; counter < {num_hops}; ++counter'):
          shr_mem_addr = f'{thread_idx_x}'
          shr_mem_addr += f' + counter * {self._num_threads} + k * {dest_data_view.lead_dim}'

          glb_mem_addr = f'{thread_idx_x}'
          glb_mem_addr += f' + counter * {self._num_threads} + k * {self._src.obj.num_rows}'

          lhs = "{}[{}]".format(dest_name, glb_mem_addr)
          rhs = "{}[{}]".format(src_name,  shr_mem_addr)
          writer(self._vm.get_lexic().glb_store(lhs, rhs, self._dest._users == [self]))

      # the last hop to fill shared mem with data
      if (dest_data_view.lead_dim % self._num_threads) != 0:
        residue = dest_data_view.lead_dim - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          finial_offset = num_hops * self._num_threads
          shr_mem_addr = f'{thread_idx_x} + {finial_offset} + k * {dest_data_view.lead_dim}'
          glb_mem_addr = f'{thread_idx_x} + {finial_offset} + k * {src_data_view.lead_dim}'

          lhs = "{}[{}]".format(dest_name, glb_mem_addr)
          rhs = "{}[{}]".format(src_name,  shr_mem_addr)
          writer(self._vm.get_lexic().glb_store(lhs, rhs, self._dest._users == [self]))

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{s>g}}({self._src.name});'


class StoreRegToShrMemColumn(AbstractInstruction):
  def __init__(self,
               context: Context,
               dest: Symbol,
               src: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(StoreRegToShrMemColumn, self).__init__(context)

    #if src.stype != SymbolType.SharedMem:
    #  raise InternalError('store: operand `src` is not in shr mem.')

    #if dest.stype != SymbolType.Global:
    #  raise InternalError('store: operand `dest` is not in glb mem.')

    src.add_user(self)
    dest.add_user(self)

    self._dest = dest
    self._src = src
    self._alpha = alpha
    self._beta = beta
    self._num_threads = num_threads
    self._is_ready = True

  def gen_code(self, writer):
    dest_matrix = self._dest.obj
    dest_name = self._dest.name
    precision = self._vm.fp_as_str()

    with writer.If(self.gen_mask_threads(self._num_threads)):
      writer.Pragma("unroll")
      with writer.For(f'int k = 0; k < {dest_matrix.get_actual_num_rows()}; ++k'):
        rhs = "{}[{} * {} + k]".format(dest_name,
                                       self._vm.get_lexic().thread_idx_x,
                                       dest_matrix.num_rows)

        real_suffix = 'f' if precision == "float" else ''

        src_access = '' if self._src.obj.size == 1 else '[k]'
        if not isinstance(self._alpha, float):
          lhs = f'{self._alpha} * {self._src.name}{src_access}'
        else:
          if self._alpha == 1.0:
            lhs = f'{self._src.name}{src_access}'
          else:
            lhs = f'{self._alpha}{real_suffix} * {self._src.name}{src_access}'

        if not isinstance(self._beta, float):
          lhs += f' + {self._beta} * {rhs}'
        else:
          if self._beta != 0.0:
            if self._beta == 1.0:
              lhs += f' + {rhs}'
            else:
              const = f'{self._beta}{real_suffix}'
              lhs += f' + {const} * {rhs}'

        writer(f'{rhs} = {lhs};')

  def __str__(self) -> str:
    return 'not implemented'
