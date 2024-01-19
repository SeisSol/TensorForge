from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
import math

def round_up_to_nearest_vec_length(n, vec_length):
    return math.ceil(n / vec_length) * vec_length
  
class StoreRegToGlb(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               src: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(StoreRegToGlb, self).__init__(vm)

    if dest.stype != SymbolType.Global:
      raise InternalError('store: operand `dest` is not in glb mem.')

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not a register obj')

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
      with writer.For(f'int n = 0; n < {dest_matrix.get_actual_num_cols()}; ++n'):
        rhs = "{}[{} + {} * n]".format(dest_name,
                                       self._vm.get_lexic().thread_idx_x,
                                       dest_matrix.num_rows)

        real_suffix = 'f' if precision == "float" else ''

        src_access = '' if self._src.obj.size == 1 else '[n]'
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


class StoreShrMemToGlb(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               src: Symbol,
               alpha: float,
               beta: float,
               num_compute_threads: int,
               num_active_threads: int):
    super(StoreShrMemToGlb, self).__init__(vm)

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

  def gen_code(self, writer):
    dest_matrix = self._dest.obj
    
    dest_name = self._dest.name
    src_name = self._src.name
    precision = self._vm.fp_as_str()
    vec_unit_length = self._vm._hw_descr.vec_unit_length
    #nearest_multiple_of_vec_unti_length = round_up_to_nearest_vec_length(n=self._num_threads, vec_length=vec_unit_length)

    thread_idx_x = self._vm.get_lexic().thread_idx_x
    num_hops = int(dest_matrix.get_actual_num_rows() / self._num_threads)

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view

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

          rhs = "{}[{}]".format(dest_name, glb_mem_addr)
          lhs = "{}[{}]".format(src_name,  shr_mem_addr)
          writer(f'{rhs} = {lhs};')

      # the last hop to fill shared mem with data
      if (dest_data_view.lead_dim % self._num_threads) != 0:
        residue = dest_data_view.lead_dim - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          finial_offset = num_hops * self._num_threads
          shr_mem_addr = f'{thread_idx_x} + {finial_offset} + k * {dest_data_view.lead_dim}'
          glb_mem_addr = f'{thread_idx_x} + {finial_offset} + k * {src_data_view.lead_dim}'

          rhs = "{}[{}]".format(dest_name, glb_mem_addr)
          lhs = "{}[{}]".format(src_name,  shr_mem_addr)
          writer(f'{rhs} = {lhs};')

  def __str__(self) -> str:
    return 'not implemented'


class StoreRegToShrMemColumn(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               src: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(StoreRegToShrMemColumn, self).__init__(vm)

    #if src.stype != SymbolType.SharedMem:
    #  raise InternalError('store: operand `src` is not in shr mem.')

    #if dest.stype != SymbolType.Global:
    #  raise InternalError('store: operand `dest` is not in glb mem.')

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
