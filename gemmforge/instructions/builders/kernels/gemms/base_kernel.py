from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.builders import GetElementPtrBuilder
from gemmforge.instructions.builders import RegistersAllocBuilder
from gemmforge.instructions import StoreRegToGlb
from abc import abstractmethod
import math

class BaseGemmKernelBuilder(AbstractBuilder):
  """ This is the base class for building complete gemm kernels."""

  def _deduce_num_threads(self):
    if self._trans_a:
      lead_dim_length = self._mat_a.get_actual_num_cols()
    else:
      lead_dim_length = self._mat_a.get_actual_num_rows()
    num_vector_units_required = math.ceil(lead_dim_length / self._hw_descr.vec_unit_length)
    self._num_compute_threads = lead_dim_length
    self._num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length
    return (self._num_compute_threads, self._num_active_threads)

  def __init__(self, **kwargs):
    super(BaseGemmKernelBuilder, self).__init__(kwargs['vm'], kwargs['symbol_table'])
    self._trans_a = kwargs['trans_a']
    self._trans_b = kwargs['trans_b']
    self._mat_a = kwargs['mat_a']
    self._mat_b = kwargs['mat_b']
    self._mat_c = kwargs['mat_c']
    self._alpha = kwargs['alpha']
    self._beta = kwargs['beta']
    self._hw_descr = kwargs['hw_descr']
    self._deduce_num_threads()

    self._reg_array_obj = None
    self._shr_mem_obj = None
    self._shr_mem_loads = []

  def get_reg_array_obj(self):
    return self._reg_array_obj

  def get_shr_mem_obj(self):
    return self._shr_mem_obj

  def get_shr_mem_loads(self):
    return self._shr_mem_loads

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_cols(), 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

  @abstractmethod
  def build_kernel(self):
    pass

  def build_epilogue(self):
    store = StoreRegToGlb(self._vm,
                          self._symbol_table[self._mat_c],
                          self._symbol_table[self._reg_array_obj],
                          self._alpha,
                          self._beta,
                          self._num_compute_threads)
    self._instructions.append(store)


  def build(self):
    self.build_prologue()
    self.build_kernel()
    self.build_epilogue()