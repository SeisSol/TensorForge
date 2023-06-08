from gemmforge.instructions.builders.alloctor_builder import RegistersAllocBuilder
from gemmforge.instructions.builders.ptr_manip_builder import GetElementPtrBuilder
from gemmforge.instructions.store import StoreRegToGlb, StoreRegToGlbIterateByRow
from .base_kernel import BaseGemmKernelBuilder
from gemmforge.instructions.builders import ShrMemAllocBuilder
from gemmforge.instructions.builders import ShrMemBasedSparseDenseGemmBuilder
from gemmforge.instructions.builders import RegisterOnlySparseDenseGemmBuilder
from gemmforge.basic_types import ShrMemObject
import math


class ShrMemBasedSparseDenseGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is a class for building shared-memory-based gemm kernels.
  This type of gemm kernels perform well on Nvidia and AMD GPUs"""

  def __init__(self, **kwargs):
    super(ShrMemBasedSparseDenseGemmKernelBuilder, self).__init__(**kwargs)
    self._deduce_num_threads()

  def _deduce_num_threads(self):
    if self._trans_a:
      lead_dim_length = self._mat_a.get_actual_num_rows()
    else:
      lead_dim_length = self._mat_a.get_actual_num_cols()
    num_vector_units_required = math.ceil(lead_dim_length / self._hw_descr.vec_unit_length)
    self._num_compute_threads = lead_dim_length
    self._num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length
    return (self._num_compute_threads, self._num_active_threads)

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_rows(), 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

  def build_epilogue(self):
    store = StoreRegToGlbIterateByRow(self._vm,
                                      self._symbol_table[self._mat_c],
                                      self._symbol_table[self._reg_array_obj],
                                      self._alpha,
                                      self._beta,
                                      self._num_compute_threads)
    self._instructions.append(store)

  def build_kernel(self):
    # create shared mem
    if self._mat_a.get_values() == None:
      builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
      builder.build(size=None)
      self._instructions.extend(builder.get_instructions())
      self._shr_mem_obj = builder.get_resultant_obj()
    else:
      self._shr_mem_obj = ShrMemObject(name=None, size=0)

    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = ShrMemBasedSparseDenseGemmBuilder(self._vm,
                                                self._symbol_table,
                                                self._reg_array_obj,
                                                self._shr_mem_obj,
                                                self._num_active_threads)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj],
                  mat_a=self._mat_a)

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())


class RegisterOnlySparseDenseGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is a class for building default gemm kernels.
  This type of gemm kernels perform well on Nvidia and AMD GPUs"""

  def __init__(self, **kwargs):
    super(RegisterOnlySparseDenseGemmKernelBuilder, self).__init__(**kwargs)

  def build_kernel(self):
    # generate the rest instructions i.e., load to shr. mem, compute, store
    self._shr_mem_obj = ShrMemObject(name=None, size=0)
    builder = RegisterOnlySparseDenseGemmBuilder(self._vm,
                                                 self._symbol_table,
                                                 self._reg_array_obj,
                                                 self._shr_mem_obj,
                                                 self._num_active_threads)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj],
                  mat_a=self._mat_a)

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())
