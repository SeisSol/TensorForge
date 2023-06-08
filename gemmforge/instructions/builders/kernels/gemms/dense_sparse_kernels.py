from .base_kernel import BaseGemmKernelBuilder
from gemmforge.instructions.builders import ShrMemAllocBuilder
from gemmforge.instructions.builders import ShrMemBasedDenseSparseGemmBuilder
from gemmforge.instructions.builders import RegisterOnlyDenseSparseGemmBuilder
from gemmforge.basic_types import ShrMemObject


class ShrMemBasedDenseSparseGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is a class for building shared-memory-based gemm kernels.
  This type of gemm kernels perform well on Nvidia and AMD GPUs"""

  def __init__(self, **kwargs):
    super(ShrMemBasedDenseSparseGemmKernelBuilder, self).__init__(**kwargs)

  def build_kernel(self):
    if not self._trans_a and self._mat_b.get_values() != None:
      self._shr_mem_obj = ShrMemObject(name=None, size=0)
    else:
      builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
      builder.build(size=None)
      self._instructions.extend(builder.get_instructions())
      self._shr_mem_obj = builder.get_resultant_obj()

    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = ShrMemBasedDenseSparseGemmBuilder(self._vm,
                                                self._symbol_table,
                                                self._reg_array_obj,
                                                self._shr_mem_obj,
                                                self._num_active_threads)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj],
                  mat_b=self._mat_b)

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())


class RegisterOnlyDenseSparseGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is a class for building default gemm kernels.
  This type of gemm kernels perform well on Nvidia and AMD GPUs"""

  def __init__(self, **kwargs):
    super(RegisterOnlyDenseSparseGemmKernelBuilder, self).__init__(**kwargs)

  def build_kernel(self):
    # generate the rest instructions i.e., load to shr. mem, compute, store
    self._shr_mem_obj = ShrMemObject(name=None, size=0)
    builder = RegisterOnlyDenseSparseGemmBuilder(self._vm,
                                                 self._symbol_table,
                                                 self._reg_array_obj,
                                                 self._shr_mem_obj,
                                                 self._num_active_threads)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj],
                  mat_b=self._mat_b)

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())
