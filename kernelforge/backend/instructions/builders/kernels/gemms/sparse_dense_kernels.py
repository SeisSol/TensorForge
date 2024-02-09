from copy import deepcopy
from kernelforge.instructions.builders.alloctor_builder import RegistersAllocBuilder
from kernelforge.instructions.builders.ptr_manip_builder import GetElementPtrBuilder
from kernelforge.instructions.store import StoreRegToGlb, StoreRegToShrMemColumn, StoreShrMemToGlb
from kernelforge.instructions.sync_threads import SyncThreads
from kernelforge.matrix.sparse import SparseMatrix
from .base_kernel import BaseGemmKernelBuilder
from kernelforge.instructions.builders import ShrMemAllocBuilder
from kernelforge.instructions.builders import ShrMemBasedSparseDenseGemmBuilder
from kernelforge.instructions.builders import RegisterOnlySparseDenseGemmBuilder
from kernelforge.basic_types import ShrMemObject
import math


class ShrMemBasedSparseDenseGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is a class for building shared-memory-based gemm kernels.
  This type of gemm kernels perform well on Nvidia and AMD GPUs"""

  def __init__(self, **kwargs):
    super(ShrMemBasedSparseDenseGemmKernelBuilder, self).__init__(**kwargs)
    self._deduce_num_threads()

  def _deduce_num_threads(self):
    if self._trans_b:
      lead_dim_length = self._mat_b.get_actual_num_rows()
    else:
      lead_dim_length = self._mat_b.get_actual_num_cols()
    num_vector_units_required = math.ceil(lead_dim_length / self._hw_descr.vec_unit_length)
    self._num_compute_threads = lead_dim_length
    self._num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length
    return (self._num_compute_threads, self._num_active_threads)

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      if isinstance(symbol.obj, SparseMatrix) and symbol.obj.get_values() != None:
        continue
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_rows(), 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

  def build_epilogue(self):
    store_to_shr = StoreRegToShrMemColumn(self._vm,
                            self._symbol_table[self._mat_c],
                            self._symbol_table[self._reg_array_obj],
                            self._alpha,
                            self._beta,
                            self._num_compute_threads)
    self._instructions.append(store_to_shr)
    # insert sync here
    self._instructions.append(SyncThreads(self._vm, self._num_compute_threads))

    # We need to find the original glb_C symbol, as the matrix C is now shr_mem_1
    # and currently there is no support to save the original name of the pointer,
    # This might be necessary in future clean ups to clean
    c_glob = None
    l = self._symbol_table.all()
    i = 0
    for level in l:
      for obj, symbol in level.items():
        if symbol.name == "glb_C":
          c_glob = symbol
      i += 1

    store_to_glb = StoreShrMemToGlb(self._vm,
                            c_glob,
                            self._symbol_table[self._mat_c],
                            self._alpha,
                            self._beta,
                            self._num_compute_threads,
                            self._num_active_threads)
    self._instructions.append(store_to_glb)

  def build_kernel(self):
    if self._trans_b and self._mat_a.get_values() != None:
      builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
      builder.build(size=self._mat_c.num_rows * self._mat_c.num_cols)
      self._instructions.extend(builder.get_instructions())
      self._shr_mem_obj = builder.get_resultant_obj()
    else:
      builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
      builder.build(size=None)
      self._instructions.extend(builder.get_instructions())
      self._shr_mem_obj = builder.get_resultant_obj()

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
                  intermediate_dest=self._symbol_table[self._mat_c],
                  register_dest=self._symbol_table[self._reg_array_obj],
                  mat_a=self._mat_a,
                  beta=self._beta)

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())


class RegisterOnlySparseDenseGemmKernelBuilder(BaseGemmKernelBuilder):
  def __init__(self, **kwargs):
    raise Exception("TODO: the optimization strategy of sparse x dense might not work for register only")
    super(RegisterOnlySparseDenseGemmKernelBuilder, self).__init__(**kwargs)

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      if isinstance(symbol.obj, SparseMatrix) and symbol.obj.get_values() != None:
        continue
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_rows(), 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

  def build_kernel(self):
    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
    builder.build(size=None)
    self._instructions.extend(builder.get_instructions())
    self._shr_mem_obj = builder.get_resultant_obj()

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
