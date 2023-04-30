from .dense_kernels import ShrMemBasedDenseGemmKernelBuilder
from .dense_kernels import RegisterOnlyDenseGemmKernelBuilder
from .dense_sparse_kernels import ShrMemBasedDenseSparseGemmKernelBuilder
from .dense_sparse_kernels import RegisterOnlyDenseSparseGemmKernelBuilder
from enum import Enum


class GemmKernelType(Enum):
  AUTO = 0
  SHR_MEM_BASED = 1
  REGISTER_ONLY_BASED = 2
  DENSE_SPARSE_SHR_MEM_BASED = 3
  DENSE_SPARSE_REGISTER_ONLY_BASED = 4

  @classmethod
  def to_str(cls, value):
    if value == "auto":
      return GemmKernelType.AUTO
    if value == "shr_mem":
      return GemmKernelType.SHR_MEM_BASED
    elif value == "register_only":
      return GemmKernelType.REGISTER_ONLY_BASED
    elif value == "dense_sparse_shr_mem":
      return GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED
    elif value == "dense_sparse_register_only":
      return GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED
    else:
      RuntimeError('unknown representation of gemm kernel type as `str`')


class GemmKernelsFactory:
  def __init__(self, **kwargs):
    print(kwargs)
    self._kwargs = kwargs
    self._vm = kwargs['vm']
    self._hw_descr = self._vm.get_hw_descr()
    self._gemm_kernel_type = kwargs['gemm_kernel_type']

    self._sparse_b = False
    if 'sparse_b' in kwargs:
      self._sparse_b = True
    self._sparse_a = False
    if 'sparse_a' in kwargs:
      self._sparse_a = True

  def _auto_select(self):
    model = self._hw_descr.model
    if model == 'pvc':
      if self._sparse_b:
        return GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED
      else:
        return GemmKernelType.REGISTER_ONLY_BASED
    else:
      if self._sparse_b:
        return GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED
      else:
        return GemmKernelType.SHR_MEM_BASED

  def get_builder(self):
    if self._gemm_kernel_type == GemmKernelType.AUTO:
      self._gemm_kernel_type = self._auto_select()

    if self._gemm_kernel_type == GemmKernelType.SHR_MEM_BASED:
      return ShrMemBasedDenseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.REGISTER_ONLY_BASED:
      return RegisterOnlyDenseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED:
      return RegisterOnlyDenseSparseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED:
      return ShrMemBasedDenseSparseGemmKernelBuilder(**self._kwargs)
    else:
      raise RuntimeError('unknown gemm type')

  def gemm_kernel_type(self):
    """returns a concrete kernel type. It is relevant if the user requested
    to perform auto-selection"""
    return self._gemm_kernel_type