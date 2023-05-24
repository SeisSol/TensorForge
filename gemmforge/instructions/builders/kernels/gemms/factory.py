from .dense_kernels import ShrMemBasedDenseGemmKernelBuilder
from .dense_kernels import RegisterOnlyDenseGemmKernelBuilder
from .dense_sparse_kernels import ShrMemBasedDenseSparseGemmKernelBuilder
from .dense_sparse_kernels import RegisterOnlyDenseSparseGemmKernelBuilder
from .sparse_dense_kernels import ShrMemBasedSparseDenseGemmKernelBuilder
from .sparse_dense_kernels import RegisterOnlySparseDenseGemmKernelBuilder
from gemmforge.matrix import SparseMatrix, DenseMatrix
from enum import Enum


class GemmKernelType(Enum):
  AUTO = 0
  SHR_MEM_BASED = 1
  REGISTER_ONLY_BASED = 2
  DENSE_SPARSE_SHR_MEM_BASED = 3
  DENSE_SPARSE_REGISTER_ONLY_BASED = 4
  SPARSE_DENSE_SHR_MEM_BASED = 5
  SPARSE_DENSE_REGISTER_ONLY_BASED = 6

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
    elif value == "sparse_dense_shr_mem":
      return GemmKernelType.SPARSE_DENSE_SHR_MEM_BASED
    elif value == "sparse_dense_register_only":
      return GemmKernelType.SPARSE_DENSE_REGISTER_ONLY_BASED
    else:
      RuntimeError('unknown representation of gemm kernel type as `str`')


class GemmKernelsFactory:
  def __init__(self, **kwargs):
    self._kwargs = kwargs
    self._vm = kwargs['vm']
    self._hw_descr = self._vm.get_hw_descr()
    self._gemm_kernel_type = kwargs['gemm_kernel_type']

    self._mat_a = kwargs['mat_a']
    self._mat_b = kwargs['mat_b']
    self._sparse_b = False
    self._sparse_a = False
    if isinstance(self._mat_b, SparseMatrix):
      self._sparse_b = True
    if isinstance(self._mat_b, DenseMatrix):
      self._sparse_a = True

  def _auto_select(self):
    model = self._hw_descr.model
    both_sparse_error = "Gemmforge does not support both matrix A and B being sparse"
    if model == 'pvc':
      if self._sparse_a and self._sparse_b:
        raise Exception(both_sparse_error)
      elif self._sparse_b:
        return GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED
      elif self._sparse_a:
        return GemmKernelType.SPARSE_DENSE_REGISTER_ONLY_BASED
      else:
        return GemmKernelType.REGISTER_ONLY_BASED
    else:
      if self._sparse_a and self._sparse_b:
        raise Exception(both_sparse_error)
      elif self._sparse_b:
        return GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED
      elif self._sparse_a:
        return GemmKernelType.SPARSE_DENSE_SHR_MEM_BASED
      else:
        return GemmKernelType.SHR_MEM_BASED

  def get_builder(self):
    if self._gemm_kernel_type == GemmKernelType.AUTO:
      self._gemm_kernel_type = self._auto_select()

    if self._gemm_kernel_type == GemmKernelType.SHR_MEM_BASED:
      return ShrMemBasedDenseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.REGISTER_ONLY_BASED:
      return RegisterOnlyDenseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED:
      return ShrMemBasedDenseSparseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED:
      return RegisterOnlyDenseSparseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.SPARSE_DENSE_SHR_MEM_BASED:
      return ShrMemBasedSparseDenseGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.SPARSE_DENSE_REGISTER_ONLY_BASED:
      return RegisterOnlySparseDenseGemmKernelBuilder(**self._kwargs)
    else:
      raise RuntimeError('unknown gemm type')

  def gemm_kernel_type(self):
    """returns a concrete kernel type. It is relevant if the user requested
    to perform auto-selection"""
    return self._gemm_kernel_type