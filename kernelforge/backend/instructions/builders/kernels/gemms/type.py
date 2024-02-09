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