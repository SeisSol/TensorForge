from gemmforge.vm import VM
from ..matrix import DenseMatrix
from .gemm.nvidia import NvidiaGemmThreadPolicy
from .csa.nvidia import NvidiaCsaThreadPolicy


class TheadPolicyFactory:
  ALLOWED_MANUFACTURES = ['nvidia', 'amd', 'intel']

  def __init__(self):
    pass

  @classmethod
  def get_gemm_policy(cls,
                      vm: VM,
                      reals_per_op: int,
                      num_threads: int,
                      op1: DenseMatrix,
                      op2: DenseMatrix,
                      res: DenseMatrix):
    default_policy = NvidiaGemmThreadPolicy(vm,
                                            reals_per_op,
                                            num_threads,
                                            op1,
                                            op2,
                                            res)
    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')

  @classmethod
  def get_csa_policy(cls,
                     vm: VM,
                     num_threads: int,
                     op1: DenseMatrix,
                     op2: DenseMatrix):
    default_policy = NvidiaCsaThreadPolicy(vm, num_threads, op1, op2)
    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')
