from kernelforge.common.vm.vm import VM
from ..matrix import DenseMatrix
from .gemm.generic import GenericGemmThreadPolicy
from .gemm.only_register_based import OnlyRegisterBasedThreadPolicy
from .csa.generic import GenericCsaThreadPolicy


class TheadPolicyFactory:
  ALLOWED_MANUFACTURES = ['nvidia', 'amd', 'intel']

  def __init__(self):
    pass

  @classmethod
  def get_gemm_policy(cls,
                      vm: VM,
                      shr_mem_per_op: int,
                      num_threads: int,
                      op1: DenseMatrix,
                      op2: DenseMatrix,
                      res: DenseMatrix):

    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      if shr_mem_per_op == 0:
        return OnlyRegisterBasedThreadPolicy(vm,
                                             num_threads,
                                             op1,
                                             op2,
                                             res)
      else:
        return GenericGemmThreadPolicy(vm,
                                       shr_mem_per_op,
                                       num_threads,
                                       op1,
                                       op2,
                                       res)
    else:
      raise RuntimeError('unknown manufacturer')

  @classmethod
  def get_csa_policy(cls,
                     vm: VM,
                     num_threads: int,
                     op1: DenseMatrix,
                     op2: DenseMatrix):
    default_policy = GenericCsaThreadPolicy(vm, num_threads, op1, op2)
    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')
