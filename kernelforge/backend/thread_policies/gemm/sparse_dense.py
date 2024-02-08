from gemmforge.thread_policies.gemm.generic import GenericGemmThreadPolicy
from ..abstract_thread_policy import AbstractGemmLikeThreadPolicy, DenseMatrix, SparseMatrix
from gemmforge.vm import VM


class GenericSparseDenseGemmThreadPolicy(AbstractGemmLikeThreadPolicy):
  def __init__(self,
               vm: VM,
               shr_mem_per_op: int,
               num_threads: int,
               op1: SparseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(vm, num_threads, op1, op2, res)
    self._shr_mem_per_op = shr_mem_per_op
    self._g = GenericGemmThreadPolicy(vm, shr_mem_per_op, num_threads, op1, op2, res)

  def _estimate_num_registers_per_mult(self, accumulator_length):
    return self._g._estimate_num_registers_per_mult(accumulator_length)

  def get_num_ops_per_block(self):
    accumulator_length = self._res.get_actual_num_rows()
    hwfactor = 1.0
    if accumulator_length < self._vm._hw_descr.vec_unit_length:
        hwfactor = float(int(self._vm._hw_descr.vec_unit_length / accumulator_length))
    max_num_regs_per_thread = self._estimate_num_registers_per_mult(accumulator_length)

    hw_descr = self._vm.get_hw_descr()
    shr_mem_bytes = self._shr_mem_per_op * self._vm.bytes_per_real()
    mults_wrt_shr_mem = hw_descr.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = hw_descr.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))

    # Note the factor is experimentally derived, but the results indicate
    # the factor does not have a huge impact on the performance
    el_count_if_dense = self._op1.get_actual_num_cols() * self._op1.get_actual_num_rows()
    el_count_if_sparse = self._op1.get_el_count()
    factor = (el_count_if_dense / el_count_if_sparse)
    factor /= 2.0
    # To prevent using too much shared memory or threads per thread block
    # If the matrices are too sparse
    if factor > 1.00/0.15:
      factor = 1.00/0.15
    if factor < 1.00:
      factor = 1.0
    factor_int = int(hwfactor*factor + 0.25)
    return max(int(factor_int * mults_per_sm / hw_descr.max_block_per_sm), 1)


class SparseDenseOnlyRegisterBasedThreadPolicy(AbstractGemmLikeThreadPolicy):
  def __init__(self,
               vm: VM,
               num_threads: int,
               op1: SparseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(vm, num_threads, op1, op2, res)

  def _estimate_num_registers_per_mult(self, accumulator_length):
    # Note: derived experimentally
    factor = self._vm.bytes_per_real() / 4
    return factor * (self._vm._hw_descr.vec_unit_length  + accumulator_length)

  def get_num_ops_per_block(self):
    accumulator_length = self._op1.get_actual_num_max_nonzero_rows()
    max_num_regs_per_thread = self._estimate_num_registers_per_mult(accumulator_length)

    hw_descr = self._vm.get_hw_descr()
    mults_per_sm = hw_descr.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)

    return max(int(mults_per_sm / hw_descr.max_block_per_sm), 1)
