from ..abstract_thread_policy import AbstractGemmLikeThreadPolicy, DenseMatrix
from kernelforge.vm import VM


class OnlyRegisterBasedThreadPolicy(AbstractGemmLikeThreadPolicy):
  def __init__(self,
               vm: VM,
               num_threads: int,
               op1: DenseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(vm, num_threads, op1, op2, res)

  def _estimate_num_registers_per_mult(self, accumulator_length):
    # Note: derived experimentally
    factor = self._vm.bytes_per_real() / 4
    return factor * (32 + accumulator_length)

  def get_num_ops_per_block(self):

    accumulator_length = self._res.get_actual_num_cols()
    max_num_regs_per_thread = self._estimate_num_registers_per_mult(accumulator_length)

    hw_descr = self._vm.get_hw_descr()
    mults_per_sm = hw_descr.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)

    return max(int(mults_per_sm / hw_descr.max_block_per_sm), 1)
