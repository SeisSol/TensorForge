from ..abstract_thread_policy import AbstractGemmLikeThreadPolicy, DenseMatrix
from gemmforge.vm import VM


class GenericGemmThreadPolicy(AbstractGemmLikeThreadPolicy):
  def __init__(self,
               vm: VM,
               shr_mem_per_op: int,
               num_threads: int,
               op1: DenseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(vm, num_threads, op1, op2, res)
    self._shr_mem_per_op = shr_mem_per_op

  def _estimate_num_registers_per_mult(self, accumulator_length):
    # Note: derived experimentally
    factor = self._vm.bytes_per_real() / 4
    return factor * (self._vm._hw_descr.vec_unit_length + accumulator_length)

  def get_num_ops_per_block(self):
    accumulator_length = self._res.get_actual_num_cols()
    max_num_regs_per_thread = self._estimate_num_registers_per_mult(accumulator_length)

    hw_descr = self._vm.get_hw_descr()
    shr_mem_bytes = self._shr_mem_per_op * self._vm.bytes_per_real()
    mults_wrt_shr_mem = hw_descr.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = hw_descr.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))

    return max(int(mults_per_sm / hw_descr.max_block_per_sm), 1)
