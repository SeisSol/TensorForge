from kernelforge.common.matrix import Matrix, DenseMatrix
from kernelforge.common.basic_types import Addressing, GeneralLexicon
from kernelforge.common.vm import VM
from kernelforge.backend.symbol import Symbol


def generate_tmp_matrix(op1: Matrix, op2: Matrix, trans_op1: bool = False, trans_op2: bool = False):
  m = op1.get_actual_num_cols() if trans_op1 else op1.get_actual_num_rows()
  n = op2.get_actual_num_rows() if trans_op2 else op2.get_actual_num_cols()
  res = DenseMatrix(num_rows=m,
                    num_cols=n,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, m, n],
                    is_tmp=True)
  return res


def get_2d_block_id(vm: VM):
  return f'{vm.lexic.thread_idx_y} + {vm.lexic.block_dim_y} * {vm.lexic.block_idx_x}'


def get_extra_offset_name(symbol: Symbol):
  return f'{symbol.name}{GeneralLexicon.EXTRA_OFFSET}'
