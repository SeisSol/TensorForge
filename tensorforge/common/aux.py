from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.basic_types import Addressing, GeneralLexicon
from tensorforge.common.vm.vm import VM
from tensorforge.backend.symbol import Symbol
from typing import List

def generate_tmp_tensor(ops: List[SubTensor], target: List[List[int]], alias=None):
  rank = 0
  for itarget in target:
    for jtarget in itarget:
      rank = max(jtarget + 1, rank)
  shape = [0] * rank
  for i, itarget in enumerate(target):
    for j, jtarget in enumerate(itarget):
      if jtarget >= 0:
        shape[jtarget] = ops[i].bbox.sizes()[j]
  res = Tensor(shape=shape,
                    addressing=Addressing.PTR_BASED,
                    bbox=None,
                    is_tmp=True,
                    alias=alias)
  return res

def generate_tmp_matrix(op1: Tensor, op2: Tensor, trans_a: bool = False, trans_b: bool = False):
  target_a = [-1, 0] if trans_a else [0, -1]
  target_b = [1, -1] if trans_b else [-1, 1]
  return generate_tmp_tensor([op1, op2], [target_a, target_b])

def get_2d_block_id(vm: VM):
  return f'{vm.lexic.thread_idx_y} + {vm.lexic.block_dim_y} * {vm.lexic.block_idx_x}'


def get_extra_offset_name(symbol: Symbol):
  return f'{symbol.name}{GeneralLexicon.EXTRA_OFFSET}'
