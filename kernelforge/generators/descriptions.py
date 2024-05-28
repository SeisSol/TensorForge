from kernelforge.common.exceptions import GenerationError
from kernelforge.common.context import Context
from kernelforge.common.basic_types import DataFlowDirection, FloatingPointType
from kernelforge.backend.instructions.builders.kernels.gemms.type import GemmKernelType
from kernelforge.common.operation import Operation
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.basic_types import Addressing
from .optree import Assignment

from typing import List, Union

class OperationDescription:
  pass

class MultilinearDescr(OperationDescription):
  def __init__(self, dest: Tensor, ops: List[Tensor], target, permute, add: bool = False,
                strict_match: bool = False,
                prefer_align: bool = False):
    self.dest = dest
    self.ops = ops
    self.target = target
    self.permute = permute
    self._strict_match = False
    self.add = add

    self.dest.tensor.set_data_flow_direction(DataFlowDirection.SINK)
    for op in self.ops:
      op.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

  def _lead_dim(self):
    return self.dest.bbox.sizes()[0]
  
  def _analyze(self):
    pass

  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._lead_dim())
    return num_threads, self._lead_dim()

  def get_accumulator_size(self):
    accsize = 1
    for s in self.dest.bbox.sizes()[1:]:
      accsize *= s
    return accsize

  def is_strict_match(self):
    return self._strict_match
  
  def matrix_list(self):
    return [self.dest] + [op for op in self.ops]
  
  def __str__(self):
    desttarget = [i for i in range(self.dest.bbox.rank())]
    return f'{self.dest}{desttarget} = {"×".join(f"{op}{optarget}" for op, optarget in zip(self.ops, self.target))}'

class ElementwiseDescr(OperationDescription):
  def __init__(self, oplist: List[Assignment],
                strict_match: bool = False,
                prefer_align: bool = False):
    self.oplist = oplist
    self._strict_match = False

    for op in oplist:
      for tensor in op.tensors(outtensors=True, intensors=False):
        tensor.set_data_flow_direction(DataFlowDirection.SINK)
      for tensor in op.tensors(outtensors=False, intensors=True):
        tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    return vul, vul

  def get_accumulator_size(self):
    return 0

  def is_strict_match(self):
    return self._strict_match
  
  def matrix_list(self):
    return [tensor for op in self.oplist for tensor in op.tensors()]
  
  def __str__(self):
    return '; '.join(str(op) for op in self.oplist)

class GemmDescr(MultilinearDescr):
  def __init__(self,
               trans_a,
               trans_b,
               a,
               b,
               c,
               alpha=1.0,
               beta=0.0,
               kernel_type: GemmKernelType = GemmKernelType.AUTO,
               strict_match: bool = False,
               prefer_align: bool = False):
    # target_a = [-1, 0] if trans_a else [0, -1]
    # target_b = [1, -1] if trans_b else [-1, 0]
    target_a = [0, -1]
    target_b = [-1, 1]
    permute_a = [1, 0] if trans_a else [0, 1]
    permute_b = [1, 0] if trans_b else [0, 1]
    # assert beta == 0.0
    # super(GemmDescr, self).__init__(c, [a, b, alpha, beta], [target_a, target_b, [], []], strict_match, prefer_align)
    if alpha == 1.0:
      super(GemmDescr, self).__init__(c, [a, b], [target_a, target_b], [permute_a, permute_b], strict_match, prefer_align)
    else:
      alpha_tensor = Tensor([], Addressing.SCALAR, data=[alpha] if isinstance(alpha, (float, int)) else None)
      super(GemmDescr, self).__init__(c, [a, b, alpha_tensor], [target_a, target_b, []], [permute_a, permute_b, []], strict_match, prefer_align)

class ForDescr:
  pass

class IfDescr:
  def __init__(self, condition, subdescr):
    self.condition = condition
    self.subdescr = subdescr

class ConsecutiveDescr:
  pass
