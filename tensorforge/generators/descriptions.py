from tensorforge.common.exceptions import GenerationError
from tensorforge.common.context import Context
from tensorforge.common.basic_types import DataFlowDirection, FloatingPointType
from tensorforge.common.operation import Operation
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.basic_types import Addressing
from .optree import Assignment

from typing import List

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
    if self._lead_dim() <= 32:
      num_threads = 32
    if self._lead_dim() <= 16:
      num_threads = 16
    if self._lead_dim() <= 8:
      num_threads = 8
    if self._lead_dim() <= 4:
      num_threads = 4
    if self._lead_dim() <= 2:
      num_threads = 2
    if self._lead_dim() <= 1:
      num_threads = 1
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
        tensor.tensor.set_data_flow_direction(DataFlowDirection.SINK)
      for tensor in op.tensors(outtensors=False, intensors=True):
        tensor.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    vul = 64 # FIXME:
    return vul, vul

  def get_accumulator_size(self):
    return 0

  def is_strict_match(self):
    return self._strict_match
  
  def matrix_list(self):
    return [tensor for op in self.oplist for tensor in op.tensors()]
  
  def __str__(self):
    return '; '.join(str(op) for op in self.oplist)

class ElementwiseDescrSingular(ElementwiseDescr):
  def __init__(self, dest: Tensor, ops: List[Tensor]):
    super().__init__(oplist=[Assignment(dest, )])

class ReductionDescr(OperationDescription):
  def __init__(self, dest: Tensor, var: Tensor, dims: List[int], op):
    self.dest = dest
    self.var = var
    self._strict_match = False

    var.set_data_flow_direction(DataFlowDirection.SOURCE)
    dest.set_data_flow_direction(DataFlowDirection.SINK)

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    return vul, vul

  def get_accumulator_size(self):
    return 0

  def is_strict_match(self):
    return self._strict_match
  
  def matrix_list(self):
    return [self.var, self.dest]
  
  def __str__(self):
    return f'{self.var} -> {self.dest}'

class GemmDescr(MultilinearDescr):
  def __init__(self,
               trans_a,
               trans_b,
               a,
               b,
               c,
               alpha=1.0,
               beta=0.0,
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
      alpha_tensor = SubTensor(Tensor([], Addressing.SCALAR, data=[alpha] if isinstance(alpha, (float, int)) else None))
      super(GemmDescr, self).__init__(c, [a, b, alpha_tensor], [target_a, target_b, []], [permute_a, permute_b, []], strict_match, prefer_align)

class ForDescr:
  pass

class IfDescr:
  def __init__(self, condition, subdescr):
    self.condition = condition
    self.subdescr = subdescr

class ConsecutiveDescr:
  pass
