from tensorforge.common.exceptions import GenerationError
from tensorforge.common.context import Context
from tensorforge.common.basic_types import DataFlowDirection, Datatype
from tensorforge.common.operation import Operation
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.basic_types import Addressing
from .optree import Assignment

from typing import List

class OperationDescription:
  def barrier(self):
    return False

class MultilinearDescr(OperationDescription):
  def __init__(self, dest: Tensor, ops: List[Tensor], target, permute, add: bool = False,
                strict_match: bool = False,
                prefer_align: bool = False):
    self.dest = dest
    self.ops = ops
    self.target = target
    self.permute = permute
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

  def matrix_list(self):
    return [self.dest] + [op for op in self.ops]

  def __str__(self):
    desttarget = [i for i in range(self.dest.bbox.rank())]
    return f'{self.dest}{desttarget} {"+" if self.add else ""}= {"×".join(f"{op}{optarget}" for op, optarget in zip(self.ops, self.target))}'

class ElementwiseDescr(OperationDescription):
  def __init__(self, oplist: List[Assignment],
                strict_match: bool = False,
                prefer_align: bool = False):
    self.oplist = oplist

    for op in oplist:
      for tensor in op.tensors(outtensors=True, intensors=False):
        tensor.tensor.set_data_flow_direction(DataFlowDirection.SINK)
      for tensor in op.tensors(outtensors=False, intensors=True):
        tensor.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    vul = 64 # FIXME:
    return vul, vul

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
    self.op = op

    var.set_data_flow_direction(DataFlowDirection.SOURCE)
    dest.set_data_flow_direction(DataFlowDirection.SINK)

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    return vul, vul

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
    add = True if beta == 1 else False

    assert beta in (0, 1)

    if alpha == 1.0:
      super(GemmDescr, self).__init__(c, [a, b], [target_a, target_b], [permute_a, permute_b], add, strict_match, prefer_align)
    else:
      # Inherit datatype from the destination so the synthetic scalar
      # always has a concrete type. Without this, Symbol.get_fptype()
      # raised in every alpha != 1 case (see Symbol.get_fptype docstring).
      dest_dtype = getattr(c.tensor, 'datatype', None)
      alpha_tensor = SubTensor(Tensor(
          [], Addressing.SCALAR,
          data=[alpha] if isinstance(alpha, (float, int)) else None,
          datatype=dest_dtype,
      ))
      super(GemmDescr, self).__init__(c, [a, b, alpha_tensor], [target_a, target_b, []], [permute_a, permute_b, []], add, strict_match, prefer_align)

class ForDescr:
  pass

class IfDescr:
  def __init__(self, condition, subdescr):
    self.condition = condition
    self.subdescr = subdescr

  def __str__(self):
    return f'if ({self.condition}): {self.subdescr}'

class ConsecutiveDescr:
  pass

class BarrierDescription(OperationDescription):
  def barrier(self):
    return True

  def trueBarrier(self):
    return False

  def matrix_list(self):
    return []

  def get_num_threads(self, ctx):
    return 32, 32

class GridFenceDescr(BarrierDescription):
  def __str__(self):
    return 'fence'

  def trueBarrier(self):
    return False

class GridBarrierDescr(BarrierDescription):
  def __str__(self):
    return 'barrier'

  def trueBarrier(self):
    return True

class RegionDescription(OperationDescription):
  def __init__(self, name):
    self.name = name

  def matrix_list(self):
    return []

  def get_num_threads(self, ctx):
    return 32, 32

  def __str__(self):
    return f'region "{self.name}"'
