from kernelforge.common.exceptions import GenerationError
from kernelforge.common.context import Context
from kernelforge.common.basic_types import DataFlowDirection, FloatingPointType
from kernelforge.backend.instructions.builders.kernels.gemms.type import GemmKernelType
from kernelforge.common.operation import Operation
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.basic_types import Addressing

from typing import List, Union

class OperationDescription:
  pass


class GemmDescr(OperationDescription):
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
    self.trans_a = trans_a
    self.trans_b = trans_b
    self.mat_a = a
    self.mat_a.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_b = b
    self.mat_b.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_c = c
    self.mat_c.set_data_flow_direction(DataFlowDirection.SINK)

    self.alpha = alpha
    self.beta = beta

    self.kernel_type = kernel_type

    self._m = None
    self._n = None
    self._k = None
    self._strict_match = strict_match
    self.prefer_align = prefer_align

    self._check()
    self._analyze()

  def _analyze(self):
    if self.trans_a:
      self._m = self.mat_a.get_actual_num_cols()
      self._k = self.mat_a.get_actual_num_rows()
    else:
      self._m = self.mat_a.get_actual_num_rows()
      self._k = self.mat_a.get_actual_num_cols()

    if self.trans_b:
      self._n = self.mat_b.get_actual_num_rows()
    else:
      self._n = self.mat_b.get_actual_num_cols()

  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._m)
    return num_threads, self._m

  def get_accumulator_size(self):
    return self._n

  def is_strict_match(self):
    return self._strict_match

  def __str__(self):
    suffix_a = '^T' if self.trans_a else ''
    suffix_b = '^T' if self.trans_b else ''
    op1 = f'{self.alpha} * {self.mat_a}{suffix_a} x {self.mat_b}{suffix_b}'
    op2 = '' if self.beta == 0 else f' + {self.beta} * {self.mat_c}'
    return f'{self.mat_c} = {op1}{op2}'

  def _check(self):
    try:
      # check whether C and A match each other
      if self.trans_a:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (NoTrans) do not match")

      # check whether C and B match each other
      if self.trans_b:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (NoTrans) do not match")

      # the contraction length of A and B can be different
      # due to the fact that matrices in a matrix chain can be aligned
      # in a slightly different way e.g., see yateto
      if self._strict_match:
        # check whether A and B match each other
        if self.trans_a:
          if self.trans_b:
            if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_cols():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (Trans) and B (Trans) do not match")
          else:
            if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_rows():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (Trans) and B (NoTrans) do not match")
        else:
          if self.trans_b:
            if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (NoTrans) and B (Trans) do not match")
          else:
            if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (NoTrans) and B (NoTrans) do not match")

    except GenerationError as err:
      print(self.mat_a.gen_descr())
      print(self.mat_b.gen_descr())
      print(self.mat_c.gen_descr())
      raise err

  def compute_flops(self):
    flops = (2 * self._k - 1) * self._m * self._n
    if self.beta != 0:
      flops += self._m * self._n
    return flops
  
  def matrix_list(self):
    return (self.mat_a, self.mat_b, self.mat_c)

class CSADescr(OperationDescription):
  def matrix_list(self):
    return (self.mat_a, self.mat_c)

  def __init__(self,
               trans_a,
               a,
               c,
               alpha=1.0,
               beta=0.0,
               strict_match: bool = False,
               prefer_align: bool = False):

    self.trans_a = trans_a
    self.trans_b = None

    self.mat_a = a
    self.mat_a.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_c = c
    self.mat_c.set_data_flow_direction(DataFlowDirection.SINK)

    self.alpha = alpha
    self.beta = beta

    self._m = None
    self._n = None
    self._strict_match = strict_match
    self.prefer_align = prefer_align

    self._check()
    self._analyze()

  def _analyze(self):
    if self.trans_a:
      self._m = self.mat_a.get_actual_num_cols()
      self._n = self.mat_a.get_actual_num_rows()
    else:
      self._m = self.mat_a.get_actual_num_rows()
      self._n = self.mat_a.get_actual_num_cols()

  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._m)
    return num_threads, self._m

  def get_accumulator_size(self):
    return self._n

  def is_strict_match(self):
    return self._strict_match

  def __str__(self):
    suffix_a = '^T' if self.trans_a else ''
    op1 = f'{self.alpha} * {self.mat_a}{suffix_a}'
    op2 = '' if self.beta == 0 else f' + {self.beta} * {self.mat_c}'
    return f'{self.mat_c} = {op1}{op2}'

  def _check(self):
    try:
      # check whether C and A match each other
      if self.trans_a:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_cols():
          raise GenerationError("Cannot generate an assignment "
                                "with given parameters. Matrix C and A (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_rows():
          raise GenerationError("Cannot generate an assignment "
                                "with given parameters. Matrix C and A (NoTrans) do not match")

      # check whether C and A match each other (second round)
      if self.trans_a:
        if self.mat_c.get_actual_num_cols() != self.mat_a.get_actual_num_rows():
          raise GenerationError("Cannot generate an assignment "
                                "with given parameters. Matrix C and A (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_cols() != self.mat_a.get_actual_num_cols():
          raise GenerationError("Cannot generate an assignment "
                                "with given parameters. Matrix C and A (NoTrans) do not match")

    except GenerationError as err:
      print(self.mat_a.gen_descr())
      print(self.mat_c.gen_descr())
      raise err

  def compute_flops(self):
    aflops = self._m * self._n
    cflops = 0
    if self._beta != 0:
      cflops = self._m * self._n
    return aflops * cflops

class PointwiseDescr(OperationDescription):
  def __init__(self,
               trans_a,
               trans_b,
               a,
               b,
               c,
               alpha=1.0,
               beta=0.0,
               operation = Operation,
               strict_match: bool = False,
               prefer_align: bool = False):
    self.trans_a = trans_a
    self.trans_b = trans_b
    self.mat_a = a
    self.mat_a.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_b = b
    self.mat_b.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_c = c
    self.mat_c.set_data_flow_direction(DataFlowDirection.SINK)

    self.alpha = alpha
    self.beta = beta

    self.operation = operation

    self._m = None
    self._n = None
    self._strict_match = strict_match
    self.prefer_align = prefer_align

    self._check()
    self._analyze()

  def _analyze(self):
    if self.trans_a:
      self._m = self.mat_a.get_actual_num_cols()
    else:
      self._m = self.mat_a.get_actual_num_rows()

    if self.trans_b:
      self._n = self.mat_b.get_actual_num_rows()
    else:
      self._n = self.mat_b.get_actual_num_cols()

  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._m)
    return num_threads, self._m

  def get_accumulator_size(self):
    return self._n

  def is_strict_match(self):
    return self._strict_match

  def __str__(self):
    suffix_a = '^T' if self.trans_a else ''
    suffix_b = '^T' if self.trans_b else ''
    op1 = f'{self.alpha} * {self.mat_a}{suffix_a} x {self.mat_b}{suffix_b}'
    op2 = '' if self.beta == 0 else f' + {self.beta} * {self.mat_c}'
    return f'{self.mat_c} = {op1}{op2}'

  def _check(self):
    try:
      # check whether C and A match each other
      if self.trans_a:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (NoTrans) do not match")

      # check whether C and B match each other
      if self.trans_b:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (NoTrans) do not match")

      # the contraction length of A and B can be different
      # due to the fact that matrices in a matrix chain can be aligned
      # in a slightly different way e.g., see yateto
      if self._strict_match:
        # check whether A and B match each other
        if self.trans_a:
          if self.trans_b:
            if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_cols():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (Trans) and B (Trans) do not match")
          else:
            if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_rows():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (Trans) and B (NoTrans) do not match")
        else:
          if self.trans_b:
            if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (NoTrans) and B (Trans) do not match")
          else:
            if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (NoTrans) and B (NoTrans) do not match")

    except GenerationError as err:
      print(self.mat_a.gen_descr())
      print(self.mat_b.gen_descr())
      print(self.mat_c.gen_descr())
      raise err

  def compute_flops(self):
    flops = (2 * self._k - 1) * self._m * self._n
    if self.beta != 0:
      flops += self._m * self._n
    return flops
  
  def matrix_list(self):
    return (self.mat_a, self.mat_b, self.mat_c)

class PointwiseAssociative:
  pass

class PointwiseAdd(PointwiseAssociative):
  ASSOCIATIVE = True
  NEUTRAL = 0

class PointwiseMul(PointwiseAssociative):
  ASSOCIATIVE = True
  NEUTRAL = 1



class PointwiseDescr2:
  def __init__(self, C, As, op):
    self._C = C
    self._As = As
    self._op = op

    self._m = self._C.shape[0]
    self._n = self._C.shape[1]

    self._check()
  
  def _check(self):
    for A in self._As:
      assert self._C.get_actual_shape() == A.get_actual_shape()
  
  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._m)
    return num_threads, self._m

  def get_accumulator_size(self):
    return self._n

  def is_strict_match(self):
    return self._strict_match
  
  def matrix_list(self):
    return [self._C] + [A for A in self._As]

class ReductionDescr:
  def __init__(self, C, A, op):
    self._C = C
    self._A = A

class MultilinearDescr:
  def __init__(self, dest: Tensor, ops: List[Tensor], target, permute, add: bool = False,
                strict_match: bool = False,
                prefer_align: bool = False):
    self.dest = dest
    self.ops = ops
    self.target = target
    self.permute = permute
    self._strict_match = False
    self.add = add

    self.dest.set_data_flow_direction(DataFlowDirection.SINK)
    for op in self.ops:
      op.set_data_flow_direction(DataFlowDirection.SOURCE)

  def _lead_dim(self):
    return self.dest.get_actual_shape()[0]
  
  def _analyze(self):
    pass

  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._lead_dim())
    return num_threads, self._lead_dim()

  def get_accumulator_size(self):
    accsize = 1
    for s in self.dest.get_actual_shape()[1:]:
      accsize *= s
    return accsize

  def is_strict_match(self):
    return self._strict_match
  
  def matrix_list(self):
    return [self.dest] + [op for op in self.ops]
  
  def __str__(self):
    destdim = len(self.dest.shape)
    desttarget = [i for i in range(destdim)]
    return f'{self.dest}{desttarget} = {"×".join(f"{op}{optarget}" for op, optarget in zip(self.ops, self.target))}'

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
  pass

class ConsecutiveDescr:
  pass
