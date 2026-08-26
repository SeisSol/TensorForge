# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import numpy as np

from tensorforge.common.exceptions import GenerationError, InternalError
from tensorforge.common.context import Context
from tensorforge.common.basic_types import DataFlowDirection, Datatype
from tensorforge.common.operation import Operation, ReductionOperator
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.basic_types import Addressing

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
  """One scalar operation applied pointwise: ``dest = op(*srcs)``.

  Previously this carried a list of ``optree.Assignment``, i.e. an expression
  tree, and unified the iteration space across all of them via
  ``Assignment.getRanges``.  That unification keyed ranges by negative integers
  and asserted ``-i-1 in ranges``, a convention the test harness needed twenty
  lines of prose to reproduce.

  With a single operation the iteration space is not derived at all: elementwise
  means every tensor operand has the destination's shape, so the space *is*
  ``dest.bbox``.  Compound expressions become several instructions over
  temporaries, which has the side benefit that the intermediate is a ``Symbol``
  the allocator can see -- an optree ``TempVar`` was a writer-allocated name
  invisible to every pass.
  """

  # Derived from the optree helpers that used to build these nodes, so the
  # arity a caller may pass is checked rather than discovered at emit time.
  UNARY = frozenset({
      Operation.ABS, Operation.ACOS, Operation.ACOSH, Operation.ASIN,
      Operation.ASINH, Operation.ATAN, Operation.ATANH, Operation.CBRT,
      Operation.COS, Operation.COSH, Operation.EXP, Operation.EXPM1,
      Operation.LOG, Operation.LOGP1, Operation.NEG, Operation.NOT,
      Operation.RCBRT, Operation.RCP, Operation.ROUND, Operation.RSQRT,
      Operation.SIN, Operation.SINH, Operation.SQRT, Operation.TAN,
      Operation.TANH, Operation.COPY, Operation.CEIL, Operation.FLOOR,
      Operation.SIGN, Operation.TRUNC, Operation.ERF, Operation.GAMMA,
  })
  BINARY = frozenset({
      Operation.ADD, Operation.AND, Operation.DIV, Operation.EQ, Operation.GE,
      Operation.GT, Operation.LE, Operation.LT, Operation.MAX, Operation.MIN,
      Operation.MOD, Operation.MUL, Operation.NEQ, Operation.OR, Operation.POW,
      Operation.SUB, Operation.SHL, Operation.SHR, Operation.SHRS,
      Operation.XOR,
  })

  def __init__(self,
               op: Operation,
               dest,
               srcs: List,
               strict_match: bool = False,
               prefer_align: bool = False):
    self.op = op
    self.dest = dest
    # A source is either a tensor or a scalar constant.  `pow_int` is the only
    # current user of the latter.
    self.srcs = list(srcs)
    self.strict_match = strict_match
    self.prefer_align = prefer_align

    expected = 1 if op in self.UNARY else 2 if op in self.BINARY else None
    if expected is None:
      raise InternalError(f'elementwise: unknown arity for {op}')
    if len(self.srcs) != expected:
      raise InternalError(
          f'elementwise: {op.name} takes {expected} operand(s), '
          f'got {len(self.srcs)}')

    dest.tensor.set_data_flow_direction(DataFlowDirection.SINK)
    for src in self.tensor_srcs():
      src.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

    for src in self.tensor_srcs():
      if list(src.bbox.sizes()) != list(dest.bbox.sizes()):
        raise InternalError(
            f'elementwise: operand shape {list(src.bbox.sizes())} does not '
            f'match destination {list(dest.bbox.sizes())}; a shape-changing '
            f'operation is not elementwise')

  def tensor_srcs(self) -> List:
    return [s for s in self.srcs if not isinstance(s, (int, float, np.integer,
                                                      np.floating))]

  def scalar_srcs(self) -> List:
    return [s for s in self.srcs if isinstance(s, (int, float, np.integer,
                                                  np.floating))]

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    return vul, vul

  def matrix_list(self):
    # Sources first, destination last.  Operand *order* here determines the
    # launcher's parameter order via Generator._name_operands, and the old
    # optree path yielded Assignment.tensors() = inputs ++ outputs.  Putting
    # dest first would silently rotate the kernel ABI.
    return self.tensor_srcs() + [self.dest]

  @staticmethod
  def _name(x):
    inner = getattr(x, 'tensor', None)
    return getattr(inner, 'alias', None) or getattr(x, 'alias', None) or str(x)

  def __str__(self):
    args = ', '.join(self._name(s) for s in self.srcs)
    return f'{self._name(self.dest)} = {self.op.name.lower()}({args})'

class ReductionDescr(OperationDescription):
  """``dest = reduce(op, var, dims)`` -- deliberately *not* an ElementwiseDescr.

  A reduction changes shape: its iteration space is the source's, while the
  destination's is that minus ``dims``.  Folding it into ElementwiseDescr would
  require re-introducing exactly the range unification that was just removed,
  so it keeps its own descriptor carrying ``op`` and ``dims``.
  """

  def __init__(self, dest, var, dims: List[int], op: ReductionOperator,
               prefer_align: bool = False):
    self.dest = dest
    self.var = var
    self.dims = list(dims)
    self.op = op
    self.prefer_align = prefer_align

    rank = var.bbox.rank()
    for d in self.dims:
      if not 0 <= d < rank:
        raise InternalError(
            f'reduction: axis {d} out of range for a rank-{rank} operand')
    if len(set(self.dims)) != len(self.dims):
      raise InternalError(f'reduction: repeated axis in {self.dims}')

    kept = [var.bbox.size(i) for i in range(rank) if i not in self.dims]
    got = list(dest.bbox.sizes())
    # a full reduction may legitimately land in a rank-1 buffer of size 1
    if got not in (kept, [1]) and kept != []:
      raise InternalError(
          f'reduction: destination shape {got} does not match the source '
          f'shape {list(var.bbox.sizes())} with axes {self.dims} removed '
          f'({kept})')

    var.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)
    dest.tensor.set_data_flow_direction(DataFlowDirection.SINK)

  def get_num_threads(self, context: Context):
    vul = context.get_vm().get_hw_descr().vec_unit_length
    return vul, vul

  def matrix_list(self):
    return [self.var, self.dest]

  @staticmethod
  def _name(x):
    inner = getattr(x, 'tensor', None)
    return getattr(inner, 'alias', None) or getattr(x, 'alias', None) or str(x)

  def __str__(self):
    return (f'{self._name(self.dest)} = {self.op}'
            f'({self._name(self.var)}, dims={self.dims})')

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
    # Transposition belongs in `target`, not in `permute`.
    #
    # `target[i][j]` says which loop index dimension j of operand i carries:
    # >= 0 is an output index, < 0 a contraction index.  That mapping is what
    # MultilinearInstruction._analyze reads to build the loop ranges --- and it
    # reads it *without* consulting `permute`.  Encoding a transpose only in
    # `permute` therefore left `_analyze` pairing the wrong dimensions: for
    # `trans_a` with a non-square operand it took the output extent for the
    # contraction extent, so the sum ran over the wrong length and the result
    # came out short by whatever the two dimensions differed by.
    #
    # This is also the convention everything else already uses:
    # `generate_tmp_matrix` writes `[-1, 0] if trans_a`, and yateto's
    # `factory.getIndices` derives `target` from the index letters while
    # emitting `permute` as the identity throughout.
    target_a = [-1, 0] if trans_a else [0, -1]
    target_b = [1, -1] if trans_b else [-1, 1]
    # `permute` still carries the transpose for the staging decisions
    # (GlbToShrLoader's `is_transpose`); deriving those from `target` too is a
    # separate step.
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
          # Shape `()`, not `(1,)`: the tensor is `[]`, and `value()` indexes
          # it with the empty tuple.
          data=(np.array(alpha, dtype=float)
                if isinstance(alpha, (float, int)) else None),
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
