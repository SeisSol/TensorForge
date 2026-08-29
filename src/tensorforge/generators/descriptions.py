# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import numpy as np

from tensorforge.common.exceptions import GenerationError, InternalError
from tensorforge.common.context import Context
from tensorforge.common.basic_types import DataFlowDirection, Datatype
from tensorforge.common.operation import Operation, ReductionOperator
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.basic_types import Addressing

from typing import List

class OperationDescription:
  def barrier(self):
    return False

  def reads(self) -> List:
    """The views this operation reads, tensor-carrying ones only.

    A scalar constant is not a read of anything, so it does not appear here
    even where a descriptor accepts one as an operand.
    """
    return []

  def writes(self):
    """The view this operation writes, or None if it writes nothing."""
    return None

  def effective_boxes(self):
    """`(reads, write)` after any range narrowing, in tensor coordinates.

    `reads` maps tensor -> BoundingBox, `write` is the destination's box.
    `None` when the descriptor's shapes do not line up well enough to say.

    Declared and effective part company only where an operation derives its
    iteration range from its operands rather than from its destination, which
    is the contraction's case and nobody else's: a pointwise operation and a
    reduction both iterate exactly what they declare.  So this is the answer
    for everything except `MultilinearDescr`, which overrides it.
    """
    dest = self.writes()
    if dest is None:
      return None
    reads = {}
    for op in self.reads():
      tensor = getattr(op, 'tensor', None)
      if tensor is None:
        continue
      box = op.storage_box()
      prev = reads.get(tensor)
      reads[tensor] = box if prev is None else prev.unite(box)
    return reads, dest.storage_box()

class MultilinearDescr(OperationDescription):
  def __init__(self, dest: Tensor, ops: List[Tensor], target, permute, add: bool = False,
                strict_match: bool = False,
                prefer_align: bool = False):
    self.dest = dest
    self.ops = ops
    self.target = target
    self.permute = permute
    # `add` says whether this operation accumulates, and --- when it is a list
    # --- which of the destination's indices the tensor being added carries,
    # the way `target` does for an operand.  yateto states only the bool today,
    # so `True` means "all of them", which is what the destination read back
    # always has.  Keeping the two apart matters because a tensor with fewer
    # indices than the destination is a broadcast, and indexing it with the
    # ones it does not have reads somewhere else entirely.
    self.add_dims = list(add) if isinstance(add, (list, tuple)) else None
    self.add = bool(add) if not isinstance(add, (list, tuple)) else True

    self.dest.tensor.set_data_flow_direction(DataFlowDirection.SINK)
    for op in self.ops:
      op.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

  def _lead_dim(self):
    return self.dest.bbox.sizes()[0]

  def _analyze(self):
    pass

  def lead_width(self, context: Context) -> int:
    """How many adjacent lead-dimension elements one lane holds.

    Legal only where *every* matrix indexed by the lead dimension proves the
    alignment: the destination and any operand whose axis 0 is the lead
    dimension are all read and written through the same wide cast, so the
    weakest of them decides.  An operand that is not indexed by the lead
    dimension -- `B` in `C[m,n] += A[m,k] B[k,n]` -- is a broadcast and is
    splatted rather than loaded wide, so it does not constrain anything.
    """
    from tensorforge.backend.instructions.memory import vectorize
    if not vectorize.lead_vectorize_supported(context):
      return 1
    align = min([getattr(m.tensor, 'alignment', 0) or 0
                 for m in self.matrix_list()] or [0])
    fp = context.fp_type.size()
    return vectorize.lead_threads_and_width(
        self._lead_dim(), fp, align, blocking=vectorize.LEAD_BLOCKING)[1]

  def scalar_num_threads(self, context: Context) -> int:
    """The lane count this operator would have had without vectorisation.

    What `RegmaxBlockPolicy` has to divide by.  Sizing `mults_per_block` from
    the *reduced* lane count would double the mults, double the shared memory
    per block and halve the occupancy -- the whole win spent on memory.
    Dividing by the count the operator started with keeps the mults where
    they were and makes the block smaller instead, which is the arrangement
    that leaves blocks per SM unchanged or better.

    Computed by the same ladder `get_num_threads` uses rather than by a
    formula that looks equivalent: `context.align` rounds 20 up to 32 and 35
    up to 64, while the ladder caps at 32, and using the wrong one moves
    `mults_per_block` on every operator whose lead dimension is not a power
    of two.
    """
    return self._thread_ladder(context)

  def _thread_ladder(self, context: Context) -> int:
    num_threads = context.align(num=self._lead_dim())
    for cap in (32, 16, 8, 4, 2, 1):
      if self._lead_dim() <= cap:
        num_threads = cap
    return num_threads

  def get_num_threads(self, context: Context):
    from tensorforge.backend.instructions.memory import vectorize
    if vectorize.lead_vectorize_supported(context):
      fp = context.fp_type.size()
      align = min([getattr(m.tensor, 'alignment', 0) or 0
                   for m in self.matrix_list()] or [0])
      threads, width = vectorize.lead_threads_and_width(
          self._lead_dim(), fp, align,
          blocking=vectorize.LEAD_BLOCKING)
      if width > 1:
        # The extent still has to be covered: the loop bound is in elements
        # and the lane count is what it is divided by, so this returns the
        # *lane* count and the width travels separately.
        return threads, self._lead_dim()
    return self._thread_ladder(context), self._lead_dim()

  def matrix_list(self):
    return [self.dest] + [op for op in self.ops]

  def reads(self):
    return list(self.ops)

  def writes(self):
    return self.dest

  def effective_boxes(self):
    """The intersection `MultilinearInstruction._analyze` will perform.

    A declared operand box is an upper bound on what that operand contributes:
    `_analyze` intersects the boxes of everything sharing a target index, and
    the destination's, and iterates only that.  So an accumulation onto the
    whole destination from an operand spanning half of it writes half, and a
    read declared over the whole tensor from such an operand reads half.

    Comparing declared reads against actual writes therefore compares two
    different things, and the elastic ADER kernels are where that bites:
    `t += Q_face * c` declares the whole tensor on every term while each term
    covers only the rows its own face touches.  Replaying the intersection
    here is what makes both sides the same kind of statement.
    """
    ranges = {}

    def narrow(t, lo, hi):
      prev = ranges.get(t)
      ranges[t] = (max(prev[0], lo), min(prev[1], hi)) if prev else (lo, hi)

    ops = list(self.ops or [])
    targets = list(self.target or [])
    dest = self.dest
    if dest is None or len(ops) != len(targets):
      return None
    for op, target in zip(ops, targets):
      if getattr(op, 'bbox', None) is None or len(target) != op.bbox.rank():
        return None
      for j, t in enumerate(target):
        narrow(t, op.bbox.lower()[j], op.bbox.upper()[j])
    for j in range(dest.bbox.rank()):
      narrow(j, dest.bbox.lower()[j], dest.bbox.upper()[j])

    reads = {}
    for op, target in zip(ops, targets):
      tensor = getattr(op, 'tensor', None)
      if tensor is None:
        continue
      box = BoundingBox(
          [ranges[t][0] + op.offset[j] for j, t in enumerate(target)],
          [ranges[t][1] + op.offset[j] for j, t in enumerate(target)])
      prev = reads.get(tensor)
      reads[tensor] = box if prev is None else prev.unite(box)
    return reads, BoundingBox(
        [ranges[j][0] + dest.offset[j] for j in range(dest.bbox.rank())],
        [ranges[j][1] + dest.offset[j] for j in range(dest.bbox.rank())])

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

  def reads(self):
    return self.tensor_srcs()

  def writes(self):
    return self.dest

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

  def reads(self):
    return [self.var]

  def writes(self):
    return self.dest

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
