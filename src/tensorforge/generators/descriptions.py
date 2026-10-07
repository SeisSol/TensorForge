# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import numpy as np

from tensorforge.common.exceptions import GenerationError, InternalError
from tensorforge.common.basic_types import DataFlowDirection
from tensorforge.common.operation import Operation, ReductionOperator
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.basic_types import Addressing

from typing import List, Optional

class GuardLiteral:
  """One term of a guard: a rank-0 condition tensor, taken or negated.

  `version` distinguishes successive values of the same tensor. yateto writes
  a condition tensor and may write it again later, and two literals over the
  same tensor at different versions are different values -- without the
  version they would compare equal and a guard could be simplified away
  against a value it never had.
  """

  __slots__ = ('tensor', 'version', 'negated')

  def __init__(self, tensor, version=0, negated=False):
    self.tensor = tensor
    self.version = version
    self.negated = negated
    # A condition is read, whether or not an operation also reads it: a
    # tensor only a guard names would otherwise have no direction, and its
    # parameter would come out writable where every other operand the kernel
    # only reads is `const`.
    # Written as well elsewhere in the kernel, it becomes SOURCESINK.
    underlying = getattr(tensor, 'tensor', tensor)
    if hasattr(underlying, 'set_data_flow_direction'):
      underlying.set_data_flow_direction(DataFlowDirection.SOURCE)

  def key(self):
    return (id(getattr(self.tensor, 'tensor', self.tensor)),
            self.version, self.negated)

  def __str__(self):
    name = getattr(getattr(self.tensor, 'tensor', None), 'alias', None) or str(self.tensor)
    return f'{"!" if self.negated else ""}{name}@{self.version}'


class OperationDescription:
  #: The guard this operation runs under: a conjunction of `GuardLiteral`,
  #: or None for one that always runs. An empty conjunction is also always.
  #:
  #: A field rather than a wrapping descriptor, because everything that walks
  #: a section -- `SectionPlan`, residency, the temporaries -- asks each
  #: descriptor for its geometry through `reads`, `writes` and
  #: `effective_boxes` and deliberately does not know what kinds there are. A
  #: wrapper would have to be unwrapped by every one of those walks, and a
  #: walk that forgot would read a guarded write as an unconditional one,
  #: which is the failure that leaves no trace. Grouping neighbors that share
  #: a guard is then a pass over the list, which is where it belongs.
  condition = None

  def guarded(self):
    """Whether this operation runs under a guard that is not always true."""
    return bool(self.condition)

  def condition_reads(self) -> List:
    """The views the guard reads.

    Kept apart from `reads()`: those are the operation's operands and a
    builder resolves them as such, while these are read to decide whether the
    operation runs at all. A section still has to stage them, so whatever
    walks the section has to see both.
    """
    return [literal.tensor for literal in (self.condition or ())]

  def barrier(self):
    return False

  def operations(self) -> List:
    """The operations this descriptor stands for, itself by default.

    A descriptor that holds others -- a loop over a body -- is not the unit
    whose geometry can be asked for: it reads and writes whatever its
    iterations do, and `writes()` has one destination to give.  So whoever
    needs the operations asks for them and gets them expanded, rather than
    learning which kinds contain which.
    """
    return [self]

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

#: Index letters for `summary`: output axes, then summed ones.
_OUT_INDICES = 'ijlmnoqr'
_SUM_INDICES = 'kpstuvwxyz'


def _ints(values):
  return [int(v) for v in values]


def operand_name(view) -> str:
  """The name a view's tensor has in the kernel (`m0`, `t1`, `v0`)."""
  tensor = getattr(view, 'tensor', None)
  if tensor is None:
    return str(view)
  return str(tensor.name or tensor.alias)


def _slice_note(view) -> str:
  """`@{..}` for the part of its tensor a view takes, or '' for all of it."""
  tensor = view.tensor
  try:
    whole = tensor.get_bbox()
    lower = [l + o for l, o in zip(view.bbox.lower(), view.offset)]
    upper = [u + o for u, o in zip(view.bbox.upper(), view.offset)]
    if _ints(lower) == _ints(whole.lower()) and _ints(upper) == _ints(whole.upper()):
      return ''
  except (AttributeError, TypeError):
    return ''
  return '@' + '×'.join(f'{{{l}..{u}}}' for l, u in zip(_ints(lower), _ints(upper)))


def view_dict(view, data=False, pack=False) -> dict:
  """One operand as data: the tensor it views and the part it takes.

  The fields `tools/host/dump_descriptors.py` writes, so that its files and
  the kernel's `tensorforge-meta` line describe an operand alike.
  `data` and `pack` are the values and the storage order, which are large and
  asked for only where they are needed.
  """
  t = view.tensor
  out = dict(name=t.name or t.alias, alias=t.alias,
             shape=_ints(t.shape),
             ashape=_ints(t.get_actual_shape()),
             tbbox=[_ints(t.bbox.lower()), _ints(t.bbox.upper())],
             bbox=[_ints(view.bbox.lower()), _ints(view.bbox.upper())],
             offset=_ints(view.offset),
             addressing=str(t.addressing),
             is_tmp=bool(t.is_tmp),
             storage=int(t.storage_volume()),
             sliced=bool(getattr(view, 'sliced', False)))
  if pack:
    out['pack'] = (list(t.storage_map()) if t.storage_map() is not None
                   else None)
  if data:
    out['data'] = (t.data.tolist() if getattr(t.data, 'tolist', None)
                   else (list(t.data) if t.data is not None else None))
  return out


def _index_letters(rank, targets):
  """A letter per index: the output's first, in its order, then the rest."""
  letters, out, summed = {}, iter(_OUT_INDICES), iter(_SUM_INDICES)
  ids = sorted({i for t in targets for i in t} | set(range(rank)),
               key=lambda i: (not 0 <= i < rank, i if i >= 0 else -i))
  for i in ids:
    pool = out if 0 <= i < rank else summed
    letters[i] = next(pool, None) or f'x{len(letters)}'
  return letters


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

  def _analyze(self):
    pass

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

  def summary(self) -> str:
    """Index notation with the kernel's names: `m2[i,j] += m0[i,k] × m1[k,j]`.

    `__str__` stays as it is -- `tools/host/parse_generated.py` reads it back
    out of older kernels -- and this is what the kernel's comment block shows:
    which index is summed, and which part of a tensor a view takes, where
    `__str__` spells every box in full and every index as a number.
    """
    rank = self.dest.bbox.rank()
    letters = _index_letters(rank, self.target)
    def term(view, idx):
      if not hasattr(view, 'tensor'):
        return str(view)
      inner = ','.join(letters[i] for i in idx)
      return f'{operand_name(view)}[{inner}]{_slice_note(view)}'
    dest = term(self.dest, range(rank))
    ops = ' × '.join(term(op, t) for op, t in zip(self.ops, self.target))
    return f'{dest} {"+=" if self.add else "="} {ops}'

  def to_dict(self, data=False, pack=False) -> dict:
    keep = [(o, t, p) for o, t, p in zip(self.ops, self.target, self.permute)
            if hasattr(o, 'tensor')]
    return dict(kind='multilinear',
                dest=view_dict(self.dest, data, pack),
                ops=[view_dict(o, data, pack) for o, _, _ in keep],
                target=[_ints(t) for _, t, _ in keep],
                permute=[_ints(p) for _, _, p in keep],
                add=bool(self.add))

class ElementwiseDescr(OperationDescription):
  """One scalar operation applied pointwise: ``dest = op(*srcs)``.

  With a single operation the iteration space is not derived at all: elementwise
  means every tensor operand has the destination's shape, so the space *is*
  ``dest.bbox``.  Compound expressions become several instructions over
  temporaries, which has the side benefit that the intermediate is a ``Symbol``
  the allocator can see -- a writer-allocated name would be invisible to every
  pass.

  That is the form the builder takes.  A frontend may state more, and
  `generators.legalize` rewrites it into that form:

  ``target``
      For each operand, the destination axes its own axes are, in its own
      order, as `MultilinearDescr.target` states them; a negative number is
      an axis the destination does not carry.  An operand may then carry
      fewer of the destination's axes, or the same ones in another order,
      and its box marks where it can be non-zero -- it reads zero outside.
      ``None`` is every operand on the destination's axes, in its shape.
  ``add``
      Add onto the destination rather than overwrite it, as
      `MultilinearDescr.add`.
  ``alpha``
      A factor without axes, the result is scaled by.
  """

  # The arity of each operation, so that what a caller passes is checked here
  # rather than discovered at emit time.
  UNARY = frozenset({
      Operation.ABS, Operation.ACOS, Operation.ACOSH, Operation.ASIN,
      Operation.ASINH, Operation.ATAN, Operation.ATANH, Operation.CBRT,
      Operation.COS, Operation.COSH, Operation.EXP, Operation.EXPM1,
      Operation.LOG, Operation.LOG1P, Operation.NEG, Operation.NOT,
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
  #: yes, no, condition -- in that order, as yateto sends `where`.
  TERNARY = frozenset({Operation.SELECT})

  def __init__(self,
               op: Operation,
               dest,
               srcs: List,
               strict_match: bool = False,
               prefer_align: bool = False,
               target: Optional[List] = None,
               add=False,
               alpha=None):
    self.op = op
    self.dest = dest
    # A source is either a tensor or a scalar constant.
    self.srcs = list(srcs)
    self.strict_match = strict_match
    self.prefer_align = prefer_align
    self.target = None if target is None else [
        None if axes is None else list(axes) for axes in target]
    self.add_dims = list(add) if isinstance(add, (list, tuple)) else None
    self.add = bool(add) if not isinstance(add, (list, tuple)) else True
    self.alpha = alpha

    expected = (1 if op in self.UNARY else 2 if op in self.BINARY
                else 3 if op in self.TERNARY else None)
    if expected is None:
      raise InternalError(f'elementwise: unknown arity for {op}')
    if len(self.srcs) != expected:
      raise InternalError(
          f'elementwise: {op.name} takes {expected} operand(s), '
          f'got {len(self.srcs)}')
    if self.target is not None:
      if len(self.target) != len(self.srcs):
        raise InternalError(
            f'elementwise: {len(self.target)} axis lists for '
            f'{len(self.srcs)} operand(s)')
      for src, axes in zip(self.srcs, self.target):
        if hasattr(src, 'bbox') and len(axes or []) != src.bbox.rank():
          raise InternalError(
              f'elementwise: {len(axes or [])} axes stated for an operand of '
              f'rank {src.bbox.rank()}')

    if not self.legal():
      # What it stands for is built and states the data flow
      # (`generators.legalize`).
      return

    dest.tensor.set_data_flow_direction(DataFlowDirection.SINK)
    for src in self.tensor_srcs():
      src.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)

    for src in self.tensor_srcs():
      # An operand without axes is a broadcast: read with an empty index at
      # every point, like a constant source but from memory.
      if src.bbox.rank() and list(src.bbox.sizes()) != list(dest.bbox.sizes()):
        raise InternalError(
            f'elementwise: operand shape {list(src.bbox.sizes())} does not '
            f'match destination {list(dest.bbox.sizes())}; a shape-changing '
            f'operation is not elementwise')

  def legal(self) -> bool:
    """Whether the builder takes this as it is: operands in the
    destination's shape, overwriting it, unscaled."""
    return self.target is None and not self.add and self.alpha is None

  def target_of(self, position: int) -> List[int]:
    """The destination axes operand `position` is stated on, in its own
    axis order."""
    if self.target is not None and self.target[position] is not None:
      return list(self.target[position])
    src = self.srcs[position]
    return list(range(src.bbox.rank())) if hasattr(src, 'bbox') else []

  def add_mask(self):
    """`add` as it was stated: a mask over the destination's axes, or a
    bool."""
    return self.add_dims if self.add_dims is not None else self.add

  def tensor_srcs(self) -> List:
    return [s for s in self.srcs if not isinstance(s, (int, float, np.integer,
                                                      np.floating))]

  def scalar_srcs(self) -> List:
    return [s for s in self.srcs if isinstance(s, (int, float, np.integer,
                                                  np.floating))]

  def reads(self):
    return self.tensor_srcs()

  def writes(self):
    return self.dest

  def matrix_list(self):
    # Sources first, destination last.  Operand *order* here determines the
    # launcher's parameter order via Generator._name_operands, so putting dest
    # first would silently rotate the kernel ABI.
    return self.tensor_srcs() + [self.dest]

  @staticmethod
  def _name(x):
    inner = getattr(x, 'tensor', None)
    return getattr(inner, 'alias', None) or getattr(x, 'alias', None) or str(x)

  def __str__(self):
    args = ', '.join(self._name(s) for s in self.srcs)
    return f'{self._name(self.dest)} = {self.op.name.lower()}({args})'

  def summary(self) -> str:
    return str(self)

  def to_dict(self, data=False, pack=False) -> dict:
    """Every operand has the destination's shape, so every axis lines up."""
    srcs = self.tensor_srcs()
    axes = list(range(len(self.dest.bbox.sizes())))
    return dict(kind='elementwise',
                op=self.op.name,
                dest=view_dict(self.dest, data, pack),
                ops=[view_dict(o, data, pack) for o in srcs],
                target=[list(axes) if o.bbox.rank() else [] for o in srcs],
                permute=[list(axes) if o.bbox.rank() else [] for o in srcs],
                scalars=[float(v) for v in self.scalar_srcs()],
                add=False)

class ReductionDescr(OperationDescription):
  """``dest = reduce(op, var, dims)`` -- deliberately *not* an ElementwiseDescr.

  A reduction changes shape: its iteration space is the source's, while the
  destination's is that minus ``dims``.  Folding it into ElementwiseDescr would
  require unifying ranges across operands, which a pointwise operation does
  without, so it keeps its own descriptor carrying ``op`` and ``dims``.
  """

  def __init__(self, dest, var, dims: List[int], op: ReductionOperator,
               prefer_align: bool = False, add=False, alpha=None):
    self.dest = dest
    self.var = var
    self.dims = list(dims)
    self.op = op
    self.prefer_align = prefer_align
    #: Add onto the destination rather than overwrite it, and the factor the
    #: result is scaled by -- what the builder does not take, and
    #: `generators.legalize` rewrites (`ElementwiseDescr`).
    self.add = add is not False and add is not None
    self.alpha = alpha

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

    if not self.legal():
      return
    var.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)
    dest.tensor.set_data_flow_direction(DataFlowDirection.SINK)

  def legal(self) -> bool:
    """Whether the builder takes this as it is: overwriting, unscaled."""
    return not self.add and self.alpha is None

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

  def summary(self) -> str:
    return str(self)

  def to_dict(self, data=False, pack=False) -> dict:
    """`dims` are axes of the operand; the ones that survive keep their
    order, so the operand maps onto the destination in order with the reduced
    axes numbered negative, the way a contraction states it."""
    kept, contracted = [], -1
    for axis in range(self.var.bbox.rank()):
      if axis in self.dims:
        kept.append(contracted)
        contracted -= 1
      else:
        kept.append(len([a for a in kept if a >= 0]))
    return dict(kind='reduction',
                op=str(self.op),
                dest=view_dict(self.dest, data, pack),
                ops=[view_dict(self.var, data, pack)],
                target=[kept],
                permute=[list(range(self.var.bbox.rank()))],
                add=False)

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
    # `permute` would therefore leave `_analyze` pairing the wrong dimensions:
    # for `trans_a` with a non-square operand it would take the output extent
    # for the contraction extent, so the sum would run over the wrong length
    # and the result come out short by whatever the two dimensions differ by.
    #
    # This is also the convention everything else uses:
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
    # A multilinear either overwrites its destination or adds onto it, so
    # beta has two values it can express.  Any other scales C before the
    # product is added, which is an operation of its own ahead of this one.
    if beta not in (0, 1):
      raise GenerationError(
          f'GemmDescr: beta = {beta} is neither 0 (overwrite C) nor 1 (add '
          f'onto C); scale C with a MultilinearDescr of its own first.')
    add = beta == 1

    if alpha == 1.0:
      super(GemmDescr, self).__init__(c, [a, b], [target_a, target_b], [permute_a, permute_b], add, strict_match, prefer_align)
    else:
      # Inherit datatype from the destination so the synthetic scalar
      # always has a concrete type. Without this, Symbol.get_fptype()
      # would raise in every alpha != 1 case (see Symbol.get_fptype docstring).
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

class BarrierDescription(OperationDescription):
  def barrier(self):
    return True

  def trueBarrier(self):
    return False

  def matrix_list(self):
    return []

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

  def __str__(self):
    return f'region "{self.name}"'


def contraction_lengths(descrs) -> List[int]:
  """The extent of every contracted axis of a list, a merged run written
  out: one per multilinear operand axis that the destination does not have
  (`target` -1)."""
  out = []
  for d in (op for descr in descrs for op in descr.operations()):
    target = getattr(d, 'target', None)
    if target is None or len(getattr(d, 'ops', ())) < 2:
      continue
    for op, axes in zip(d.ops, target):
      box = getattr(op, 'bbox', None)
      if box is None:
        continue
      for axis, t in enumerate(axes):
        if t == -1 and axis < box.rank():
          out.append(int(box.size(axis)))
  return out
