# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Several consecutive elements of the batch in one multiplication
(`Options.stack_elements`).

An operation of the batch is the same for every element: the operators the
batch shares are read by all of them, and what an element owns -- its degrees
of freedom, its flux solver -- is read and written by it alone.  So `k`
consecutive elements computed together are one operation over operands with
one more axis: every operand an element owns gains a last axis of extent `k`,
the stack, and every one the batch shares stays as it is.  A product against
a shared operator then comes out `k` times wider -- `Q[56x9] = A[56x56] x
Q[56x9]` becomes `Q[56x9x2] = A[56x56] x Q[56x9x2]`, the same instruction
stream over a wider right-hand side -- and a product between two operands an
element owns becomes `k` independent ones side by side, which is what packed
arithmetic takes two at a time.

This module states that as a rewrite of the operation list, the only place
the stack is visible to the builders: they see wider tensors and build them
as they build any.  What the stack means for memory is the traversal's and
the addressing's business:

* the batch loop visits a stack per trip, `ceil(numElements / k)` of them
  (`count`), and binds for each element of the stack whether it exists and
  is not masked (`BatchLoop`);
* an operand of the stack is reached through a pointer per element, and the
  last index of an access picks the pointer instead of adding to the address
  (`ptr_manip.GetElementPtr`, `Symbol.build_address`): an element owns its
  memory, and two elements' memory is not one tensor with a stride between
  them -- under `Addressing.PTR_BASED` it is two allocations;
* a write runs under its element's mask, so a stack the batch ends inside,
  or an element its flag masks, writes nothing for it (`Symbol.store`).

A pointwise operation and a reduction take the stack as one more axis of
their images (`_Stacker.elementwise`, `_Stacker.reduction`).

What is not stacked is refused here, by name, rather than built wrong: an
operation other than a contraction, a pointwise operation or a reduction, a
guard (a condition an element owns would have to hold per element of the
stack), an operand an element owns without axes, sparse or in a storage
order of its own, a destination the batch shares, and a full reduction into
a buffer of one.
"""

from typing import Dict, List

from tensorforge.common.basic_types import Addressing, GeneralLexicon
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor

from .descriptions import (BarrierDescription, ElementwiseDescr,
                           MultilinearDescr, OperationDescription,
                           ReductionDescr, RegionDescription)

#: The stacks a multiplication may compute.  Two elements is what a 16-wide
#: matrix instruction or a packed pair of lanes takes at once, and the
#: register cost grows with every element added.
STACKS = (1, 2)


def per_element(tensor) -> bool:
  """Whether every element of the batch has its own `tensor`: strided or
  through a pointer of its own, the two addressings that take the element
  index.  The batch-constant ones (`NONE`) and a scalar passed by value
  (`SCALAR`) are the batch's."""
  return getattr(tensor, 'addressing', None) in (Addressing.STRIDED,
                                                 Addressing.PTR_BASED)


def factor(tensor) -> int:
  """How many elements `tensor`'s last axis holds side by side: its stack,
  1 for a tensor that is not stacked."""
  return int(getattr(tensor, 'stack', 1) or 1)


def count(section_index: int, k: int) -> str:
  """The trips the batch loop of a section makes over `numElements` elements
  in stacks of `k`: `ceil(numElements / k)`, as text, which is where the
  element count lives (a kernel parameter)."""
  name = f'{GeneralLexicon.NUM_ELEMENTS}{section_index}'
  if k == 1:
    return name
  return f'(({name} + {k - 1}) / {k})'


def mask_key(e: int) -> str:
  """What the batch loop publishes the mask of element `e` of its stack under
  (`BatchLoop.batch_indices`), for the element pointers that pick by it."""
  return f'stackMask{e}'


def stack(descrs: List[OperationDescription], k: int) -> List[OperationDescription]:
  """`descrs` with `k` consecutive elements computed together.

  The identity for `k == 1`, and for a list already stacked -- a generator
  built again from the list another one was built from (`Generator._sibling`)
  is handed the rewritten list, and stacking it twice would stack the stack.
  """
  if k not in STACKS:
    raise GenerationError(
        f'stack_elements = {k}: a multiplication stacks one or two elements '
        f'(one of {", ".join(map(str, STACKS))})')
  if k == 1 or any(factor(t) > 1 for t in _tensors(descrs)):
    return descrs
  stacker = _Stacker(k)
  return [stacker.descr(d) for d in descrs]


def _tensors(descrs):
  for d in descrs:
    for op in d.operations():
      for view in ([op.writes()] if op.writes() is not None else []) + list(op.reads()):
        tensor = getattr(view, 'tensor', None)
        if tensor is not None:
          yield tensor


class _Stacker:
  def __init__(self, k: int):
    self.k = k
    #: id of a tensor -> its stacked twin.  One twin per tensor, whichever
    #: operation names it first: a tensor two operations share is one buffer,
    #: and two twins of it would be two.
    self.tensors: Dict[int, Tensor] = {}

  def refuse(self, what: str) -> GenerationError:
    return GenerationError(f'stack_elements = {self.k}: {what}')

  def tensor(self, t: Tensor) -> Tensor:
    twin = self.tensors.get(id(t))
    if twin is not None:
      return twin
    name = t.alias or t.name
    if t.rank0:
      # The stack would be its only axis, and the lanes are spread over the
      # first: the elements of the stack would land on different lanes, where
      # every other operand holds all of them in each lane.
      raise self.refuse(f'{name} is one number per element; stacked it would '
                        f'spread the stack over the lanes')
    if not t.is_dense():
      raise self.refuse(f'{name} is sparse, and an element owns it')
    if t.storage_order is not None or t.storage_parts != 1:
      raise self.refuse(f'{name} is stored in an order of its own')
    if getattr(t, 'passed_by_value', False):
      raise self.refuse(f'{name} is passed by value, and an element owns it')
    lower, upper = list(t.bbox.lower()), list(t.bbox.upper())
    twin = Tensor(list(t.shape) + [self.k], t.addressing,
                  BoundingBox(lower + [0], upper + [self.k]),
                  alias=t.alias, is_tmp=t.is_tmp, datatype=t.datatype,
                  alignment=t.alignment, residence=t.residence)
    #: The stack, on the twin: its last axis is `k` elements of the batch,
    #: each in memory of its own (`factor`).
    twin.stack = self.k
    # Which way the data flows, as the tensor says now.  The descriptions
    # rebuilt over the twins set what their constructors set -- a
    # destination is a sink -- and a direction given after construction is
    # not theirs to know: `add=True` promotes its destination to read *and*
    # written that way, and a twin that was only a sink went unfilled while
    # the kernel accumulated onto it (NaN on sm_120 and gfx1150 alike).
    if t.direction is not None:
        twin.set_data_flow_direction(t.direction)
    self.tensors[id(t)] = twin
    return twin

  def view(self, v):
    tensor = getattr(v, 'tensor', None)
    if tensor is None or not per_element(tensor):
      return v
    return SubTensor(self.tensor(tensor),
                     BoundingBox(list(v.bbox.lower()) + [0],
                                 list(v.bbox.upper()) + [self.k]),
                     offset=list(v.offset) + [0], sliced=v.sliced)

  def descr(self, d: OperationDescription) -> OperationDescription:
    if isinstance(d, (BarrierDescription, RegionDescription)):
      return d
    if not isinstance(d, (MultilinearDescr, ElementwiseDescr,
                          ReductionDescr)):
      raise self.refuse(f'{type(d).__name__} `{d}` is no operation a stack '
                        f'takes')
    if d.guarded():
      raise self.refuse(f'`{d}` runs under a guard, which an element decides '
                        f'for itself')
    if not per_element(d.dest.tensor):
      raise self.refuse(f'`{d}` writes {d.dest.tensor.alias}, which the batch '
                        f'shares')
    if isinstance(d, ElementwiseDescr):
      return self.elementwise(d)
    if isinstance(d, ReductionDescr):
      return self.reduction(d)
    # The stack is a new index of the destination, its last; every operand
    # an element owns carries it there too, and an operand the batch shares
    # does not -- it is read the same for each element of the stack.
    rank = d.dest.bbox.rank()
    ops, target, permute = [], [], []
    for op, axes, perm in zip(d.ops, d.target, d.permute):
      if per_element(getattr(op, 'tensor', None)):
        ops.append(self.view(op))
        target.append(list(axes) + [rank])
        permute.append(list(perm) + [len(perm)])
      else:
        ops.append(op)
        target.append(list(axes))
        permute.append(list(perm))
    add = d.add if d.add_dims is None else list(d.add_dims) + [rank]
    return MultilinearDescr(self.view(d.dest), ops, target, permute, add=add)

  def elementwise(self, d: ElementwiseDescr) -> ElementwiseDescr:
    """A pointwise operation over a stack: what an element owns carries the
    stack, the destination's last axis; an operand the batch shares is read
    alike for each element of it -- on the destination's other axes, which
    `legalize` copies it onto -- and one without axes, or a number, at every
    point as before."""
    rank = d.dest.bbox.rank()
    srcs, target, broadcast = [], [], False
    for src in d.srcs:
      tensor = getattr(src, 'tensor', None)
      if tensor is None:
        srcs.append(src)
        target.append(None)
      elif per_element(tensor):
        srcs.append(self.view(src))
        target.append(list(range(rank + 1)))
      elif src.bbox.rank() == 0:
        srcs.append(src)
        target.append([])
      else:
        srcs.append(src)
        target.append(list(range(rank)))
        broadcast = True
    return ElementwiseDescr(d.op, self.view(d.dest), srcs,
                            strict_match=d.strict_match,
                            prefer_align=d.prefer_align,
                            target=target if broadcast else None)

  def reduction(self, d: ReductionDescr) -> ReductionDescr:
    """A reduction over a stack: the stack is an axis the reduction keeps,
    the last of the operand and of the destination alike."""
    if not per_element(d.var.tensor):
      raise self.refuse(f'`{d}` reduces {d.var.tensor.alias}, which the batch '
                        f'shares, into {d.dest.tensor.alias}, which an element '
                        f'owns')
    kept = [d.var.bbox.size(i) for i in range(d.var.bbox.rank())
            if i not in d.dims]
    if list(d.dest.bbox.sizes()) != kept:
      # A full reduction into a buffer of one: stacked, the buffer of one
      # would be a buffer of `k` with one more axis than the stack keeps.
      raise self.refuse(f'`{d}` lands in a destination shaped otherwise than '
                        f'what it keeps')
    return ReductionDescr(self.view(d.dest), self.view(d.var), list(d.dims),
                          d.op, prefer_align=d.prefer_align)
