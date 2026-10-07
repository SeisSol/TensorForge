# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a descriptor states that the builders do not take, as descriptors
they do.

A frontend states an operation the way its source language does.  yateto
names axes, so an operand of a pointwise operation may carry fewer of them,
or the same ones in another order; an operation may add onto its result or
be scaled by a factor; and an operand's box marks where it can be non-zero,
so it reads zero outside.  The builders take less: an `ElementwiseDescr`
over operands of the destination's shape that overwrites it, and a
`ReductionDescr` that overwrites.  Getting from the one to the other is a
question about descriptors, not about yateto, so it is answered here, once,
for every frontend: a descriptor list from the Python API or a test case is
rewritten exactly as one from yateto's reader.

Per descriptor, in list order:

* a `SELECT` whose condition is one value per batch element becomes two
  guarded multilinears, one per branch (`_hoist`);
* an elementwise operation or a reduction that accumulates writes a scratch,
  which a multilinear adds on afterwards (`_accumulator`);
* an operand on other axes than the destination is copied onto them
  (`_conform`);
* where operand boxes cut the destination, it is cut with them, and an
  operand is read in a piece it covers and is 0 in one it does not
  (`_cells`);
* a factor is a multilinear over what was written (`_scale`).

A scratch tensor is named for what it holds -- `_conform`, `_accum`,
`_pieces` -- and numbered across the list, so the names, and with them the
kernels, depend on the list alone.  A list with nothing to rewrite comes back
with the same descriptors in it.
"""

from __future__ import annotations

import copy
import itertools
from typing import List

import numpy as np

from tensorforge.common.basic_types import Addressing
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import Operation
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 GuardLiteral,
                                                 MultilinearDescr,
                                                 ReductionDescr)


def legalize(descr_list: List) -> List:
  """`descr_list` with every descriptor the builders take as it is, and
  every other one replaced by those it stands for."""
  return _Legalizer().run(descr_list)


def _is_num(x) -> bool:
  """A constant operand, as `ElementwiseDescr.scalar_srcs` counts one."""
  return isinstance(x, (int, float, np.integer, np.floating))


def _empty(view) -> bool:
  return any(int(size) == 0 for size in view.bbox.sizes())


def _multilinear(dest, ops, target, add=False):
  """`dest = ops...` over the given axes, overwriting or adding on; each
  operand in its own axis order."""
  return MultilinearDescr(dest, ops, target,
                          [list(range(len(t))) for t in target], add=add,
                          strict_match=False, prefer_align=False)


class _Legalizer:
  def __init__(self):
    #: Numbers the scratch tensors, so that two in one list do not share a
    #: name.
    self._scratch = 0
    #: Numbers the ternaries hoisted into guards: each gets a guard version
    #: of its own (`_hoist`).
    self._hoisted = 0

  def run(self, descr_list):
    out = []
    for descr in descr_list:
      if isinstance(descr, ElementwiseDescr) and self._hoistable(descr):
        out += self._hoist(descr)
      elif isinstance(descr, ElementwiseDescr) and not descr.legal():
        out += self._under(descr, self._elementwise(descr))
      elif isinstance(descr, ReductionDescr) and not descr.legal():
        out += self._under(descr, self._reduction(descr))
      else:
        out.append(descr)
    return out

  @staticmethod
  def _under(descr, produced):
    """What `descr` stands for, under its guard."""
    for each in produced:
      each.condition = descr.condition
    return produced

  # -- a ternary with one condition per element -------------------------- #

  @staticmethod
  def _hoistable(d):
    """Whether a ternary's condition is one value per batch element, and not
    the tensor the ternary writes -- the second half of a hoist reads the
    condition after the first half has written the result.  Nor where a
    branch is a number, or stores nothing (`0.0 * C`): the select takes that
    as the number 0, and a copy of it would have no cells to copy."""
    if d.op != Operation.SELECT:
      return False
    yes, no, cond = d.srcs
    if _is_num(cond) or d.target_of(2) or cond.tensor is d.dest.tensor:
      return False
    return not any(_is_num(branch) or (d.target_of(position) and _empty(branch))
                   for position, branch in enumerate((yes, no)))

  def _hoist(self, d):
    """`dest = cond ? yes : no` with a rank-0 condition, as two statements.

    `dest = yes` under the guard and `cond`, `dest = no` under the guard and
    not `cond`, so the condition is read once where each region opens, the
    branch not taken is not computed at all, and neither is a select per
    entry.  Each half is a single-operand multilinear: that is what
    broadcasts a branch of lower rank, permutes, accumulates and takes a
    factor.

    A guard version only groups neighboring statements under one guard, so a
    hoist's own is a negative one: a frontend's start at zero, and two
    hoists over the same condition tensor must not merge into one region.
    """
    self._hoisted += 1
    version = -self._hoisted
    yes, no, cond = d.srcs
    out = []
    for position, branch, negated in ((0, yes, False), (1, no, True)):
      ops, axes = [branch], [d.target_of(position)]
      if d.alpha is not None:
        ops, axes = ops + [copy.copy(d.alpha)], axes + [[]]
      descr = _multilinear(d.dest, ops, axes, add=d.add_mask())
      descr.condition = list(d.condition or []) + [
          GuardLiteral(cond, version, negated)]
      out.append(descr)
    return out

  # -- scratch tensors ----------------------------------------------------- #

  def _name(self, kind):
    name = f'_{kind}{self._scratch}'
    self._scratch += 1
    return name

  def _scratch_over(self, box, kind, datatype):
    tensor = Tensor(shape=[int(extent) for extent in box.upper()],
                    addressing=Addressing.PTR_BASED,
                    bbox=box,
                    alias=self._name(kind),
                    is_tmp=True,
                    datatype=datatype)
    return SubTensor(tensor, box)

  def _through_scratch(self, result, dest, kind, add):
    """A scratch tensor over `dest`'s box, and the multilinear that puts it
    into `result` afterwards: adding it on, or assigning it."""
    scratch = self._scratch_over(dest.bbox, kind,
                                 getattr(dest.tensor, 'datatype', None))
    # The scratch has the destination *view's* shape, so its axes are the
    # ones to state -- a rank-0 result is a view with an axis of extent one.
    axes = list(range(dest.bbox.rank()))
    return scratch, lambda: _multilinear(copy.copy(result), [scratch],
                                         [axes], add=axes if add else False)

  def _accumulator(self, d):
    """Where a descriptor writes that accumulates.  Neither builder adds onto
    its destination, so it writes a scratch, and a multilinear adds that on
    afterwards.  Returns the view to write and the descriptors that follow,
    as a callable: none where nothing accumulates."""
    if not d.add:
      return d.dest, lambda: []
    scratch, put = self._through_scratch(d.dest, d.dest, 'accum', add=True)
    return scratch, lambda: [put()]

  # -- an elementwise operation --------------------------------------------- #

  def _elementwise(self, d):
    out = []
    dest, accumulate = self._accumulator(d)
    args = self._conform(d, out)
    dest, cells, assign = self._assembled(d.dest, dest, args)
    for cell, cell_args in cells:
      out.append(ElementwiseDescr(d.op, cell, cell_args,
                                  strict_match=d.strict_match,
                                  prefer_align=d.prefer_align))
    out += self._scale(dest, d.alpha)
    out += assign()
    out += accumulate()
    return out

  def _conform(self, d, out):
    """Bring every operand onto the destination's axes.

    An elementwise operation applies one scalar operation cell by cell, so
    each operand has to be indexed by exactly the destination's axes, in the
    destination's order.  One stated on others -- fewer of them, or the same
    ones in another order, `t[i,j,k] = A[i,k] + B[k,j]` -- is copied into a
    scratch that is, by a multilinear over one operand with no axis
    contracted.  That is the same operation a broadcast already is, so it is
    spelled the same way rather than given a kind of its own.

    Judged on the axes and not on the extents: two operands of a square
    destination can have matching extents and still name their axes the
    other way round, and comparing shapes calls that a match.

    An operand whose box holds nothing is known to be zero everywhere
    (`0.0 * C` stores nothing), and is the number 0.

    Nothing to do where no axes are stated: every operand is then on the
    destination's.
    """
    if d.target is None:
      return list(d.srcs)
    rank = d.dest.bbox.rank()
    conformed = []
    for position, arg in enumerate(d.srcs):
      if _is_num(arg):
        conformed.append(arg)
        continue
      axes = d.target_of(position)
      if axes and _empty(arg):
        conformed.append(0)
        continue
      if axes == list(range(rank)) or (not axes
                                       and getattr(arg.tensor, 'rank0', False)):
        # The same axes, or none at all: an operand without axes is read
        # with an empty index and broadcast by the pointwise operation.
        conformed.append(arg)
        continue
      if any(not 0 <= axis < rank for axis in axes):
        raise NotImplementedError(
          f'an elementwise operand is indexed by axes {axes}, of which its '
          f'rank-{rank} destination does not carry the negative ones; an '
          f'axis that survives in no operand of a pointwise operation has '
          f'nothing to iterate.')
      conformed.append(self._conform_one(d, arg, axes, out))
    return conformed

  def _conform_one(self, d, arg, axes, out):
    """One operand, copied onto the destination's axes.

    The scratch takes the destination's box rather than a box of its own, so
    that the copy writes and the operation reads the same cells in the same
    coordinates.  Its datatype is the operand's: a copy does not convert, and
    a sum over booleans stays boolean.
    """
    dest = self._scratch_over(d.dest.bbox, 'conform',
                              getattr(arg.tensor, 'datatype', None))
    out.append(_multilinear(dest, [arg], [axes]))
    return dest

  def _assembled(self, result, dest, args):
    """Where the pieces `_cells` cuts an assignment into are written.

    An assignment to the tensor itself through a narrower window defines the
    whole tensor, zeros outside the window (`SubTensor.owed_zeros`), and each
    `ElementwiseDescr` keeps that promise on its own.  A piece of one would
    zero its siblings' parts with it.  So pieces that owe zeros are assembled
    in a scratch over the window, which owes nothing, and one multilinear
    assigns the scratch to the destination, zeros included -- the shape an
    accumulating one already takes (`_accumulator`).

    Returns the view to write, the pieces, and a callable returning the
    assignment, which is nothing where the pieces write the destination.
    """
    cells = _cells(dest, args)
    if len(cells) == 1:
      # one piece is the whole window (an operand's box reaching past it was
      # clamped to it): it is the destination, and owes what the destination
      # owes -- a piece marked as a slice would owe nothing
      return dest, [(dest, cells[0][1])], lambda: []
    if dest.owed_zeros() is None:
      return dest, cells, lambda: []
    scratch, assign = self._through_scratch(result, dest, 'pieces', add=False)
    return scratch, _cells(scratch, args), lambda: [assign()]

  # -- a reduction -------------------------------------------------------- #

  def _reduction(self, d):
    dest, accumulate = self._accumulator(d)
    out = [ReductionDescr(dest, d.var, d.dims, d.op,
                          prefer_align=d.prefer_align)]
    out += self._scale(dest, d.alpha)
    out += accumulate()
    return out

  # -- a factor ------------------------------------------------------------ #

  @staticmethod
  def _scale(written, alpha):
    """`written` scaled in place, as an operation of its own.

    Neither an elementwise operation nor a reduction carries a factor, and
    folding one into a reduction would change what it starts from.  Applying
    it afterwards is a multilinear over what was written and the factor --
    for one that accumulates, the scratch, since the factor multiplies what
    was computed and not what it is added to.
    """
    if alpha is None:
      return []
    axes = list(range(written.bbox.rank()))
    return [_multilinear(written, [written, copy.copy(alpha)], [axes, []])]


def _cells(dest, args):
  """`dest` cut into boxes on which each operand is all or nothing.

  An operand's box marks where it can be non-zero -- a table storing three of
  eight entries has a box of three -- and a pointwise operation over the whole
  destination reads zero outside it: `exp` of it is one, `x + 0` is `x`,
  `0 > 0` is false.  An `ElementwiseDescr` runs one operation over one box and
  takes no operand of another shape.  So the destination is cut at every edge
  an operand's box has inside it; in each piece an operand either covers it
  and is read there, or misses it and is the number 0.  One piece, the
  destination itself, where every box agrees.
  """
  def boxed(arg):
    return (hasattr(arg, 'bbox') and arg.bbox.rank() > 0
            and arg.bbox.rank() == dest.bbox.rank())
  lower, upper = list(dest.bbox.lower()), list(dest.bbox.upper())
  boxes = [arg.bbox for arg in args if boxed(arg)]
  if all(list(box.lower()) == lower and list(box.upper()) == upper
         for box in boxes):
    return [(dest, args)]
  cuts = []
  for dim in range(len(lower)):
    points = {lower[dim], upper[dim]}
    for box in boxes:
      points.update(min(max(int(p), lower[dim]), upper[dim])
                    for p in (box.lower()[dim], box.upper()[dim]))
    points = sorted(points)
    cuts.append(list(zip(points, points[1:])))
  cells = []
  for pieces in itertools.product(*cuts):
    lo = [piece[0] for piece in pieces]
    hi = [piece[1] for piece in pieces]
    cell = copy.copy(dest)
    cell.bbox = BoundingBox(lo, hi)
    # a piece owns its part and nothing else: as the tensor itself through a
    # narrower window, it would owe zeros over its siblings' parts
    # (`SubTensor.owed_zeros`), which `_assembled` keeps for the whole
    cell.sliced = True
    cell_args = []
    for arg in args:
      if not boxed(arg):
        cell_args.append(arg)
      elif all(arg.bbox.lower()[k] <= lo[k] and hi[k] <= arg.bbox.upper()[k]
               for k in range(len(lo))):
        part = copy.copy(arg)
        part.bbox = BoundingBox(lo, hi)
        cell_args.append(part)
      else:
        cell_args.append(0)
    cells.append((cell, cell_args))
  return cells
