# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

from tensorforge.interface import YatetoInterface as yi
from tensorforge.common.basic_types import Addressing, Datatype, DataFlowDirection, Residence
from tensorforge.common.context import Context
from tensorforge.common.helper import generate_tmp_tensor
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.matrix.spp import FullSPP, BoundingBoxSPP, ListSPP
from tensorforge.common.matrix.boundingbox import BoundingBox as BBox
from tensorforge.generators.generator import Generator as TensorForgeGenerator
from tensorforge.generators.descriptions import MultilinearDescr, ElementwiseDescr, ReductionDescr, GridBarrierDescr, GridFenceDescr, RegionDescription, GuardLiteral
from tensorforge.common.operation import Operation
from tensorforge.common.operation import AddOperator, MulOperator, MinOperator, MaxOperator, AndOperator, OrOperator, XorOperator

from tensorforge.ir.data.variable import TensorView, TensorAlloc
from tensorforge.ir.data.variable import TensorData
from tensorforge.ir.logical.compute import Multilinear
from tensorforge.ir.type import BaseDatatype
from tensorforge.ir.data.memory import Logical

import numpy as np
import re

from contextlib import contextmanager

class Reader:
  """What every reading of yateto's output has in common.

  yateto reaches this side in two ways -- a whole kernel as data, or one
  operation at a time as the objects its own codegen works with -- and what
  separates them is only how an operand is understood. Both meet a tensor
  once and name it from then on, both produce descriptors in the order the
  operations arrived, and both prefix everything with the region they are in.
  """

  def __init__(self):
    self._cache = {}

    # to be replaced by the IR list
    self._descr_list = []

    # TODO: maybe remove again
    self._prefix = ""

  def result(self):
    """The descriptors read so far, and the tensors they name.

    The list itself rather than a copy: a reader handed one operation at a
    time is not finished when it is first asked for its result.
    """
    return self._descr_list, self._cache

  # NOTE: regions and barriers have no place in the description yet -- yateto
  #       has never called for one -- so these are unreachable. When they are
  #       needed they belong in the operations list, as entries of their own.
  def switch_region(self, barrier):
    if barrier:
      self._descr_list += [GridBarrierDescr()]
    else:
      self._descr_list += [GridFenceDescr()]

  def set_region_name(self, name):
    self._prefix = f"{name}."
    self._descr_list += [RegionDescription(name)]


class DescriptionReader(Reader):
  """Turns a kernel description into TensorForge tensors and descriptors.

  Split off from the emitting half because the two answer different
  questions and share only their result: this one knows what yateto means
  and nothing about how a kernel is built, `KernelEmitter` the other way
  round.
  """

  def __init__(self, arch, attrs=None):
    super().__init__()
    self._arch = arch
    #: The attributes yateto attached to this kernel, or None when yateto has
    #: no attribute channel to attach them with.  Passed on untouched; the
    #: Generator is what reads them.
    self._attrs = attrs

    self._ir_list = []
    self._tensor_list = {}

    # TODO: maybe remove again
    self._prefix = ""

    #: Numbers the scratch tensors this reader introduces, so that two of them
    #: in one kernel do not land on the same name.
    self._scratch = 0

  #: yateto names its operations after the class that implements them; the
  #: enum here is spelled differently and is not a superset.  What is missing
  #: is named in `add_operation_new` rather than mapped to something close.
  ELEMENTWISE_OPS = {
    'Sin': Operation.SIN, 'Cos': Operation.COS, 'Tan': Operation.TAN,
    'Asin': Operation.ASIN, 'Acos': Operation.ACOS, 'Atan': Operation.ATAN,
    'Sinh': Operation.SINH, 'Cosh': Operation.COSH, 'Tanh': Operation.TANH,
    'Asinh': Operation.ASINH, 'Acosh': Operation.ACOSH,
    'Atanh': Operation.ATANH,
    'Log': Operation.LOG, 'Exp': Operation.EXP,
    'Log1p': Operation.LOGP1, 'Expm1': Operation.EXPM1,
    'Sqrt': Operation.SQRT, 'Cbrt': Operation.CBRT, 'Abs': Operation.ABS,
    'Min': Operation.MIN, 'Max': Operation.MAX, 'Pow': Operation.POW,
    'Div': Operation.DIV, 'Add': Operation.ADD, 'Mul': Operation.MUL,
    'And': Operation.AND, 'Or': Operation.OR, 'Xor': Operation.XOR,
    'Not': Operation.NOT,
    'CmpEq': Operation.EQ, 'CmpNe': Operation.NEQ,
    'CmpLt': Operation.LT, 'CmpLe': Operation.LE,
    'CmpGt': Operation.GT, 'CmpGe': Operation.GE,
  }

  #: A reduction carries an operator object, not an enum member: the neutral
  #: element it starts from is type-dependent and only the operator knows it.
  REDUCTION_OPS = {
    'Add': AddOperator, 'Mul': MulOperator,
    'Min': MinOperator, 'Max': MaxOperator,
    'And': AndOperator, 'Or': OrOperator, 'Xor': XorOperator,
  }

  def convert_op(self, name, dest=None):
    """The elementwise operation yateto spelled as `name`.

    A cast is a copy. There is no enum member carrying a target type, and
    there does not need to be: yateto types the destination of a cast with
    exactly the type cast to, so writing the value into it converts it. The
    destination is checked rather than trusted, because a copy into the wrong
    type would silently be a different cast.
    """
    cast = re.fullmatch(r'Cast<(.+)>', name)
    if cast is not None:
      target = Datatype.ytt2enum(cast.group(1))
      actual = None if dest is None else getattr(self._cache.get(f'{self._prefix}{dest["name"]}'), 'datatype', None)
      if actual != target:
        raise NotImplementedError(
          f'a cast to {cast.group(1)} writing into a destination of type '
          f'{actual}: the conversion here is the one the store performs, so '
          f'the two have to agree.')
      return Operation.COPY
    if name not in self.ELEMENTWISE_OPS:
      raise NotImplementedError(
        f'yateto operation {name!r} has no counterpart here. The ternary '
        f'select (it would need a third operand through `Lexic.get_operation`) '
        f'and the logical, as opposed to bitwise, negation are the ones yateto '
        f'can currently emit and this side cannot express.')
    return self.ELEMENTWISE_OPS[name]

  def convert_reduction_op(self, name):
    """The reduction operator yateto spelled as `name`."""
    if name not in self.REDUCTION_OPS:
      raise NotImplementedError(
        f'yateto reduces over {name!r}, which is not one of the operators a '
        f'reduction can start from here ({", ".join(sorted(self.REDUCTION_OPS))}).')
    return self.REDUCTION_OPS[name]()

  def convert_condition(self, condition):
    """A guard, as yateto exports it: a conjunction of literals, or nothing.

    `None` is the guard that never holds -- yateto has already decided the
    statement is dead. An empty list is the guard that always holds. Every
    other list is a conjunction, each literal naming a rank-0 condition
    tensor, the version of it that is meant, and whether it is negated.
    """
    if condition is None:
      return None
    return [GuardLiteral(self.tensor_ref(literal['tensor']),
                         literal['version'],
                         literal['negated'])
            for literal in condition]

  #: Operations that a multilinear describes exactly. A pointwise product is
  #: a multilinear over which no axis is contracted, and a sum over one axis
  #: is a multilinear with a single operand over which that axis is. Both go
  #: that way rather than to their own descriptor: the multilinear path
  #: carries the optimisations, and it broadcasts an operand of lower rank,
  #: which the elementwise one does not.
  AS_MULTILINEAR = {('elementwise', 'Mul'), ('reduction', 'Add')}

  def _linear_layout(self, result, args):
    """Where each operand's axes land, in the numbering a multilinear uses.

    Non-negative numbers are axes of the result, in its own order; negative
    ones are axes contracted away, numbered as they are met. This is the
    numbering yateto builds for the operations it already sends as
    multilinear, so an operation converted here is indistinguishable from one
    that arrived that way.
    """
    axis = {index: position for position, index in enumerate(result['indices'])}
    contracted = -1
    for arg in args:
      for index in arg['indices']:
        if index not in axis:
          axis[index] = contracted
          contracted -= 1
    target = [[axis[index] for index in arg['indices']] for arg in args]
    permute = [list(range(len(arg['indices']))) for arg in args]
    return target, permute

  @staticmethod
  def _accumulates(add):
    """Whether an operation adds onto its destination rather than overwriting.

    yateto states the accumulation as a mask over the destination's axes, so
    the value is either `False` -- overwrite -- or the axes the accumulated
    value spans. A rank-0 destination has no axes, so the mask that
    accumulates onto a scalar is the empty one, and `bool` answers the
    opposite of the question there. `False` is the only value that overwrites.
    """
    return add is not False

  def _reduction_dims(self, result, arg):
    """The axes of `arg` that the reduction removes.

    yateto keeps the surviving axes in the operand's own order, so the axes
    it dropped are enough to describe the reduction -- no permutation comes
    with it. Checking that here means a change on the far side surfaces as a
    message rather than as transposed results.
    """
    src = list(arg['indices'])
    dst = list(result['indices'])
    dims = [i for i, index in enumerate(src) if index not in dst]
    kept = [index for index in src if index in dst]
    if kept != dst:
      raise NotImplementedError(
        f'the reduction keeps axes {kept} but its result is indexed {dst}; '
        f'a reduction that also permutes is not expressible as a '
        f'ReductionDescr.')
    return dims

  def _conform(self, result, argrefs, args):
    """Bring every operand of a pointwise operation onto the destination's axes.

    An elementwise operation applies one scalar operation cell by cell, so
    each operand has to be indexed by exactly the destination's axes, in the
    destination's order. yateto does not require that: it names the axes, and
    an operand may carry fewer of them, or the same ones in another order --
    `t[i,j,k] = A[i,k] + B[k,j]` is an elementwise sum over a semiring.

    An operand that does not already match is copied into a scratch tensor
    that does, by a multilinear over one operand with no axis contracted.
    That is the same operation a broadcast already is on this path, so it is
    spelled the same way rather than given a kind of its own.

    Judged on the axes and not on the extents. Two operands of a square
    destination can have matching extents and still name their axes the other
    way round, and comparing shapes calls that a match -- which computes the
    transpose of what was asked for and says nothing.
    """
    axes = list(result['indices'])
    conformed = []
    for ref, arg in zip(argrefs, args):
      indices = list(ref['indices'])
      if indices == axes:
        conformed.append(arg)
        continue
      missing = [index for index in indices if index not in axes]
      if missing:
        raise NotImplementedError(
          f'an elementwise operand is indexed by {missing}, which its '
          f'destination {axes} does not carry; an axis that survives in no '
          f'operand of a pointwise operation has nothing to iterate.')
      conformed.append(self._conform_one(result, arg, axes, indices))
    return conformed

  def _conform_one(self, result, arg, axes, indices):
    """One operand, copied onto the destination's axes.

    The scratch takes the destination occurrence's box rather than a box of
    its own, so that the copy writes and the operation reads the same cells
    in the same coordinates. Its datatype is the operand's: a copy does not
    convert, and a sum over booleans stays boolean.
    """
    box = self.tensor_ref(result).bbox
    name = f'{self._prefix}_conform{self._scratch}'
    self._scratch += 1

    tensor = Tensor(shape=[int(extent) for extent in box.upper()],
                    addressing=Addressing.PTR_BASED,
                    bbox=box,
                    alias=name,
                    is_tmp=True,
                    datatype=getattr(arg.tensor, 'datatype', None))
    self._cache[name] = tensor
    dest = SubTensor(tensor, box)

    target = [[axes.index(index) for index in indices]]
    permute = [list(range(len(indices)))]
    self._descr_list.append(MultilinearDescr(dest,
                                             [arg],
                                             target,
                                             permute,
                                             add=False,
                                             strict_match=False,
                                             prefer_align=False))
    return dest

  def _accumulator(self, result, dest, add):
    """Where a non-accumulating descriptor writes when yateto wanted a sum.

    Neither `ElementwiseDescr` nor `ReductionDescr` accumulates; both
    overwrite their destination. An operation that yateto marked as
    accumulating therefore writes a scratch tensor, and a multilinear adds
    that onto the destination afterwards -- the same two-step shape a scaled
    result already takes.

    Returns the destination to write and a callable that appends the
    accumulation, which is nothing when there is none.
    """
    if not self._accumulates(add):
      return dest, lambda: None

    box = dest.bbox
    name = f'{self._prefix}_accum{self._scratch}'
    self._scratch += 1
    tensor = Tensor(shape=[int(extent) for extent in box.upper()],
                    addressing=Addressing.PTR_BASED,
                    bbox=box,
                    alias=name,
                    is_tmp=True,
                    datatype=getattr(dest.tensor, 'datatype', None))
    self._cache[name] = tensor
    scratch = SubTensor(tensor, box)

    # The scratch has the destination *view's* shape, so its axes are the
    # ones to state -- not `result['indices']`, which counts a rank-0 result
    # as having none while the view carries it as an axis of extent one.
    axes = list(range(box.rank()))

    def accumulate():
      self._descr_list.append(MultilinearDescr(self.tensor_ref(result),
                                               [scratch],
                                               [axes],
                                               [axes],
                                               add=axes,
                                               strict_match=False,
                                               prefer_align=False))

    return scratch, accumulate

  def add_operation_new(self, d):
    kind = d['type']
    result = self.tensor_ref(d['result'])
    args = [self.tensor_ref(arg) for arg in d['args']]
    condition = self.convert_condition(d['condition'])

    if condition is None:
      # the guard never holds; yateto kept the statement only so that the
      # tensors it names stay in the kernel's signature
      return 0
    linear = d.get('linear') or {}
    # `add` is a mask over the destination's axes, or False. Not a bool: the
    # empty mask means a destination without axes is accumulated onto, and
    # `bool([])` says the opposite.
    add = linear.get('add', False)
    accumulates = add is not False and add is not None
    # the guard covers every descriptor this operation turns into, the
    # scaling that may follow included
    first = len(self._descr_list)

    if kind == 'multilinear':
      # the scale is already one of `args` whenever it is not one -- yateto
      # appends it as a rank-0 operand with an empty target -- so `alpha`
      # here is the same value a second time and is deliberately unused.
      target = [list(t) for t in d['target']]
      permute = [list(p) for p in d['permute']]
      self._descr_list.append(MultilinearDescr(result,
                                               args,
                                               target,
                                               permute,
                                               add=add,
                                               strict_match=False,
                                               prefer_align=False))
    elif (kind, d.get('optype')) in self.AS_MULTILINEAR:
      argrefs = list(d['args'])
      if self._is_named_scalar(linear.get('alpha')):
        # a multilinear takes its factor as one more operand over no axis,
        # which is how yateto sends a scaled contraction as well
        argrefs = argrefs + [linear['alpha']]
        args = args + [self.tensor_ref(linear['alpha'])]
      target, permute = self._linear_layout(d['result'], argrefs)
      self._descr_list.append(MultilinearDescr(result,
                                               args,
                                               target,
                                               permute,
                                               add=add,
                                               strict_match=False,
                                               prefer_align=False))
    elif kind == 'elementwise':
      dest, accumulate = self._accumulator(d['result'], result, add)
      args = self._conform(d['result'], d['args'], args)
      self._descr_list.append(ElementwiseDescr(self.convert_op(d['optype'], d['result']),
                                               dest,
                                               args,
                                               strict_match=False,
                                               prefer_align=False))
      self._append_scaling(d['result'], linear.get('alpha'), dest)
      accumulate()
    elif kind == 'reduction':
      dest, accumulate = self._accumulator(d['result'], result, add)
      assert len(args) == 1
      self._descr_list.append(ReductionDescr(dest,
                                             args[0],
                                             self._reduction_dims(d['result'], d['args'][0]),
                                             self.convert_reduction_op(d['optype']),
                                             prefer_align=False))
      self._append_scaling(d['result'], linear.get('alpha'), dest)
      accumulate()
    else:
      raise NotImplementedError(f'yateto exported an operation of type {kind!r}')

    for descr in self._descr_list[first:]:
      descr.condition = condition

    return 0# self._descr_list[-1].get_flops()

  def _append_scaling(self, result, alpha, view=None):
    """Scale a result in place, as an operation of its own.

    Neither an elementwise operation nor a reduction carries a factor, and
    folding one into a reduction would change what it starts from. Applying
    it afterwards is a multilinear over the result and the factor, which is
    the same shape of operation yateto sends for a scaled contraction.
    """
    if not self._is_named_scalar(alpha):
      return
    # `view` is what the operation actually wrote. It is the destination
    # itself for an operation that overwrites, and the scratch for one whose
    # result is still to be added on -- the factor multiplies what was
    # computed, not what it will be added to.
    scaled = self.tensor_ref(result) if view is None else view
    axes = list(range(scaled.bbox.rank()))
    self._descr_list.append(MultilinearDescr(scaled,
                                             [scaled,
                                              self.tensor_ref(alpha)],
                                             [axes, []],
                                             [axes, []],
                                             add=False,
                                             strict_match=False,
                                             prefer_align=False))

  def _is_named_scalar(self, alpha):
    """Whether `alpha` is a runtime argument rather than the constant one.

    yateto always sends a scale, and a reference to it carries only a name --
    the values came with `add_tensor`, so the answer is in the cache. A scale
    that is literally one is the same as no scale at all.
    """
    if alpha is None:
      return False
    tensor = self._cache.get(f'{self._prefix}{alpha["name"]}')
    data = None if tensor is None else getattr(tensor, 'data', None)
    if data is None:
      return True
    values = list(data.values()) if isinstance(data, dict) else list(np.ravel(data))
    return values != [1] and values != [1.0]

  def tensor_ref(self, d):
    """One occurrence of a tensor, as the operation names it.

    Three things belong to the occurrence and not to the tensor, and all
    three used to be dropped here: the box the equivalent sparsity pattern
    marks out -- which is the range the operation runs over, and regularly
    much smaller than the storage -- the shift that a slicing operand
    imposes, and whether it is a slice at all.

    Coordinates: the box is in the space the operand names, the shift maps
    that space onto the storage, and the two stay apart. Boxes are
    intersected across operands further down, which only means anything if
    every operand contributes its box in the same space; the shift is a pure
    addressing constant and is applied where an address is formed.
    """
    name = f'{self._prefix}{d["name"]}'

    assert(name in self._cache)
    tensor = self._cache[name]

    box = d.get('bbox')
    # An all-zero pattern marks out no box at all, and a description from a
    # yateto that predates the field states none either.
    bbox = tensor.bbox if box is None else BBox(list(box[0]), list(box[1]))
    offset = list(d.get('offset') or [0] * bbox.rank())

    selectors = d.get('offset_from')
    if selectors and any(selectors):
      offset = self._selected_offset(offset, selectors)

    return SubTensor(tensor, bbox, offset, sliced=bool(d.get('sliced')))

  def _selected_offset(self, offset, selectors):
    """An offset whose shift on some axis is only known once it runs.

    That is how one of several matrices is selected: the family is a tensor
    with one axis more, and the shift along that axis is a value rather than
    a number. `VarOffset` is what carries the pair.

    Two things this does not yet do, and they are the next step rather than
    an oversight. The selected axis has to leave the box arithmetic --
    `SubTensor.storage_box` folds the offset into the box and
    `MultilinearDescr.effective_boxes` adds it to the loop ranges, neither of
    which a value can join. And the alignment a layout promises about a
    column only survives a shift along the leading axis if the stride
    preserves it, which is decided where the layout is, not here.
    """
    raise NotImplementedError(
      f'operand selected at run time along axis '
      f'{[i for i, sel in enumerate(selectors) if sel]}: the shift arrives '
      f'as data and `VarOffset` can carry it, but the box arithmetic still '
      f'treats every offset as a number.')

  def add_tensor(self, d):
    name = d['name']
    name = f'{self._prefix}{name}'

    datatype = Datatype.ytt2enum(d['datatype'])

    datatype_new = BaseDatatype.ytt2enum(d['datatype'])

    shape = d['storage']['shape']
    storagetype = d['storage']['type']

    residence = Residence.str2residence(d['residence'])

    addressingStr = d['addressing']
    if addressingStr == '&':
      addressing = Addressing.NONE
    elif addressingStr == 'n*N+o&':
      addressing = Addressing.STRIDED
    elif addressingStr == 'n&+o&':
      addressing = Addressing.PTR_BASED
    elif addressingStr == '':
      addressing = Addressing.SCALAR
    elif addressingStr is None and residence is Residence.CODE:
      # There is no formula because there is no parameter. `NONE` is still
      # what the operand *is* -- one and the same datum for every batch
      # element -- and the paths that ask read the residence for the rest.
      addressing = Addressing.NONE
    else:
      # An unhandled spelling used to leave `addressing` unbound, and the
      # first read of it blamed a line that had nothing to do with it.
      raise NotImplementedError(
        f'tensor {name}: the description states addressing '
        f'{addressingStr!r} with residence {residence}, which this frontend '
        f'has no reading for.')

    if storagetype == 'full':
      spp = FullSPP(shape)
      bbox = None
    if storagetype == 'bbox':
      starts = d['storage']['start'] or [0]
      sizes = d['storage']['sizes'] or [1]
      lower = list(starts)
      upper = [start + size for start, size in zip(starts, sizes)]
      bbox = BBox(lower, upper)
      spp = FullSPP(shape)#BoundingBoxSPP(bbox)
    if storagetype == 'spp':
      bbox = None
      # NOTE: ListSPP numbers the entries in the order they arrive, and that
      #       numbering is the address; yateto sends them in its storage order.
      spp = ListSPP([tuple(entry) for entry in d['storage']['entries']], shape)

    values = self._values(d['values'])
    is_temporary = d['flags']['temporary']
    is_constant = d['flags']['constant']

    self._cache[name] = Tensor(shape, addressing, bbox, name, is_temporary, spp,
                               values, datatype, d.get('alignment', 0),
                               residence=residence)

    # as the tensor carries it: an axisless one with its axis of extent one
    carried = self._cache[name]
    self._tensor_list[name] = TensorData(datatype_new, list(carried.shape),
                                         carried.spp, values=values)

  @staticmethod
  def _values(values):
    """The constant data a tensor carries, if it carries any.

    Two shapes reach here and `Tensor` wants a different thing for each: a
    dense run of values as an array, and a handful of named entries as a
    map from coordinate to value. Neither is a list, which `Tensor` refuses
    on purpose.
    """
    if values is None:
      return None
    if values['kind'] == 'flat':
      return np.asarray(values['data'])
    if values['kind'] == 'entries':
      return {tuple(index): value for index, value in values['data']}
    raise NotImplementedError(f'unknown value kind {values["kind"]!r}')

  def read(self, description):
    """Read a whole kernel: every tensor first, then every operation.

    Tensors first because an operation names them, and the description
    lists them in the order they were met, so nothing forward-references.
    """
    version = description.get('version')
    if version != YatetoFrontend.INTERFACE_VERSION:
      raise NotImplementedError(
        f'the description states interface version {version}, this side '
        f'reads {YatetoFrontend.INTERFACE_VERSION}.')
    for tensor in description['tensors']:
      self.add_tensor(tensor)
    for operation in description['operations']:
      self.add_operation_new(operation)
    return self.result()


class TermReader(Reader):
  """Turns yateto's own term objects into TensorForge tensors and descriptors.

  This is the route a yateto takes that sends one operation at a time as the
  objects its codegen works with, rather than a kernel as data. A term states
  its indices, its memory layout and its equivalent sparsity pattern, and
  everything a description states outright is derived from those here: the
  box an occurrence runs over, the shift a slicing operand imposes, whether
  it is a slice at all, and what alignment may be claimed for it.

  yateto is never imported. A term is whatever was handed over, and the few
  places that have to tell one kind of layout from another ask for the type's
  name.
  """

  def __init__(self, arch):
    super().__init__()
    self._arch = arch
    self._tmp_matrices = {}

  def add_operation(self, dest, ops, target, permute, add):
    """One contraction, product, permutation or copy, as a multilinear.

    Every operation that arrives this way is linear in its operands, so
    there is one descriptor kind to build and the return value is the flop
    count yateto adds up -- nothing counts them here yet.
    """
    self._cache_matrices(dest, ops, target, permute)
    can_be_aligned = self._can_be_aligned(dest, ops, target, permute)
    destdims = [i for i in range(len(dest.indices))]
    self._descr_list.append(
      MultilinearDescr(self.get_tensor(dest, can_be_aligned, destdims),
                       [self.get_tensor(op, can_be_aligned, optarget)
                        for op, optarget in zip(ops, target)],
                       target, permute, add=add,
                       strict_match=False,
                       prefer_align=can_be_aligned))
    return 0

  @staticmethod
  def is_scalar(op):
    """Whether this operand is a named scalar rather than a tensor.

    Asked of the object, since recognising yateto's `Scalar` by type would
    mean importing yateto. It carries no memory layout, and a factor that is
    a literal arrives as a number instead.
    """
    return not hasattr(op, 'memoryLayout') and not isinstance(op, (float, int))

  def _datatype(self, source):
    if hasattr(source, 'datatype'):
      stype = Datatype.ytt2enum(source.datatype)
    else:
      stype = None
    if hasattr(self._arch, 'typename'):
      fptype = Datatype.str2enum(self._arch.typename)
    else:
      fptype = None

    assert not (stype is None and fptype is None)

    return stype if stype is not None else fptype

  def _can_be_aligned(self, dest, ops, target, permute):
    # TODO: useful?
    aligned = dest.memoryLayout.alignedStride()
    for i, op in enumerate(ops):
      if 0 in target[i]:
        aligned &= dest.memoryLayout.alignedStride() and permute[i][0] == 0

    return aligned

  def get_tensor(self, op, can_be_aligned, dims):
    if isinstance(op, (float, int)):
      return SubTensor(tensor = Tensor([], Addressing.SCALAR, data = np.array(op)))
    elif self.is_scalar(op):
      return SubTensor(self._cache[f'{self._prefix}{op.name()}'])
    else:
      tensor = self._cache[f'{self._prefix}{op.name}']
      currentPreShape = BBox([s for s, _ in op.eqspp.nnzbounds()], [e+1 for _, e in op.eqspp.nnzbounds()])

      # Two shifts act on a yateto tensor, in opposite directions, and they must
      # not be conflated:
      #
      #   * the memory bounding box (`tml.bbox()`) restricts what is *stored*.
      #     It lives in storage coordinates and is subtracted when an address is
      #     formed (see Symbol.access_address).
      #   * a MemoryLayoutView adds a slicing offset: the view's own index space
      #     is [0, end-start), mapped to the base by `relidx`.
      #
      # `currentPreShape` is derived from eqspp, which is defined over the
      # *view* shape --- so it is already in logical coordinates and stays
      # there.  Bounding boxes become loop ranges and are intersected across
      # operands (MultilinearInstruction._analyze); that intersection is only
      # meaningful if every operand contributes it in the same, shared logical
      # index space.  The offset is a pure addressing constant and is applied at
      # the access site only.
      tml = op.memoryLayout
      offset = [0] * currentPreShape.rank()
      # a view means the operand names a slice, not the tensor; see
      # SubTensor.sliced.  The offset alone does not carry it: `subslice` from
      # index 0 produces a view with a zero shift.
      sliced = type(tml).__name__ == 'MemoryLayoutView'
      while type(tml).__name__ == 'MemoryLayoutView':
        # relidx() adds this view's `start` in the one dimension it slices;
        # nested views compose, so this accumulates the full logical->storage shift
        offset = list(tml.relidx(offset))
        tml = tml.base
      tml = tml.storage()

      if can_be_aligned and currentPreShape.rank() > 0 and tml.alignedStride():
        # Alignment is a property of the *address*, so snap in storage
        # coordinates and pull the result back into logical ones.  Widening is
        # sound because the entries gained are zero by eqspp; it must not,
        # however, reach past what is actually stored.
        storeRange = tml.bbox()[0]
        newLower = max(self._arch.alignedLower(currentPreShape._lower[0] + offset[0]),
                       storeRange.start)
        newUpper = min(self._arch.alignedUpper(currentPreShape._upper[0] + offset[0]),
                       storeRange.stop)

        currentPreShape._lower = tuple([newLower - offset[0]] + list(currentPreShape._lower[1:]))
        currentPreShape._upper = tuple([newUpper - offset[0]] + list(currentPreShape._upper[1:]))

      # invariant tying the two coordinate systems together: bbox + offset must
      # land inside what the storage layout actually holds
      storeBox = tml.bbox()
      for j, (lo, hi) in enumerate(zip(currentPreShape.lower(), currentPreShape.upper())):
        assert lo >= hi or (storeBox[j].start <= lo + offset[j] and hi + offset[j] <= storeBox[j].stop), \
            f'{op.name}: logical bbox [{lo},{hi}) + offset {offset[j]} escapes ' \
            f'storage [{storeBox[j].start},{storeBox[j].stop}) in dim {j}'

      return SubTensor(tensor, currentPreShape, offset, sliced=sliced)

  def make_tensor(self, op, can_be_aligned, dims):
    if isinstance(op, (float, int)):
      return Tensor([], Addressing.SCALAR, data = np.array(op))
    if self.is_scalar(op):
      entry = self._add_scalar(op)
      entry_name = op.name()
    else:
      entry = self._get_tensorforge_matrix(op)
      entry_name = op.name

    entry_name = f'{self._prefix}{entry_name}'

    if not (entry_name in self._cache and entry.is_same(self._cache[entry_name])):
      self._cache[entry_name] = entry

  def _cache_matrices(self, dest, ops, target, permute):
    can_be_aligned = self._can_be_aligned(dest, ops, target, permute)

    # no add onto a matrix that doesn't exist (TODO: check if that's always the case)
    assert not(dest.is_temporary and dest in ops)

    for op, optarget in zip(ops, target):
      self.make_tensor(op, can_be_aligned, optarget)

    self.make_tensor(dest, can_be_aligned, [i for i in range(len(dest.indices))])

    if dest.is_temporary: # (dest is never a scalar---for the time being)
      self._tmp_matrices[f'{self._prefix}{dest.name}'] = self._cache[f'{self._prefix}{dest.name}']

  def _add_scalar(self, scalar):
    name = f'{self._prefix}{scalar.name()}'
    tensor = Tensor([], Addressing.SCALAR, alias=name, datatype=self._datatype(scalar.datatype))
    self._tmp_matrices[name] = tensor
    return self._tmp_matrices[name]

  @staticmethod
  def _deduce_addressing(term):
    if term.is_compute_constant:
      return Addressing.NONE
    if term.is_temporary:
      return Addressing.STRIDED
    else:
      return Addressing.PTR_BASED

  @staticmethod
  def _storage(tml):
    if type(tml).__name__ == 'MemoryLayoutView':
      return tml.storage()
    return tml

  def _get_tensorforge_matrix(self, tensor):
    tml = self._storage(tensor.memoryLayout)

    shape=[rng.stop for rng in tml.bbox()]
    bboxrange=tml.bbox()

    addr_mode = self._deduce_addressing(tensor) if tensor.addressing is None else tensor.addressing
    if tensor.is_temporary and tensor.name in self._tmp_matrices:
      return self._tmp_matrices[tensor.name]

    if type(tml).__name__ == 'DenseMemoryLayout':
      pattern = None
    else:
      # from zero rather than from the box's lower corner: `entries` numbers
      # what it returns relative to the ranges it is given, and the pattern
      # has to be stated over the shape, which is what `ListSPP` indexes
      ranges = [range(0, shape[i]) for i in range(len(shape))]
      pattern = tml.entries(*ranges)

    alignment = 16 if len(tensor.memoryLayout.shape()) > 0 and tensor.memoryLayout.alignedStride() else 0

    return yi.gen_matrix(shape,
                         bboxrange,
                         addressing=addr_mode,
                         name=f'{self._prefix}{tensor.name}',
                         is_tmp=tensor.is_temporary,
                         permute=None,
                         pattern=pattern,
                         values = tensor.values,
                         datatype = self._datatype(tensor.datatype),
                         alignment = alignment)


class KernelEmitter:
  """Runs TensorForge over what the reader built and writes the call site."""

  def __init__(self, arch, attrs, descr_list, cache):
    #: What the generated routine ends up being called. Only known once it
    #: has been generated, since it is derived from what was generated.
    self.base_name = None
    self._arch = arch
    self._attrs = attrs
    self._descr_list = descr_list
    self._cache = cache

  def generate(self, cpp, routineCache):
    if hasattr(self._arch, 'typename'):
      fptype = Datatype.str2enum(self._arch.typename)
    else:
      fptype = None

    context = Context(arch=self._arch.name,
                      backend=self._arch.backend,
                      fp_type=fptype)

    # print(self._ir_list)

    tensorforge_generator = TensorForgeGenerator(self._descr_list, context,
                                                attrs=self._attrs)
    tensorforge_generator.generate()

    cpp(f'{self._gen_call_site(tensorforge_generator)}')
    routine_name = tensorforge_generator.get_base_name()
    self.base_name = routine_name

    routineCache.addRoutine(routine_name, TensorForgeWriter(tensorforge_generator, context.get_vm().get_headers()))

  def _gen_call_site(self, generator):
    mat_name_map = {}
    offset_name_map = {}
    for name, matrix in self._cache.items():
      if matrix.direction == DataFlowDirection.SOURCE and matrix.addressing != Addressing.SCALAR:
        datatype = matrix.datatype
        assert datatype is not None
        ptr_type = f'const {datatype}{matrix.addressing.to_pointer()}'
        mat_name_map[name] = f'const_cast<{ptr_type}>({name})'
      else:
        mat_name_map[name] = name

      if matrix.is_tmp or matrix.addressing == Addressing.NONE:
        offset_name_map[name] = '0'
      else:
        parts = name.split('.')
        assert len(parts) <= 2
        varname = f'extraOffset_{parts[-1]}'
        if len(parts) == 2:
          offset_name_map[name] = f'{parts[0]}.{varname}'
        else:
          offset_name_map[name] = varname

    return generator.generate_call_site(mat_name_map,
                                        offset_name_map)

class TensorForgeWriter:
  def __init__(self, tensorforge_generator, headers):
    self._headers = list(headers) + list(tensorforge_generator.get_helper_headers())
    self._generator = tensorforge_generator
    self._basename = self._generator.get_base_name()

  def target(self):
    return 'gpu'

  def __eq__(self, other):
    # The routine cache keys on the name and asks this when a name arrives
    # twice.  Comparing names would make the answer yes by construction --
    # the name is how the question got here -- so the second routine would be
    # dropped whatever it contained.  Comparing the source makes it a real
    # question, and the two kernels are interchangeable exactly when a
    # compiler cannot tell them apart.
    if isinstance(other, TensorForgeWriter):
      return (self._basename == other._basename
              and self._generator.unnamed_source()
                  == other._generator.unnamed_source())
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, 'a', encoding='utf-8') as file:
      file.write(kernel)
      file.write(launcher)

    return self._generator.get_header()

class Recorded:
  """One kernel, as `capture` hands it over.

  `name` is a property rather than a value because it is not known when the
  kernel arrives: it is derived from what gets generated, so it stays `None`
  for a kernel that never does.
  """

  __slots__ = ('description', 'descrs', 'emitter')

  def __init__(self, description):
    #: What yateto sent, when it sent a description at all. `None` for a
    #: kernel that arrived as terms, one operation at a time -- there is no
    #: description of it to keep, only what was built.
    self.description = description
    #: What was built from it, once it has been read. `None` for a
    #: description that could not be read at all.
    self.descrs = None
    self.emitter = None

  @property
  def name(self):
    return None if self.emitter is None else self.emitter.base_name


class YatetoFrontend:
  #: The version of yateto's export interface this reads. yateto refuses an
  #: exporter that speaks an older one, because the fields added since would
  #: be dropped silently rather than missed loudly.
  INTERFACE_VERSION = 7

  def __init__(self, arch, attrs=None):
    """The routine exporter yateto instantiates, once per kernel.

    ``attrs`` carries the per-kernel switches from ``Generator.add``.  A
    yateto that does not know about them calls this with ``arch`` alone, and
    the default then selects the kernel surface that predates the channel --
    a flag mask on every kernel, checked against ``nullptr``.
    """
    self._arch = arch
    self._attrs = attrs
    self._description = None
    self._emitter = None
    #: Set once a kernel arrives as terms; see `add_linear_operation`.
    self._terms = None
    self._recorded = None

  #: Set by `capture()`. Every kernel that goes through here is offered to
  #: it, once it has been built and therefore has a name.
  _sink = None

  @classmethod
  @contextmanager
  def capture(cls, sink):
    """Record every kernel handed over inside this block.

    `sink(recorded)` gets a `Recorded`: the description yateto sent, the
    descriptors built from it, and what the routine ends up being called.
    A supported way in, so that tooling wanting to see a codegen run does
    not have to patch a method belonging to another class -- which is what
    it used to do, and why it only ever saw the one kind of descriptor the
    patch happened to know about.

    It fires as the kernel arrives, not once it is built, so a kernel that
    fails to build is recorded too. Those are the ones worth having: a
    failure needs the input that produced it. The name is only known once
    there is a generated routine to name, so it reads as `None` until then.
    """
    previous = cls._sink
    cls._sink = sink
    try:
      yield
    finally:
      cls._sink = previous

  def _record(self, description):
    """Offer this kernel to whoever is capturing, if anyone is."""
    sink = type(self)._sink
    if sink is None:
      return None
    recorded = Recorded(description)
    sink(recorded)
    return recorded

  def add_kernel(self, description):
    """The whole kernel, as data, in one call."""
    # Recorded before it is read, so that a description this cannot read is
    # recorded too -- that is precisely the one worth having.
    recorded = self._record(description)

    reader = DescriptionReader(self._arch, self._attrs)
    descr_list, cache = reader.read(description)
    self._description = description
    self._emitter = KernelEmitter(self._arch, self._attrs, descr_list, cache)

    if recorded is not None:
      recorded.descrs = descr_list
      recorded.emitter = self._emitter

  def add_linear_operation(self, dest, ops, target, permute, add):
    """One operation of a kernel that arrives as terms rather than as data.

    Nothing announces such a kernel -- the first operation is what starts
    it, and it is complete only once `generate` asks for it -- so the reader
    is made here and kept.

    What is recorded is the descriptor list itself, which the reader goes on
    appending to, so a capture fills in as the rest of the kernel arrives.
    """
    if self._terms is None:
      self._terms = TermReader(self._arch)
      self._recorded = self._record(None)
      if self._recorded is not None:
        self._recorded.descrs, _ = self._terms.result()
    return self._terms.add_operation(dest, ops, target, permute, add)

  def generate(self, cpp, cache):
    if self._emitter is None and self._terms is not None:
      descr_list, tensors = self._terms.result()
      self._emitter = KernelEmitter(self._arch, self._attrs, descr_list, tensors)
      if self._recorded is not None:
        self._recorded.emitter = self._emitter
    if self._emitter is None:
      raise NotImplementedError(
        'generate() before a kernel arrived: there is nothing to build.')
    self._emitter.generate(cpp, cache)
