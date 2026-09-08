# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

from tensorforge.interface import YatetoInterface as yi
from tensorforge.common.basic_types import Addressing, Datatype, DataFlowDirection
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

class DescriptionReader:
  """Turns a kernel description into TensorForge tensors and descriptors.

  Split off from the emitting half because the two answer different
  questions and share only their result: this one knows what yateto means
  and nothing about how a kernel is built, `KernelEmitter` the other way
  round. The old single class also carried a `V1` in its name that had
  stopped being true.
  """

  def __init__(self, arch, attrs=None):
    self._arch = arch
    #: The attributes yateto attached to this kernel, or None when yateto has
    #: no attribute channel to attach them with.  Passed on untouched; the
    #: Generator is what reads them.
    self._attrs = attrs
    self._cache = {}
    self._tmp_matrices = {}

    # to be replaced by the IR list
    self._descr_list = []

    self._ir_list = []
    self._tensor_list = {}

    # TODO: maybe remove again
    self._prefix = ""

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

  def _is_phantom(self, ref):
    """Whether this reference names a rank-0 tensor carried as extent one.

    Its one axis is the destination's axis 0, which is what every rank-0
    tensor in the operation shares -- there is only ever one of them.
    """
    if len(ref['indices']) > 0:
      return False
    tensor = self._cache.get(f'{self._prefix}{ref["name"]}')
    return tensor is not None and tensor.addressing != Addressing.SCALAR

  def _fixup_phantom(self, args, target, permute):
    """Give a rank-0 operand the axis its extent-one shape now has."""
    for i, arg in enumerate(args):
      if self._is_phantom(arg) and len(target[i]) == 0:
        target[i] = [0]
        permute[i] = [0]
    return target, permute

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
    return self._fixup_phantom(args, target, permute)

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
    add = linear.get('add', False)
    # the guard covers every descriptor this operation turns into, the
    # scaling that may follow included
    first = len(self._descr_list)

    if kind == 'multilinear':
      # the scale is already one of `args` whenever it is not one -- yateto
      # appends it as a rank-0 operand with an empty target -- so `alpha`
      # here is the same value a second time and is deliberately unused.
      target, permute = self._fixup_phantom(d['args'],
                                           [list(t) for t in d['target']],
                                           [list(p) for p in d['permute']])
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
      if add:
        raise NotImplementedError(
          'an elementwise operation that accumulates onto its destination; '
          'ElementwiseDescr overwrites.')
      self._descr_list.append(ElementwiseDescr(self.convert_op(d['optype'], d['result']),
                                               result,
                                               args,
                                               strict_match=False,
                                               prefer_align=False))
      self._append_scaling(d['result'], linear.get('alpha'))
    elif kind == 'reduction':
      if add:
        raise NotImplementedError(
          'a reduction that accumulates onto its destination; '
          'ReductionDescr overwrites.')
      assert len(args) == 1
      self._descr_list.append(ReductionDescr(result,
                                             args[0],
                                             self._reduction_dims(d['result'], d['args'][0]),
                                             self.convert_reduction_op(d['optype']),
                                             prefer_align=False))
      self._append_scaling(d['result'], linear.get('alpha'))
    else:
      raise NotImplementedError(f'yateto exported an operation of type {kind!r}')

    for descr in self._descr_list[first:]:
      descr.condition = condition

    return 0# self._descr_list[-1].get_flops()

  def _append_scaling(self, result, alpha):
    """Scale a result in place, as an operation of its own.

    Neither an elementwise operation nor a reduction carries a factor, and
    folding one into a reduction would change what it starts from. Applying
    it afterwards is a multilinear over the result and the factor, which is
    the same shape of operation yateto sends for a scaled contraction.
    """
    if not self._is_named_scalar(alpha):
      return
    axes = [0] if self._is_phantom(result) else list(range(len(result['indices'])))
    self._descr_list.append(MultilinearDescr(self.tensor_ref(result),
                                             [self.tensor_ref(result),
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
    offset = d.get('offset') or [0] * bbox.rank()

    return SubTensor(tensor, bbox, offset, sliced=bool(d.get('sliced')))

  def add_tensor(self, d):
    name = d['name']
    name = f'{self._prefix}{name}'

    datatype = Datatype.ytt2enum(d['datatype'])

    datatype_new = BaseDatatype.ytt2enum(d['datatype'])

    shape = d['storage']['shape']
    storagetype = d['storage']['type']

    addressingStr = d['addressing']
    if addressingStr == '&':
      addressing = Addressing.NONE
    elif addressingStr == 'n*N+o&':
      addressing = Addressing.STRIDED
    elif addressingStr == 'n&+o&':
      addressing = Addressing.PTR_BASED
    elif addressingStr == '':
      addressing = Addressing.SCALAR

    if addressing != Addressing.SCALAR and len(shape) == 0:
      # A tensor without axes still holds one element per batch entry, and
      # every path below indexes a destination by at least one axis. Carrying
      # it as an axis of extent one is the representation `ReductionDescr`
      # already allows for a full reduction. A scalar is different: it is
      # passed by value and is never indexed.
      shape = [1]

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
                               values, datatype, d.get('alignment', 0))

    self._tensor_list[name] = TensorData(datatype_new, shape, spp, values=values)

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
    return self._descr_list, self._cache


class KernelEmitter:
  """Runs TensorForge over what the reader built and writes the call site."""

  def __init__(self, arch, attrs, descr_list, cache):
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
    if isinstance(other, TensorForgeWriter):
      return self._basename == other._basename
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

class YatetoFrontend:
  #: The version of yateto's export interface this reads. yateto refuses an
  #: exporter that speaks an older one, because the fields added since would
  #: be dropped silently rather than missed loudly.
  INTERFACE_VERSION = 3

  def __init__(self, arch, attrs=None):
    """The routine exporter yateto instantiates, once per kernel.

    ``attrs`` carries the per-kernel switches from ``Generator.add``.  A
    yateto that does not know about them calls this with ``arch`` alone, and
    the default then selects the kernel surface that predates the channel --
    a flag mask on every kernel, checked against ``nullptr``.
    """
    self._arch = arch
    self._attrs = attrs
    self._emitter = None

  def add_kernel(self, description):
    """The whole kernel, as data, in one call."""
    reader = DescriptionReader(self._arch, self._attrs)
    descr_list, cache = reader.read(description)
    self._emitter = KernelEmitter(self._arch, self._attrs, descr_list, cache)

  def generate(self, cpp, cache):
    if self._emitter is None:
      raise NotImplementedError(
        'generate() before add_kernel(): there is nothing to build.')
    self._emitter.generate(cpp, cache)
