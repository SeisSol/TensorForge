
from tensorforge.interface import YatetoInterface as yi
from tensorforge.common.basic_types import Addressing, Datatype, DataFlowDirection
from tensorforge.common.context import Context
from tensorforge.common.helper import generate_tmp_tensor
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.matrix.spp import FullSPP, BoundingBoxSPP, ListSPP
from tensorforge.common.matrix.boundingbox import BoundingBox as BBox
from tensorforge.generators.generator import Generator as TensorForgeGenerator
from tensorforge.generators.descriptions import MultilinearDescr, ElementwiseDescr, GridBarrierDescr, GridFenceDescr, RegionDescription

from tensorforge.ir.data.variable import TensorView, TensorAlloc
from tensorforge.ir.data.variable import TensorData
from tensorforge.ir.logical.compute import Multilinear
from tensorforge.ir.type import BaseDatatype
from tensorforge.ir.data.memory import Logical

class GpuKernelGeneratorV1:
  def __init__(self, arch):
    self._arch = arch
    self._cache = {}
    self._tmp_matrices = {}

    # to be replaced by the IR list
    self._descr_list = []

    self._ir_list = []
    self._tensor_list = {}

    # TODO: maybe remove again
    self._prefix = ""

  def add_operation(self, dest, ops, target, permute, add):
    self._cache_matrices(dest, ops, target, permute)
    can_be_aligned = self._can_be_aligned(dest, ops, target, permute)
    self._descr_list.append(MultilinearDescr(self.get_tensor(dest, can_be_aligned, [i for i in range(len(dest.indices))]),
                              [self.get_tensor(op, can_be_aligned, optarget) for op, optarget in zip(ops, target)],
                              target, permute, add=add,
                                strict_match=False,
                                prefer_align=can_be_aligned))
    return 0# self._descr_list[-1].get_flops()

  def add_operation_new(self, d):
    result = self.tensor_ref(d['result'])
    args = [self.tensor_ref(arg) for arg in d['args']]

    condition_raw = d['condition']
    condition = [self.tensor_ref(var) for clause in condition_raw for var in clause]
    # condition = self.tensor_ref(d['condition'])

    if d['type'] == 'reduction':
      assert len(args) == 1
      op = self.convert_op(d['optype'])

    if d['type'] == 'elementwise':
      op = self.convert_op(d['optype'])

    if d['type'] == 'matmul':
      pass

    if 'linear' in d['type']:
      alpha = self.tensor_ref(d['linear']['alpha'])
      add = d['linear']['add']

    if d['type'] == 'multilinear':
      target = d['target']
      permute = d['permute']

      # TODO

      alpha = self.tensor_ref(d['linear']['alpha'])
      add = d['linear']['add']

      # ElementwiseDescr()
      self._descr_list.append(MultilinearDescr(result,
                              args,
                              target, permute, add=add,
                                strict_match=False,
                                prefer_align=False))

      result = self.tensor_ref_new(d['result'])
      args = [self.tensor_ref_new(arg) for arg in d['args']]

      condition_raw = d['condition']
      condition = [self.tensor_ref_new(var) for clause in condition_raw for var in clause]

      self._ir_list.append(Multilinear(result, None, None, args, target, add))

    return 0# self._descr_list[-1].get_flops()

  def convert_op(self):
    pass

  def is_scalar(self, op):
    # a bit hacky...
    return not hasattr(op, 'memoryLayout') and not isinstance(op, (float, int)) #TODO: isinstance(op, Scalar):

  def get_tensor(self, op, can_be_aligned, dims):
    if isinstance(op, (float, int)):
      return SubTensor(tensor = Tensor([], Addressing.SCALAR, data = [op]))
    elif self.is_scalar(op):
      return SubTensor(self._cache[f'{self._prefix}{op.name()}'])
    else:
      tensor = self._cache[f'{self._prefix}{op.name}']
      currentPreShape = BBox([s for s, _ in op.eqspp.nnzbounds()], [e+1 for _, e in op.eqspp.nnzbounds()])

      tml = op.memoryLayout
      relidx = [0] * currentPreShape.rank()
      if type(tml).__name__ == 'MemoryLayoutView':
        while tml != tml.storage():
          relidx = tml.relidx(relidx)
          tml = tml.base
        currentPreShape = BBox([x + relidx[i] for i, x in enumerate(currentPreShape._lower)], [x + relidx[i] for i, x in enumerate(currentPreShape._upper)])

        tml = tml.storage()

      if can_be_aligned:
        for i, dim in enumerate(dims):
          if i == 0 and tml.alignedStride(): # previously: dim == 0
            # a bit hacky right now...
            newLower = self._arch.alignedLower(currentPreShape._lower[i])
            newUpper = self._arch.alignedUpper(currentPreShape._upper[i])

            # TODO: check this condition again
            newUpper = min(newUpper, tml.bbox()[0].stop)

            currentPreShape._lower = tuple([newLower] + list(currentPreShape._lower[1:]))
            currentPreShape._upper = tuple([newUpper] + list(currentPreShape._upper[1:]))

      return SubTensor(tensor, currentPreShape, relidx)

  def add_scalar(self, ops, statements, indices):
    indicesIndexed = {}
    for i,op in enumerate(ops):
      self.make_tensor(op, False, None)
      indicesIndexed[op.name() if self.is_scalar(op) else op.name] = indices[i]

    def assigner(pretensor):
      if self.is_scalar(pretensor):
        self.make_tensor(pretensor, False, None)
        indicesIndexed[pretensor.name()] = []
        subTensor = SubTensor(self._cache[f'{self._prefix}{pretensor.name()}'], BBox([], []))
      else:
        bbox = BBox([s for s, _ in pretensor.eqspp().nnzbounds()], [e+1 for _, e in pretensor.eqspp().nnzbounds()])
        subTensor = SubTensor(self._cache[f'{self._prefix}{pretensor.name()}'], bbox)
      return subTensor, indicesIndexed[pretensor.name()]

    for statement in statements:
      statement.assignTensor(assigner)

    self._descr_list.append(ElementwiseDescr(statements,
                                strict_match=False,
                                prefer_align=False))
    return 0

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

  def generate(self, cpp, routineCache):
    if hasattr(self._arch, 'typename'):
      fptype = Datatype.str2enum(self._arch.typename)
    else:
      fptype = None

    context = Context(arch=self._arch.name,
                      backend=self._arch.backend,
                      fp_type=fptype)

    # print(self._ir_list)

    tensorforge_generator = TensorForgeGenerator(self._descr_list, context)
    tensorforge_generator.generate()

    cpp(f'{self._gen_call_site(tensorforge_generator)}')
    routine_name = tensorforge_generator.get_base_name()

    routineCache.addRoutine(routine_name, TensorForgeWriter(tensorforge_generator, context.get_vm().get_headers()))

  def _can_be_aligned(self, dest, ops, target, permute):
    # TODO: useful?
    aligned = dest.memoryLayout.alignedStride()
    for i, op in enumerate(ops):
      if 0 in target[i]:
        aligned &= dest.memoryLayout.alignedStride() and permute[i][0] == 0

    return aligned

  def make_tensor(self, op, can_be_aligned, dims):
    if isinstance(op, (float, int)):
      return Tensor([], Addressing.SCALAR, data = [op])
    if self.is_scalar(op):
      entry = self._add_scalar(op)
      entry_name = op.name()
    else:
      entry = self._get_tensorforge_matrix(op)
      entry_name = op.name

    entry_name = f'{self._prefix}{entry_name}'

    if not (entry_name in self._cache and entry.is_same(self._cache[entry_name])):
      self._cache[entry_name] = entry

  def tensor_ref(self, d):
    name = d['name']
    eqspp = d['spp']

    name = f'{self._prefix}{name}'

    assert(name in self._cache)

    return SubTensor(self._cache[name], self._cache[name].bbox)

  def tensor_ref_new(self, d):
    name = d['name']
    eqspp = d['spp']

    assert(name in self._cache)

    # TODO: bbox

    return TensorAlloc(name, self._tensor_list[name], Logical())

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

    if storagetype == 'full':
      spp = FullSPP(shape)
      bbox = None
    if storagetype == 'bbox':
      starts = d['storage']['start']
      sizes = d['storage']['sizes']
      lower = list(starts)
      upper = [start + size for start, size in zip(starts, sizes)]
      bbox = BBox(lower, upper)
      spp = FullSPP(shape)#BoundingBoxSPP(bbox)
    if storagetype == 'spp':
      bbox = None
      spp = ListSPP(d['storage']['entries'])

    values = d['values']
    is_temporary = d['flags']['temporary']
    is_constant = d['flags']['constant']

    self._cache[name] = Tensor(shape, addressing, bbox, name, is_temporary, spp, values, datatype)

    self._tensor_list[name] = TensorData(datatype_new, shape, spp, values=values)

  def _cache_matrices(self, dest, ops, target, permute):
    can_be_aligned = self._can_be_aligned(dest, ops, target, permute)

    # no add onto a matrix that doesn't exist (TODO: check if that's always the case)
    assert not(dest.is_temporary and dest in ops)

    for op, optarget in zip(ops, target):
      self.make_tensor(op, can_be_aligned, optarget)

    if dest.is_temporary: # (dest is never a scalar---for the time being)
      self.make_tensor(dest, can_be_aligned, [i for i in range(len(dest.indices))])
      self._tmp_matrices[f'{self._prefix}{dest.name}'] = self._cache[f'{self._prefix}{dest.name}']
    else:
      self.make_tensor(dest, can_be_aligned, [i for i in range(len(dest.indices))])



  def _add_scalar(self, scalar):
    name = f'{self._prefix}{scalar.name()}'
    tensor = Tensor([], Addressing.SCALAR, alias=name, datatype=self._datatype(scalar.datatype))
    self._tmp_matrices[name] = tensor # SubTensor(tensor, tensor.bbox)
    return self._tmp_matrices[name]

  def deduce_addresing(self, term):
    if term.is_compute_constant:
      return Addressing.NONE
    if term.is_temporary:
      return Addressing.STRIDED
    else:
      return Addressing.PTR_BASED

  def _storage(self, tml):
    if type(tml).__name__ == 'MemoryLayoutView':
      return tml.storage()
    return tml

  def _get_tensorforge_matrix(self, tensor):
    tml = self._storage(tensor.memoryLayout)

    shape=[rng.stop for rng in tml.bbox()]
    bboxrange=tml.bbox()

    addr_mode = self.deduce_addresing(tensor) if tensor.addressing is None else tensor.addressing
    if tensor.is_temporary and tensor.name in self._tmp_matrices:
      return self._tmp_matrices[tensor.name]

    if type(tml).__name__ == 'DenseMemoryLayout':
      pattern = None
    else:
      #ranges = []
      #for i in range(len(shape)):
      #  ranges += [range(tml.bbox()[i].start, tml.bbox()[i].stop)]
      ranges = []
      for i in range(len(shape)):
        ranges += [range(0, shape[i])]
      pattern = tml.entries(*ranges)
      # incorrect:
      # pattern = tensor.eqspp.as_ndarray()

    return yi.gen_matrix(shape,
                               bboxrange,
                               addressing=addr_mode,
                               name=f'{self._prefix}{tensor.name}',
                               is_tmp=tensor.is_temporary,
                               permute=None,
                               pattern=pattern,
                               values = tensor.values,
                               datatype = self._datatype(tensor.datatype))

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

  def _append_operation(self, op):
    if isinstance(op, (float, int)):
      return Tensor([], Addressing.SCALAR, data = op)
    elif self.is_scalar(op):
      return self._cache[f'{self._prefix}{op.name()}']
    else:
      return self._cache[f'{self._prefix}{op.name}']

  def switch_region(self, barrier):
    if barrier:
      self._descr_list += [GridBarrierDescr()]
    else:
      self._descr_list += [GridFenceDescr()]

  def set_region_name(self, name):
    self._prefix = f"{name}."
    self._descr_list += [RegionDescription(name)]

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
  def __init__(self, arch):
    self.generator = GpuKernelGeneratorV1(arch)

  def generate(self, cpp, cache):
    self.generator.generate(cpp, cache)

  def add_linear_operation(self, dest, ops, target, permute, add):
    # legacy gateway
    return self.generator.add_operation(dest, ops, target, permute, add)

  def region_switch(self, barrier):
    self.generator.switch_region(barrier)
    return 0

  def set_region_name(self, name):
    self.generator.set_region_name(name)

  def add_operation(self, description):
    return self.generator.add_operation_new(description)

  def add_tensor(self, tensor):
    return self.generator.add_tensor(tensor)
