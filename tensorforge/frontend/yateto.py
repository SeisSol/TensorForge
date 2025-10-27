
from tensorforge.interface import YatetoInterface as yi
from tensorforge.common.basic_types import Addressing, FloatingPointType, DataFlowDirection
from tensorforge.common.context import Context
from tensorforge.common.aux import generate_tmp_tensor
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.matrix.spp import FullSPP, BoundingBoxSPP, ListSPP
from tensorforge.common.matrix.boundingbox import BoundingBox as BBox
from tensorforge.generators.descriptions import ElementwiseDescr
from tensorforge.generators.generator import Generator as TensorForgeGenerator
from tensorforge.generators.descriptions import MultilinearDescr

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
      return SubTensor(self._cache[op.name()])
    else:
      tensor = self._cache[op.name]
      currentPreShape = BBox([s for s, _ in op.eqspp.nnzbounds()], [e+1 for _, e in op.eqspp.nnzbounds()])

      tml = op.memoryLayout
      if type(op.memoryLayout).__name__ == 'MemoryLayoutView':
        relidx = op.memoryLayout.relidx([0] * len(currentPreShape._lower))
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

      return SubTensor(tensor, currentPreShape)

  def add_scalar(self, ops, statements, indices):
    indicesIndexed = {}
    for i,op in enumerate(ops):
      self.make_tensor(op, False, None)
      indicesIndexed[op.name() if self.is_scalar(op) else op.name] = indices[i]
    
    def assigner(pretensor):
      if self.is_scalar(pretensor):
        self.make_tensor(pretensor, False, None)
        indicesIndexed[pretensor.name()] = []
        subTensor = SubTensor(self._cache[pretensor.name()], BBox([], []))
      else:
        bbox = BBox([s for s, _ in pretensor.eqspp().nnzbounds()], [e+1 for _, e in pretensor.eqspp().nnzbounds()])
        subTensor = SubTensor(self._cache[pretensor.name()], bbox)
      return subTensor, indicesIndexed[pretensor.name()]
    
    for statement in statements:
      statement.assignTensor(assigner)
    
    self._descr_list.append(ElementwiseDescr(statements,
                                strict_match=False,
                                prefer_align=False))
    return 0

  def generate(self, cpp, routineCache):
    if hasattr(self._arch, 'typename'):
      fptype = FloatingPointType.str2enum(self._arch.typename)
    else:
      fptype = FloatingPointType.FLOAT

    context = Context(arch=self._arch.name,
                      backend=self._arch.backend,
                      fp_type=fptype)

    # print(self._ir_list)

    tensorforge_generator = TensorForgeGenerator(self._descr_list, context)
    tensorforge_generator.register()

    cpp(f'{self._gen_call_site(tensorforge_generator)}')
    routine_name = tensorforge_generator.get_base_name()
    tensorforge_generator.generate()

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
      entry = self._get_tensorforge_matrix(tensor=op,
                                          shape=[rng.stop for rng in self._storage(op.memoryLayout).bbox()],
                                          bboxrange=self._storage(op.memoryLayout).bbox())
      entry_name = op.name
    
    if not (entry_name in self._cache and entry.is_same(self._cache[entry_name])):
      self._cache[entry_name] = entry
  
  def tensor_ref(self, d):
    name = d['name']
    eqspp = d['spp']

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
    datatype = FloatingPointType.ytt2enum(d['datatype'])

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
      if dest.name not in self._tmp_matrices:
        self._cache[dest.name] = self._gen_tmp_matix(ops, target, permute, dest.name, can_be_aligned)
    else:
      self.make_tensor(dest, can_be_aligned, [i for i in range(len(dest.indices))])

  def _add_scalar(self, scalar):
    tensor = Tensor([], Addressing.SCALAR, alias=scalar.name(), datatype=scalar.datatype)
    self._tmp_matrices[scalar.name()] = tensor # SubTensor(tensor, tensor.bbox)
    return self._tmp_matrices[scalar.name()]
  
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

  def _get_tensorforge_matrix(self, tensor, shape, bboxrange):
    tml = self._storage(tensor.memoryLayout)

    addr_mode = self.deduce_addresing(tensor) if tensor.addressing is None else tensor.addressing
    if tensor.is_temporary:
      if not tensor.name in self._tmp_matrices:
        raise RuntimeError(f'expected tmp. tensor {tensor.name} to be cached '
                           f'while code generation for fused-gemms')
      else:
        return self._tmp_matrices[tensor.name]

    if type(tml).__name__ == 'DenseMemoryLayout':
      pattern = None
    else:
      rowRange = range(tml.bbox()[0].start, tml.bbox()[0].stop)
      colRange = range(tml.bbox()[1].start, tml.bbox()[1].stop)
      pattern = tml.entries(rowRange, colRange)
      # incorrect:
      # pattern = tensor.eqspp.as_ndarray()

    return yi.gen_matrix(shape,
                               bboxrange,
                               addressing=addr_mode,
                               name=tensor.name,
                               is_tmp=tensor.is_temporary,
                               permute=None,
                               pattern=pattern,
                               values = tensor.values,
                               datatype = tensor.datatype)

  def _gen_tmp_matix(self, ops, target, permute, res_name, can_be_aligned):
    # TODO: ignore scalars here?
    tmp_matrix = generate_tmp_tensor(ops=[self.get_tensor(op, can_be_aligned, optarget) for op, optarget in zip(ops, target)],
                                     target=target, alias=res_name) # permute?
    self._tmp_matrices[res_name] = tmp_matrix
    return tmp_matrix

  def _gen_call_site(self, generator):
    mat_name_map = {}
    offset_name_map = {}
    for name, matrix in self._cache.items():
      if matrix.direction == DataFlowDirection.SOURCE and matrix.addressing != Addressing.SCALAR:
        datatype = FloatingPointType.str2enum(self._arch.typename) if matrix.datatype is None else matrix.datatype
        ptr_type = f'const {datatype}{matrix.addressing.to_pointer()}'
        mat_name_map[name] = f'const_cast<{ptr_type}>({name})'
      else:
        mat_name_map[name] = name

      if matrix.is_tmp or matrix.addressing == Addressing.NONE:
        offset_name_map[name] = '0'
      else:
        offset_name_map[name] = f'extraOffset_{name}'
    
    return generator.generate_call_site(mat_name_map,
                                        offset_name_map,
                                        'numElements',
                                        'flags',
                                        'streamPtr')
  
  def _append_operation(self, op):
    if isinstance(op, (float, int)):
      return Tensor([], Addressing.SCALAR, data = op)
    elif self.is_scalar(op):
      return self._cache[op.name()]
    else:
      return self._cache[op.name]

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

  def add_operation(self, description):
    return self.generator.add_operation_new(description)

  def add_tensor(self, tensor):
    return self.generator.add_tensor(tensor)
