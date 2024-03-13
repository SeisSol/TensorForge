import enum
from typing import Union, List
from copy import deepcopy
from kernelforge.common.matrix.boundingbox import BoundingBox
from functools import reduce
from kernelforge.common.context import Context
from kernelforge.common.basic_types import FloatingPointType, Addressing

class SymbolType(enum.Enum):
  Batch = 1
  Global = 2
  SharedMem = 3
  Register = 4
  Scratch = 5
  Scalar = 6
  Data = 7

def determine_dim_index(term, index, shape, permute):
  divpos = reduce(lambda x,y: shape[x]*shape[y], permute[:index], 1)
  modpos = shape[permute[index]]
  return f'((({term}) / {divpos}) % {modpos})'

class SparseDataView:
  def __init__(self, shape: List[int], permute: Union[List[int], None], ssp):
    pass

class DataView:
  def __init__(self, shape: List[int], permute: Union[List[int], None], bbox: BoundingBox = None):
    self.shape = shape
    if permute is None:
      permute = [i for i in range(len(shape))]
    if bbox is None:
      bbox = BoundingBox([0] * len(shape), shape)
    self._permute = permute
    self._bbox = bbox

  def get_bbox(self):
    return deepcopy(self._bbox)

  def reset_bbox(self, bbox):
    self._bbox = bbox
    self._offset = self.get_offset()

  def get_offset(self):
    addr = 0
    for i, s in reversed(zip(self._bbox.lower()[1:], self.shape[:-1])):
      addr = s * (i + addr)
    addr = self._bbox.lower()[0] + addr
    return addr

  def rank(self):
    return len(self.shape)

  def get_volume(self):
    volume = 1
    for s in self.shape:
      volume *= s
    return volume

  def get_dim_size(self, index):
    assert index >= 0 and index < len(self.shape)
    return self._bbox.size(index)
  
  def get_dim_strides(self, mask=[], bbox=False):
    # TODO: permute? Yes or no? Also, unify SPPs.
    strides = [1] * self.rank()
    current = 1
    for i, size in enumerate(self.get_bbox().sizes() if bbox else self.shape):
      if i not in mask:
        strides[i] = current
        current *= size
    return strides

  def get_dimension_addressing(self, lead_dim, nonlead_dim):
    return [determine_dim_index(lead_dim, i, self.shape, self._permute) for i in range(self._lead_dim_len)] + [f'k{i}' for i in range(len(self._permute) - self._lead_dim_len)]
    # + [determine_dim_index(nonlead_dim, i, self.shape, self._permute[self._lead_dim_len:]) for i in range(len(self._permute) - self._lead_dim_len)]

  def get_address(self, lead_dim, nonlead_dim):
    index = self.get_dimension_addressing(lead_dim, nonlead_dim)
    addr = '0'
    for i, s in reversed(zip(index[1:], self.shape[:-1])):
      addr = f'{s} * ({i} + {addr})'
    addr = f'{index[0]} + {addr}'
    if self._offset:
      addr = f'{self._offset} + {addr}'
    return addr

  def __str__(self):
    return f'shape: {self.shape}, permute: {self._permute}'

class OldDataView:
  def __init__(self, rows: int, columns: int, is_transposed: bool, bbox: List[int] = None):
    self._rows = rows
    self._columns = columns
    self.is_transposed = is_transposed
    if not bbox:
      bbox = [0, 0, rows, columns]
    self._bbox = bbox
    self._lead_dim = self.get_lead_dim()
    self._offset = self.get_offset()

  def get_bbox(self):
    return deepcopy(self._bbox)

  def reset_bbox(self, bbox):
    assert bbox[2] - bbox[0] <= self._rows
    assert bbox[3] - bbox[1] <= self._columns
    self._bbox = bbox
    self._offset = self.get_offset()

  def get_offset(self):
    return self._bbox[0] + self._bbox[1] * self._lead_dim

  def get_volume(self):
    return self._rows * self._columns

  def get_lead_dim(self):
    return self._rows

  def get_dim_size(self, index):
    assert index >= 0 and index < 2
    return self._bbox[2 + index] - self._bbox[index]

  def get_address(self, row_idx, column_idx):
    addr = f'{row_idx} + {column_idx} * {self._lead_dim}'
    if self._offset:
      addr = f'{self._offset} + {addr}'
    return addr

  def __str__(self):
    return f'rows: {self.rows}, cols: {self.columns}, lid: {self.lead_dim}, trans: {self.is_transposed}'

class Immediate:
  def __init__(self, value, fptype: FloatingPointType):
    self._value = value
    self._type = fptype
  
  def write(context: Context):
    return self._type.literal(self._value)

class Variable:
  def __init__(self, name):
    self._name = name

  def write(context: Context):
    return self._name

class LeadIndex:
  def __init__(self, lane):
    self._lane = lane

  def write(context: Context):
    return f'({context.get_vm().get_lexic().thread_idx_x} % {self._lane})'

class Symbol:
  def __init__(self,
               name: str,
               stype: SymbolType,
               obj):
    self.name = name
    self.stype = stype
    self.obj = obj
    self.data_view: Union[DataView, None] = None
    self.fptype: Union[FloatingPointType, None] = None
    self.lead_dims = [0] # has only an effect for register storage
    self._users = []
  
  def clone(self):
    cloned = Symbol(self.name, self.stype, self.obj)
    cloned.data_view = deepcopy(self.data_view)
    cloned.fptype = self.fptype
    cloned._users = [user for user in self._users]
    cloned.lead_dims = [ld for ld in self.lead_dims]
    return cloned

  def get_fptype(self, context: Context):
    # TODO: make obsolete
    if self.fptype is not None:
      return self.fptype
    else:
      return context.fp_type

  def address(self):
    if self.stype == SymbolType.Scalar:
      return f'&{self.name}'
    else:
      return f'{self.name}'
  
  def access_address(self, context: Context, index: List[Union[str, int]]):
    if self.stype == SymbolType.Global or self.stype == SymbolType.Batch or self.stype == SymbolType.SharedMem:
      # lead_dim + nonlead_dim
      dimstr = " + ".join(f"{var} * {stride}" for var, stride in zip(index, self.data_view.get_dim_strides()))
      return dimstr if len(dimstr) > 0 else "0"
    if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      # nonlead_dim only
      filtered_index = map(lambda x: x[1], filter(lambda x: x[0] not in self.lead_dims, enumerate(index)))
      dimstr = " + ".join(f"{var} * {stride}" for var, stride in zip(filtered_index, self.data_view.get_dim_strides(self.lead_dims, True)))
      return dimstr if len(dimstr) > 0 else "0"
    raise NotImplementedError('Not supposed to be called')

  def access(self, context: Context, index: List[Union[str, int]]):
    if self.stype == SymbolType.Global or self.stype == SymbolType.Batch or self.stype == SymbolType.SharedMem or self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      return f'{self.name}[{self.access_address(context, index)}]'
    if self.stype == SymbolType.Scalar:
      return f'{self.name}'
    if self.stype == SymbolType.Data:
      return self.get_fptype.literal(self.obj.value(runIdx))
  
  def encode_values(self, pos, runIdx, writer, context: Context, variable, index: List[Union[str, int]], nontemp):
    if pos == len(index):
      value = self.obj.value(runIdx)
      if value is not None:
        writer(f'{variable} = {self.get_fptype.literal(value)};')
      # TODO: variable reference if no value present (self.obj.index(runIdx)?)
    if isinstance(index[pos], int):
      runIdx[pos] = index[pos]
    else:
      if True: # sparse/data
        # TODO: check sparsity pattern here for which ifs are worth it
        for i in range(self.data_view._shape[pos]):
          myIndex[pos] = i
          with writer.If(f'{index[pos]} == {i}'):
            self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp)

  def load(self, writer, context: Context, variable, index: List[Union[str, int]], nontemp):
    if self.stype == SymbolType.Data:
      writer(f'{self.get_fptype(context)} {variable} = {self.get_fptype(context).literal(0)};')
      self.encode_values(0, [0] * len(index), writer, context, variable, index, nontemp)
    else:
      pre_access = self.access(context, index)
      if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
        assert len(self.lead_dims) == 1
        if index[self.lead_dims[0]] != context.get_vm().get_lexic().thread_idx_x:
          access = context.get_vm().get_lexic().broadcast_sync(pre_access, index[self.lead_dims[0]], -1)
        else:
          access = pre_access
      else:
        access = pre_access
      if self.stype == SymbolType.Global:
        writer(f'{self.get_fptype(context)} {variable};')
        writer(context.get_vm().get_lexic().glb_load(variable, access, nontemp))
      else:
        writer(f'{self.get_fptype(context)} {variable} = {access};')
  
  def store(self, writer, context, variable, index: List[Union[str, int]], nontemp):
    assert self.stype != SymbolType.Scalar and self.stype != SymbolType.Data

    access = self.access(context, index)

    if self.stype == SymbolType.Global:
      assign = context.get_vm().get_lexic().glb_store(access, variable, nontemp)
    else:
      assign = f'{access} = {variable};'
    if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      assert len(self.lead_dims) == 1
      if index[self.lead_dims[0]] == context.get_vm().get_lexic().thread_idx_x:
        writer(assign)
      else:
        with writer.If(f'{context.get_vm().get_lexic().thread_idx_x} == {index[self.lead_dims[0]]}'):
          writer(assign)
    else:
      writer(assign)

  def add_user(self, user):
    self._users.append(user)

  def get_user_list(self):
    # set by instructions
    return self._users

  def get_first_user(self):
    return self._users[0]
  
  def get_last_user(self):
    return self._users[-1]

  def __str__(self):
    return f'name: {self.name}, type: {self.stype}, lead: {self.lead_dims}'

