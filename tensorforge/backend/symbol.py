import enum
from typing import Union, List
from copy import deepcopy
from tensorforge.common.matrix.boundingbox import BoundingBox
from functools import reduce
from tensorforge.common.context import Context
from tensorforge.common.basic_types import FloatingPointType, Addressing
from .writer import Writer

from tensorforge.common.matrix.spp import BoundingBoxSPP

import numpy as np

class SymbolType(enum.Enum):
  Batch = 1
  Global = 2
  SharedMem = 3
  Register = 4
  Scratch = 5
  Scalar = 6
  Data = 7,
  WarpwideSource = 8,
  WarpwideAccumulator = 9

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
    strides = []
    current = 1
    for i, size in enumerate(self.get_bbox().sizes() if bbox else self.shape):
      if i not in mask:
        strides += [current]
        current *= size
    return strides

  def get_dim_offsets(self, mask=[], bbox=False):
    # TODO: permute? Yes or no? Also, unify SPPs.
    offsets = []
    for i, start in enumerate(self.get_bbox().lower()):
      if i not in mask:
        offsets += [start]
    return offsets

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

class Immediate:
  def __init__(self, value, fptype: FloatingPointType):
    self._value = value
    self._type = fptype
  
  def is_thread_dependent(self):
    return False
  
  def write_nonlead(self):
    return self._type.literal(self._value)

  def write(self, context: Context):
    return self._type.literal(self._value)

class Variable:
  def __init__(self, name, fptype: FloatingPointType):
    self._name = name
    self._type = fptype

  def is_thread_dependent(self):
    return False

  def write_nonlead(self):
    return self._name

  def write(self, context: Context):
    return self._name

class LeadIndex:
  # TODO: make nonlead a variable
  def __init__(self, nonlead, block, stride):
    self._nonlead = nonlead
    self._block = block
    self._stride = stride
  
  def is_thread_dependent(self):
    return True
  
  def write_nonlead(self):
    return f'{self._nonlead}'

  def write(self, context: Context):
    if self._block > 1:
      return f'(({context.get_vm().get_lexic().thread_idx_x} / {self._stride}) % {self._block}) + {self._nonlead} * {self._block}'
    elif self._block == 1:
      return f'{self._nonlead}'

class LeadLoop:
  def __init__(self, name, start, end, threads, stride, unroll=False):
    self.start = start
    self.end = end
    self.unroll = unroll
    self.threads = threads
    self.var = name
    self.stride = stride

  def write(self, context: Context, writer: Writer, inner):
    actualstart = self.start // self.threads
    realstart = (self.start + self.threads - 1) // self.threads
    realend = (self.end) // self.threads
    actualend = (self.end + self.threads - 1) // self.threads

    leadExpr = f'({context.get_vm().get_lexic().thread_idx_x} / {self.stride}) % {self.threads}'

    if actualstart >= actualend:
      pass
    if actualstart == realend:
      index = LeadIndex(actualstart, self.threads, self.stride)
      startIdx = self.start - actualstart * self.threads
      endIdx = self.end - realend * self.threads
      if startIdx > 0:
        with writer.If(f'{leadExpr} >= {startIdx} && {leadExpr} < {self.end - realend * self.threads}'):
          inner([index])
      else:
        with writer.If(f'{leadExpr} < {self.end - realend * self.threads}'):
          inner([index])
    else:
      if self.start % self.threads != 0:
        index = LeadIndex(actualstart, self.threads, self.stride)
        with writer.If(f'{leadExpr} >= {self.start - actualstart}'):
          inner([index])
      if self.unroll:
        for value in range(realstart, realend):
          index = LeadIndex(value, self.threads, self.stride)
          inner([index])
      elif realstart < realend:
        # writer.insert_pragma_unroll()
        var = self.var
        with writer.For(f'int {var} = {realstart}; {var} < {realend}; {var} += 1'):
          index = LeadIndex(var, self.threads, self.stride)
          inner([index])
      if self.end % self.threads != 0:
        index = LeadIndex(actualend - 1, self.threads, self.stride)
        with writer.If(f'{leadExpr} < {self.end - realend * self.threads}'):
          inner([index])

class Loop:
  def __init__(self, name, start, end, step=1, unroll=False):
    self.start = start
    self.end = end
    self.step = step
    self.unroll = unroll
    self.var = name

  def write(self, context: Context, writer: Writer, inner):
    if self.unroll:
      for value in range(self.start, self.end, self.step):
        inner([Immediate(value, FloatingPointType.INT)])
        #inner([value])
    elif self.start < self.end:
      # writer.insert_pragma_unroll()
      var = self.var
      with writer.For(f'int {var} = {self.start}; {var} < {self.end}; {var} += {self.step}'):
        inner([Variable(var, FloatingPointType.INT)])
        #inner([var])

# TODO: add leading
class LinearizedLoop:
  def __init__(self, loops, blocksize = 1):
    self.loops = loops
    self.blocksize = blocksize
  
  def write(self, context: Context, writer: Writer, inner):
    totalloopsize = 1
    multiplies = [0] * len(self.loops)
    loopsize = [0] * len(self.loops)
    for i, loop in enumerate(self.loops):
      multiplies[i] = totalloopsize
      loopsize[i] = (loop.end - loop.start) // loop.step
      totalloopsize *= loopsize[i]
    
    loopvar = 'var'
    loopvar2 = 'var2'

    # the pragma bears great control over the application speed. And the compile time.
    writer.insert_pragma_unroll()
    with writer.For(f'int {loopvar} = 0; {loopvar} < {totalloopsize}; {loopvar} += {self.blocksize}'):
      if self.blocksize == 1:
        writer(f'int {loopvar2} = {loopvar};')
      else:
        writer(f'int {loopvar2} = {loopvar} + ({context.get_vm().get_lexic().thread_idx_x} % {self.blocksize});')
      for i, loop in enumerate(self.loops):
        writer(f'int {loop.var} = (({loopvar2} / {multiplies[i]}) % {loopsize[i]}) * {loop.step} + {loop.start};')
      inner([Variable(loop.var, FloatingPointType.INT) for loop in self.loops])

class MultiLoop:
  pass

class SparseLoop:
  pass

def write_loops(context: Context, writer: Writer, loops: List[Loop], inner):
  def write_loops_inner(context: Context, writer: Writer, loops: List[Loop], inner, varlist):
    if len(loops) == 0:
      with writer.Scope():
        inner(varlist)
    else:
      inner_next = lambda v: write_loops_inner(context, writer, loops[1:], inner, varlist + v)
      loops[0].write(context, writer, inner_next)
  write_loops_inner(context, writer, loops, inner, [])

class Symbol:
  def __init__(self,
               name: str,
               stype: SymbolType,
               obj):
    self.name = name
    self.stype = stype
    self.obj = obj
    self.data_view: Union[DataView, None] = None
    self.datatype: Union[FloatingPointType, None] = None
    self.num_threads = None
    self.lead_dims = [0] # has only an effect for register storage
    self._users = []
  
  def clone(self):
    cloned = Symbol(self.name, self.stype, self.obj)
    cloned.data_view = deepcopy(self.data_view)
    cloned.datatype = self.datatype
    cloned._users = [user for user in self._users]
    cloned.lead_dims = [ld for ld in self.lead_dims]
    return cloned

  def get_fptype(self, context: Context):
    # TODO: make obsolete
    if self.datatype is not None:
      return self.datatype
    elif self.obj is not None and self.obj.datatype is not None:
      return self.obj.datatype
    else:
      return context.fp_type

  def address(self):
    if self.stype == SymbolType.Scalar:
      return f'&{self.name}'
    else:
      return f'{self.name}'

  def access_address(self, context: Context, index: List[Union[str, int, Immediate, Variable, LeadIndex]]):
    if self.stype == SymbolType.Global or self.stype == SymbolType.Batch or self.stype == SymbolType.SharedMem:
      writevar = lambda var: f'{var}' if isinstance(var, (str, int, float, np.int64)) else var.write(context)
      # lead_dim + nonlead_dim
      # TODO: really ref self.obj.bbox.lower() here?
      # self.obj.bbox.lower()
      writeOffset = lambda i,var,offset,stride: f"({writevar(var)} - {offset}) * {stride}"
      dimstr = " + ".join(writeOffset(i,var,offset,stride) for i, (var, offset, stride) in enumerate(zip(index, self.data_view.get_dim_offsets(), self.data_view.get_dim_strides())))
      return dimstr if len(dimstr) > 0 else "0"
    if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      writevar = lambda var: f'{var}' if isinstance(var, (str, int, float, np.int64)) else var.write_nonlead()
      writeOffset = lambda i,var,offset,stride: f"({writevar(var)} - {offset}) * {stride}"
      writeNoOffset = lambda i,var,offset,stride: f"{writevar(var)} * {stride}"
      writers = [0] * self.data_view.rank()
      strides = [0] * self.data_view.rank()
      stride = 1
      for i in range(self.data_view.rank()):
        strides[i] = stride
        if isinstance(index[i], LeadIndex):
          stride *= (self.data_view.get_dim_size(i) + self.num_threads - 1) // self.num_threads
          writers[i] = writeNoOffset
        else:
          stride *= self.data_view.get_dim_size(i)
          writers[i] = writeOffset
      dimstr = " + ".join(writer(i,var,offset,stride) for i, (var, offset, stride, writer) in enumerate(zip(index, self.data_view.get_dim_offsets(), strides, writers)))
      return dimstr if len(dimstr) > 0 else "0"
    raise NotImplementedError('Not supposed to be called')

  def access(self, context: Context, index: List[Union[str, int, Immediate, Variable, LeadIndex]]):
    if self.stype == SymbolType.Global or self.stype == SymbolType.Batch or self.stype == SymbolType.SharedMem or self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      return f'{self.name}[{self.access_address(context, index)}]'
    if self.stype == SymbolType.Scalar:
      return f'{self.name}'
    if self.stype == SymbolType.Data:
      return self.get_fptype(context).literal(self.obj.value(runIdx))
  
  def encode_values(self, pos, runIdx, writer, context: Context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp, leadidx):
    if pos == len(index):
      if self.stype == SymbolType.Data:
        value = self.obj.value(runIdx)
        if value is not None:
          writer(f'{variable} = {self.get_fptype(context).literal(value)};')
      else:
        # TODO: unite with access_address
        if leadidx is None:
          value = self.obj.linear_index(runIdx)
          if value is not None:
            writer(f'{variable} = {self.name}[{value}];')
        else:
          strindex = index[leadidx].write(context)
          rngs = []
          rng = None
          startValue = None
          for i in range(self.data_view.get_dim_size(leadidx)):
            runIdx[leadidx] = i
            value = self.obj.linear_index(runIdx)
            if value is not None:
              if rng is None:
                rng = i
                startValue = value
              elif rng is not None and (value - startValue) != (i - rng) * index[leadidx]._stride:
                rngs += [(rng, i)]
                rng = i
                startValue = value
            elif value is None and rng is not None:
              rngs += [(rng, i)]
              rng = None
              startValue = None
          if rng is not None:
            rngs += [(rng, self.data_view.get_dim_size(leadidx))]

          if len(rngs) > 0:
            idxvar = writer.varalloc()
            writer(f'const int {idxvar} = {strindex};')
            for rngS, rngE in rngs:
              runIdx[leadidx] = rngS
              value = self.obj.linear_index(runIdx)
              with writer.If(f'{idxvar} >= {rngS} && {idxvar} < {rngE}'):
                writer(f'{variable} = {self.name}[{value - rngS} + {idxvar}];')
    else:
      offset = self.data_view.get_dim_offsets()[pos]
      if isinstance(index[pos], int):
        runIdx[pos] = index[pos]
        self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
      elif isinstance(index[pos], Immediate):
        runIdx[pos] = index[pos]._value
        self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
      elif pos == leadidx:
        self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
      else:
        # TODO: move block sparsity one level up
        strindex = f'{index[pos]}' if isinstance(index[pos], (str, int, float, np.int64)) else index[pos].write(context)
        if True: # sparse/data
          # TODO: check sparsity pattern here for which ifs are worth it
          for i in range(self.data_view.get_dim_size(pos)):
            runIdx[pos] = i
            with writer.If(f'({strindex} - {offset}) == {runIdx[pos]}'):
              self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)

  def load(self, writer, context: Context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp):
    if self.stype == SymbolType.Data or (not self.obj.is_dense() and not isinstance(self.obj.spp, BoundingBoxSPP)):
      writer(f'{self.get_fptype(context)} {variable} = {self.get_fptype(context).literal(0)};')

      # treat the lead index last for better sparsity handling
      leadidx = None
      for i,idx in enumerate(index):
        if isinstance(idx, LeadIndex):
          if leadidx is None:
            leadidx = idx
          else:
            leadidx = None
            break

      if leadidx is not None:
        leadidxidx = index.index(leadidx)
      else:
        leadidxidx = None
      self.encode_values(0, [0] * len(index), writer, context, variable, index, nontemp, leadidxidx)
    else:
      pre_access = self.access(context, index)
      if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
        assert len(self.lead_dims) == 1
        idx = index[self.lead_dims[0]]
        if not isinstance(idx, LeadIndex):
          writevar = lambda var: f'{var}' if isinstance(var, (str, int, float, np.int64)) else var.write(context)
          access = context.get_vm().get_lexic().broadcast_sync(pre_access, writevar(idx), -1)
        else:
          access = pre_access
      else:
        access = pre_access
      if self.stype == SymbolType.Global:
        writer(f'{self.get_fptype(context)} {variable};')
        writer(context.get_vm().get_lexic().glb_load(variable, access, nontemp))
      else:
        writer(f'{self.get_fptype(context)} {variable} = {access};')
  
  def store(self, writer, context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp):
    assert self.stype != SymbolType.Data

    access = self.access(context, index)

    if self.stype == SymbolType.Global:
      assign = context.get_vm().get_lexic().glb_store(access, variable, nontemp)
    else:
      assign = f'{access} = {variable};'
    if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      assert len(self.lead_dims) == 1
      if isinstance(index[self.lead_dims[0]], LeadIndex):
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
  
  def __repr__(self):
    return self.__str__()

class SymbolView:
  def __init__(self, symbol, view = None):
    self.symbol = symbol
    self.bbox = view
    if view is None:
      self.bbox = symbol.data_view.get_bbox()
  
  def __str__(self):
    return f'{self.symbol} {self.bbox}'
  
  def __repr__(self):
    return self.__str__()
