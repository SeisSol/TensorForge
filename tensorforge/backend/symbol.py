import enum
from typing import Union, List
from copy import deepcopy
from tensorforge.common.matrix.boundingbox import BoundingBox
from functools import reduce
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Datatype, Addressing
from tensorforge.common.exceptions import GenerationError
from .writer import Writer
from tensorforge.backend.pir.core import BOOL, INDEX

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
  def __init__(self, value, fptype: Datatype):
    self._value = value
    self._type = fptype

  def is_thread_dependent(self):
    return False

  def write_nonlead(self):
    return self._type.literal(self._value)

  def write(self, context: Context):
    return self._type.literal(self._value)

  def nonlead(self):
    return self._value

  def lead(self):
    return self._value

class Variable:
  def __init__(self, name, fptype: Datatype):
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
    if context.get_vm().get_lexic().simd_mode:
      return f'({self._nonlead} * {self._block})'
    elif self._block > 1:
      return f'(({context.get_vm().get_lexic().thread_idx_x} / {self._stride}) % {self._block}) + {self._nonlead} * {self._block}'
    elif self._block == 1:
      return f'{self._nonlead}'

  def nonlead(self):
    return self._nonlead

  def lead(self):
    return self._nonlead * self._block

class VarOffset:
  def __init__(self, variable, offset):
    self.variable = variable
    self.offset = offset

  def is_thread_dependent(self):
    return self.variable.is_thread_dependent()

  def write_nonlead(self):
    # TODO: lead
    return f'({self.variable.write_nonlead()} + {self.offset})'

  def write(self, context: Context):
    # TODO: lead
    return f'({self.variable.write(context)} + {self.offset})'

def add_offset(x, offset):
  if offset == 0:
    return x
  elif isinstance(x, (float, int, np.int64)):
    return x + offset
  elif isinstance(x, Immediate):
    return Immediate(x._value + offset, x._type)
  elif isinstance(x, VarOffset):
    return VarOffset(x.variable, x.offset + offset)
  else:
    return VarOffset(x, offset)

class LeadLoop:
  def __init__(self, name, start, end, threads, stride, unroll=False):
    self.start = start
    self.end = end
    self.unroll = unroll
    self.threads = threads
    self.var = name
    self.stride = stride

  def _lead(self, context: Context, writer):
    """`(tid / stride) % threads` --- as IR values, or as text on the legacy path.

    This is where the first thread-dependent value enters the IR: `thread_id`
    is the one non-uniform source, so everything derived from it is marked
    non-uniform and the barrier-in-divergent-region check becomes live.
    """
    if hasattr(writer, 'op'):
      tid = writer.thread_id('x')
      lane = writer.op('div', INDEX, tid, self.stride, hint='lane')
      return writer.op('rem', INDEX, lane, self.threads, hint='lead')
    lex = context.get_vm().get_lexic()
    return f'({lex.thread_idx_x} / {self.stride}) % {self.threads}'

  def _guard(self, writer, lead, lo, hi):
    """`lead >= lo && lead < hi`, with either bound optional."""
    if hasattr(writer, 'op'):
      cond = None
      if lo is not None:
        cond = writer.op('ge', BOOL, lead, lo, hint='g')
      if hi is not None:
        upper = writer.op('lt', BOOL, lead, hi, hint='g')
        cond = upper if cond is None else writer.op('and', BOOL, cond, upper,
                                                    hint='g')
      return writer.if_(cond)
    parts = []
    if lo is not None:
      parts.append(f'{lead} >= {lo}')
    if hi is not None:
      parts.append(f'{lead} < {hi}')
    return writer.If(' && '.join(parts))

  def write(self, context: Context, writer: Writer, inner):
    actualstart = self.start // self.threads
    realstart = (self.start + self.threads - 1) // self.threads
    realend = (self.end) // self.threads
    actualend = (self.end + self.threads - 1) // self.threads

    lead = self._lead(context, writer)
    tail = self.end - realend * self.threads

    if actualstart >= actualend:
      pass
    if actualstart == realend:
      index = LeadIndex(actualstart, self.threads, self.stride)
      startIdx = self.start - actualstart * self.threads
      with self._guard(writer, lead, startIdx if startIdx > 0 else None, tail):
        inner([index])
    else:
      if self.start % self.threads != 0:
        index = LeadIndex(actualstart, self.threads, self.stride)
        with self._guard(writer, lead, self.start - actualstart, None):
          inner([index])
      if self.unroll:
        for value in range(realstart, realend):
          inner([LeadIndex(value, self.threads, self.stride)])
      elif realstart < realend:
        if hasattr(writer, 'for_'):
          loop = writer.for_(realstart, realend, 1, unroll=True, hint=self.var)
          with loop:
            inner([LeadIndex(str(loop.induction), self.threads, self.stride)])
        else:
          var = self.var
          with writer.For(f'int32_t {var} = {realstart}; {var} < {realend}; '
                          f'{var} += 1', True):
            inner([LeadIndex(var, self.threads, self.stride)])
      if self.end % self.threads != 0:
        index = LeadIndex(actualend - 1, self.threads, self.stride)
        with self._guard(writer, lead, None, tail):
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
        inner([Immediate(value, Datatype.I32)])
    elif self.start < self.end:
      # a real `for` region instead of a text block.
      # The body stays raw for now -- `inner` interpolates the induction
      # variable into strings -- but the loop itself is now a node the passes
      # can reason about, and every loader and store that goes through
      # `write_loops` gets it at once.
      if hasattr(writer, 'for_'):
        loop = writer.for_(self.start, self.end, self.step,
                           unroll=True, hint=self.var)
        with loop:
          inner([Variable(str(loop.induction), Datatype.I32)])
      else:
        var = self.var
        with writer.For(f'int32_t {var} = {self.start}; {var} < {self.end}; '
                        f'{var} += {self.step}', True):
          inner([Variable(var, Datatype.I32)])

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
    with writer.For(f'int32_t {loopvar} = 0; {loopvar} < {totalloopsize}; {loopvar} += {self.blocksize}', True):
      if self.blocksize == 1:
        writer(f'int32_t {loopvar2} = {loopvar};')
      else:
        writer(f'int32_t {loopvar2} = {loopvar} + ({context.get_vm().get_lexic().thread_idx_x} % {self.blocksize});')
      for i, loop in enumerate(self.loops):
        writer(f'int32_t {loop.var} = (({loopvar2} / {multiplies[i]}) % {loopsize[i]}) * {loop.step} + {loop.start};')
      inner([Variable(loop.var, Datatype.I32) for loop in self.loops])

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
    self.datatype: Union[Datatype, None] = None
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

  def get_fptype(self):
    """Resolve this symbol's floating-point type.

    .. deprecated::
        Callers should pass an explicit :class:`Datatype` through their
        own context rather than reaching into the symbol. This wrapper
        will be removed once the backend's instruction templates have
        been threaded with explicit dtype arguments.

    Resolution order is ``self.datatype`` first, then the underlying
    tensor object's ``datatype``. A missing datatype now raises a
    descriptive error instead of an opaque ``assert False`` — that
    case almost always means a synthetic operand (e.g. the scalar
    constructed inside ``GemmDescr.__init__`` for ``alpha != 1``) was
    built without a ``datatype=`` keyword.
    """
    if self.datatype is not None:
      return self.datatype
    if self.obj is not None and getattr(self.obj, 'datatype', None) is not None:
      return self.obj.datatype
    obj_descr = repr(self.obj) if self.obj is not None else 'None'
    raise GenerationError(
        f"Symbol {self.name!r} has no datatype set. "
        f"Underlying object: {obj_descr}. "
        f"Either pass datatype= when constructing the Tensor or set "
        f"Symbol.datatype explicitly before code generation."
    )

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
      writeLeadOffset = lambda i,var,offset,stride: f"({writevar(var)} - {offset // self.num_threads}) * {stride}"
      writers = [0] * self.data_view.rank()
      strides = [0] * self.data_view.rank()
      stride = 1
      for i in range(self.data_view.rank()):
        strides[i] = stride
        if isinstance(index[i], LeadIndex):
          stride *= (self.data_view.get_dim_size(i) + self.num_threads - 1) // self.num_threads
          writers[i] = writeLeadOffset
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
      return self.get_fptype().literal(self.obj.value(runIdx))

  def encode_values(self, pos, runIdx, writer, context: Context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp, leadidx):
    wrote = False
    if pos == len(index):
      if self.stype == SymbolType.Data:
        value = self.obj.value(runIdx)
        if value is not None:
          writer(f'{variable} = {self.get_fptype().literal(value)};')
          wrote = True
      else:
        # TODO: unite with access_address
        if leadidx is None:
          value = self.obj.linear_index(runIdx)
          if value is not None:
            writer(f'{variable} = {self.name}[{value}];')
            wrote = True
        else:
          offset = self.data_view.get_dim_offsets()[leadidx]
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
            writer(f'const int32_t {idxvar} = {strindex} - {offset};')

            lead = index[leadidx]
            bndS = lead._nonlead * lead._block
            bndE = (lead._nonlead + 1) * lead._block

            for rngS, rngE in rngs:
              runIdx[leadidx] = rngS
              value = self.obj.linear_index(runIdx)

              if rngS <= bndS and rngE >= bndE:
                writer(f'{variable} = {self.name}[{value - rngS} + {idxvar}];')
                wrote = True
              elif rngE > bndS and rngS < bndE:
                with writer.If(f'{idxvar} >= {rngS} && {idxvar} < {rngE}'):
                  writer(f'{variable} = {self.name}[{value - rngS} + {idxvar}];')
                  wrote = True
    else:
      if isinstance(index[pos], (int, np.int32, np.int64)):
        runIdx[pos] = index[pos]
        wrote |= self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
      elif isinstance(index[pos], Immediate):
        runIdx[pos] = index[pos]._value
        wrote |= self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
      elif pos == leadidx:
        wrote |= self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
      else:
        # TODO: move block sparsity one level up
        strindex = f'{index[pos]}' if isinstance(index[pos], (str, int, float, np.int64)) else index[pos].write(context)
        if True: # sparse/data
          # TODO: check sparsity pattern here for which ifs are worth it
          offset = self.data_view.get_dim_offsets()[pos]
          for i in range(self.data_view.get_dim_size(pos)):
            runIdx[pos] = i
            with writer.If(f'({strindex} - {offset}) == {runIdx[pos]}'):
              wrote |= self.encode_values(pos + 1, runIdx, writer, context, variable, index, nontemp, leadidx)
    return wrote

  def load_linear(self, writer, context: Context, variable, index, vec = 1):
    if context.get_vm().get_lexic().simd_mode:
      writer(f'{context.get_vm().get_lexic().get_simd(self.get_fptype(), self.num_threads)} {variable}({index});')
    else:
      if self.stype == SymbolType.Register:
        access = f'{self.name}[{index // self.num_threads}]'
      else:
        access = f'{self.name}[{index} + threadIdx.x * {vec}]'

      if vec == 1:
        writer(f'{self.get_fptype()} {variable} = {access};')
      else:
        writer(f'tensorforge::VectorT<{self.get_fptype()}, {vec}> {variable} = *(tensorforge::VectorT<{self.get_fptype()}, {vec}>*)&{access};')

  def store_linear(self, writer, context: Context, variable, index, vec = 1):
    if context.get_vm().get_lexic().simd_mode:
      pass
      # TODO:
      # writer(f'{context.get_vm().get_lexic().get_simd(self.get_fptype(), self.num_threads)} {variable}({index});')
    else:
      if self.stype == SymbolType.Register:
        access = f'{self.name}[{index // self.num_threads}]'
      else:
        access = f'{self.name}[{index} + threadIdx.x * {vec}]'

      if vec == 1:
        writer(f'{access} = {variable};')
      else:
        convert = f'*(tensorforge::VectorT<{self.get_fptype()}, {vec}>*)&'
        writer(f'{convert}{access} = {convert}{variable};')

  def load(self, writer, context: Context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp):
    if self.stype == SymbolType.Data or (not self.obj.is_dense() and not isinstance(self.obj.spp, BoundingBoxSPP)):
      writer(f'{self.get_fptype()} {variable} = {self.get_fptype().literal(0)};')

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
      return self.encode_values(0, [0] * len(index), writer, context, variable, index, nontemp, leadidxidx)
    else:
      pre_access = self.access(context, index)
      if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
        assert len(self.lead_dims) == 1
        idx = index[self.lead_dims[0]]
        if isinstance(idx, (float, int, np.int32, np.int64)) or not idx.is_thread_dependent():
          if isinstance(idx, (float, int, np.int32, np.int64)):
            idx = Immediate(idx, Datatype.I32)
          # doesn't work
          if isinstance(idx, Variable):
            writevar = idx.write_nonlead()
            pre_access = self.access(context, index)
            access = context.get_vm().get_lexic().broadcast(pre_access, writevar, self.num_threads)
          else:
            index2 = list(index)
            index2[self.lead_dims[0]] = LeadIndex(idx._value // self.num_threads, self.num_threads, 1)
            pre_access = self.access(context, index2)

            writevar = idx._value % self.num_threads
            access = context.get_vm().get_lexic().broadcast(pre_access, writevar, self.num_threads)
        else:
          access = pre_access
      else:
        access = pre_access
      if context.get_vm().get_lexic().simd_mode:
        writer(f'{context.get_vm().get_lexic().get_simd(self.get_fptype(), 16)} {variable}({access});')
      elif self.stype == SymbolType.Global:
        writer(f'{self.get_fptype()} {variable};')
        writer(context.get_vm().get_lexic().glb_load(variable, access, nontemp))
      else:
        writer(f'{self.get_fptype()} {variable} = {access};')
      return True

  def store(self, writer, context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp, atomic=None):
    assert self.stype != SymbolType.Data

    access = self.access(context, index)

    if context.get_vm().get_lexic().simd_mode:
      if self.stype == SymbolType.Global:
        writer(f'{variable}.copy_to({access});')
      else:
        writer(f'{variable} = {access};')
    else:
      if self.stype == SymbolType.Global:
        if atomic:
          assign = context.get_vm().get_lexic().atomic_store(access, variable, None, self.get_fptype())
        else:
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
  def __init__(self, symbol, view = None, offset = None):
    self.symbol = symbol
    self.bbox = view
    if view is None:
      self.bbox = symbol.data_view.get_bbox()
    self.offset = offset or ([0] * self.bbox.rank())

  def __str__(self):
    return f'{self.symbol} {self.bbox}'

  def __repr__(self):
    return self.__str__()
