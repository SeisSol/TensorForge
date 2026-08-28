# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import enum
from typing import Union, List
from copy import deepcopy
from tensorforge.common.matrix.boundingbox import BoundingBox
from functools import reduce
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Datatype, Addressing
from tensorforge.common.exceptions import GenerationError
from .writer import Writer
from tensorforge.backend.pir.core import (BOOL, INDEX, SCALAR_LAYOUT, Effect,
                                          LaneAxis, RegisterLayout, ScalarType)

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
    # physical extent, i.e. including any staging padding --- see
    # get_dim_strides for why this is `shape` and not the bounding box
    volume = 1
    for s in self.shape:
      volume *= s
    return volume

  def get_dim_size(self, index):
    assert index >= 0 and index < len(self.shape)
    return self._bbox.size(index)

  def get_dim_slots(self, index, num_threads):
    """Per-thread slots a thread-distributed dimension occupies.

    `ceil(u/T) - floor(l/T)`, i.e. whole thread-blocks are rebased away and the
    ragged ends survive as predicates (see LeadLoop.write).  This is *not*
    `ceil((u-l)/T)` as soon as [l,u) straddles a block boundary --- for
    l=31, u=33, T=32 the two give 2 and 1 --- and the allocation side
    (MultilinearInstruction._iregs, MultilinearBuilder._alloc_register_array)
    has always used the former.  Addressing has to agree with allocation, or
    the next dimension aliases onto this one.
    """
    assert index >= 0 and index < len(self.shape)
    lower = self._bbox.lower()[index]
    upper = self._bbox.upper()[index]
    return -(-upper // num_threads) - lower // num_threads

  def get_dim_strides(self, mask=[]):
    """Strides of the buffer this view describes.

    `shape` is the *physical extent* --- the stride basis --- and `bbox` is the
    live coordinate range inside it, with address 0 at `bbox.lower()`.  The two
    are not interchangeable: a shared-memory staging buffer is padded against
    bank conflicts and for alignment (GlbToShrLoader._get_bounding_box_dense
    builds `dst_shape` from `_next_size()`), so its extent is strictly larger
    than its bounding box and the padding has to show up in the strides.

    For a global tensor the extent *is* the bounding box, since memory spans
    `upper - lower`.  That is established where the view is constructed
    (Tensor.get_actual_shape), not here.
    """
    # TODO: permute? Yes or no? Also, unify SPPs.
    strides = []
    current = 1
    for i, size in enumerate(self.shape):
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

  def build(self, writer, context: Context):
    return self._value

  def build_nonlead(self, writer, context: Context):
    return self._value

class Variable:
  # `value` is the IR value the name came from.  Carrying it is what lets the
  # address arithmetic keep a def-use edge to the loop induction variable --
  # without it a passed-through name looks like a constant to LICM, which
  # would happily hoist a loop-dependent address out of its loop.
  def __init__(self, name, fptype: Datatype, value=None):
    self._name = name
    self._type = fptype
    self._value = value

  def is_thread_dependent(self):
    return False

  def write_nonlead(self):
    return self._name

  def write(self, context: Context):
    return self._name

  def build(self, writer, context: Context):
    return self._name if self._value is None else self._value

  def build_nonlead(self, writer, context: Context):
    return self.build(writer, context)

class LeadIndex:
  """An index into a dimension that is spread across the lanes of a wave.

  ``idx = ((tid / stride) % block) + nonlead * block`` -- so `block` and
  `stride` describe the *distribution* (which lane holds what) and `nonlead`
  picks the slot within a lane.  Two `LeadIndex` with the same block/stride
  address the same register image at possibly different offsets; two with
  different block/stride do not, and moving between them is a shuffle.

  That distinction is what `layout()` hands out, and it is the only reason
  this class is comparable: a pass that wants to know whether two values may
  be treated as one needs to ask, and asking used to mean re-deriving the
  answer from a generated string.
  """

  # TODO: make nonlead a variable
  def __init__(self, nonlead, block, stride, value=None):
    self._nonlead = nonlead
    self._block = block
    self._stride = stride
    self._value = value

  def layout(self) -> RegisterLayout:
    """The one-axis register layout this index addresses.

    Deliberately drops `nonlead` and `value`: those say *which* element, not
    *how the dimension is distributed*.
    """
    return RegisterLayout((LaneAxis(self._block, self._stride),))

  def same_layout(self, other) -> bool:
    """Do the two indices address the same distribution?

    An index that is not a `LeadIndex` is not distributed at all, so it only
    matches a degenerate (`block == 1`) lead index.
    """
    if isinstance(other, LeadIndex):
      return self.layout() == other.layout()
    return self._block == 1

  def _key(self):
    return (self._nonlead, self._block, self._stride, self._value)

  def __eq__(self, other):
    # Structural, including `nonlead`: this is value equality of the *index*,
    # which is a stronger statement than `same_layout`.
    return isinstance(other, LeadIndex) and self._key() == other._key()

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self._key())

  def __repr__(self):
    tail = '' if self._value is None else f', value={self._value!r}'
    return (f'LeadIndex({self._nonlead!r}, block={self._block}, '
            f'stride={self._stride}{tail})')

  def is_thread_dependent(self):
    return True

  def write_nonlead(self):
    return f'{self._nonlead}'

  def write(self, context: Context):
    if self._block > 1:
      return f'(({context.get_vm().get_lexic().thread_idx_x} / {self._stride}) % {self._block}) + {self._nonlead} * {self._block}'
    elif self._block == 1:
      return f'{self._nonlead}'

  def nonlead(self):
    return self._nonlead

  def lead(self):
    return self._nonlead * self._block

  def build_nonlead(self, writer, context: Context):
    return self._nonlead if self._value is None else self._value

  def build(self, writer, context: Context):
    nl = self.build_nonlead(writer, context)
    if self._block > 1:
      # The lane's share of this dimension, asked of the builder rather than
      # spelled out here.  The arithmetic used to be inline -- `(tid/stride) %
      # block` -- which is the SPMD answer written down as though it were the
      # only one there is.  It is not: an explicitly vectorised lowering holds
      # the whole dimension in one register and contributes nothing to the
      # address, and the difference belongs where the model is known.
      lane = writer.lane_offset(self._block, self._stride, hint='lane')
      return writer.op('add', INDEX, lane,
                       writer.op('mul', INDEX, nl, self._block, hint='lead'),
                       hint='lead')
    return nl

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

  def build(self, writer, context: Context):
    return writer.op('add', INDEX, self.variable.build(writer, context),
                     self.offset, hint='off')

  def build_nonlead(self, writer, context: Context):
    return writer.op('add', INDEX,
                     self.variable.build_nonlead(writer, context),
                     self.offset, hint='off')

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

def layout_of(index, num_threads=None):
  """The register layout a loaded value ends up with, from its index list.

  A load is where a distribution enters the IR: the index expression already
  says which lane holds what, and `LeadIndex` has said so all along --- it
  just printed the answer instead of returning it.  Everything downstream is
  either derived from one of these or produced by an explicit relayout, so a
  load that stays untracked leaves its whole consumer chain untracked.

  Several distributed dimensions give a multi-axis layout, in dimension
  order -- a fused operator can put four lanes on each entry of dimension 0
  and run dimension 1 alongside, so lane `l` holds `(l % 4, l // 4)`.  That is
  `LaneAxis(4, 1)` beside `LaneAxis(16, 4)`, and the two together are a
  bijection: `holders` returns one lane.

  The result is only handed out when the axes `tiles()` the wave.  Axes that
  do not tile describe something this function has not established -- partial
  replication, an unusual nesting order -- and `None`, meaning *unknown*, is
  the safe answer: it only ever stops a pass from acting, whereas a guess can
  make it act wrongly.
  """
  leads = [unwrap_lead(i) for i in index]
  leads = [x[0] for x in leads if x is not None]
  if not leads:
    # Thread-independent: every lane computes the same address, so every lane
    # loads the same element.  That is *replicated*, which is a layout --- and
    # `None` here said *unknown*, the strictly weaker claim, on every load of
    # a scalar or a broadcast operand.  The distinction is invisible to the
    # SPMD emitter (both spell the value `float x`) and load-bearing for an
    # explicitly vectorised one, where replicated is `T` and unknown is a
    # value that cannot be given a type at all.
    return SCALAR_LAYOUT
  if len(leads) == 1:
    return leads[0].layout()
  if num_threads is None:
    return None
  layout = RegisterLayout(tuple(l.layout().axis(0) for l in leads))
  return layout if layout.tiles(num_threads) else None


def unwrap_lead(index):
  """Peel `VarOffset` wrappers and report the accumulated shift.

  Returns `(lead_index, shift)` when a `LeadIndex` sits underneath, else
  `None`.  `add_offset` wraps a slicing offset around whatever index it is
  handed, which is exactly right for a global or shared-memory symbol: the
  address is built from the full index expression, so the shift is just
  another constant in the sum.

  Registers distribute the lead dimension across lanes --- element `s` lives
  in lane `s % T`, slot `s // T`.  A shift of `q*T` moves whole blocks and so
  is a change of slot with the lane untouched; a remainder would move data
  between lanes, which is a shuffle and not something an address can express.
  Hence the shift has to be split off here and divided by `T` before it is
  applied to the block index, and only multiples of `T` are representable.
  """
  shift = 0
  while isinstance(index, VarOffset):
    shift += index.offset
    index = index.variable
  if isinstance(index, LeadIndex):
    return index, shift
  return None

class LeadLoop:
  """Loop over a thread-distributed dimension, with guards for the ragged ends.

  `neutral`, when given, is the value a masked-out lane may contribute instead
  of being guarded --- so that if-conversion can drop the guard rather than
  merely predicate what is inside it.  It is a property of the whole
  consumer chain, not of a single operator: for `sum(prod(...))` it is 0,
  because 0 absorbs the product *and* is the sum's neutral.  Where no such
  value exists (a `max` of products, say), it stays None and the guard has to
  survive.
  """

  def __init__(self, name, start, end, threads, stride, unroll=False,
               neutral=None):
    self.start = start
    self.end = end
    self.unroll = unroll
    self.threads = threads
    self.var = name
    self.stride = stride
    self.neutral = neutral

  def _lead(self, context: Context, writer):
    """`(tid / stride) % threads` --- as IR values, or as text on the legacy path.

    This is where the first thread-dependent value enters the IR: `thread_id`
    is the one non-uniform source, so everything derived from it is marked
    non-uniform and the barrier-in-divergent-region check becomes live.
    """
    tid = writer.thread_id('x')
    lane = writer.op('div', INDEX, tid, self.stride, hint='lane')
    return writer.op('rem', INDEX, lane, self.threads, hint='lead')

  def _guard(self, writer, lead, lo, hi):
    """`lead >= lo && lead < hi`, with either bound optional."""
    cond = None
    if lo is not None:
      cond = writer.op('ge', BOOL, lead, lo, hint='g')
    if hi is not None:
      upper = writer.op('lt', BOOL, lead, hi, hint='g')
      cond = upper if cond is None else writer.op('and', BOOL, cond, upper,
                                                  hint='g')
    return writer.if_(cond, attrs=(('neutral', self.neutral),)
                      if self.neutral is not None else ())

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
        # the guard compares against a *lane*, so the bound has to be the
        # in-block remainder.  Without the `* self.threads` this only happened
        # to be right while actualstart == 0, i.e. start < threads; for
        # start=37, threads=32 it read `lead >= 36` and dropped the head block.
        with self._guard(writer, lead,
                         self.start - actualstart * self.threads, None):
          inner([index])
      if self.unroll:
        for value in range(realstart, realend):
          inner([LeadIndex(value, self.threads, self.stride)])
      elif realstart < realend:
        loop = writer.for_(realstart, realend, 1, unroll=True, hint=self.var)
        with loop:
          inner([LeadIndex(str(loop.induction), self.threads, self.stride,
                           loop.induction)])
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
      loop = writer.for_(self.start, self.end, self.step,
                         unroll=True, hint=self.var)
      with loop:
        inner([Variable(str(loop.induction), Datatype.I32, loop.induction)])

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

    # the pragma bears great control over the application speed. And the compile time.
    outer = writer.for_(0, totalloopsize, self.blocksize, unroll=True,
                        hint='var')
    with outer:
      flat = outer.induction
      if self.blocksize != 1:
        lane = writer.op('rem', INDEX, writer.thread_id('x'), self.blocksize,
                         hint='lane')
        flat = writer.op('add', INDEX, flat, lane, hint='var2')
      idx = []
      for i, loop in enumerate(self.loops):
        # Skip the identities up front rather than letting `fold` remove them:
        # the last op carries the `escapes` marker and so is exempt from
        # folding, and `x + 0` would survive as noise.
        steps = []
        if multiplies[i] != 1:
          steps.append(('div', multiplies[i]))
        steps.append(('rem', loopsize[i]))
        if loop.step != 1:
          steps.append(('mul', loop.step))
        if loop.start != 0:
          steps.append(('add', loop.start))
        v = flat
        for j, (name, operand) in enumerate(steps):
          v = writer.op(name, INDEX, v, operand, hint=loop.var,
                        escapes=(j == len(steps) - 1))
        idx.append(Variable(str(v), Datatype.I32, v))
      inner(idx)

class MultiLoop:
  pass

class SparseLoop:
  pass

def write_loops(context: Context, writer: Writer, loops: List[Loop], inner):
  def write_loops_inner(context: Context, writer: Writer, loops: List[Loop], inner, varlist):
    if len(loops) == 0:
      inner(varlist)
    else:
      inner_next = lambda v: write_loops_inner(context, writer, loops[1:], inner, varlist + v)
      loops[0].write(context, writer, inner_next)
  write_loops_inner(context, writer, loops, inner, [])

def _operands(variable, addrs):
  # the stored value first: it is what `{0}` refers to
  out = [variable] if not isinstance(variable, (str, int, float, type(None))) else []
  return tuple(out + [a for a in addrs if not isinstance(a, (int, str))])


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
    #: Which axis of this symbol is spread across the lanes.
    #:
    #: Read by `load` and `store` for `Register` and `Scratch` symbols, and by
    #: `GlbToShrLoader` when it writes a register image; for the other symbol
    #: types nothing consults it, since the lane axis of a global or shared
    #: tensor is whatever the reading instruction chooses to distribute.
    #:
    #: The default is a guess, and every site that creates a register image
    #: now overrides it -- `multilinear_builder` for both the staged operands
    #: and the destination accumulator.  The compute instructions read it from
    #: here rather than keeping their own copy.
    self.lead_dims = [0]
    #: How this symbol's register image is distributed across the wave, when
    #: that is known.  Set by whoever fills it -- `store_linear` is the only
    #: filler that does so today -- and reported by `load_linear`, so that a
    #: reader does not have to restate a fact about a write it cannot see.
    #: `None` is *unknown*, never *not distributed*.
    self.layout = None
    #: The PIR value for this buffer, and the builder that made it.
    #:
    #: A value belongs to the body it was built into.  With one body per loop
    #: body most buffers are allocated and used inside one, so the value
    #: reaches its consumers and the C++ name is redundant; a buffer that
    #: outlives the body -- the shared arena, or a tile shared between two
    #: batch loops -- is allocated in one builder and read in another, and
    #: there the value is meaningless and the name is all there is.  Holding
    #: the builder alongside the value is what tells those two cases apart,
    #: rather than assuming the common one and emitting a dangling id in the
    #: rare one.
    self._pir_buffer = None
    self._users = []

  def set_pir_buffer(self, builder, value) -> None:
    self._pir_buffer = (self._builder_uid(builder), value)

  @staticmethod
  def _builder_uid(builder):
    """A token that is never handed to a second builder.

    `id()` is not one.  A body that has been finished and collected frees its
    address, and the next builder can be given the same address -- so a
    buffer declared in the pre-loop preload was recognised as belonging to
    the batch loop's body, and its reads were emitted against a value whose
    declaration was in another scope.  That compiles to a name that does not
    exist, which the corpus renders happily and only the syntax check catches.
    """
    return getattr(builder, 'uid', None)

  def pir_buffer(self, builder):
    """The value, if it belongs to the body currently being built."""
    if self._pir_buffer is None:
      return None
    owner, value = self._pir_buffer
    uid = self._builder_uid(builder)
    return value if uid is not None and owner == uid else None

  def clone(self):
    cloned = Symbol(self.name, self.stype, self.obj)
    cloned.data_view = deepcopy(self.data_view)
    cloned.datatype = self.datatype
    cloned.layout = self.layout
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

  def build_address(self, writer, context: Context, index):
    """The address as IR values instead of a string.

    Same arithmetic as `access_address` --- sum of `(index - offset) * stride`
    --- but as nodes, so `fold` removes the `- 0` and `* 1` terms the string
    path always wrote out, CSE shares subexpressions between the loads of one
    body, and LICM can lift the loop-invariant part.
    """
    def arith(name, a, b, py):
      # fold right here when both sides are numbers: an address that is fully
      # static should be a literal in the generated source, not a temporary
      # that `fold` collapses and `pin` then has to keep alive
      if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(py(a, b))
      return writer.op(name, INDEX, a, b, hint='a')

    def term(var, offset, stride, lead=False):
      v = (var if isinstance(var, (str, int, float, np.int64))
           else (var.build_nonlead(writer, context) if lead
                 else var.build(writer, context)))
      # a slicing offset can push the subtracted constant negative; emitting
      # `x - -3` is legal C but pointless, so pick the operator to match
      if offset > 0:
        v = arith('sub', v, offset, lambda x, y: x - y)
      elif offset < 0:
        v = arith('add', v, -offset, lambda x, y: x + y)
      if stride != 1:
        v = arith('mul', v, stride, lambda x, y: x * y)
      return v

    if self.stype in (SymbolType.Global, SymbolType.Batch, SymbolType.SharedMem):
      parts = [term(var, off, st) for var, off, st in
               zip(index, self.data_view.get_dim_offsets(),
                   self.data_view.get_dim_strides())]
    elif self.stype in (SymbolType.Register, SymbolType.Scratch):
      parts = []
      stride = 1
      offsets = self.data_view.get_dim_offsets()
      for i in range(self.data_view.rank()):
        lead = unwrap_lead(index[i])
        if lead is not None:
          lead_index, shift = lead
          assert shift % self.num_threads == 0, (
              f'{self.name}: lead-dimension slicing offset {shift} is not a '
              f'multiple of {self.num_threads}; only whole thread-blocks can '
              f'be applied to a register-resident operand')
          # address = index - lower + shift, and on the lead dimension every
          # one of those three lives in units of whole blocks
          parts.append(term(lead_index,
                            offsets[i] // self.num_threads
                            - shift // self.num_threads,
                            stride, lead=True))
          stride *= self.data_view.get_dim_slots(i, self.num_threads)
        else:
          parts.append(term(index[i], offsets[i], stride, lead=True))
          stride *= self.data_view.get_dim_size(i)
    else:
      raise NotImplementedError('Not supposed to be called')

    if not parts:
      return 0
    total = parts[0]
    for p in parts[1:]:
      total = arith('add', total, p, lambda x, y: x + y)
    return total

  def address_value(self, writer, context: Context,
                    index: List[Union[str, int, Immediate, Variable, LeadIndex]]):
    """The address as an operand, not as a name inside a string.

    `access_address` pins its result, because the name is interpolated into
    raw text that the IR cannot see into: unpinned, folding would rewrite the
    value the text still refers to by name.  An address handed to `Op.LOAD` or
    `Op.STORE` is a real operand, so the pin is not only unnecessary, it is
    the thing standing between the address arithmetic and every pass that
    could improve it --- the same `i * 18 + j` recomputed at sixteen loads
    stays sixteen computations while it is pinned.
    """
    return self.build_address(writer, context, index)

  def access_address(self, context: Context, index: List[Union[str, int, Immediate, Variable, LeadIndex]], writer=None, out=None):
    if writer is not None:
      # the result's name goes into raw text, so pin it against DCE and folding
      v = writer.pin(self.build_address(writer, context, index))
      if out is not None:
        out.append(v)
      return str(v)
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
      def writeOffset(var, offset, stride):
        if offset > 0:
          return f"({writevar(var)} - {offset}) * {stride}"
        elif offset < 0:
          return f"({writevar(var)} + {-offset}) * {stride}"
        return f"({writevar(var)}) * {stride}"
      terms = []
      stride = 1
      offsets = self.data_view.get_dim_offsets()
      for i in range(self.data_view.rank()):
        lead = unwrap_lead(index[i])
        if lead is not None:
          lead_index, shift = lead
          assert shift % self.num_threads == 0, (
              f'{self.name}: lead-dimension slicing offset {shift} is not a '
              f'multiple of {self.num_threads}; only whole thread-blocks can '
              f'be applied to a register-resident operand')
          terms.append(writeOffset(lead_index,
                                   offsets[i] // self.num_threads
                                   - shift // self.num_threads, stride))
          stride *= self.data_view.get_dim_slots(i, self.num_threads)
        else:
          terms.append(writeOffset(index[i], offsets[i], stride))
          stride *= self.data_view.get_dim_size(i)
      dimstr = " + ".join(terms)
      return dimstr if len(dimstr) > 0 else "0"
    raise NotImplementedError('Not supposed to be called')

  def access(self, context: Context, index: List[Union[str, int, Immediate, Variable, LeadIndex]], writer=None, out=None,
             base: str = None):
    # `base` overrides the pointer, not the symbol: a rotating shared-memory
    # buffer declares its pointer at the stage consumers read and is filled at
    # a different one.  See AbstractShrMemWrite.write_base().
    name = base or self.name
    if self.stype == SymbolType.Global or self.stype == SymbolType.Batch or self.stype == SymbolType.SharedMem or self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      return f'{name}[{self.access_address(context, index, writer, out)}]'
    if self.stype == SymbolType.Scalar:
      return f'{name}'
    if self.stype == SymbolType.Data:
      return self.get_fptype().literal(self.obj.value(index))

  def encode_values(self, pos, runIdx, writer, context: Context, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp, leadidx):
    wrote = None
    if pos == len(index):
      if self.stype == SymbolType.Data:
        # constant value (data) load

        value = self.obj.value(runIdx)
        if value is not None:
          wrote = writer.const(value, ScalarType(self.get_fptype()))
          # writer(f'{variable} = {self.get_fptype().literal(value)};')
      else:
        # TODO: unite with access_address
        if leadidx is None:
          # scalar load

          value = self.obj.linear_index(runIdx)
          if value is not None:
            wrote = writer.load(self, value,
                             type_=ScalarType(self.get_fptype()), hint='data',
                             layout=None)
            # writer.access_stmt(f'{variable} = {self.name}[{value}];', self, Effect.READ)
        else:
          # SIMD block-aligned sparsity

          offset = self.data_view.get_dim_offsets()[leadidx]
          strindex = self.build_address(writer, context, index[leadidx])
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
            idxvar = writer.op('sub', INDEX, strindex, offset, hint='idx')

            lead = index[leadidx]
            bndS = lead._nonlead * lead._block
            bndE = (lead._nonlead + 1) * lead._block

            for rngS, rngE in rngs:
              runIdx[leadidx] = rngS
              value = self.obj.linear_index(runIdx)

              validx = writer.op('add', INDEX, (value - rngS), idxvar)

              if rngS <= bndS and rngE >= bndE:
                assert wrote is None
                wrote = writer.load(self, validx, type_=ScalarType(self.get_fptype()), hint='data',
                                          layout=layout_of(validx, self.num_threads))
                # writer.access_stmt(f'{variable} = {self.name}[{validx}];', self, Effect.READ)
              elif rngE > bndS and rngS < bndE:
                cond1 = writer.op('ge', BOOL, idxvar, rngS)
                cond2 = writer.op('lt', BOOL, idxvar, rngE)
                cond = writer.op('and', BOOL, cond1, cond2, hint='cond')
                with sel.then():
                  local_load = writer.load(self, validx, type_=ScalarType(self.get_fptype()), hint='data',
                                          layout=layout_of(validx, self.num_threads))
                  sel.yield_(local_load)
                  # writer.access_stmt(f'{variable} = {self.name}[{validx}];', self, Effect.READ)
                with sel.otherwise():
                  if wrote is None:
                    sel.yield_(writer.const(0.0, ScalarType(self.get_fptype())))
                  else:
                    sel.yield_(wrote)

                wrote = sel.result
    else:
      if isinstance(index[pos], (int, np.int32, np.int64)):
        runIdx[pos] = index[pos]
        wrote = self.encode_values(pos + 1, runIdx, writer, context, index, nontemp, leadidx)
      elif isinstance(index[pos], Immediate):
        runIdx[pos] = index[pos]._value
        wrote = self.encode_values(pos + 1, runIdx, writer, context, index, nontemp, leadidx)
      elif pos == leadidx:
        wrote = self.encode_values(pos + 1, runIdx, writer, context, index, nontemp, leadidx)
      else:
        # unroll one dimension (speculatively)
        offset = self.data_view.get_dim_offsets()[pos]
        for i in range(self.data_view.get_dim_size(pos)):
          runIdx[pos] = i

          with writer.speculative() as spec:
            if isinstance(index[pos], (str, int, float, np.int64)):
              idxvar = index[pos] - offset
            else:
              strindex = self.build_address(writer, context, index[leadidx])
              idxvar = writer.op('sub', INDEX, strindex, offset, hint='idx')
            cond = writer.op('eq', BOOL, idxvar, runIdx[pos], hint='cond')

            sel = writer.if_else(cond)

            with sel.then():
              wrote_here = self.encode_values(pos + 1, runIdx, writer, context, index, nontemp, leadidx)
              if wrote_here is None:
                spec.discard()
                sel.yield_(writer.const(0.0, ScalarType(self.get_fptype())))
              else:
                sel.yield_(wrote_here)
            with sel.otherwise():
              if wrote is None:
                sel.yield_(writer.const(0.0, ScalarType(self.get_fptype())))
              else:
                sel.yield_(wrote)

          if wrote_here is not None:
            wrote = sel.result

    return wrote

  def load_linear(self, writer, context: Context, variable, index, vec = 1):
    addrs = []
    if self.stype == SymbolType.Register:
      addr = index // self.num_threads
    else:
      addr = f'{index} + threadIdx.x * {vec}'
    access = f'{self.name}[{addr}]'

    if variable is None:
      # Structured: the consumer takes the value rather than a name it
      # allocated beforehand.  Same seam as `load`, and the reason is the
      # same -- an operand handed to a vendor intrinsic has to have a
      # definition point, or the def-use edge to this read does not exist.
      buf = self.pir_buffer(writer) if vec == 1 else None
      if buf is not None and hasattr(writer, 'load'):
        # Addressed rather than named, the pair to `store_linear`.  There
        # were two structured mechanisms here: `load_expr` wraps a string
        # in a value so the def-use edge exists, and `load` makes the
        # buffer an operand so the *access* is known too.  The second
        # subsumes the first wherever the buffer is a value in this body.
        #
        # `addr` rather than a formula rebuilt here.  The first version
        # inlined `index // self.num_threads`, which is the *register*
        # address; the day the shared window became a value that branch
        # started taking shared symbols too, and `num_threads` is None for
        # those.  One address, computed once, for whichever branch runs.
        #
        # The layout claim is carried across unchanged: it is recorded by
        # the fill in `_record_linear_layout` and only reported here, since
        # neither address has a lane term to derive one from.
        return writer.load(buf, addr, hint='lin', layout=self.layout)
      from tensorforge.backend.pir.core import ScalarType
      type_ = (ScalarType(self.get_fptype()) if vec == 1
               else ScalarType(self.get_fptype(), vec))
      text = (access if vec == 1
              else f'*(tensorforge::VectorT<{self.get_fptype()}, {vec}>*)&{access}')
      # Whatever the fill recorded, unchanged: this read cannot see the
      # distribution, so it reports rather than derives.
      return writer.load_expr(text, type_, self, hint='lin',
                              layout=self.layout)

    if vec == 1:
      writer.access_stmt(f'{self.get_fptype()} {variable} = {access};', self, Effect.READ, args=_operands(variable, addrs))
    else:
      writer(f'tensorforge::VectorT<{self.get_fptype()}, {vec}> {variable} = *(tensorforge::VectorT<{self.get_fptype()}, {vec}>*)&{access};')
    return None

  def _record_linear_layout(self, index, vec):
    """Note the distribution a linearized fill leaves behind.

    `GlbToRegLoader` stages a flat run: it reads `glb[i + threadIdx.x * g]`
    into a temporary and stores that temporary at `reg[i // num_threads]`,
    with `i` stepping by `num_threads * g`.  So register slot `s` holds
    element `s * num_threads + t` on lane `t` --- which is `LaneAxis(threads,
    1)`, the same map `LeadIndex` writes as
    `((tid / stride) % block) + slot * block` with `block = num_threads` and
    `stride = 1`.

    The claim is recorded *here*, where the write that makes it true is
    emitted, rather than re-derived in `load_linear` from the read.  The read
    is `reg[i // num_threads]` and carries no lane term at all: nothing about
    the distribution is recoverable from it, so a reader stating one would be
    restating a fact owned by another file --- the arrangement that produced
    two wrong layout claims in the relayout table already.

    Only for registers, and only for the shape the loader actually emits.
    Anything else leaves `layout` alone, which keeps it `None`, which means
    unknown.
    """
    from tensorforge.backend.pir.core import LaneAxis, RegisterLayout
    if self.stype != SymbolType.Register or self.num_threads is None:
      return
    if not isinstance(index, int) or index % (self.num_threads * vec) != 0:
      # A fill that does not start on a slot boundary distributes something
      # this function has not established.
      return
    layout = RegisterLayout((LaneAxis(self.num_threads, 1),))
    if self.layout is not None and self.layout != layout:
      # Two fills disagreeing about the same register image is a defect, and
      # a silent overwrite would hand the second one's claim to consumers of
      # the first.  Unknown is the safe answer.
      self.layout = None
      return
    self.layout = layout

  def store_linear(self, writer, context: Context, variable, index, vec = 1,
                   base: str = None):
    # `base` overrides the pointer written through, without changing the
    # symbol.  A rotating shared-memory buffer declares its pointer at the
    # stage consumers read and fills a different one -- see
    # AbstractShrMemWrite.write_base().
    name = base or self.name
    addrs = []
    self._record_linear_layout(index, vec)
    if self.stype == SymbolType.Register:
      addr = index // self.num_threads
    else:
      addr = f'{index} + threadIdx.x * {vec}'
    access = f'{name}[{addr}]'

    if vec == 1:
      # `base` is the symbol's own name for everything except a rotating
      # shared buffer, and every caller passes it rather than leaving it
      # None -- so testing for None alone silently never fires.  When it is
      # a real override the value is the wrong buffer: it addresses the
      # stage consumers read while this write fills a different one.
      own_base = base is None or base == self.name
      buf = self.pir_buffer(writer) if own_base else None
      if buf is not None and hasattr(writer, 'store'):
        # Structured: the buffer is an operand, so the write declares what
        # it touches by construction instead of by a hand-passed alias
        # base.  The emitted text is the same while the allocation still
        # carries its `extern` name; it stops being the same on the commit
        # that drops the name, which is the point.
        writer.store(buf, variable, addr)
      elif not isinstance(variable, (str, int, float)):
        # The value came from a structured read, so it has no C++ name yet
        # and may never get one -- the emitter decides whether to inline it
        # into this very statement.  Formatting it in at build time would
        # take that decision away and print a name that was never declared.
        writer.access_stmt(f'{access} = {{0}};', self, Effect.WRITE,
                           args=(variable,), fmt=True)
      else:
        writer.access_stmt(f'{access} = {variable};', self, Effect.WRITE, args=_operands(variable, addrs))
    else:
      convert = f'*(tensorforge::VectorT<{self.get_fptype()}, {vec}>*)&'
      writer.access_stmt(f'{convert}{access} = {convert}{variable};', self, Effect.WRITE, args=_operands(variable, addrs))

  def load(self, writer, context: Context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp):
    addrs = []
    if self.stype == SymbolType.Data or (not self.obj.is_dense() and not isinstance(self.obj.spp, BoundingBoxSPP)):
      if variable is None:
        leadidx = None
        for i, idx in enumerate(index):
          if isinstance(idx, LeadIndex):
            if leadidx is None:
              leadidx = idx
            else:
              leadidx = None
              break
        leadidxidx = index.index(leadidx) if leadidx is not None else None
        return self.encode_values(0, [0] * len(index), writer, context,
                                index, nontemp, leadidxidx)
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
      assert False
      return self.encode_values(0, [0] * len(index), writer, context, index, nontemp, leadidxidx)
    else:
      pre_access = self.access(context, index, writer, addrs)
      if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
        assert len(self.lead_dims) == 1
        idx = index[self.lead_dims[0]]
        if isinstance(idx, (float, int, np.int32, np.int64)) or not idx.is_thread_dependent():
          if isinstance(idx, (float, int, np.int32, np.int64)):
            idx = Immediate(idx, Datatype.I32)
          # doesn't work
          if isinstance(idx, Variable):
            writevar = idx.write_nonlead()
            pre_access = self.access(context, index, writer, addrs)
            access = context.get_vm().get_lexic().broadcast(pre_access, writevar, self.num_threads)
          else:
            index2 = list(index)
            index2[self.lead_dims[0]] = LeadIndex(idx._value // self.num_threads, self.num_threads, 1)
            pre_access = self.access(context, index2, writer, addrs)

            writevar = idx._value % self.num_threads
            access = context.get_vm().get_lexic().broadcast(pre_access, writevar, self.num_threads)
        else:
          access = pre_access
      else:
        access = pre_access
      if variable is None:
        # structured: the consumer takes the value, not a name
        from tensorforge.backend.pir.core import ScalarType
        if access is pre_access and self.stype in (
                SymbolType.Register, SymbolType.Scratch,
                SymbolType.SharedMem, SymbolType.Batch, SymbolType.Global):
          # The dereference itself, with the address as an operand rather than
          # as a name spliced into a string.  Everything the string version
          # declared -- the symbol it reads, the effect, the layout -- survives;
          # what it could not say is that `base` and the address are *operands*,
          # so a pass could neither see the def-use edge to the address nor
          # recognise two reads of the same place.
          #
          # Two kinds stay on the text path.  A broadcast (`access is not
          # pre_access`) is a load wrapped in a vendor intrinsic, and splitting
          # it here would leave a named temporary the source does not have.  A
          # Scalar is not a subscripted access at all -- `access` returns the
          # bare name -- so `Op.LOAD` would invent a `[0]` that never existed.
          return writer.load(self, self.address_value(writer, context, index),
                             type_=ScalarType(self.get_fptype()), hint='data',
                             layout=layout_of(index, self.num_threads), nontemporal = nontemp)
        return writer.load_expr(
            access, ScalarType(self.get_fptype()), self,
            args=_operands(variable, addrs),
            hint='data', layout=layout_of(index, self.num_threads))
      if self.stype == SymbolType.Global:
        writer(f'{self.get_fptype()} {variable} = {context.get_vm().get_lexic().glb_load(variable, access, nontemp)};', self, Effect.READ, args=_operands(variable, addrs))
      else:
        writer.access_stmt(f'{self.get_fptype()} {variable} = {access};', self, Effect.READ, args=_operands(variable, addrs))
      return True

  def store(self, writer, context, variable, index: List[Union[str, int, Immediate, Variable, LeadIndex]], nontemp, atomic=None,
            base: str = None):
    addrs = []
    assert self.stype != SymbolType.Data

    access = self.access(context, index, writer, addrs, base=base)

    fmt = not isinstance(variable, (str, int, float))
    var = '{0}' if fmt else variable
    if self.stype == SymbolType.Global:
      if atomic:
        assign = context.get_vm().get_lexic().atomic_store(access, var, None, self.get_fptype())
      else:
        assign = context.get_vm().get_lexic().glb_store(access, var, nontemp)
    else:
      assign = f'{access} = {var};'

    kind = Effect.ATOMIC if atomic else Effect.WRITE
    if self.stype == SymbolType.Register or self.stype == SymbolType.Scratch:
      assert len(self.lead_dims) == 1
      if isinstance(index[self.lead_dims[0]], LeadIndex):
        from tensorforge.backend.pir.core import Value as _Value
        if base is None and isinstance(variable, _Value):
          # The symmetric case to the structured load: the destination
          # address and the stored value are operands, not names inside a
          # string.  A pass can now see that this write and a later read
          # touch the same place, and the address arithmetic is foldable
          # instead of pinned behind a name the text refers to.
          #
          # `base` is an override of the pointer name -- a rotating buffer
          # writing to a stage other than its own -- which `Op.STORE` cannot
          # express, since its base *is* the symbol.  A non-Value variable is
          # a literal, and the spelling a literal gets is the emitter's to
          # decide on the text path; routing it here would change `0` into
          # `0.0f` or the reverse for reasons unrelated to this change.
          writer.store(self, variable,
                       self.address_value(writer, context, index))
        else:
          writer.access_stmt(assign, self, kind, args=_operands(variable, addrs), fmt=fmt)
      else:
        with writer.If(f'{context.get_vm().get_lexic().thread_idx_x} == {index[self.lead_dims[0]]}'):
          writer.access_stmt(assign, self, kind, args=_operands(variable, addrs), fmt=fmt)
    else:
      writer.access_stmt(assign, self, kind, args=_operands(variable, addrs), fmt=fmt)

  def add_user(self, user):
    self._users.append(user)

  def replace_user(self, old, new) -> bool:
    """Swap a user in place, keeping its position in the list.

    Position matters: ShrMemOpt sizes a region from get_first_user() and lets
    that instruction emit the declaration.  A pass that rewrites an instruction
    by appending its replacement leaves the *replaced* one first, so the region
    is sized from a stale object -- observably, a rotating buffer sized for one
    stage and then overlapping its neighbour.
    """
    for i, user in enumerate(self._users):
      if user is old:
        self._users[i] = new
        return True
    return False

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
