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
from tensorforge.common.exceptions import GenerationError, InternalError
from .writer import Writer
from tensorforge.backend.pir.core import (BOOL, INDEX, SCALAR_LAYOUT, Effect,
                                          LaneAxis, RegisterLayout, ScalarType)

from tensorforge.common.matrix.spp import BoundingBoxSPP

import numpy as np

#: A wide access that does not prove an alignment but is spelled with a type
#: that declares only element alignment.  Legal at any base; the compiler
#: splits it rather than emitting a transfer the hardware needs aligned.
RELAXED = 'relaxed'


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
    # `BoundingBox` has no mutating API -- `_lower`/`_upper` are tuples and
    # every operation returns a new box -- so the defensive deepcopy that used
    # to be here bought nothing and cost ~700k copies on a single large GEMM.
    return self._bbox

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

  @staticmethod
  def split_lead_shift(shift: int, num_threads: int, width: int = 1):
    """A slicing shift on the lead axis, as ``(whole slots, lanes)``.

    The two halves are not interchangeable and only one of them is always
    expressible.  Moving by whole slots keeps every element in the lane that
    held it, so it is a change of register index; the remainder moves data
    *between* lanes, which under SPMD is a shuffle and not something an
    address can say -- hence the assertion that has always guarded this.

    When the work-item holds the whole wave there are no lanes to move
    between: the remainder is simply where the vector starts inside the slot
    run, and the address can say it.  Which is the difference the assertion
    now tests for rather than forbidding outright.
    """
    span = num_threads * width
    return shift // span, (shift % span) // width

  def lead_lanes(self, explicit_simd: bool, num_threads: int) -> int:
    """How many register entries one slot of a lead dimension occupies.

    One in SPMD: the lane is the thread, so a thread's private array holds a
    single entry per slot and the other lanes' entries live in the other
    threads' arrays.  `num_threads` when the work-item holds the whole wave:
    every lane's entry is in *this* array, so a slot is a run of that many.

    On the type this is the same number twice -- the allocation's size and the
    address's stride -- and they have to agree or the next dimension aliases
    onto this one.  There were already two copies of the slot-count formula
    for that reason (see `get_dim_slots`), and this is deliberately not a
    third: both sides call here.
    """
    return num_threads if explicit_simd else 1

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
  def __init__(self, nonlead, block, stride, value=None, width=1,
               offset=0):
    self._nonlead = nonlead
    self._block = block
    self._stride = stride
    self._value = value
    #: A slicing shift, in *elements*.
    #:
    #: This used to be a `VarOffset` wrapped around the index, and 649 of the
    #: 661 offsets in the corpus wrapped a `LeadIndex` -- but the wrapper is
    #: the wrong place for it, and not only because it is nearly always this
    #: one case.  The offset's unit depends on which view of the index you
    #: take: `write`/`build` answer in elements, `write_nonlead`/`build_nonlead`
    #: answer in *slots*, and converting between them needs `block` and
    #: `width`, which only this class has.  `VarOffset.write_nonlead` added an
    #: element count to a slot index and got `2 + 32` where the answer is `4`;
    #: nothing called it, so nothing noticed.
    #:
    #: `unwrap_lead` keeps handing the raw element shift to its caller, which
    #: is what the register paths convert.  Moving that conversion in here is
    #: the next step and the one that makes an ESIMD base offset expressible:
    #: a shift that is not a whole number of slots moves data between lanes,
    #: which is a shuffle in SPMD and simply a different vector base when the
    #: work-item holds the wave.
    self._offset = offset
    #: How many *adjacent* elements this lane holds at this index.  With
    #: `width > 1` the map above scales uniformly:
    #:
    #:     idx = width * (((tid / stride) % block) + nonlead * block) + c
    #:
    #: with `c` in `[0, width)` naming a component *inside the value* rather
    #: than a position in the address -- the whole vector is one access.
    #:
    #: This is what turns the per-lane element set from strided into
    #: adjacent.  At `width == 1` a lane holds `lane, lane + block,
    #: lane + 2*block, ...`, which are `block` apart and cannot be one wide
    #: access however the base is aligned; at `width == 2` it holds
    #: `2*lane, 2*lane+1` and then the pair `2*block` further on.  Cyclic
    #: becomes blocked-by-`width`, and that is the whole content of the
    #: change -- `layout()` is unchanged, because *which lane holds what
    #: share* is still `LaneAxis(block, stride)`; the width lives on the
    #: value's `ScalarType.length`, where `LaneAxis` says packing belongs.
    self._width = width
    if width < 1:
      raise InternalError(f'lead width must be >= 1, got {width}')

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

  @property
  def width(self) -> int:
    return self._width

  def _key(self):
    return (self._nonlead, self._block, self._stride, self._value,
            self._width, self._offset)

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
    wide = '' if self._width == 1 else f', width={self._width}'
    off = '' if self._offset == 0 else f', offset={self._offset}'
    return (f'LeadIndex({self._nonlead!r}, block={self._block}, '
            f'stride={self._stride}{tail}{wide}{off})')

  def is_thread_dependent(self):
    return True

  def write_nonlead(self):
    return f'{self._nonlead}'

  def write(self, context: Context):
    if self._block > 1:
      inner = (f'(({context.get_vm().get_lexic().thread_idx_x} / '
               f'{self._stride}) % {self._block}) + '
               f'{self._nonlead} * {self._block}')
      out = inner if self._width == 1 else f'({inner}) * {self._width}'
    elif self._block == 1:
      out = (f'{self._nonlead}' if self._width == 1
             else f'{self._nonlead} * {self._width}')
    else:
      return None
    # Elements, so the shift goes on as it stands.  The slot view below
    # deliberately does not apply it: there it would need dividing, and the
    # register paths do that themselves off `unwrap_lead`.
    return out if self._offset == 0 else f'({out} + {self._offset})'

  def offset(self):
    return self._offset

  def with_offset(self, offset):
    return LeadIndex(self._nonlead, self._block, self._stride, self._value,
                     self._width, offset)

  def nonlead(self):
    return self._nonlead

  def lead(self):
    return self._nonlead * self._block * self._width

  def build_nonlead(self, writer, context: Context):
    """The slot, in the units the *register* address is counted in.

    Floats, not vectors.  A register-resident lead dimension addresses
    `r[slot]` directly -- the lane term is implicit in "each lane has its own
    array" -- so at width `w` one slot covers `w` consecutive floats and the
    next one starts `w` further on.  Returning the bare slot number put slot
    1 at float 1, overlapping the second half of slot 0.

    Invisible in everything generated so far: every operator in the corpus
    has at most one slot per lane once the lane count is chosen for the
    width, so `slot` and `slot * w` are both 0.  It bites at the first
    operator whose lead dimension is longer than `threads * w` -- 120 over 32
    lanes is the first, and it is an ordinary order-5 shape.
    """
    nl = self._nonlead if self._value is None else self._value
    if self._width == 1:
      return nl
    if isinstance(nl, (int, np.integer)):
      return int(nl) * self._width
    return writer.op('mul', INDEX, nl, self._width, hint='vslot')

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
      addr = writer.op('add', INDEX, lane,
                       writer.op('mul', INDEX, nl, self._block, hint='lead'),
                       hint='lead')
      # The scaling is on the *whole* index, not on the lane term alone: a
      # lane's share starts `width` elements after its neighbour's, and the
      # next slot starts `width * block` after this one.  Scaling only the
      # lane would interleave the slots into each other's lanes.
      if self._width > 1:
        addr = writer.op('mul', INDEX, addr, self._width, hint='vlead')
      return self._shift(writer, addr)
    if self._width > 1:
      return self._shift(writer, writer.op('mul', INDEX, nl, self._width,
                                           hint='vlead'))
    return self._shift(writer, nl)

  def _shift(self, writer, addr):
    """Apply the slicing offset to the *element* view.

    Only here.  `build_nonlead` answers in slots, where the same number would
    have to be divided by `block * width` first -- and the register callers do
    that themselves off `unwrap_lead`, because only they know whether a
    remainder is a shuffle they cannot express.
    """
    if self._offset == 0:
      return addr
    return writer.op('add', INDEX, addr, self._offset, hint='off')

class VarOffset:
  """A plain index plus a constant.

  Never a `LeadIndex` -- that combination lives in `LeadIndex._offset`, and
  the reason is a unit mismatch this class cannot resolve: the offset counts
  elements, `write_nonlead` answers in slots, and converting between the two
  needs `block` and `width`, which are the lead index's and not the wrapper's.
  Before the merge this method returned `2 + 32` where the answer was `4`; it
  was unreachable, so nothing found it.  The assertion is what keeps it that
  way now that `add_offset` folds instead of wrapping.
  """

  def __init__(self, variable, offset):
    if isinstance(variable, LeadIndex):
      raise InternalError(
          'a lead index carries its own offset; wrapping one in a VarOffset '
          'puts an element count where a slot index is expected. Use '
          'add_offset(), which folds.')
    self.variable = variable
    self.offset = offset

  def is_thread_dependent(self):
    return self.variable.is_thread_dependent()

  def write_nonlead(self):
    # No unit conversion needed: the wrapped index is never distributed, so
    # its slot view and its element view are the same number.
    return f'({self.variable.write_nonlead()} + {self.offset})'

  def write(self, context: Context):
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
  elif isinstance(x, LeadIndex):
    # Folded in, not wrapped.  See `LeadIndex._offset`: a wrapper cannot
    # convert the shift between the element view and the slot view, because
    # the conversion needs `block` and `width`.
    return x.with_offset(x.offset() + offset)
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


def lead_width_of(index) -> int:
  """The per-lane width the lead index of this access carries.

  One number for the whole access: the lead dimension is the only one that
  is distributed, so at most one index in the list can be wide, and an access
  whose lead index is a plain integer -- a broadcast, a sliced constant -- is
  scalar however wide its neighbours are.
  """
  for idx in index:
    lead = unwrap_lead(idx)
    if lead is not None:
      return lead[0].width
  return 1


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
    # The contract is unchanged -- `(index, shift in elements)` -- but the
    # shift now usually comes off the index itself rather than off a wrapper.
    # The index handed back is the one *without* it applied, because the
    # register callers convert the shift to slots and would otherwise count
    # it twice.
    return index.with_offset(0), shift + index.offset()
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
               neutral=None, width=1):
    self.start = start
    self.end = end
    self.unroll = unroll
    self.threads = threads
    self.var = name
    self.stride = stride
    self.neutral = neutral
    #: Elements each lane holds adjacently at one slot -- see `LeadIndex`.
    #: One slot then covers `threads * width` elements instead of `threads`.
    self.width = width

  def _lead(self, context: Context, writer):
    """Which element of the distributed dimension this lane is at.

    This is where the first thread-dependent value enters the IR: it is the
    one non-uniform source, so everything derived from it is marked
    non-uniform and the barrier-in-divergent-region check becomes live.

    The arithmetic used to be written out here as `(tid / stride) % threads`.
    That is the SPMD answer, and it is only *an* answer: the explicitly
    vectorised lowering holds every element of the dimension in one register
    and returns all `threads` indices at once, which makes the guard below a
    mask instead of a branch.  Asking the builder is what lets the two
    differ without this function knowing which one it is talking to.
    """
    return writer.lane_index(self.threads, self.stride, hint='lead')

  def _narrow_possible(self, writer) -> bool:
    """Whether this lowering can replace a guard with a narrower vector."""
    return bool(getattr(writer, '_explicit_simd', lambda: False)())

  def _narrow(self, writer, slot, lo, hi, elem_lo, elem_hi):
    """A guarded block as a shorter vector, or None to keep the guard.

    A guard on the lead axis is a *ragged edge*: the dimension is 12 elements
    wide, the wave is 16, and lanes 12..15 are masked off.  In SPMD that mask
    is unavoidable -- the wave is 16 threads whatever the operand looks like.
    Explicitly vectorised, the vector width is a compile-time choice, so the
    honest answer is a 12-wide vector and no mask at all.

    `lo`/`hi` are lane bounds and `elem_lo`/`elem_hi` the element range the
    block actually covers.  Both are passed rather than re-derived: the three
    call sites (a single-slot block, a wave-unaligned head, a ragged tail)
    each intersect the range differently, and deriving it here once got two of
    the three wrong before it was passed in.

    Returns `(extent in lanes, base offset in elements)`.  The slot goes into
    the offset, so the index is always slot zero -- `nonlead * block` stops
    being the right base as soon as `block` is the narrowed extent rather than
    the slot stride, and folding the slot in removes that trap instead of
    documenting it.
    """
    if not self._narrow_possible(writer):
      return None
    if hi is None:
      hi = self.threads
    base = lo or 0
    if hi <= base or (hi - base) >= self.threads and slot == 0 and base == 0:
      return None
    if (hi - base) * self.width != elem_hi - elem_lo:
      # The lane bounds are *ceilings* (see `_lane_hi`): at a ragged end one
      # lane holds a vector half inside the box, and the guard is what stops
      # its extra component from being stored.  Narrowing removes the guard,
      # so it may only be done where there is no such lane -- where the vector
      # extent covers the range exactly.
      #
      # Never true on the current corpus, because narrowing only meets
      # `width == 1` today.  It is a precondition and not an observation: the
      # width policy is free to offer a width that does not divide, and then
      # the two features would silently agree to store past the box.
      return None
    # Neither bound is a mask.  A lower one is a vector that *starts* later,
    # an upper one is a vector that stops earlier, and a later slot is a
    # vector that starts a whole slot run in.  All three are a base offset in
    # elements, which `LeadIndex` carries since the offset moved out of
    # `VarOffset` -- and which `split_lead_shift` can put into a register
    # address now that the leftover lanes have a run to sit in.
    return hi - base, elem_lo

  def _lane_lo(self, offset):
    """The first lane whose vector reaches element `offset`.

    Floor, so the lane whose vector *straddles* the bound is included and
    computes one component from outside the box.  See `_lane_hi` for why that
    is the right direction.
    """
    return offset // self.width

  def _peel(self, inner, first, offset):
    """Emit the leftover components as plain element indices.

    `first` is the element the last whole vector ended at.  Each leftover is
    handed to `inner` as an integer, which is a *fixed* element of a
    distributed dimension -- `Symbol.load` broadcasts it from the lane that
    owns it and `Symbol.store` guards the write to that lane, both of which
    already existed for sliced constants.  So the tail is scalar FMAs on the
    existing machinery rather than a component mask on a new one, and there
    are at most `width - 1` of them per lead loop.
    """
    for c in range(self._peeled(offset)):
      inner([first + c])

  def _peeled(self, offset):
    """The `offset % width` components that no whole vector covers.

    Returned as *element* indices, so the caller can hand them to `inner` as
    plain integers: a fixed element of a distributed dimension is exactly
    what `Symbol.load`'s broadcast path and `Symbol.store`'s
    one-lane-owns-it guard already handle, and reusing them costs at most
    `width - 1` scalar operations at the end of a lead loop.

    Peeling rather than over-computing.  The earlier arrangement let the
    straddling lane carry a component from outside the box and relied on
    nothing reading it back -- true for the destination, whose own guard is
    at element granularity, and true only by accident for the *source*, whose
    read left the operand window whenever the window was sized for the exact
    extent.  A scalar tail has neither problem and costs one FMA.
    """
    return offset % self.width

  def _lane_hi(self, offset):
    """One past the last lane whose vector lies wholly below element `offset`.

    Floor now, not ceiling: the straddling lane is excluded here and its
    valid components come back as `_peeled`.

    That is deliberate, and it rests on two conditions the width policy
    checks rather than this loop:

    * the extra component is never *stored*.  It lands in an accumulator
      slot that the destination's own guard -- which is at element
      granularity -- does not write.  What it holds is arithmetic on data
      from outside the box, and that is only contained because nothing reads
      it back.
    * the operand window covers the rounded-up extent, so the read that
      produces it stays inside the tile.  Reading `ceil(extent / (threads *
      width)) * threads * width` elements from a window sized for `extent`
      would leave the buffer, which is a correctness problem rather than a
      wasted lane.

    """
    return offset // self.width

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
    # A slot is `threads * width` elements wide, so every division that turns
    # an element range into a slot range takes the product.  At `width == 1`
    # this is the arithmetic that was here before, character for character.
    span = self.threads * self.width
    actualstart = self.start // span
    realstart = (self.start + span - 1) // span
    realend = (self.end) // span
    actualend = (self.end + span - 1) // span

    # Eager in SPMD, on demand where a guard may be narrowed away.
    #
    # Every branch below uses the lane index in SPMD, so building it up front
    # costs nothing and keeps the value numbering exactly where it was --
    # deferring it renumbers 83 snapshot files for no change in meaning.
    # Under narrowing it can go unused, and then it must not be built: a
    # `rawexpr` is opaque text, so `dce` cannot know it is free of effects and
    # leaves a `simd<int, 16>` in the output that nothing reads.
    _lead_cache = ([] if self._narrow_possible(writer)
                   else [self._lead(context, writer)])

    def lead():
      if not _lead_cache:
        _lead_cache.append(self._lead(context, writer))
      return _lead_cache[0]
    tail = self.end - realend * span

    if actualstart >= actualend:
      pass
    if actualstart == realend:
      startIdx = self.start - actualstart * span
      lo = self._lane_lo(startIdx) if startIdx > 0 else None
      hi = self._lane_hi(tail)
      # one slot, and the range is the loop's own
      narrowed = self._narrow(writer, actualstart, lo, hi,
                              self.start, self.end)
      if narrowed is not None:
        extent, base = narrowed
        inner([LeadIndex(0, extent, self.stride, width=self.width,
                         offset=base)])
      elif hi > 0 or lo is not None:
        index = LeadIndex(actualstart, self.threads, self.stride,
                          width=self.width)
        with self._guard(writer, lead(), lo, hi):
          inner([index])
      self._peel(inner, actualstart * span + hi * self.width, tail)
    else:
      if self.start % span != 0:
        # the guard compares against a *lane*, so the bound has to be the
        # in-block remainder.  Without the `* self.threads` this only happened
        # to be right while actualstart == 0, i.e. start < threads; for
        # start=37, threads=32 it read `lead >= 36` and dropped the head block.
        lo = self._lane_lo(self.start - actualstart * span)
        # The head block of a wave-unaligned range, and the same shape as the
        # ragged tail seen from the other end: a vector that starts later
        # rather than one that stops earlier.  `DataView.split_lead_shift`
        # is what makes it addressable -- the leftover lanes are a
        # displacement inside the slot run, which exists only when the run is
        # longer than one entry.
        narrowed = self._narrow(writer, actualstart, lo, None,
                                self.start, (actualstart + 1) * span)
        if narrowed is not None:
          extent, base = narrowed
          inner([LeadIndex(0, extent, self.stride,
                           width=self.width, offset=base)])
        else:
          index = LeadIndex(actualstart, self.threads, self.stride,
                            width=self.width)
          with self._guard(writer, lead(), lo, None):
            inner([index])
      if self.unroll:
        for value in range(realstart, realend):
          inner([LeadIndex(value, self.threads, self.stride,
                           width=self.width)])
      elif realstart < realend:
        loop = writer.for_(realstart, realend, 1, unroll=True, hint=self.var)
        with loop:
          inner([LeadIndex(str(loop.induction), self.threads, self.stride,
                           loop.induction, width=self.width)])
      if self.end % span != 0:
        hi = self._lane_hi(tail)
        if hi > 0:
          # the tail block starts at its slot boundary and stops at `end`
          narrowed = self._narrow(writer, actualend - 1, None, hi,
                                  (actualend - 1) * span, self.end)
          if narrowed is not None:
            extent, base = narrowed
            inner([LeadIndex(0, extent, self.stride, width=self.width,
                             offset=base)])
          else:
            index = LeadIndex(actualend - 1, self.threads, self.stride,
                              width=self.width)
            with self._guard(writer, lead(), None, hi):
              inner([index])
        self._peel(inner, realend * span + hi * self.width, tail)

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

  def _linear_claim(self, index, vec: int):
    """The alignment this linearized access can prove, in bytes.

    `None` for a scalar access: there is no cast, so there is nothing to
    prove, and the verifier only asks of wide ones.  For a wide one the
    answer combines the two facts that make the cast legal -- how far the
    base is aligned, and how far into it the hop starts.

    The offset term is the one that is easy to forget.  A 16-byte aligned
    base does not make `&buf[i]` 16-byte aligned for every `i`; the hop
    offsets `plan_hops` produces are multiples of `threads * width`, which
    carries it, but a caller that picked an offset by hand would not, and
    this is where that shows up rather than at runtime.
    """
    if vec == 1:
      return None
    elem = self.get_fptype().size()
    base = self.linear_align_bytes()
    if self.stype == SymbolType.Register:
      # `RELAXED`, not a number.  A private address cannot be made to name a
      # contiguous 16 bytes -- AMDGPU interleaves the lanes at dword
      # granularity -- so there is no alignment to prove here and demanding
      # one would only force an `alignas` that pads the scratch frame without
      # buying a wide transfer.  The access is spelled with the relaxed
      # vector type instead, which is well-defined at element alignment.
      return RELAXED
    # `index + threadIdx.x * vec`: the lane term is a multiple of `vec` by
    # construction, so only the base offset can break the alignment.
    off = index if isinstance(index, int) else None
    if off is None:
      # A symbolic offset proves nothing.  Answering the element size makes
      # `verify` reject any wide access built on one, which is the intent:
      # the width was chosen somewhere that could not see the offset.
      return elem
    if off == 0:
      return base
    byte_off = off * elem
    return min(base, byte_off & -byte_off)   # largest power of two dividing it

  def linear_align_bytes(self) -> int:
    """How far the base of a linearized access is *provably* aligned, in bytes.

    Provably.  An answer of `elem.size()` is not "4-byte aligned", it is "no
    promise beyond the element", and `widths_for` turns both into the same
    permission: width 1 only.  That asymmetry is deliberate and is the same
    one `lane_span` makes by raising instead of returning 1 -- an unproven
    alignment and a natural one are the same number and opposite facts, and
    the reinterpret cast a wide access needs must not be reachable by
    defaulting.

    Where the promises come from:

    * **Global/Data**: whatever the frontend attached.  `yateto.py` sets 16
      when the memory layout reports an aligned stride, 0 otherwise, and 0
      has to mean unknown rather than "unaligned" -- a batched tensor's base
      moves by the batch stride, so a promise about element 0 of matrix 0 is
      only a promise about every matrix if the stride carries it, which is
      exactly what `alignedStride()` reports.
    * **SharedMem/Scratch**: 16, and not by assumption -- `_suballocate`
      rounds every window start up to `16 // elem.size()` elements for this
      reason, so the property holds however the windows were requested.
    * **Register**: the element size, and that is not a limitation to work
      around.  A register array is in the private address space, which
      AMDGPU interleaves per lane at dword granularity, so no alignment of a
      private address makes a wide access contiguous -- and in the case that
      matters the array is promoted and has no address.  A wide *register*
      access is therefore spelled with the relaxed vector type instead of
      demanding an alignment, and the width is decided by the other end.
    """
    elem = self.get_fptype().size()
    if self.stype in (SymbolType.SharedMem, SymbolType.Scratch):
      return max(elem, 16)
    if self.stype in (SymbolType.Global, SymbolType.Data, SymbolType.Batch):
      return max(elem, getattr(self.obj, 'alignment', 0) or 0)
    return elem

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
          simd = bool(getattr(writer, '_explicit_simd', lambda: False)())
          lanes = self.data_view.lead_lanes(simd, self.num_threads)
          slot_shift, lane_shift = DataView.split_lead_shift(
              shift, self.num_threads, lead_index.width)
          assert simd or lane_shift == 0, (
              f'{self.name}: lead-dimension slicing offset {shift} is not a '
              f'multiple of {self.num_threads}; only whole thread-blocks can '
              f'be applied to a register-resident operand')
          # address = index - lower + shift, and on the lead dimension the
          # first two live in units of whole slots.  So does the shift, except
          # for the lanes it leaves over -- those are a displacement *within*
          # the slot run, which only exists when the run is more than one
          # entry long.
          parts.append(term(lead_index,
                            offsets[i] // self.num_threads - slot_shift,
                            stride * lanes, lead=True))
          if lane_shift:
            parts.append(lane_shift * stride)
          stride *= self.data_view.get_dim_slots(i, self.num_threads) * lanes
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
          # Unconditional here, unlike the structured path above: this is the
          # SPMD spelling, where a lane is a thread and a sub-slot shift is a
          # shuffle.  An explicitly vectorised kernel does not reach it.
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
      buf = self.pir_buffer(writer)
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
        #
        # The width rides on the *type* rather than on a second parameter.
        # `ScalarType.length` is where a lane holding several consecutive
        # elements already lived (`LaneAxis` says so in as many words: packing
        # is a vector type over the slot dimension, not a lane axis), and the
        # ESIMD emitter already reads it -- `span * (length or 1)` is its
        # `simd<>` width.  So the vectorised path needs no new state, only the
        # type it always had and an emitter that spells the access for it.
        from tensorforge.backend.pir.core import ScalarType
        ltype = (ScalarType(self.get_fptype()) if vec == 1
                 else ScalarType(self.get_fptype(), vec))
        return writer.load(buf, addr, type_=ltype, hint='lin',
                           layout=self.layout,
                           align=self._linear_claim(index, vec))
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
      #
      # No longer gated on `vec == 1`.  A wide write used to be excluded
      # from this branch and spelled as a cast into a raw string, which put
      # the one access that moves the most bytes outside every pass's view.
      # The width is on the stored value's type, so the emitter spells the
      # cast and the buffer stays an operand.
      writer.store(buf, variable, addr,
                   align=self._linear_claim(index, vec))
    elif vec != 1:
      # Unstructured fallback: a rotating buffer writes through an alias
      # base, so there is no buffer value to make an operand of.
      convert = f'*(tensorforge::VectorT<{self.get_fptype()}, {vec}>*)&'
      writer.access_stmt(f'{convert}{access} = {convert}{variable};', self, Effect.WRITE, args=_operands(variable, addrs))
    elif not isinstance(variable, (str, int, float)):
      # The value came from a structured read, so it has no C++ name yet
      # and may never get one -- the emitter decides whether to inline it
      # into this very statement.  Formatting it in at build time would
      # take that decision away and print a name that was never declared.
      writer.access_stmt(f'{access} = {{0}};', self, Effect.WRITE,
                         args=(variable,), fmt=True)
    else:
      writer.access_stmt(f'{access} = {variable};', self, Effect.WRITE, args=_operands(variable, addrs))

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
          # The width the lead index carries, on the type -- the address is
          # already scaled by it, because `LeadIndex.build` multiplies.
          #
          # `RELAXED` rather than a byte count, for every space and not only
          # for registers.  A shared window *is* 16-byte aligned and the
          # address *is* a multiple of the width, but a sliced operand adds a
          # constant that this call cannot see the divisibility of, and a
          # claim that is wrong is worse than one that is weak.  The relaxed
          # type is legal at any base; tightening it needs the constant part
          # of the address modelled, which is its own step.
          w = lead_width_of(index)
          ltype = (ScalarType(self.get_fptype()) if w == 1
                   else ScalarType(self.get_fptype(), w))
          return writer.load(self, self.address_value(writer, context, index),
                             type_=ltype, hint='data',
                             align=None if w == 1 else RELAXED,
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

    kind = Effect.ATOMIC if atomic else Effect.WRITE
    from tensorforge.backend.pir.core import Value as _Value
    lead = index[self.lead_dims[0]] if len(self.lead_dims) == 1 else None
    # `unwrap_lead`, not `isinstance`: a slicing offset wraps the lead index
    # in a `VarOffset`, and `build_address` has always peeled that -- so the
    # only thing the narrower test achieved was to send a sliced store back to
    # the text path, where its address is a pinned name instead of an operand.
    #
    # `base` no longer disqualifies.  It overrides the pointer *name*, which
    # `Op.STORE` now carries as an attribute; the base it attributes accesses
    # to is still the symbol, which is what a rotating buffer's stages are.
    structured = (not atomic and isinstance(variable, _Value)
                  and lead is not None and unwrap_lead(lead) is not None
                  and self.stype in (SymbolType.Register, SymbolType.Scratch,
                                     SymbolType.Global, SymbolType.SharedMem))

    # Decided *before* the text address is built, not after.  `self.access()`
    # emits the address as IR ops and the last of them carries `escapes`, so
    # it survives DCE whether or not anything reads it -- building one for a
    # store that then takes the structured path leaves a second, identical
    # address chain in the output next to the one `address_value` produces.
    # That waste already existed for registers; extending the structured path
    # to global memory would have doubled it rather than exposed it.
    if not structured:
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

    if structured:
      # The symmetric case to the structured load: the destination address and
      # the stored value are operands, not names inside a string.  A pass can
      # now see that this write and a later read touch the same place, and the
      # address arithmetic is foldable instead of pinned behind a name the
      # text refers to.
      #
      # Global joins Register and Scratch here.  It could not before, because
      # the nontemporal hint was resolved into a finished statement by
      # `lexic.glb_store` at this call site, leaving nothing structured to
      # emit; `Op.STORE` carries the hint now and the emitter asks the lexic.
      #
      # Three cases stay on the text path, each for its own reason.  `base` is
      # an override of the pointer name -- a rotating buffer writing to a
      # stage other than its own -- which `Op.STORE` cannot express, since its
      # base *is* the symbol.  A non-Value variable is a literal, and the
      # spelling a literal gets is the emitter's to decide; routing it here
      # would change `0` into `0.0f` or the reverse for reasons unrelated to
      # this change.  And an atomic goes through `atomic_store`, which returns
      # an expression rather than a statement.
      # The width comes off the *stored value*, as it does in the emitter:
      # the buffer is typed by its element and would narrow every wide write
      # to its first component.  `RELAXED` for the same reason as in `load`.
      wide = getattr(getattr(variable, 'type', None), 'length', None)
      writer.store(self, variable,
                   self.address_value(writer, context, index),
                   align=None if wide is None else RELAXED,
                   nontemporal=bool(nontemp), pointer=base)
    elif (self.stype in (SymbolType.Register, SymbolType.Scratch)
          and not isinstance(lead, LeadIndex)):
      # One named element of a dimension that lives in the registers, so
      # exactly one lane holds it and the others must not write.
      #
      # Deliberately *not* extended to Global alongside the branch above.
      # Global memory is shared: every lane addresses it directly and there is
      # no lane that uniquely owns an element, so the guard would be wrong --
      # and it also formats the index with `{...}`, which for a `VarOffset`
      # interpolates a Python object repr (address included) straight into the
      # generated source.  Unreachable today for register-like symbols, which
      # is why nothing has caught it; see `VarOffset.__str__`.
      with writer.If(f'{context.get_vm().get_lexic().thread_idx_x} == {lead}'):
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
