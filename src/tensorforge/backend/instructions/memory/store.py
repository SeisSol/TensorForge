# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import List, Optional, Tuple
import numpy as np
from tensorforge.common.context import Context
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.backend.data_types import RegMemObject
from tensorforge.backend.symbol import Symbol, SymbolType, DataView, write_loops, LeadLoop, Loop, Immediate, add_offset
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from . import AbstractShrMemWrite, element_blocks
from .hints import cache_hint, readers
from ..abstract_instruction import AbstractInstruction


def _padded_row(sizes, context) -> List[int]:
  """The buffer's physical extent: its box, with the leading dimension rounded
  up to `Options.stage_row_bytes`.

  A `DataView` already separates the two -- `shape` is the extent the strides
  are built from and `bbox` the live range inside it -- because a staged
  operand is padded against bank conflicts.  A temporary written out of
  registers is padded only by this, and it is the one the damage step reads
  widest: five buffers of `(125, 6)` carrying a third of all its shared
  traffic, where unpadded every pack is refused because 125 is odd and so
  every row above the first starts at an odd element.

  Only the leading dimension, because it is the one a wide access runs along
  *and* the stride of every dimension above it.  The pad itself is never read:
  a group that does not fit whole in the box stays scalar.
  """
  sizes = list(sizes)
  opts = context.get_user_options()
  want = opts.stage_row_bytes
  if not sizes:
    return sizes
  if not want:
    if (getattr(opts, 'mma_fragment_ld', False)
            and getattr(opts, 'tensor_cores', None)
            and str(getattr(context.target.hw, 'vendor', '')).lower()
            == 'nvidia'):
      sizes[0] = fragment_ld(sizes[0], context)
    return sizes
  unit = max(1, want // context.fp_type.size())
  sizes[0] = -(-sizes[0] // unit) * unit
  return sizes


def fragment_ld(rows: int, context) -> int:
  """The smallest leading dimension at or above `rows` under which NVIDIA's
  matrix path reads a B fragment out of the buffer without a bank conflict
  (`Options.mma_fragment_ld`); `rows` where none within a bank's width is.

  A B fragment of `mma.sync` has lane `(t, g) = (lane % 4, lane // 4)` read
  column `g` at row `t` (plus four per further register), so a register's
  read is `t + ld * g` over the warp -- four adjacent elements eight columns
  apart.  At `ld = 56` in double precision those land on eight bank pairs in
  a half-warp's phase instead of sixteen, a two-way conflict on every read;
  at 60 on sixteen.  The store that fills the buffer writes a row a lane,
  adjacent whatever the stride, and every other reader addresses through the
  view's strides, so the padding is never read and costs a few elements a
  column."""
  from tensorforge.backend.pir.banks import ways
  size = context.fp_type.size()
  banks = getattr(context.target.hw, 'shmem_banks', None) or 32
  pattern = [(t, g) for g in range(8) for t in range(4)]
  for ld in range(int(rows), int(rows) + banks + 1):
    if ways([t + ld * g for t, g in pattern], size, 1, banks) == 1:
      return ld
  return int(rows)


def _contains(outer, inner) -> bool:
  """Whether box `inner` lies inside box `outer`."""
  return all(ol <= il and iu <= ou for il, iu, ol, ou in zip(
      inner.lower(), inner.upper(), outer.lower(), outer.upper()))


def _intersect(a, b):
  """The common part of two boxes, or None where they share no cell."""
  lower = [max(x, y) for x, y in zip(a.lower(), b.lower())]
  upper = [min(x, y) for x, y in zip(a.upper(), b.upper())]
  if any(l >= u for l, u in zip(lower, upper)):
    return None
  return BoundingBox(lower, upper)


def _split_axes_of(symbol):
  """`{axis: (block, stride)}` of a register image with a two-level lead
  (`Options.lead_split`), in lane order; None for a one-axis image."""
  if (getattr(symbol, 'lead_axes', None) is None
      or len(getattr(symbol, 'lead_dims', ())) < 2):
    return None
  layout = symbol.register_layout()
  if layout is None:
    return None
  return {dim: (layout.axis(n).block, layout.axis(n).stride)
          for n, dim in enumerate(symbol.lead_dims)}


def _split_loops(split, bbox):
  """The nest over `bbox` of a two-level image: a lead loop per axis at the
  block and lane stride the image states, a plain loop elsewhere."""
  loops = []
  for i in range(bbox.rank()):
    if i in split:
      block, stride = split[i]
      loops += [LeadLoop(f'i{i}', bbox.lower()[i], bbox.upper()[i], block,
                         stride)]
    else:
      loops += [Loop(f'i{i}', bbox.lower()[i], bbox.upper()[i], 1)]
  return loops


def _complement(lo, hi, own_lo, own_hi):
  """Disjoint boxes covering `[lo, hi)` less `[own_lo, own_hi)`."""
  lo, hi = list(lo), list(hi)
  boxes = []
  for d in range(len(lo)):
    a = min(max(lo[d], own_lo[d]), hi[d])
    b = max(min(hi[d], own_hi[d]), a)
    if lo[d] < a:
      boxes.append((lo[:d] + [lo[d]] + lo[d + 1:], hi[:d] + [a] + hi[d + 1:]))
    if b < hi[d]:
      boxes.append((lo[:d] + [b] + lo[d + 1:], hi[:d] + [hi[d]] + hi[d + 1:]))
    lo[d], hi[d] = a, b
    if a == b:
      break
  return boxes


class StoreRegToShr(AbstractShrMemWrite):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               shr_mem: Symbol,
               num_threads: int,
               lead_width: int = 1,
               dest_bbox=None,
               dest_offset=None,
               clear: bool = False,
               clear_within=None):
    super(StoreRegToShr, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in registers')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is not registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.SharedMem:
      raise InternalError('store: operand `dest` is not in shared mem.')

    if not isinstance(dest.obj, Tensor):
      raise InternalError(f'store: operand `dest` is not a matrix, instead: {type(dest.obj)}')

    src.add_user(self)
    dest.add_user(self)
    shr_mem.add_user(self)

    # `dest_bbox` is the whole buffer, `dest_offset` where this store lands in
    # it.  The two differ as soon as a tensor is assembled from several writes:
    # sizing the view to `src` alone would give every slice its own idea of how
    # big the shared buffer is, and the last one written would win.
    self._dest_offset = (list(dest_offset) if dest_offset is not None
                         else [0] * src.data_view.get_bbox().rank())
    buffer_bbox = dest_bbox if dest_bbox is not None else src.data_view.get_bbox()
    # Whether this store writes the whole buffer or one slice of it -- see
    # `partial_defs`.  Decided here, from the two boxes the constructor is
    # given, and biased towards "part": calling a whole write partial costs a
    # longer live range, calling a slice whole lets the allocator hand an
    # earlier slice's cells to another buffer.
    self._partial = (list(src.data_view.get_bbox().sizes())
                     != list(buffer_bbox.sizes())
                     or any(o != 0 for o in self._dest_offset))
    #: Whether this store also zeroes the rest of the buffer -- see
    #: `SectionPlan.zero_first`.  It is the temporary's first write then, and
    #: it defines the whole buffer rather than a slice of it.
    #:
    #: `clear_within` confines that to a box of the buffer: an assignment owes
    #: zeros only over the box it promised to define
    #: (`MultilinearBuilder._assignment_zero_box`), and a slice of a buffer
    #: other operations assemble must not touch their parts.  A store that
    #: confined has no zero left to write clears nothing, and it defines the
    #: whole buffer only where the box covers it.
    self._clear_within = None
    if clear and clear_within is not None:
      src_bbox = src.data_view.get_bbox()
      own = BoundingBox([l + o for l, o in zip(src_bbox.lower(), self._dest_offset)],
                        [u + o for u, o in zip(src_bbox.upper(), self._dest_offset)])
      within = _intersect(buffer_bbox, clear_within)
      clear = within is not None and not _contains(own, within)
      self._clear_within = within
    self._clear = clear
    if clear and (clear_within is None or _contains(clear_within, buffer_bbox)):
      self._partial = False
    dest.data_view = DataView(_padded_row(buffer_bbox.sizes(), context),
                              permute=None,
                              bbox=buffer_bbox)

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._shr_mem: Symbol = shr_mem
    self._num_threads: int = num_threads
    #: See `gen_code_inner`: the register image is blocked by this.
    self._lead_width: int = lead_width
    view: DataView = self._dest.data_view
    self._shm_volume: int = view.get_volume()

  def transfer_granule(self) -> int:
    """Elements one access of this store writes: the lead width the register
    image is blocked by (`_store`), whose lane holds that many consecutive
    rows and writes them in one access.  The buffer's permutation keeps
    them together (`AbstractShrMemWrite._swizzle`)."""
    return max(1, self._lead_width)

  def partial_defs(self):
    """The buffer, when this store is one of several that assemble it.

    `mixed/ml_slices_then_ew` writes `tmp[:, 0:4]` and `tmp[:, 4:8]` in two
    stores.  Taken as a whole definition, the second would kill the first, so
    the first half would be dead until then; the allocator could give that
    stretch to a buffer read in between, which the first store then
    overwrites.
    """
    return (self._dest,) if self._partial else ()

  def gen_code_inner(self, writer: Writer) -> None:
    # The zeros go out on the clearing nest's lanes, which are not the lanes
    # that wrote or will write the other cells: a store into this buffer
    # before the zeros, or after them, writes some of the same cells from
    # other lanes.  Both wait for a barrier (`pir.barriers`), which two
    # stores into a buffer nobody clears do not -- the zeros are what the
    # buffer holds where nothing else is computed, and the order of the
    # writes is the only thing that says so.
    buf = (self._dest.pir_buffer(writer)
           if getattr(self, '_clear', False) and hasattr(writer, 'mark')
           else None)
    if buf is not None:
      writer.mark('clears', buf)
    self._store(writer)
    if buf is not None:
      writer.mark('cleared', buf)

  def _store(self, writer: Writer) -> None:
    src_bbox = self._src.data_view.get_bbox()
    if getattr(self, '_clear', False):
      self._clear_rest(writer, src_bbox)
    if src_bbox.rank() == 0:
      # one value, the same on every lane: no loop to distribute
      value = self._src.load(writer, self._context, None, [], False)
      self._dest.store(writer, self._context, value, [], False)
      return

    loops = []
    split = _split_axes_of(self._src)
    if split is not None:
      loops = _split_loops(split, src_bbox)
    else:
      # The width the compute instruction used, not 1.  The register image
      # is blocked by it -- lane `l` holds elements `w*l .. w*l+w-1` -- and
      # reading it back cyclically maps register float `r` to element
      # `l + r*threads` instead of `w*(l + slot*block) + c`.  Fourteen of
      # sixteen entries land in the wrong place for `w = 2`, silently: the
      # code compiles, the snapshot looks plausible, and the only thing that
      # would notice is a numerical check the host oracle cannot yet run.
      loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0],
                         self._num_threads, 1, width=self._lead_width,
                         exact_head=True)]
      for i in range(1, src_bbox.rank()):
        loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

    def inner(indices):
      value = self._src.load(writer, self._context, None, indices, False)
      self._dest.store(writer, self._context, value,
                       [add_offset(x, self._dest_offset[i])
                        for i, x in enumerate(indices)], False)

    write_loops(self._context, writer, loops, inner)

  def _clear_rest(self, writer: Writer, src_bbox) -> None:
    """Zero every cell of the buffer this store does not write itself --
    of the part `clear_within` confines it to, where it does.

    The complement only, as disjoint boxes, and never a cell the store then
    writes: the two nests map a row to a lane each in their own way -- the
    data one by the compute's width -- and a cell written twice by two lanes
    has no order between them.  What a later operation reads from here, the
    section's barriers order, as they do for every shared write.
    """
    buf = self._dest.data_view.get_bbox()
    if getattr(self, '_clear_within', None) is not None:
      buf = self._clear_within
    own_lo = [l + o for l, o in zip(src_bbox.lower(), self._dest_offset)]
    own_hi = [u + o for u, o in zip(src_bbox.upper(), self._dest_offset)]
    lo, hi = list(buf.lower()), list(buf.upper())
    boxes = []
    for d in range(buf.rank()):
      a = min(max(lo[d], own_lo[d]), hi[d])
      b = max(min(hi[d], own_hi[d]), a)
      if lo[d] < a:
        boxes.append((lo[:d] + [lo[d]] + lo[d + 1:], hi[:d] + [a] + hi[d + 1:]))
      if b < hi[d]:
        boxes.append((lo[:d] + [b] + lo[d + 1:], hi[:d] + [hi[d]] + hi[d + 1:]))
      lo[d], hi[d] = a, b
      if a == b:
        break

    def zero(indices):
      self._dest.store(writer, self._context, writer.const(0), indices, False)

    for blo, bhi in boxes:
      loops = [LeadLoop('z0', blo[0], bhi[0], self._num_threads, 1)]
      loops += [Loop(f'z{i}', blo[i], bhi[i], 1) for i in range(1, len(blo))]
      write_loops(self._context, writer, loops, zero)

  def get_dest(self) -> Symbol:
    return self._dest

  def __str__(self) -> str:
    clear = ', clear' if getattr(self, '_clear', False) else ''
    return f'{self._dest.name} = store{{r>s{clear}}}({self._shr_mem.name}, {self._src.name});'


class StoreRegToGlb(AbstractInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               num_threads: int,
               lead_width: int = 1,
               dest_offset=None,
               dest_bbox=None,
               zero_fill=True):
    super(StoreRegToGlb, self).__init__(context)

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not in reg mem')

    if not isinstance(src.obj, RegMemObject):
      raise InternalError(f'store: operand `src` is registers, instead: {type(src.obj)}')

    if dest.stype != SymbolType.Global:
      raise InternalError('store: operand `dest` is not in global memory.')

    if not isinstance(dest.obj, Tensor):
      raise InternalError('store: operand `dest` is not a matrix')

    src.add_user(self)
    dest.add_user(self)

    # The fourth view onto a global tensor, and the only one on the writing
    # side.  A prepared operand is an input today, so this is inert -- but a
    # store addressing the destination at one stride while a load addresses it
    # at another is the kind of disagreement that shows up as wrong values in
    # a later kernel rather than as a failure here.
    dest.data_view = DataView(shape=dest.obj.get_actual_shape(),
                              permute=None,
                              bbox=dest.obj.get_bbox(),
                              elem_parts=dest.obj.storage_parts,
                              owner=dest.obj)

    #if dest.data_view.get_dim_size(0) < src.data_view.get_dim_size(0):
    #  raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._num_threads: int = num_threads
    #: See `gen_code_inner`: the register image is blocked by this.
    self._lead_width: int = lead_width
    self._is_ready: bool = True
    # logical->storage shift of the destination slice.  `src` (registers) is
    # indexed logically, `dest` (global) in storage coordinates, so the shift
    # enters both the loop bounds --- which mix the two bboxes --- and the
    # store itself.
    self._dest_offset = (list(dest_offset) if dest_offset is not None
                         else [0] * dest.data_view.rank())
    # What this store promises to define, in the same logical coordinates as
    # `src`.  `_analyze` intersects the accumulator's range down to what the
    # operands support, so it can end up narrower than the descriptor declared
    # and the difference has to be zero-filled.  The *tensor's* box is not that
    # promise: a sliced write declares a small part of it, and the rest belongs
    # to other descriptors or to whatever the caller put there.
    # An accumulation defines nothing: it adds to what is there.  Only an
    # assignment undertakes to cover a box, and only then may the part
    # `_analyze` narrowed away be filled with zeros.
    self._promise = dest_bbox if zero_fill else None
    self._zero_fill = zero_fill

  def gen_ir(self, writer: Writer) -> None:
    writer.new_line()

    # With `Options.hint_outputs`, also a destination that nothing reads but
    # transfers -- a `+=` destination's own preload: an accumulated register
    # image always has other users, so without the option no output store
    # takes the hint.
    allowed = len(self._src.get_user_list()) == 1  # self._src.get_last_user() is self
    if not allowed and self._context.get_user_options().hint_outputs:
      allowed = all(getattr(r, '_src', None) is self._dest
                    for r in readers(self._dest))
    allow_nontemporal = cache_hint(self._context, allowed)

    writer.Comment(f'{self}')
    src_bbox = self._src.data_view.get_bbox()
    if not getattr(self, '_zero_fill', True):
      dest_bbox = src_bbox
    elif self._promise is not None:
      dest_bbox = self._promise
    else:
      # pull the destination's storage bbox back into logical coordinates so
      # the union below compares like with like
      raw_dest_bbox = self._dest.data_view.get_bbox()
      dest_bbox = BoundingBox(
          [l - o for l, o in zip(raw_dest_bbox.lower(), self._dest_offset)],
          [u - o for u, o in zip(raw_dest_bbox.upper(), self._dest_offset)])
    if src_bbox.rank() == 0:
      # One value, the same on every lane, and every lane writes it: the same
      # number to the same address, so there is no owner to pick.
      value = self._src.load(writer, self._context, None, [], False)
      self._dest.store(writer, self._context, value, [], allow_nontemporal)
      return
    with writer.Scope():
      split = _split_axes_of(self._src)
      if split is not None:
        self._store_split(writer, split, src_bbox, dest_bbox,
                          allow_nontemporal)
        return
      loops = []
      # The lead loop is built from the accumulator alone: it drives the thread
      # mapping, and widening it would hand `inner` a LeadIndex outside what
      # the register array holds, which the `needsLoad` test below cannot see
      # (it recognizes `Immediate` only).  What the promise covers beyond it
      # gets its own nest below, which writes zeros and never touches `src`.
      # Widened like the other two: the register image is blocked by the
      # compute width, so reading it back cyclically would put fourteen of
      # sixteen entries in the wrong place for `w = 2`.
      pad = self._pad_writable(src_bbox, dest_bbox)
      loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0],
                         self._num_threads, 1, width=self._lead_width,
                         full_lane=pad, pad=pad, exact_head=True)]
      for i in range(1, src_bbox.rank()):
        unroll = (src_bbox.lower()[i], src_bbox.upper()[i]) != (dest_bbox.lower()[i], dest_bbox.upper()[i])
        lower = min(src_bbox.lower()[i], dest_bbox.lower()[i])
        upper = max(src_bbox.upper()[i], dest_bbox.upper()[i])
        loops += [Loop(f'i{i}', lower, upper, 1, unroll)]

      def inner(indices):
        needsLoad = all(not isinstance(index, Immediate) or (src_bbox.lower()[i] <= index._value and src_bbox.upper()[i] > index._value) for i,index in enumerate(indices))
        dest_indices = [add_offset(x, self._dest_offset[i])
                        for i, x in enumerate(indices)]

        def emit():
          if needsLoad:
            value = self._src.load(writer, self._context, None, indices, False,
                                   broadcast=owner is None)
          else:
            value = writer.const(0) # TODO: dtype
          self._dest.store(writer, self._context, value, dest_indices,
                           allow_nontemporal)

        # A lead index the loop handed over as a plain integer -- the peeled
        # tail of a widened lead dimension -- names one element, and one lane
        # holds it.  Global memory has no owner of its own, so `Symbol.store`
        # cannot work this out for itself the way it does for a register
        # destination; the source symbol knows and this is where both are in
        # scope.
        #
        # The whole body goes inside the branch rather than the store alone,
        # which is what makes the shuffle disappear instead of moving.  The
        # broadcast exists so every lane has an element only one of them
        # holds; with the write guarded to that lane it is reading its own
        # register, and `readlane` is `__shfl_sync` over the full warp mask,
        # so leaving it outside would be a shuffle whose partners are in a
        # branch they do not take.
        #
        # Without the guard every lane would store the element.  Under `=`
        # that is the same value written `threads` times and the result is
        # right; under `+=` it is the contribution counted `threads` times.
        owner = self._src.owning_lane(indices) if needsLoad else None
        if owner is None:
          emit()
        else:
          with writer.If(f'{self._context.target.lexic.thread_idx_x}'
                         f' == {owner}'):
            emit()

      # As the loader: a hint the target cannot spell is none.
      hinted = bool(allow_nontemporal) and self._context.target.nontemporal(
          self._dest.get_fptype())
      found = (element_blocks.active(self._context, self)
               if not hinted else None)
      if not (found is not None and self.element_surface() is not None
              and self._block_loops(writer, found, loops, inner, src_bbox)):
        write_loops(self._context, writer, loops, inner)

      # The whole tail wrote its padding lanes' zeros already.
      threads = self._num_threads
      filled = (-(-src_bbox.upper()[0] // threads) * threads) if pad else None
      self._fill_lead_remainder(writer, src_bbox, dest_bbox, allow_nontemporal,
                                upper_from=filled)

  def element_surface(self):
    """The element surface this store writes by 2D blocks where the body
    asks for them (`Options.sycl_element_blocks`), or None: an operand of
    the element with one (`element_blocks.surface`) written out of an image
    of sixteen lanes a slot, its rows on the lanes, one element a lane, and
    no mask of a stack's elements over the writes."""
    from tensorforge.backend import elementmask
    if (not element_blocks.wanted(self._context)
        or element_blocks.target_refusal(self._context) is not None):
      return None
    found = element_blocks.surface(self._dest)
    if found is None or _split_axes_of(self._src) is not None:
      return None
    if (self._src.lead_dims != [0] or self._lead_width != 1
        or self._num_threads != element_blocks.LANES
        or self._src.lead_block(0) != element_blocks.LANES
        or elementmask.active()[0] is not None):
      return None
    bbox = self._src.data_view.get_bbox()
    if bbox.rank() != 2:
      return None
    span = element_blocks.LANES
    lo, hi = bbox.lower()[0], bbox.upper()[0]
    slots = range(lo // span, -(-hi // span))
    if not any(element_blocks.slots_inside(found, self._dest_offset[0], s)
               for s in slots):
      return None
    return found

  def _block_loops(self, writer, found, loops, inner, src_bbox) -> bool:
    """`loops` with the slots whose every lane this store writes today
    written by 2D blocks: a slot of the loop, a whole slot handed over on
    its own, a full-lane tail that writes its padding lanes' zeros (`pad`).
    The values are today's -- the image's, the zero a column past the image
    gets, the zero of a padding lane -- gathered a slot at a time and
    written in messages of eight columns (`element_blocks.chunks`).  A slot
    a guard holds to some of its lanes -- the ragged slot of an
    accumulation, which promises nothing of the rest -- is written as
    today, lane by lane, and so is a slot whose 16 rows leave the operand's
    columns.  The message writes its block whole and stands under no
    guard: the lead loop puts the slots that need one under it, and those
    are the ones written as today.

    The ragged last slot of a write that ends inside a slot -- SeisSol's
    rows 48-55, of a 56-row column or of the first 56 of 64 -- is written
    as the block that ends with its last row, 16 rows back, when the slot
    before it is one this store writes whole by blocks: its lanes are that
    slot's upper ones and this slot's written ones, each rotated into place
    down or up the sub-group and the two merged by lane
    (`Lexic.sub_group_shift`) -- the rows today's writes write, with their
    values, two messages for nine (`_overlap_tail_plan`).  Not a padded
    tail (`LeadLoop.pad`): today's write zeroes the padding rows after the
    data, which that block ends before -- inside the columns it is a slot
    written whole, as above.

    False, and nothing built, where the bound buffer is not in this body."""
    from tensorforge.backend.pir import block_reads
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType, Value
    base = self._dest.pir_buffer(writer)
    if base is None or not hasattr(writer, 'pack'):
      return False
    lead, cols = loops
    span = element_blocks.LANES
    first_row, first_col = self._dest_offset[0], self._dest_offset[1]
    c_lo, c_hi = cols.start, cols.end
    if not (0 <= c_lo + first_col and c_hi + first_col <= found.columns):
      return False
    pieces = element_blocks.chunks(c_lo, c_hi)
    elem = self._dest.get_fptype()
    start, end = lead.start, lead.end
    loop_lo, loop_hi = -(-start // span), end // span
    loop_inside = (loop_lo < loop_hi
                   and element_blocks.slots_inside(found, first_row, loop_lo)
                   and element_blocks.slots_inside(found, first_row,
                                                   loop_hi - 1))

    def whole(index, slot) -> bool:
      if getattr(index, 'first', 0):
        return False
      if index.valid is not None:
        return bool(index.pad)
      if isinstance(slot, int):
        return start <= slot * span and (slot + 1) * span <= end
      return True

    lexic = self._context.target.lexic
    tail = self._overlap_tail_plan(found, lead, loop_lo, loop_hi,
                                   loop_inside, first_row)

    def write_tail(slot, valid):
      """The ragged last slot `slot`, lanes below `valid` written today, as
      the block of the 16 rows that end with its last one, written after
      the lead loop and under no guard: lanes below `16 - valid` the upper
      lanes of the slot before -- one this store writes whole by blocks --
      rotated down, the rest this slot's rotated up, both as the image
      holds them."""
      from tensorforge.backend.symbol import LeadIndex
      ahead = span - valid
      mine = [LeadIndex(slot, lead.threads, lead.stride, width=lead.width)]
      before = [LeadIndex(slot - 1, lead.threads, lead.stride,
                          width=lead.width)]
      lane = writer.lane_index(span, hint='elane')
      low = writer.op('lt', BOOL, lane, ahead, hint='elow')
      merged = {}

      def at(w):
        q = int(w[0]._value)
        if not src_bbox.lower()[1] <= q < src_bbox.upper()[1]:
          merged[q] = writer.const(0)
          return
        own = self._src.load(writer, self._context, None, mine + w, False)
        prior = self._src.load(writer, self._context, None, before + w,
                               False)
        down = writer.rawexpr(lexic.sub_group_shift('{0}', valid), prior,
                              type_=ScalarType(elem), hint='erot', pure=True,
                              crosslane=True)
        up = writer.rawexpr(lexic.sub_group_shift('{0}', ahead, down=False),
                            own, type_=ScalarType(elem), hint='erot',
                            pure=True, crosslane=True)
        merged[q] = writer.op('select', ScalarType(elem), low, down, up,
                              hint='emerge')

      Loop(cols.var, c_lo, c_hi, cols.step, unroll=True).write(
          self._context, writer, at)
      x = slot * span + first_row + valid - span
      for q, rows in pieces:
        part = [merged[c] for c in range(q, q + rows)]
        share = (part[0] if rows == 1 else
                 writer.pack(ScalarType(elem, rows), *part, hint='eshare'))
        block_reads.store_block2d(
            writer, base, x, q + first_col, share, rows=rows, cols=span,
            width=found.pitch, height=found.columns, pitch=found.pitch,
            element=True)

    def slot_level(v):
      index = v[0]
      slot = index._nonlead
      static = isinstance(slot, (int, np.integer))
      slot = int(slot) if static else slot
      if static and tail is not None and slot == tail[0]:
        # Written by `write_tail` after the loop, under no guard.
        return
      inside = (element_blocks.slots_inside(found, first_row, slot)
                if static else loop_inside and isinstance(index._value, Value))
      if not (inside and whole(index, slot)):
        Loop(cols.var, c_lo, c_hi, cols.step, cols.unroll).write(
            self._context, writer, lambda w: inner(v + w))
        return
      values = {}

      def at(w):
        indices = v + w
        q = int(w[0]._value)
        if src_bbox.lower()[1] <= q < src_bbox.upper()[1]:
          value = self._src.load(writer, self._context, None, indices, False)
        else:
          value = writer.const(0)
        if index.valid is not None:
          held = self._dest._valid_access(
              writer, [add_offset(x, self._dest_offset[i])
                       for i, x in enumerate(indices)],
              index.valid).get('predicate')
          if held is not None:
            value = writer.op('select', ScalarType(elem), held, value,
                              writer.const(0), hint='edata')
        values[q] = value

      Loop(cols.var, c_lo, c_hi, cols.step, unroll=True).write(
          self._context, writer, at)
      if static:
        x = slot * span + first_row
      else:
        x = writer.op('mul', INDEX, index._value, span, hint='eslot')
        if first_row:
          x = writer.op('add', INDEX, x, first_row, hint='eslot')
      for q, rows in pieces:
        part = [values[c] for c in range(q, q + rows)]
        share = (part[0] if rows == 1 else
                 writer.pack(ScalarType(elem, rows), *part, hint='eshare'))
        block_reads.store_block2d(
            writer, base, x, q + first_col, share, rows=rows, cols=span,
            width=found.pitch, height=found.columns, pitch=found.pitch,
            element=True)

    lead.write(self._context, writer, slot_level)
    if tail is not None:
      write_tail(*tail)
    return True

  def _overlap_tail_plan(self, found, lead, loop_lo, loop_hi, loop_inside,
                         first_row) -> Optional[Tuple[int, int]]:
    """`(slot, valid)` for the ragged last slot of this store's lead range
    that `_block_loops` writes as the block ending with its last row, or
    None: the range ends inside a slot and no padding follows it that
    today's write zeroes (`LeadLoop.pad`), the slot before is one the store
    writes whole by blocks -- inside the operand's columns, after the
    range's start, a static slot or one of a loop of slots inside -- the
    block lies in the columns, and the target spells the rotation."""
    span = element_blocks.LANES
    start, end = lead.start, lead.end
    if (not end % span or lead.pad
        or self._context.target.lexic.sub_group_shift('{0}', 1) is None):
      return None
    slot = (end - 1) // span
    valid = end - slot * span
    before = slot - 1
    if before * span < start or not element_blocks.slots_inside(
        found, first_row, before):
      return None
    if not lead.unroll and loop_lo <= before < loop_hi and not loop_inside:
      return None
    x = slot * span + first_row + valid - span
    if x < 0 or x + span > found.rows:
      return None
    return slot, valid

  def _store_split(self, writer, split, src_bbox, dest_bbox,
                   allow_nontemporal) -> None:
    """The store of a two-level image (`Options.lead_split`): a lead loop
    per axis over what the image holds, each element written by the lane
    that holds it, and where the store zero-fills, zeros over the rest of
    the promised box, as the shared store clears around its own
    (`StoreRegToShr._clear_rest`)."""
    def inner(indices):
      value = self._src.load(writer, self._context, None, indices, False)
      self._dest.store(writer, self._context, value,
                       [add_offset(x, self._dest_offset[i])
                        for i, x in enumerate(indices)], allow_nontemporal)

    write_loops(self._context, writer, _split_loops(split, src_bbox), inner)
    if not self._zero_fill:
      return

    def zero(indices):
      self._dest.store(writer, self._context, writer.const(0),
                       [add_offset(x, self._dest_offset[i])
                        for i, x in enumerate(indices)], allow_nontemporal)

    for blo, bhi in _complement(dest_bbox.lower(), dest_bbox.upper(),
                                src_bbox.lower(), src_bbox.upper()):
      loops = [LeadLoop('z0', blo[0], bhi[0], self._num_threads, 1)]
      loops += [Loop(f'z{i}', blo[i], bhi[i], 1) for i in range(1, len(blo))]
      write_loops(self._context, writer, loops, zero)

  def _pad_writable(self, src_bbox, dest_bbox) -> bool:
    """May the ragged end of the lead dimension be written whole?

    Where this store zero-fills the rows after it anyway.  A tensor whose box
    is larger than what the multiplication defines -- `64 x 9` with 56 rows
    computed -- is promised zeros beyond (`_fill_lead_remainder`), and a
    full-width tail whose padding lanes are zeroed (`Symbol.store`) writes
    exactly those zeros, in the same write as the data: one message rather
    than two (16 + 8) and a separate nest under ESIMD, one select rather than
    a branch under SPMD.  The remainder fill then starts after the slot.

    An accumulation promises nothing -- it adds to what is there -- and a
    window that ends inside the tensor's data has the next rows after it; both
    keep the narrow write.  A tensor stored as its box has no rows after it at
    all (`get_actual_shape` is the box).
    """
    opts = self._context.get_user_options()
    if (not getattr(opts, 'full_lane_tails', False)
            or not getattr(self, '_zero_fill', True)
            or self._lead_width != 1 or not self._num_threads):
      return False
    threads = self._num_threads
    hi = src_bbox.upper()[0]
    if hi % threads == 0:
      return False
    slot_end = -(-hi // threads) * threads
    return dest_bbox.lower()[0] <= src_bbox.lower()[0] and dest_bbox.upper()[0] >= slot_end

  def _fill_lead_remainder(self, writer, src_bbox, dest_bbox,
                           allow_nontemporal, upper_from=None) -> None:
    """Zero what the promise covers beyond the accumulator's lead range.

    A tensor padded for alignment --- 20 basis functions stored as 32 --- is
    assigned over its eqspp window and read back over the padded range, so the
    padding has to be defined and nothing else defines it.  The main nest
    cannot do it: its lead loop is sized to the register array, and running it
    further would read past the end.  A separate nest over just the remainder
    can, because it writes a constant and never touches `src`.

    An accumulation never gets here: it does not zero-fill at all.
    """
    if src_bbox.rank() == 0:
      return
    gaps = [(dest_bbox.lower()[0], src_bbox.lower()[0]),
            (upper_from if upper_from is not None else src_bbox.upper()[0],
             dest_bbox.upper()[0])]
    for lo, hi in gaps:
      if lo >= hi:
        continue
      loops = [LeadLoop('z0', lo, hi, self._num_threads, 1)]
      for i in range(1, src_bbox.rank()):
        loops += [Loop(f'z{i}', min(src_bbox.lower()[i], dest_bbox.lower()[i]),
                       max(src_bbox.upper()[i], dest_bbox.upper()[i]), 1)]

      def zero(indices):
        value = self._spread_zero(writer, indices[0])
        self._dest.store(writer, self._context, value,
                         [add_offset(x, self._dest_offset[i])
                          for i, x in enumerate(indices)],
                         allow_nontemporal)

      write_loops(self._context, writer, loops, zero)

  def _spread_zero(self, writer, at):
    """The zero a remainder row is written, spread as the row index is.

    A constant is the same on every lane, which under SPMD every lane stores
    at its own address.  Under the explicit vector one work-item stores for
    all of them, at the block's first address, as wide as the value: a
    constant wrote one element, so of `dQ(1)`'s rows 10..15 at order 4 only
    row 10 became the zero its storage promises, the rest kept what the
    buffer held.  A select over the block's lanes (`lane_index`, as
    `PromisedZeros` builds its mask) carries their distribution, and the
    store is the block's vector of zeros.
    """
    zero = writer.const(0) # TODO: dtype
    from tensorforge.backend.symbol import LeadIndex
    if (not getattr(self._context.target, 'explicit_simd', False)
        or not isinstance(at, LeadIndex) or not hasattr(writer, 'lane_index')):
      return zero
    from tensorforge.backend.pir.core import BOOL
    lane = writer.lane_index(at._block, at._stride, hint='lead')
    every = writer.op('ge', BOOL, lane, 0, hint='o')
    return writer.op('select', zero.type, every, zero, zero, hint='z')

  def __str__(self) -> str:
    return f'{self._dest.name} = store{{r>g}}({self._src.name});'


class ReserveShrMem(AbstractShrMemWrite):
  """A shared buffer with a window and no transfer.

  Every other buffer is introduced by a store or a load, and `temporaries`
  says why: the window is declared by the buffer's *first user*, which has
  to know how big it is, and a compute instruction writing shared memory
  does not.  A rotation slot has no such user at all.  The members of a
  merged run's carried family are written through the loop's table
  (`Generator._rotate_members`), so the slot itself never appears as the
  destination of a transfer, and `_declare_buffers` would reach it with an
  empty user list.

  So the rotation introduces them, which is the one thing that does know
  their size: it is the family's shape, and the family has one by
  construction (`rolling._rotatable` refuses otherwise).  Nothing is
  emitted for it beyond the window -- `gen_code_inner` writes no statement
  -- and the loop then stores through the table as it would through any
  buffer.

  It defines the slot rather than using it, which is what it is: before it
  the buffer holds nothing anybody wants, and saying so is what lets the
  allocator hand the stretch in front of it to another buffer.
  """

  def __init__(self, context: Context, dest: Symbol, shr_mem: Symbol,
               bbox: BoundingBox):
    super(ReserveShrMem, self).__init__(context)
    if dest.stype != SymbolType.SharedMem:
      raise InternalError(
          f'{dest.name}: a reservation is of a shared buffer, not '
          f'{dest.stype}')
    dest.add_user(self)
    shr_mem.add_user(self)
    # The view as a store would have left it, padded the same way: a slot is
    # read through the loop's table before anything writes it -- that is what
    # a carried family *is* -- so the view cannot wait for a first write the
    # way every other temporary's does.
    dest.data_view = DataView(_padded_row(bbox.sizes(), context),
                              permute=None, bbox=bbox)
    self._dest = dest
    self._shr_mem = shr_mem
    self._shm_volume = dest.data_view.get_volume()
    if self._shm_volume <= 0:
      raise InternalError(
          f'{dest.name}: a buffer of {self._shm_volume} elements')
    self._is_ready = True

  def gen_code_inner(self, writer: Writer) -> None:
    """Nothing: the window is the whole of it, and that is emitted by
    `gen_code_declare` before this runs."""

  def get_dest(self) -> Symbol:
    return self._dest

  def get_src(self):
    return None

  def __str__(self) -> str:
    return f'{self._dest.name} = reserve{{s}}({self._shm_volume})'
