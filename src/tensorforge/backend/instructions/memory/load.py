# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import List, Union
import math
from tensorforge.common.matrix.tensor import Tensor
from . import AbstractShrMemWrite, MemoryInstruction
from tensorforge.backend.symbol import Symbol, SymbolType, DataView, write_loops, LeadLoop, Loop, add_offset
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.backend.data_types import RegMemObject
from .hints import cache_hint, readers


def _hint_allowed(loader) -> bool:
  """Whether a load from global memory may take the cache hint.

  Where it is the only user of its source.  With `Options.hint_outputs` also
  where it is the only *reader*: the one read of a `+=` destination, whose
  other users only write it.
  """
  src = loader._src
  if len(src.get_user_list()) == 1:
    return True
  if not loader._context.get_user_options().hint_outputs:
    return False
  mine = readers(src)
  return len(mine) == 1 and mine[0] is loader


# to find a number coprime to the number of shared memory banks
def _find_next_coprime(number, conumber):
  for i in range(number, number + conumber):
    if math.gcd(i, conumber) == 1:
      return i

from . import vectorize


class LoadInstruction:
  pass

class GlbToShrLoader(AbstractShrMemWrite, LoadInstruction):
  def __init__(self, **kwargs):
    super(GlbToShrLoader, self).__init__(kwargs['context'])
    self._dest = kwargs['dest']
    self._src = kwargs['src']
    self._shr_mem = kwargs['shr_mem']
    self._num_threads = kwargs['num_threads']
    #: Set again by `set_threadconfig_pre`, which is where a blockwide
    #: transfer widens `_num_threads` to the block and this one does not
    #: follow.  Initialized here so that a transfer nobody reconfigures still
    #: answers the question.
    self._lanes = kwargs['num_threads']
    #: A shared image is not blocked -- the compute path reads it by element,
    #: not by lane -- so this stays 1 unless a caller says otherwise.
    self._lead_width = kwargs.get('lead_width', 1)
    self._permute: None = kwargs['permute']
    self._manual_unroll_threshold = 4
    self._no_memcpy = kwargs['no_memcpy'] if 'no_memcpy' in kwargs else False

    if 'max_load_offset' in kwargs:
      self._max_load_offset = kwargs['max_load_offset']
    else:
      self._max_load_offset = self._num_threads

    if 'blockwide' in kwargs:
      self._blockwide = kwargs['blockwide']
    else:
      self._blockwide = False

    if 'alignment' in kwargs:
      self._alignment = kwargs['alignment']
    else:
      self._alignment = 1

    #: Copy the operand's storage scalar for scalar, as the host laid it out
    #: -- parts, their planes and any storage order included -- rather than
    #: its logical elements.  For the section prologue's preloaded operators:
    #: the image in shared memory is read with the same view as the buffer in
    #: global memory, so it has to be the same bytes.
    self._verbatim = kwargs.get('verbatim', False)

    self._check()
    self._lid_dim: Union[int, None] = None
    self._align_shm_volume: Union[int, None] = None
    self._tensor: Tensor = self._src.obj

    self._dest.add_user(self)
    self._src.add_user(self)
    self._shr_mem.add_user(self)
    self._is_ready: bool = False

    #: Whether the transfer is a `copy.async` where it can be one.  Where it
    #: cannot -- a source that is not a value of the body it is issued in --
    #: it moves its bytes with ordinary loads.
    self._use_cuda_memcpy = self._context.get_vm().get_hw_descr().vendor == 'nvidia' and not self._no_memcpy
    #: tokens issued by this transfer, for the `LoadWait` that retires them
    self._tokens = []
    self._token_owner = None
    self._issued_structured = False
    #: did this transfer actually put something in flight?  `_use_cuda_memcpy`
    #: is a static choice; the reordering path ignores it and moves the data
    #: synchronously, so the flag alone cannot tell a wait what to do.
    self._issued_async = False

    if self._permute is None:
      self._permute = [i for i in range(len(self._src.obj.shape))]

    self._needs_reorder = self._permute != [i for i in range(len(self._src.obj.shape))]

    self._get_bounding_box_dense()

  def set_threadconfig_pre(self, num_threads, mults):
    #: Lanes one work-item covers, as opposed to `_num_threads`, which is how
    #: many the whole team covers.  The two coincide except for a blockwide
    #: transfer, where the team is the block and the work-item is still one
    #: multiplication's worth -- and under an explicit vector that difference
    #: decides the width of the register the transfer moves through.
    self._lanes = num_threads
    if self._blockwide:
      self._num_threads = num_threads * mults

  def _hop_granularities(self):
    """Elements per hop this transfer will try, widest first.

    The widths `_write_datatransfer` walks, named once so that the permutation
    can ask the same question the transfer answers.

    Bounded by what is *proved* about the source address, not by what is
    likely: `cp.async` needs both ends naturally aligned to the width it
    moves, and `Symbol.linear_align_bytes` is the promise the frontend
    attached -- 16 where the layout reports an aligned stride, the element
    size where it reports nothing.  An unproved alignment and a natural one
    are the same number and opposite facts, so the element size means width 1,
    and the whole transfer drops to it rather than only its tail.

    The destination is 16-byte aligned by construction (the allocator starts
    every buffer in either arena on 16 bytes, `pir.allocate`), so it
    constrains nothing here.  Its *permutation*
    does, and that is asked at the access -- see `_swizzle_cap`.
    """
    elem = self._dest.get_fptype().size()
    limit = min(16, self._src.linear_align_bytes()) if self._use_cuda_memcpy \
        else 16
    return [m for m in [4, 2, 1] if m * elem <= limit]

  def _swizzle_cap(self, writer) -> int:
    """The widest access the destination's permutation admits, in elements.

    A permuted window moves *granules*, and an access wider than one granule
    would be permuted by its first element only -- which is what
    `PirBuilder._check_width` refuses.  So the transfer narrows instead: a
    16-byte load is worth having and a wrong one is not.

    Usually no constraint at all, because `_swizzle` grants a granule at least
    as wide as `transfer_granule()` or declines to permute.  Asked even so:
    the window may carry a permutation this transfer did not choose -- the
    matrix path sets one on its own tiles -- and the invariant then holds only
    by nobody having tried.
    """
    buf = self._destination_buffer(writer) if writer is not None else None
    swz = getattr(getattr(buf, 'type', None), 'swizzle', None)
    if swz is None:
      return 16
    return max(1, swz.granule) if swz.width > 1 else 16

  def transfer_granule(self) -> int:
    """See `AbstractShrMemWrite.transfer_granule`: the widest hop, which is
    what a permutation of the destination has to keep whole."""
    return max(self._hop_granularities())

  def _next_size(self, size):
    return _find_next_coprime(size, self._context.get_vm().get_hw_descr().shmem_banks)

  def _explicit_simd(self) -> bool:
    return bool(getattr(self._context.get_vm().get_lexic(), 'simd_mode', False))

  def _lane_span(self) -> int:
    """How many elements of one hop a single work-item carries.

    Under SPMD the answer is one: a thread moves `increment` elements and the
    team moves `_num_threads` of those. Under an explicit vector the work-item
    *is* the wave, so it carries `_lanes` granules at once and the transfer's
    register is that much wider -- which is a fact about the declaration and
    therefore has to be stated, not left in the address.
    """
    return self._lanes if self._explicit_simd() else self._num_threads

  def _linear_idx(self):
    """Where this work-item's share of a linear transfer begins.

    Under SPMD that is the thread's own index, and the lane term rides in the
    address.  Under an explicit vector there is no lane index to ask for --
    `EsimdEmitter._thread_idx('x')` refuses the question, since one work-item
    holds the whole vector -- so the share of a per-multiplication transfer
    begins at zero and the distribution has moved into the type.

    A blockwide transfer keeps a term, because the team really is several
    work-items: work-item `y` owns lanes `[y * lanes, (y + 1) * lanes)`, and
    those are contiguous granules, so its share begins at `y * lanes`.

    This is text rather than a PIR value, and that is why it has to be said
    here: the transfer is built by the macro layer and never passes the
    emitter that would refuse `item.get_local_id(0)`.  It would reach the
    generated kernel as an ordinary subscript instead -- an address that is
    wrong and compiles.
    """
    lexic = self._context.get_vm().get_lexic()
    if self._explicit_simd():
      if self._blockwide:
        return f'({lexic.thread_idx_y} * {self._lanes})'
      return '0'
    if self._blockwide:
      # The thread's own number in the block, so the *hardware* axes: where a
      # multiplication spans waves the lane is derived from them
      # (`Generator._lane_mapping`) and numbers the lanes of one
      # multiplication, which several threads of the block share.
      tid_x = getattr(lexic, 'raw_thread_idx_x', lexic.thread_idx_x)
      tid_y = getattr(lexic, 'raw_thread_idx_y', lexic.thread_idx_y)
      dim_x = getattr(lexic, 'raw_block_dim_x', lexic.block_dim_x)
      return f'({tid_x} + {tid_y} * {dim_x})'
    else:
      return f'{lexic.thread_idx_x}'

  def _get_bounding_box_dense(self):
    # The parts belong to the *global* tensor: the staging tile below is
    # built from what was copied into it, so its elements are single
    # scalars whatever the source was stored as.
    self._src.data_view = DataView(shape=self._tensor.get_actual_shape(),
                                   permute=None,
                                   bbox=self._tensor.get_bbox(),
                                   elem_parts=self._tensor.storage_parts,
                                   owner=self._tensor)

    src_real_shape = self._tensor.bbox.sizes()
    if self._verbatim:
      # One contiguous run of `storage_volume` scalars.  Counted in elements,
      # as the other branches do, the copy would stop at the first part of a
      # two-part operand: 3136 of 6272 scalars for a split 56x56, and the
      # kernel would multiply by whatever the arena holds behind them.
      volume = self._tensor.storage_volume()
      shape = list(self._tensor.get_actual_shape())
      self._dest.data_view = DataView(shape=shape, permute=None,
                                      bbox=self._tensor.get_bbox(),
                                      elem_parts=self._tensor.storage_parts,
                                      owner=self._tensor)
      self._shm_volume = volume
      self._read_shape = shape
      self._dst_shape = shape
      self._loop_indices = []
      self._loadsize = volume
      return
    dst_bbox = self._tensor.get_bbox() # BoundingBox([0] * len(self._tensor.shape), src_real_shape)
    dst_shape = []
    read_shape = []
    loop_indices = []
    offset = 0
    loadsize = 1
    need_transpose = True

    if self._tensor.is_dense():

      # TODO: remove distinction between tensor shape and real shape
      for i in range(len(src_real_shape)):
        # offset += self._tensor.shape[i] - src_real_shape[i]
        if offset <= self._max_load_offset:
          readshape = src_real_shape[i] # self._tensor.shape[i]
        else:
          readshape = src_real_shape[i]
          loop_indices += [i]
        if self._permute[i] == 0: # TODO: not ideal
          need_transpose = False
        if need_transpose:
          dstshape = self._next_size(readshape)
        else:
          dstshape = readshape

        # TODO: move somewhere else?
        if i == 0:
          dstshape = ((dstshape + self._alignment - 1) // self._alignment) * self._alignment

        dst_shape += [dstshape]
        read_shape += [readshape]
        if len(loop_indices) <= 1:
          loadsize *= readshape

      # cap the first loop index, we're still contiguous there
      if len(loop_indices) > 0:
        loop_indices = loop_indices[1:]

      self._shm_volume = 1
      for dsts in dst_shape:
        self._shm_volume *= dsts
    else:
      loadsize = self._tensor.memory()
      self._shm_volume = loadsize
      read_shape = list(src_real_shape)
      dst_shape = list(src_real_shape)

    self._dest.data_view = DataView(shape=dst_shape,
                                    permute=None,
                                    bbox=dst_bbox)

    self._read_shape = read_shape
    self._dst_shape = dst_shape

    self._loop_indices = loop_indices
    self._loadsize = loadsize

  def gen_code_inner(self, writer: Writer) -> None:
    allow_nontemporal = cache_hint(self._context, _hint_allowed(self))
    if self._verbatim and self._tensor.storage_volume() != self._loadsize:
      # The storage was decided after the image was sized -- an order offered
      # at emission, say -- and the reservation made for it is the old size.
      # Copying the new one would run into the next operator's image.
      raise InternalError(
          f'{self._dest.name}: {self._tensor.alias} occupies '
          f'{self._tensor.storage_volume()} scalars now, and its image in '
          f'shared memory was reserved for {self._loadsize}')


    if self._needs_reorder:
      src_bbox = self._src.data_view.get_bbox()
      loops = []
      # Same width as the compute instruction that reads this image, for the
      # same reason as in `store.py`: the register array is blocked by it, and
      # filling it cyclically while the compute reads it blocked puts most
      # entries in the wrong place without any diagnostic.
      loops += [LeadLoop('i0', src_bbox.lower()[0], src_bbox.upper()[0],
                         self._num_threads, 1, width=self._lead_width)]
      for i in range(1, src_bbox.rank()):
        loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

      def inner(indices):
        value = self._src.load(writer, self._context, None, indices, allow_nontemporal)
        self._dest.store(writer, self._context, value, indices, False)

      # The reordering path moves the data with ordinary loads and stores.
      # Nothing is in flight when it returns, so nothing may be waited for.
      self._issued_async = False
      write_loops(self._context, writer, loops, inner)
    else:
      structured_issue = self._use_cuda_memcpy and self._structured_copy(writer)
      # Nothing is in flight unless the structured route carried it, and a
      # wait for a transfer that never issued is a wait that never retires.
      self._issued_async = structured_issue
      if structured_issue:
        self._tokens = []
        self._token_owner = getattr(writer, 'uid', None)
        self._issued_structured = True

      loops = [writer.For(f'int32_t i{i} = 0; i{i} < {self._dest.data_view.shape[i]}; ++i{i}', True) for i in self._loop_indices]

      for loop in loops:
        loop.__enter__()

      index = list(self._dest.data_view.get_dim_offsets())
      for li in self._loop_indices:
        index[li] = f'i{li}'

      linscale = None
      if len(self._dst_shape) > 0 and self._dst_shape[0] != self._read_shape[0]:
        linscale = (self._read_shape[0], self._dst_shape[0])

      self._write_datatransfer(writer, 0, 0, index, self._loadsize, allow_nontemporal, linscale)

      for loop in loops[::-1]:
        loop.__exit__(None, None, None)

  def _bypass_covering(self, writer, src_offset, length) -> int:
    """Elements per access if this whole run can take the L1-bypassing form.

    Width is not only a count of accesses here.  On sm_120 the 16-byte
    `cp.async` lowers to `cp.async.cg` and `LDGSTS.E.BYPASS.128`, while the
    4- and 8-byte ones lower to `cp.async.ca` and fill L1 on the way -- so a
    staging transfer that ends in narrower accesses evicts the working set of
    every other read in the kernel, which for a kernel already at its L1
    datapath is the expensive half.  The zero-filling variant keeps the bypass
    (`LDGSTS.E.BYPASS.128.ZFILL`), which is what lets a ragged run stay on it
    instead of finishing in two narrower passes and a scalar tail.  Measured
    from the PTX and SASS this toolchain emits, not from the documentation.

    Zero -- keep the stepping -- unless all of it holds:

    * the asynchronous path is available at all, and this transfer takes it;
    * the widest admissible access is exactly sixteen bytes (`_hop_granularities`
      proves the source alignment, `_swizzle_cap` the destination's permutation)
      and the run starts on that boundary;
    * the run is the only one, so the elements the last access zero-fills are
      past the data rather than in the next row of it -- a padded row would do
      as well, and `stage_row_bytes` is how one gets asked for;
    * the window has room for them, which `align_shr_mem` gives by rounding a
      stage up to the vector unit.
    """
    if not (self._use_cuda_memcpy and self._structured_copy(writer)):
      return 0
    if self._explicit_simd() or self._loop_indices or self._needs_reorder:
      return 0
    if not self._context.get_user_options().align_shr_mem:
      return 0
    elem = self._dest.get_fptype().size()
    width = min(max(self._hop_granularities()), self._swizzle_cap(writer))
    if width * elem != 16 or src_offset % width:
      return 0
    if length % width and self._vm.get_lexic().copy_async('d', 's', 16, 4) is None:
      # The last access covers part of an element group, which only the
      # zero-filling form can do, and this target has none.
      #
      # Reachable, though a strided operand cannot show it: there the run *is*
      # the storage volume, which is the batch stride, so a stride that is a
      # multiple of sixteen bytes and a run that is not a multiple of four
      # elements are the same number twice.  `Addressing.PTR_BASED` has no
      # stride at all -- every element names its own buffer -- so the promise
      # is about those buffers and says nothing about the length.  45 elements
      # behind pointers is then a run that not only ends ragged but, at 16
      # lanes, never reaches one whole sixteen-byte round: without the fill it
      # goes to L1 in its entirety.
      return 0
    if self.stage_size() < length + width:
      return 0
    return width

  def _write_datatransfer(self, writer, src_offset, dst_offset, index, length, nontemporal, linscale=None):
    pos = 0

    granularities = self._hop_granularities()

    wide = self._bypass_covering(writer, src_offset, length)
    if wide:
      # Every access sixteen bytes, so every access bypasses L1.  The whole
      # rounds first, then the one partial round: the lanes that still have a
      # full access, and then the single lane whose access straddles the end,
      # which zero-fills the rest.  Two guards where the stepping has two
      # narrower passes and a scalar tail -- and the point is not that there
      # are fewer of them.
      pos = (length // (self._num_threads * wide)) * wide
      self._write_hop(writer, src_offset, dst_offset, index, 0, pos, wide,
                      nontemporal, linscale)
      rest = length - pos * self._num_threads
      if rest:
        whole, part = divmod(rest, wide)
        if whole:
          with writer.If(f'{self._linear_idx()} < {whole}'):
            self._write_hop(writer, src_offset, dst_offset, index, pos,
                            pos + wide, wide, nontemporal, linscale)
        if part:
          # The one lane whose access straddles the end: `part` elements of
          # source and the rest zeroed, in an access that is still sixteen
          # bytes and still bypasses L1.
          with writer.If(f'{self._linear_idx()} == {whole}'):
            self._write_hop(writer, src_offset, dst_offset, index, pos,
                            pos + wide, wide, nontemporal, linscale,
                            zfill=wide - part)
      return

    cap = self._swizzle_cap(writer)
    for vecsize in granularities:
      if vecsize <= cap and src_offset % vecsize == 0:
        num_hops = ((length - pos * self._num_threads) // (self._num_threads * vecsize)) * vecsize
        self._write_hop(writer, src_offset, dst_offset, index, pos, pos + num_hops, vecsize, nontemporal, linscale)
        pos += num_hops
    rest = length % self._num_threads
    if rest > 0:
      if self._explicit_simd():
        self._write_tail_vector(writer, src_offset, dst_offset, index, pos,
                                rest, nontemporal, linscale)
      else:
        # The tail: `length % num_threads` elements, moved by the lanes below
        # `rest`.  A guard block, not the copy's own predicate, and not for
        # want of support: `copy_async` takes one, but it has to be a Value
        # and `_linear_idx()` is text.  The block costs nothing here -- a
        # token has no C++ representation, so nothing is scoped inside it that
        # the wait needs to name.  With the linear index as a Value, the guard
        # would be the copy's predicate instead.
        with writer.If(f'{self._linear_idx()} < {rest}'):
          self._write_hop(writer, src_offset, dst_offset, index, pos, pos+1, 1, nontemporal, linscale)

  def _write_tail_vector(self, writer, src_offset, dst_offset, index, pos,
                         rest, nontemporal, linscale):
    """The tail, as narrower transfers rather than as a guard on the lane.

    `if (linear_idx < rest)` is a statement about which lanes take part, and
    under an explicit vector there is no lane to make it about: the condition
    is a scalar, so the branch is taken whole or not at all and the transfer
    inside it moves the full register width -- `num_threads` elements where
    `rest` were meant, over the top of whatever follows the tile.

    What the guard says instead is that the transfer is `rest` elements wide,
    and a width is something this lowering can spell.  So the tail becomes one
    transfer per participating work-item, each of a *compile-time* width:
    work-item `j` carries `min(lanes, rest - j * lanes)` granules beginning at
    `j * lanes`.  A runtime-varying width would not be expressible, which is
    why the split is over work-items and not over a bound.

    Per multiplication there is only ever one work-item, so the common case is
    a single transfer of `rest` and no guard at all.  Blockwide -- which is
    `preload_globals`, once per block in the prologue -- is where more than one
    chunk appears, and the guard on `y` there is a scalar branch on a scalar
    value, which is the kind this model does have.
    """
    lexic = self._context.get_vm().get_lexic()
    lanes = self._lanes
    for j in range((rest + lanes - 1) // lanes):
      width = min(lanes, rest - j * lanes)
      # No separate base: under the guard `y == j` the blockwide
      # `_linear_idx()` is `j * lanes` already, and per multiplication it is
      # zero and `j` is only ever zero.  So the address the hop builds is the
      # one this chunk wants, and only its *width* has to be narrowed.
      if self._blockwide:
        with writer.If(f'{lexic.thread_idx_y} == {j}'):
          self._write_hop(writer, src_offset, dst_offset, index, pos, pos + 1,
                          1, nontemporal, linscale, lanes=width)
      else:
        self._write_hop(writer, src_offset, dst_offset, index, pos, pos + 1,
                        1, nontemporal, linscale, lanes=width)

  def _write_hop(self, writer, src_offset, dst_offset, index, start, end,
                 increment, nontemporal, linscale, lanes=None, zfill=0):
    """`lanes` narrows the transfer's own register without touching the claim
    the fill leaves on the image.

    Two different statements, and not one number.  The register is what
    this work-item moves in one go; the claim is how the *image* is spread
    once the fill is done, which every later read of it reports.  A tail chunk
    narrows the first and must not touch the second -- two fills recording
    different claims about one image leave it unknown, and unknown is a
    declaration the explicit-vector lowering cannot write.
    """
    span = self._lane_span() if lanes is None else lanes
    if end > start:
      if increment > 1:
        vectortype = self._vm.get_lexic().get_fptype(self._dest.get_fptype(), increment)
        typeprefix = f'*({vectortype}*)&'
      else:
        typeprefix = ''

      # Not gated on `_use_cuda_memcpy`: gated, every non-NVIDIA transfer
      # would write the window as raw text -- `s0[...] = glb_load(...)` with
      # the address built by `access_address`, never passing through `store`.
      # Two consequences, and the second is the one that matters: nothing
      # could see what the transfer touches, and a permuted window would be
      # read through `load` and written around it -- the reads applying the
      # swizzle and the writes not.
      structured = self._structured_copy(writer)
      if structured and self._use_cuda_memcpy:
        # One `copy.async` per hop, carrying the hop's extent.  The vector
        # width stops being a cast on both sides of an assignment and becomes
        # `elems`, which is what the emitter needs anyway to check the
        # transfer size against `copy_async_sizes()`.
        # The same claim the synchronous branch below records.  A `copy.async`
        # distributes its destination exactly as a load-and-store pair does --
        # the engine moves the bytes, not the mapping -- so leaving it unsaid
        # here would make the answer depend on which transfer the target
        # happens to use.
        self._dest._record_linear_layout(dst_offset, increment,
                                         self._lane_span(), writer)
        dst_buf = self._destination_buffer(writer)
        src_buf = self._src.pir_buffer(writer)
        def write_load(lhs, rhs, _d=dst_buf, _s=src_buf, _n=increment,
                       _z=zfill):
          self._tokens.append(writer.copy_async(
              _d, _s, dst_index=(lhs,), src_index=(rhs,), elems=_n,
              zfill=_z))
      elif structured:
        # No async engine, so the transfer is a load and a store, spelled in
        # a way every pass can read.
        from tensorforge.backend.pir.core import (LaneAxis, RegisterLayout,
                                                   ScalarType)
        fpt = self._dest.get_fptype()
        # A staging transfer is lane-linear: `contiguous_index` is
        # `increment * linear_idx() + ...`, so lane `t` carries granule `t`,
        # and `increment` is a vector width rather than a lane stride.  The
        # ESIMD lowering needs this said -- an SPMD backend can leave the
        # distribution in the index expression, a vector one cannot.
        xfer_layout = RegisterLayout((LaneAxis(span, 1),))
        # Said on the *destination* as well, not only on the value in flight.
        # A later read of this image is `load_linear`, whose address has no
        # lane term at all -- it reports what the fill recorded and can derive
        # nothing.  Stating the claim only on the loaded value would leave the
        # symbol unknown, so every consumer of the staged image would have to
        # fail closed: invisible under SPMD, where unknown costs precision, and
        # fatal under an explicit vector, where a declaration cannot be written
        # without a distribution.
        #
        # Same call `store_linear` makes for the other fill path, so the two
        # cannot record different claims about the same shape.
        self._dest._record_linear_layout(dst_offset, increment,
                                         self._lane_span(), writer)
        dst_buf = self._dest.pir_buffer(writer)
        src_buf = self._src.pir_buffer(writer)
        def write_load(lhs, rhs, _d=dst_buf, _s=src_buf, _n=increment,
                       _t=fpt, _nt=nontemporal, _l=xfer_layout):
          ltype = ScalarType(_t) if _n == 1 else ScalarType(_t, _n)
          value = writer.load(_s, rhs, type_=ltype, hint='ld',
                              nontemporal=_nt, layout=_l)
          writer.store(_d, value, lhs)
      else:
        # `increment`, not 1: above a width of one both sides of this
        # assignment are `*(VectorT<T, N>*)&...`, so the value the hint would
        # attach to is a vector and the lexic has to be told which.
        def write_load(lhs, rhs, _t=self._dest.get_fptype(), _n=increment,
                       _nt=nontemporal):
          writer(f'{lhs} = {self._context.get_vm().get_lexic().glb_load(rhs, datatype=_t, length=_n, nontemporal=_nt)};')

      # The destination's rows are longer than the source's -- padded, so
      # that a row starts where a wide access may start.  A linear transfer
      # counts in *source* elements, so element `i` of the copy belongs at
      # `(i / read) * dst + i % read` in the image: the row number scaled up,
      # the position inside the row kept.
      #
      # Only on the destination.  Applying it to both sides is the same
      # rescaling of an address whose rows are still `read` long, which reads
      # the source at the padded stride and copies the wrong elements -- one
      # `k9` GEMM at `stage_row_bytes` 16 would come out with a checksum off
      # in the fourth digit, and every value in it a real number from the
      # wrong place, which is the failure that does not look like one.
      if linscale is None:
        indexwrapper = lambda x: x
      else:
        indexwrapper = lambda x: f'((({x}) / {linscale[0]}) * {linscale[1]} + (({x}) % {linscale[0]}))'

      if (end - start) / increment > self._manual_unroll_threshold:
        # load using a for-loop
        with writer.For(f'int32_t i = {start}; i < {end}; i += {increment}', True):
          linear = f'{increment} * {self._linear_idx()} + i * {self._num_threads}'
          dst_index, src_index = indexwrapper(linear), linear
          dest_access_index = self._dest.access_address(self._context, index, writer)
          src_access_index = self._src.access_address(self._context, index, writer)
          if structured:
            write_load(f'{dst_offset} + {dest_access_index} + {dst_index}',
                       f'{src_offset} + {src_access_index} + {src_index}')
          else:
            lhs = f'{typeprefix}{self._dest.name}[{dst_offset} + {dest_access_index} + {dst_index}]'
            rhs = f'{typeprefix}{self._src.name}[{src_offset} + {src_access_index} + {src_index}]'
            write_load(lhs, rhs)
      else:
        # load using manual loop unrolling
        for counter in range(start, end, increment):
          linear = f'{increment} * {self._linear_idx()} + {counter * self._num_threads}'
          dst_index, src_index = indexwrapper(linear), linear
          dest_access_index = self._dest.access_address(self._context, index, writer)
          src_access_index = self._src.access_address(self._context, index, writer)
          if structured:
            write_load(f'{dst_offset} + {dest_access_index} + {dst_index}',
                       f'{src_offset} + {src_access_index} + {src_index}')
          else:
            lhs = f'{typeprefix}{self._dest.name}[{dst_offset} + {dest_access_index} + {dst_index}]'
            rhs = f'{typeprefix}{self._src.name}[{src_offset} + {src_access_index} + {src_index}]'
            write_load(lhs, rhs)

  def tokens_for(self, writer):
    """The tokens this transfer issued, if they belong to the body at hand.

    Same guard as `Symbol.pir_buffer`, and for the same reason: a token is a
    value of one body, and a wait emitted into another body than the one its
    transfer was issued in cannot name it.
    """
    owner = getattr(writer, 'uid', None)
    if owner is None or owner != self._token_owner:
      return []
    return list(self._tokens)

  def _structured_copy(self, writer) -> bool:
    """Can this transfer be a `copy.async` rather than a line of text?

    Only where both ends are values in *this* body.
    """
    if not hasattr(writer, 'copy_async'):
      return False
    if self._src.pir_buffer(writer) is None:
      return False
    return self._destination_buffer(writer) is not None

  # `_swizzle` needs `_structured_copy`'s answer before this body has bound
  # anything, and cannot have it: the question is asked from inside
  # `writer.alloc`, the call that binds the destination, so at that moment
  # neither end is bound and the answer is no for every transfer in the corpus
  # -- and the same transfer says yes three times afterwards.  That ordering,
  # not anything about the copies, is why 92 of the 94 unpermuted windows are
  # unpermuted, and 63 of them would take a real swizzle (47 of those `xor32`).
  #
  # Predicting the answer instead does not work: `Addressing.NONE` is not the
  # discriminator, and a rolled or pipelined transfer is emitted in a body its
  # pointer bindings do not reach.  Guessing yes there would be caught rather
  # than shipped -- `_check_swizzles_are_total` refuses such a body -- but
  # caught is not fixed.  The fix is to decide the permutation once the body
  # exists, which means applying it where the address is *emitted* rather than
  # where it is built; `pir.banks` is written against exactly that point.

  def _destination_buffer(self, writer):
    """The value this transfer fills."""
    return self._dest.pir_buffer(writer)

  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  def _check(self) -> None:
    #if self._src.stype != SymbolType.Global:
    #  raise InternalError('shr-load: `src` operand is not in global mem.')

    if not isinstance(self._src.obj, Tensor):
      raise InternalError(f'shr-load: `src` operand is not a tensor, instead: {self._src.obj}')

    if self._dest.stype != SymbolType.SharedMem:
      raise InternalError('shr-load: `dest` operand is not in shr. mem.')

    if not isinstance(self._dest.obj, Tensor):
      raise InternalError(f'shr-load: `dest` operand is not a tensor, instead: {self._dest.obj}')

  def get_headers(self) -> List[str]:
    # The structured route lowers to the `__pipeline_*` primitives, whose
    # header carries no architecture floor.
    return ['cuda_pipeline.h'] if self._use_cuda_memcpy else []

  def __str__(self):
    return f'{self._dest.name} = load{{g>s}}({self._src.name}[{", ".join(str(p) for p in self._permute)}])'


class GlbToRegLoader(MemoryInstruction, LoadInstruction):
  def __init__(self,
               context: Context,
               src: Symbol,
               dest: Symbol,
               num_threads: int,
               linearize: bool,
               lead_width: int = 1,
               src_bbox=None,
               src_offset=None):
    super(GlbToRegLoader, self).__init__(context)

    if dest.stype != SymbolType.Register:
      raise InternalError('store: operand `dest` is not in reg mem')

    if not isinstance(dest.obj, RegMemObject):
      raise InternalError(f'store: operand `dest` is registers, instead: {type(dest.obj)}')

    if src.stype != SymbolType.Global:
      raise InternalError('store: operand `src` is not in global memory.')

    if not isinstance(src.obj, Tensor):
      raise InternalError('store: operand `src` is not a matrix')

    src.add_user(self)
    dest.add_user(self)

    # `src_bbox` is the region to load, in *logical* coordinates; `src_offset`
    # is the logical->storage shift of the operand this load stages.  Registers
    # dispatch on `isinstance(index, LeadIndex)` (Symbol.build_address), so an
    # offset cannot ride along as a VarOffset the way it can for a global or
    # shared-memory operand --- it would land in the non-lead branch and pick up
    # both the wrong divisor and the wrong stride.  It is therefore consumed
    # here: read at `x + offset`, write at `x`, and the register image is in
    # logical coordinates from then on.
    self._bbox = src_bbox if src_bbox is not None else src.obj.get_bbox()
    self._offset = list(src_offset) if src_offset is not None else [0] * self._bbox.rank()

    dest.data_view = DataView(shape=src.obj.shape,
                              permute=None,
                              bbox=self._bbox)

    # if dest.data_view.get_dim_size(0) > src.data_view.get_dim_size(0):
    #   raise InternalError('store: `src` and `dest` do not match in size aling dim `0`')

    self._dest: Symbol = dest
    self._src: Symbol = src#.clone()
    self._num_threads: int = num_threads
    #: The register image this loader fills is blocked by this; see
    #: `gen_code_inner`.
    self._lead_width: int = lead_width
    self._is_ready: bool = True
    self._linearize = linearize

  def gen_code_inner(self, writer: Writer) -> None:
    writer.new_line()

    allow_nontemporal = cache_hint(self._context, _hint_allowed(self))

    src_bbox = self._bbox

    if self._linearize:
      # a flat run over spp.count_nz() cannot express a sub-slice
      assert all(o == 0 for o in self._offset), \
          (f'{self._src.name}: linearized register load cannot apply slicing '
           f'offset {self._offset}')
      # TODO: box better?
      total_size = self._src.obj.spp.count_nz()

      # The width comes from the *source*, and only from the source.  The
      # minimum over both ends would quietly disable the whole path: the
      # destination is a register array in the private address space, where
      # AMDGPU interleaves the lanes at dword granularity, so no alignment of
      # a private address names a contiguous 16 bytes and asking that end to
      # prove one can only ever answer "4".  The register side is spelled with
      # the relaxed vector type instead -- legal at any alignment, split by
      # the compiler if the array survives to be addressed at all, and free
      # when it is promoted, which is the case that matters.
      #
      # So what is being decided here is the width of the *global* read, which
      # is the access with a real hardware alignment requirement.
      elem = self._src.get_fptype().size()
      widths = vectorize.widths_for(elem, self._src.linear_align_bytes())
      hops, tail = vectorize.plan_hops(total_size, self._num_threads, widths)

      for i, g in hops:
        # The staged value is passed on as a value rather than through a C++
        # name.  A named temporary needs a declaration, and `flatten_scopes`
        # keeps any region whose raw text declares a name: an opaque block
        # head, across which the async scheduler drops its state and nothing
        # reorders.
        staged = self._src.load_linear(writer, self._context, None, i, g,
                                       threads=self._num_threads)
        self._dest.store_linear(writer, self._context, staged, i, g,
                                threads=self._num_threads)

      if tail:
        # Fewer than `num_threads` elements, so some lanes have nothing to
        # read.  Emitted unguarded, on purpose: the lanes past the end read
        # into the neighboring matrix of the batch and their registers are
        # never consumed.  It is still a read past the tensor -- 23 of 32
        # lanes for a 9-element operand -- and at the last matrix in the batch
        # there is no neighbor.  Guarding it is a decision about the buffer,
        # not about the width; `_write_datatransfer` already predicates its
        # own tail and is the shape to copy when that decision is made.
        self._dest.store_linear(
            writer, self._context,
            self._src.load_linear(writer, self._context, None,
                                  total_size - tail, 1,
                                  threads=self._num_threads),
            total_size - tail, 1, threads=self._num_threads)

    else:
      # The lane axis is whichever dimension the destination declares, not
      # dimension 0: a transposed operand carries the lead index elsewhere, and
      # writing the image with the lane on dimension 0 while every reader
      # addresses it through `lead_dims` puts the two out of step.
      lead_pos = self._dest.lead_dims[0]
      loops = []
      for i in range(src_bbox.rank()):
        if i == lead_pos:
          # The destination's block, not the multiplication's lane count.
          # They are the same for an image spread over every lane, and differ
          # for a replicated one -- and it is the block `LeadIndex` divides
          # by (`idx = ((tid / stride) % block) + nonlead * block`), so
          # filling with the wider number writes each lane a different element
          # of an image whose readers expect every run of `block` lanes to
          # hold the same ones.
          loops += [LeadLoop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i],
                             self._dest.lead_block(lead_pos), 1,
                             width=self._lead_width)]
        else:
          loops += [Loop(f'i{i}', src_bbox.lower()[i], src_bbox.upper()[i], 1)]

      def inner(indices):
        # logical index in on the register side, storage index out on the
        # global side --- add_offset folds the (usual) zero away
        value = self._src.load(writer, self._context, None,
                       [add_offset(x, self._offset[i])
                        for i, x in enumerate(indices)], allow_nontemporal)
        self._dest.store(writer, self._context, value, indices, False)

      write_loops(self._context, writer, loops, inner)

  def __str__(self) -> str:
    return f'{self._dest.name} = load{{g>r}}({self._src.name});'

class LoadWait(MemoryInstruction, LoadInstruction):
  def __init__(self, instr):
    super(LoadWait, self).__init__(instr._context)
    self._instr = instr
    self._is_ready = True

  # A wait completes the awaited transfer, so from a data-flow point of view
  # it *is* the write.  Consumers must therefore be ordered after the wait,
  # not after the issuing load.  Once async/wait carries a real token this
  # becomes a use of that token instead.
  def awaited(self):
    return self._instr

  def defs(self):
    return self._instr.defs()

  def uses(self):
    # ...but the destination buffer is occupied from the moment the copy is
    # *issued*: the DMA writes into it while it is in flight.  Reported as a
    # def alone, the wait would read as the start of the buffer's value, and
    # whatever follows the issue would see nothing holding the buffer until
    # it.  Naming it here as well keeps the buffer used from the issue on,
    # while `defs` keeps ordering consumers after us.  It is also why the
    # wait defines nothing whole (`Generator._declare_buffers`): the value
    # starts at the issue.
    return self._instr.defs()

  def gen_ir(self, sink) -> None:
    # Nothing in flight, nothing written: neither a wait nor the comment
    # naming one.  A transfer into registers puts nothing in flight, and
    # neither does one that took the reordering path and moved its data with
    # plain loads and stores.  `_issued_async` records what the transfer did;
    # `_use_cuda_memcpy` is only what it was allowed to do.
    if isinstance(self._instr, GlbToShrLoader) and self._instr._issued_async:
      super().gen_ir(sink)

  def gen_code_inner(self, writer: Writer) -> None:
    tokens = self._instr.tokens_for(writer)
    if not tokens and self._instr._issued_structured:
      # The transfer issued structurally but into a different body, so its
      # tokens are not nameable here.  Draining is correct and merely waits
      # longer; returning without a wait would read the copy in flight.
      writer.wait()
      return
    if tokens:
      # One wait for the whole transfer.  `schedule_async` derives the count
      # from the last of them and retires the rest, so the hops need no wait
      # of their own.
      writer.wait(tokens[-1], *tokens[:-1])

  def __str__(self) -> str:
    return f'wait({self._instr});'
