# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from ..abstract_instruction import AbstractInstruction, _explicit_simd
from abc import abstractmethod
from tensorforge.backend.writer import Writer
from typing import Union
from tensorforge.common.context import Context
from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.backend.pir.core import Qual, MemSpace

class MemoryInstruction(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._declare = True

  @abstractmethod
  def gen_code_inner(self, writer: Writer):
    pass

  def gen_code_declare(self, writer: Writer):
    pass

  def gen_ir(self, sink):
    # The declaration belongs outside the scope: the symbol it declares is
    # consumed by later instructions.
    if self._declare:
      self.gen_code_declare(sink)

    # No braces around the transfer: it names nothing that could clash -- its
    # temporaries are values the shared allocator numbers -- and an opaque
    # block head is a wall the async scheduler gives up its state at and
    # nothing reorders across.
    sink.Comment(self.__str__())
    self.gen_code_inner(sink)

class AbstractShrMemWrite(MemoryInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._shm_volume: int = 0
    self._declare = False
    #: Whether the buffer is the block's rather than the multiplication's --
    #: an operator preloaded once per block, a staged member of a merged run.
    self._global_offset = False
    #: What the buffer's start has to be a multiple of, in elements, beyond
    #: what every buffer of its arena starts on.
    self._place_align: Union[int, None] = None

  def stage_size(self) -> int:
    """Size of the buffer, aligned where shared allocations are."""
    user_options = self._context.get_user_options()
    if user_options.align_shr_mem:
      return self._context.align(self._shm_volume)
    return self._shm_volume

  def _window(self, writer, name: str):
    """A window into this transfer's buffer, as a value.  Placed by the
    allocator (`pir.allocate`), which is where the buffer's offset is
    decided."""
    return writer.alloc(self._dest.get_fptype(), (self.stage_size(),),
                        MemSpace.SHARED, hint=name, extern=name,
                        arena=self._arena(),
                        quals=(Qual.RESTRICT,),
                        swizzle=self._swizzle(writer), identity=self._dest,
                        place_align=self._place_align)

  def _arena(self) -> str:
    return (GeneralLexicon.TOTAL_SHR_MEM if self._global_offset
            else self._shr_mem.name)

  def gen_code_declare(self, writer: Writer) -> None:
    if self._declare:
      # The window is a value, so a read through it declares what it touches
      # instead of naming it.  `extern` because the consumers still spell
      # `s0` out, same as the register tiles.
      self._dest.set_pir_buffer(writer, self._window(writer, self._dest.name))

  #: Shared memory is 32 banks wide on both vendors, so there is nothing to
  #: gain from permuting over a longer period than that.
  _BANKS = 32

  def transfer_granule(self) -> int:
    """Elements this transfer writes in one access.

    One, unless a subclass moves more: `StoreRegToShr` walks its register
    image element by element, and every element is its own store.  What a
    transfer moves at once is the unit a permutation of this window has to
    keep whole, exactly as a wide read is (`XorSwizzle.granule`), so the
    writer states it and `_swizzle` grants at least that much.

    Stated by the writer rather than asked of the permutation, because the
    permutation is decided when the buffer is allocated and the accesses come
    later.  A blanket decline -- no swizzle at all wherever a bulk copy could
    reach the window -- would be the same fact with no number attached.
    """
    return 1

  def _swizzle(self, writer=None):
    """A row permutation for this window, when one is both legal and useful.

    A tile written a row at a time and read a column at a time costs a bank
    cycle per lane: `s0[(threadIdx.x % 32) * 32]` puts every lane in bank 0
    with a different address, which `tools/bank_conflicts.py` reports as
    32-way.  Permuting the columns per row costs nothing and clears it, and
    leaves a row-wise access exactly as good -- it reaches the same banks in a
    different order.

    Only for a power-of-two row width, which is what makes `k ^ (n % width)`
    stay inside the row.  Everything else keeps the plain layout: a tile 13
    wide has no conflict worth this, and a permutation that carried across
    rows would be a different element, not a slower one.
    """
    from tensorforge.backend.pir.core import XorSwizzle
    view = self._dest.data_view
    if view is None or len(view.shape) < 2:
      return None

    # Not under an explicit-SIMD lowering.  There a transfer is a `copy_from`
    # over a whole `simd<N>`, which reads N *contiguous* positions -- and the
    # permutation acts per element, so the vector's components arrive
    # transposed, or from outside the run once the block key exceeds N.
    #
    # A permutation over *granules* of N would keep those runs intact; the
    # elements of one vector move together and stay in order.  It costs
    # exactly the spreading it preserves, though -- granule 1 takes a
    # stride-32 column read to 1-way, granule 2 to 2-way, granule 4 to 4-way,
    # because a coarser unit has proportionally fewer distinct keys.  That is
    # a real option and a real trade, and not one taken here.
    if _explicit_simd(self._context):
      return None

    # Only when every write to this window goes through the IR.  A *raw* write
    # does not pass `store` and so would not permute while every read did:
    # `GlbToShrLoader` falls back to text when the *source* has no buffer in
    # this body -- `addressing_none` reads through `ptr_glb_m1`, which
    # `ptr_manip` binds only on the structured path.
    #
    # It is the text that is the problem, not the bulk.  A structured
    # `copy_async` permutes its destination like any other access
    # (`PirBuilder.copy_async`), and the width it moves is granted above, so a
    # window filled by one is as permutable as a window filled element by
    # element.
    #
    # Asked here rather than caught later on purpose: the guard at `finish`
    # can only raise by then, because the permutation is already baked into
    # every index it emitted.  Declining is the only response available before
    # that, and it needs the same question asked earlier.
    # Asked of the loader, not of every writer: it is the loader that decides
    # whether the transfer goes through `store`, and it is its question.
    # Asking `self._src.pir_buffer(...)` instead would be a proxy that happens
    # to agree for `GlbToShrLoader` and never does for `StoreRegToShr`, whose
    # source is a register and has no buffer by construction -- and a proxy
    # that agrees today is a proxy that breaks quietly.
    if writer is not None and hasattr(self, '_structured_copy'):
      if not self._structured_copy(writer):
        return None

    # The width has to divide the *volume*, not merely be the row width.  The
    # permutation maps each block of `width` elements onto itself, so a buffer
    # whose last block is partial has indices that permute past its end -- 728
    # elements swizzled at 32 puts eight of them into the next window.  Shared
    # memory, silently, which is the worst failure available here.
    #
    # The largest power of two dividing the volume is safe by construction and
    # is also the better choice: a 16x16 tile takes 32 rather than its row
    # width 16, and its column read is 1-way rather than 2-way; a 56x13 window
    # takes 8 where a row-width rule would decline outright, 4-way rather than
    # 8-way.
    #
    # An odd volume yields 1, which is no swizzle -- and that is the right
    # answer rather than a fallback.  A row width coprime with 32 already
    # spreads a column read over every bank, so permuting it would move
    # elements the plain layout had placed well: strides 9 and 13 are 1-way
    # untouched and 2- or 3-way under any width.
    volume = 1
    for n in view.shape:
      volume *= n

    # The granule is what an access may take in one go, from either side: a
    # reader's `k_width`, and this transfer's own hop (`transfer_granule`).
    # It has to be decided here because the permutation is decided here, and
    # the accesses come later and cannot change it -- so the window grants the
    # unit rather than an access claiming it.  At `k_width` 1 and a transfer
    # that writes one element at a time, which is the default and every
    # recorded kernel, the granule is 1.
    #
    # The transfer's own hop is a *requirement*, not a wish: a copy that moves
    # four elements into a window permuted per element writes them where the
    # readers will not look.  Where the volume cannot carry a granule that
    # wide, the answer is no permutation at all -- for the buffers that
    # actually need the granule, not for every buffer a bulk copy could reach.
    #
    # Granted as wide as asked, even where that leaves no width to permute --
    # a 180-element window takes granule 4 and then width 1, which is no
    # permutation at all.  That is the right end of the trade rather than a
    # failure of it: the alternative is a narrower granule, which does not make
    # the wide read slower, it makes it a scalar read again.  The permutation
    # is worth a few bank cycles; the access it would forbid is worth a load.
    need = max(1, self.transfer_granule())
    granule = 1
    want = max(need, getattr(self._context.get_user_options(), 'k_width', 1) or 1)
    while granule * 2 <= want and volume % (granule * 2) == 0:
      granule *= 2
    if granule < need:
      return None

    width = 1
    while width * 2 <= self._BANKS and volume % (width * granule * 2) == 0:
      width *= 2
    if width < 2:
      return None
    return XorSwizzle(width, granule)

  def compute_shared_mem_size(self) -> int:
    # What a block-wide buffer takes of the block's arena
    # (`Generator._settle_storage`).
    # `int`, because a sparse operand's stage size is a numpy count, and it
    # rides through every offset into `LaunchConfig.shared_elements`, where
    # `json.dumps` in the kernel metadata refuses it -- unless
    # `align_shr_mem` happens to round it back into an `int` on the way.
    return int(self.stage_size())

  def set_window(self, first: bool, block: bool,
                 align: Union[int, None] = None) -> None:
    """Whether this transfer declares its buffer's window -- the first of
    its users does -- and whether the buffer is the block's.  Where it sits
    is the allocator's to decide."""
    self._is_ready = True
    self._declare = first
    self._global_offset = block
    if align:
      self._place_align = align

  @abstractmethod
  def get_dest(self):
    pass
