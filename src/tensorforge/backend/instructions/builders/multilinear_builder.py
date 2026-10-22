# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Tuple
from tensorforge.common.basic_types import Addressing
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.memory.load import GlbToShrLoader, GlbToRegLoader
from tensorforge.backend.instructions.memory.store import StoreRegToGlb, StoreRegToShr
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.backend.instructions.compute.promise import PromisedZeros
from tensorforge.backend.instructions.compute.multilinear import MultilinearInstruction
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.exceptions import InternalError, GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.backend.instructions.builders.operation_builder import OperationBuilder
from tensorforge.backend.instructions.abstract_instruction import _explicit_simd
from tensorforge.backend.placement import (Placement, ResultPlacement,
                                           choose_operand_placement,
                                           choose_result_placement,
                                           legal_operand_placements,
                                           legal_result_placements)
from tensorforge.common.operation import AddOperator, MulOperator


class MultilinearBuilder(OperationBuilder):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #: What this hardware prefers among the placements that are legal.  Only
    #: preferences: nothing here can make a correct kernel incorrect, and the
    #: legality half is asked separately at each decision.  An accumulation
    #: is built onto what its destination holds, whatever the hardware
    #: prefers: whether it reaches memory as an atomic add is the body's to
    #: decide once it knows who reads the sum (`pir.atomics`).
    self._policy = self._context.target.prefs.placement

  def _k_width(self, descr) -> int:
    """How many reduction steps one body should cover for this operator.

    Decided per operator rather than per section, unlike the lead width: the
    reduction does not touch the register image that the descriptors of a
    section share, so one operator taking a wider group does not constrain
    its neighbors.
    """
    width = self._context.get_user_options().k_width
    return max(1, width)

  def resolve_operands(self, descr):
    """Stage every operand where this operation wants to read it.

    Not the base's answer: a contraction does not have to settle a value back
    into memory to use it.  It consults the residency, takes a register image
    where the lane axis agrees, and stages afresh where it does not.
    """
    self._descr = descr
    self._dest_obj = descr.dest
    self._ops = [self.view_of(op) for op in descr.ops]

    self._add = descr.add
    # yateto states `add` as a bool today; the array form is meant to say which
    # of the destination's indices the tensor being added carries.  Accumulating
    # onto fewer indices than the destination has is a broadcast accumulation,
    # and `prev` here is the destination read back --- at the destination's full
    # rank --- so there is nothing to index it with along a missing one.  Take
    # the array when it agrees with the destination and refuse when it does not,
    # rather than reading somewhere else and calling it an answer.
    add_dims = getattr(descr, 'add_dims', None)
    if add_dims is not None and set(add_dims) != set(range(self._dest_obj.bbox.rank())):
      raise GenerationError(
          f'accumulating onto indices {sorted(add_dims)} of a rank-'
          f'{self._dest_obj.bbox.rank()} destination is a broadcast '
          f'accumulation, which is not implemented')

    self._mem_regions = [None] * len(self._ops)
    # per operand: which of its dimensions is spread across lanes
    self._lead_pos = [0] * len(self._ops)
    # where the lead loop runs, decided ahead of the operands, which are
    # staged in the lanes it reads them in
    self._theta = None
    self._theta = self._lead_origin_shift()

    for i in range(len(self._ops)):
        self._make_load_op(i)
    self._insert_sync_block()
    return self._mem_regions

  def alloc_destination(self, descr, operands):
    """The accumulator, in the shifted origin the lead loop runs in.

    An accumulation stages what it adds onto here (`_get_target_symbol`).
    A load of it into registers goes out behind the product, right ahead
    of the epilogue that adds it (`_make_compute`), with the declaration of
    the registers it fills: the product reads it nowhere, and its registers
    are free while the product runs.  Anything
    else stays where it is -- a load into shared memory, which the barrier
    behind the operands orders ahead of the epilogue's reads, and the store
    that empties a register image the destination may not be read from.
    """
    self._bias_loads = []
    if self._add:
      mark = len(self._instructions)
      self._get_target_symbol()
      staged = self._instructions[mark:]
      if (any(isinstance(x, GlbToRegLoader) for x in staged)
          and all(isinstance(x, (GlbToRegLoader, RegisterAlloc))
                  for x in staged)):
        self._bias_loads = staged
        del self._instructions[mark:]
    self._temp_regs = self._alloc_register_array()
    return self._temp_regs

  def emit_compute(self, descr, operands, dest) -> None:
    self._make_compute()
    self._insert_sync_block()

  def record_result(self, descr, dest) -> None:
    self._make_store()
    self._insert_sync_block()

  # TODO: check if we always can allow a direct global memory load
  def _make_load_op(self, i):
    has_lead_dim = 0 in self._descr.target[i]
    transpose = self._descr.permute[i] != [j for j in range(len(self._descr.target[i]))]

    # Which operand dimension a register image spreads across lanes: the one
    # its readers ask for lane by lane.  With another dimension on the lanes,
    # such a read finds every element it wants in the registers of one lane,
    # and no broadcast can hand each lane a different one of them.
    #
    # Where the operand carries the destination's lead index, that is the
    # dimension, and everything downstream of the staging assumes it:
    # `_stage_shift` puts the operand's rows in the lanes of the destination's
    # rows along `target[i].index(0)`, and
    # `MultilinearInstruction._check_offsets` checks the slicing remainder
    # there.  Where it does not, the readers differ.  The loop nest reads
    # fixed elements and broadcasts each from the lane that holds it, which
    # works whichever dimension is on the lanes.  The matrix paths read
    # `B(k, j)` from lane `k`, along the first contraction index (`k0`, target
    # `-1`), so that one goes on the lanes: an operand stored with it on
    # another dimension -- `Tinv(j, q)` contracted over `q`, a `B^T` -- is
    # then read in place by both.  A compressed operand keeps dimension 0,
    # since its flat fill assigns lanes by storage element and the sparse
    # accessor reads it that way (`linearize` below).
    symbol = self._ops[i].symbol
    dense = symbol.obj is None or symbol.obj.is_dense()
    target = self._descr.target[i]
    if has_lead_dim:
      lead_pos = target.index(0)
    elif -1 in target and dense:
      lead_pos = target.index(-1)
    else:
      lead_pos = 0
    self._lead_pos[i] = lead_pos

    # Whether this operand's lane axis is where a reader expects it.  A
    # legality fact and nothing else: it does not change with the vendor, and
    # the staging shift and the choice of loader both key on it even where the
    # hardware can read the operand in place anyway.
    lane_axis_needs_moving = transpose or not has_lead_dim

    addressable = not (
        symbol.stype in (SymbolType.Scalar, SymbolType.Data)
        or (isinstance(symbol.obj, Tensor) and len(symbol.obj.shape) == 0)
        or getattr(symbol.obj, 'addressing', None) == Addressing.NONE)
    image_bytes, budget = (self._register_image_cost(i, lead_pos)
                           if addressable else (None, None))
    window = 1
    for size in self._ops[i].bbox.sizes():
      window *= size
    placement = choose_operand_placement(
        legal_operand_placements(addressable=addressable,
                                 transposed=transpose,
                                 carries_lead_dim=has_lead_dim,
                                 policy=self._policy,
                                 single_element=window == 1),
        self._policy, image_bytes=image_bytes, register_budget=budget)

    name = self._ops[i].symbol.name

    # An image already staged for an earlier operation spreads whichever
    # dimension *that* operation needed across lanes, and holds its rows in
    # the lanes *that* operation read them in.  Reusing it for an operand
    # with a different lane axis -- the same tensor used once plainly and once
    # transposed -- is a cross-lane transpose, and for one whose rows this
    # operation computes in other lanes -- rows `8..16` of an image read for
    # rows `0..8` of the destination -- a cross-lane move; an address
    # expresses neither (`_image_serves`).  Both cases have a fallback
    # already:
    #   * a preload of a global input (`src is dest`) can be dropped, since
    #     global memory still holds the value and the ordinary path re-stages
    #     it in the orientation and the lanes this operand wants;
    #   * a pending store of a temporary has to go out to shared memory first,
    #     which is exactly what a shared placement already does (and what the CUDA
    #     path does for every transposed operand anyway).
    staged = self._residency.get(name)
    scalar = staged is not None and self._scalar_image(i)
    if (staged is not None and not scalar
            and not self._image_serves(i, staged, lead_pos)):
      if staged.is_preload:
        self._residency.drop(name)
      else:
        # legality, not preference: the image cannot serve, and only a copy
        # through shared memory can move a value between lanes
        placement = Placement.SHARED

    # A register image staged in the tensor's own coordinates -- a transposed
    # operand, which `_stage_shift` leaves unshifted -- holds the operand's
    # row `n + offset` in that row's lane, and the lead loop computes the
    # destination's row `n` in the lane of `n + theta`.  Where the two differ
    # by part of a round of the lanes, the image would hold every row one
    # lane group off; the copy through shared memory takes any offset.
    if (placement is Placement.REGISTER and has_lead_dim and transpose
            and not self._in_loop_lanes(i, 0)):
      placement = Placement.SHARED

    # The linearized load packs the operand flat --- storage element `f` goes
    # to lane `f % T`, slot `f // T`.  With the lane axis elsewhere the flat
    # packing and the per-dimension addressing describe different images, so
    # take the dimension-wise loader instead; it is correct for any lane axis.
    #
    # `is_dense` is the other half.  Flat packing makes the first dimension
    # the lane axis only when the first extent is a multiple of the lane count
    # -- at 16 elements over 32 lanes it does not -- and the reader,
    # `MultilinearInstruction._second_operand_is_sparse`, which picks the
    # accessor, decides the same question.  Two answers to one question, on
    # different grounds, would leave register slots nobody writes.
    #
    # So the flat fill is for compressed operands and no others: their cells
    # have no dimension-wise address in the first place, which is why the flat
    # map is the right one there and the only one there.  Both sides read the
    # same predicate.  Note this says *which map*, not how wide it is walked
    # -- a vectorized fill stays available under either.
    linearize = lane_axis_needs_moving and lead_pos == 0 and not dense

    if name in self._residency and self._resolve_reuse(i, name):
      entry = self._residency.get(name)
      if scalar:
        # One value, and every lane holds it: read where it is, whatever the
        # placement said -- a shared copy adds a store, a barrier and a load.
        self._ops[i].symbol = entry.image
      elif placement is Placement.SHARED and entry.home.stype == SymbolType.SharedMem:
        box, offset = entry.in_buffer() or (None, None)
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=entry.image,
                                                dest=entry.home,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads,
                                                dest_bbox=box,
                                                dest_offset=offset))
        self._residency.drop(name)
        self._ops[i].symbol = entry.home
      elif placement is Placement.SHARED:
        # The image is the only copy of a pending write, so it cannot be
        # dropped the way a preload can.  Its home is where its symbol says,
        # and for anything but a temporary that is global memory, which a
        # store into shared memory has no way to name.  So the write goes to
        # its home and the operand keeps its global symbol; the staging below
        # then reads it back in the orientation this operand wants.
        #
        # A round trip, and the price of moving the lane axis of a value the
        # kernel has to publish anyway.  Only a shared staging buffer of its
        # own would avoid it, and that is a placement decision rather than
        # something to settle here.
        self._instructions.extend(self._residency.flush(name))
      else:
        self._ops[i].symbol = entry.image

    if self._ops[i].symbol.stype == SymbolType.Scalar or self._ops[i].symbol.stype == SymbolType.Data \
      or (isinstance(self._ops[i].symbol.obj, Tensor) and len(self._ops[i].symbol.obj.shape) == 0): # <-- quasi-scalar
      self._mem_regions[i] = self._ops[i]
    else:

      if self._ops[i].symbol.stype == SymbolType.Global:
        if placement is Placement.SHARED and self._ops[i].symbol.obj.addressing != Addressing.NONE:
          shift = self._stage_shift(i, absorb_lead=False)
          staged, load_op = self._make_loader_and_symbol(self._stage_view(i, shift), is_transpose=self._descr.permute[i])
          self._mem_regions[i] = self._staged_region(i, staged, shift)
          self._instructions.append(load_op)
        else:
          if placement is Placement.REGISTER and self._ops[i].symbol.obj.addressing != Addressing.NONE:
            # only register-preload dense matrices for now
            shift = self._stage_shift(i, absorb_lead=not lane_axis_needs_moving)
            staged, load_op = self._make_loader_and_symbol_reg(
                self._stage_view(i, shift), linearize=linearize,
                lead_pos=lead_pos,
                carries_lead=0 in self._descr.target[i])
            self._mem_regions[i] = self._staged_region(i, staged, shift)
            self._residency.record_preload(
                self._ops[i].symbol.name, self._mem_regions[i].symbol,
                self._mem_regions[i].symbol.data_view.get_bbox(), shift)
            self._instructions.append(load_op)
          elif placement is Placement.SHARED and self._ops[i].symbol.obj.addressing != Addressing.NONE:
            # only register-preload dense matrices for now
            shift = self._stage_shift(i, absorb_lead=False)
            staged, load_op = self._make_loader_and_symbol(self._stage_view(i, shift), None)
            self._mem_regions[i] = self._staged_region(i, staged, shift)
            self._residency.record_preload(
                self._ops[i].symbol.name, self._mem_regions[i].symbol,
                self._mem_regions[i].symbol.data_view.get_bbox(), shift)
            self._instructions.append(load_op)
          else:
            # Note: operand will reside in glb. mem for gemm operation
            self._mem_regions[i] = self._ops[i]

      elif (self._ops[i].symbol.stype == SymbolType.SharedMem
            and placement is Placement.REGISTER
            and self._context.get_user_options().stage_shared):
        # Staged into this operation's registers, as a global operand is
        # (`stage_shared`).
        shift = self._stage_shift(i, absorb_lead=not lane_axis_needs_moving)
        staged, load_op = self._make_loader_and_symbol_reg(
            self._stage_view(i, shift), linearize=linearize,
            lead_pos=lead_pos,
            carries_lead=0 in self._descr.target[i])
        self._mem_regions[i] = self._staged_region(i, staged, shift)
        self._instructions.append(load_op)
      elif self._ops[i].symbol.stype == SymbolType.SharedMem or self._ops[i].symbol.stype == SymbolType.Register:
        # An operand that already sits in shared memory or in registers is
        # read where it is.  Deciding to re-stage it in a different
        # orientation would need a record of which orientation the existing
        # image has; the residency entry carries that, and `_image_serves`
        # above is what consults it.
        self._mem_regions[i] = self._ops[i]
      else:
        raise InternalError(f'gemm-builder: op{i} ({self._ops[i].symbol.name}) must be either in shr or glb mem, given: {self._ops[i].symbol.stype}')

  def _union_of(self, i):
    view = self._ops[i]
    union = self._plan.operand_union(view.symbol.name,
                                     getattr(view.symbol, 'obj', None))
    if union is None:
      union = BoundingBox([l + o for l, o in zip(view.bbox.lower(), view.offset)],
                          [u + o for u, o in zip(view.bbox.upper(), view.offset)])
    return union

  def _register_image_cost(self, i, lead_pos):
    """`(bytes, budget)` of staging operand `i` in registers, or `(None,
    None)` where there is nothing to check.

    The image as the loader allocates it: the operand's box, the lead
    dimension rounded up to whole slots of `num_threads` lanes.

    Under the explicit-SIMD lowering one work-item holds the whole image, and
    the image alone is held against the thread's register file.  Under SPMD a
    lane holds `1 / num_threads` of it, which alone never comes near the file;
    what does is several of them at once.  So there the figure is this
    image's share per lane plus every register image already resident, held
    against `Options.preload_register_share` of the lane's file -- and nothing
    is checked where that share is 0.
    """
    budget = getattr(self._context.target.hw,
                     'max_reg_per_thread', None)
    if budget is None:
      return None, None
    view = self._ops[i]
    sizes = [u - l for l, u in zip(view.bbox.lower(), view.bbox.upper())]
    threads = self._num_threads or 1
    if sizes:
      sizes[lead_pos] = -(-sizes[lead_pos] // threads) * threads
    volume = 1
    for n in sizes:
      volume *= n
    dtype = getattr(view.symbol.obj, 'datatype', None) or self._context.fp_type
    if _explicit_simd(self._context):
      return volume * dtype.size(), budget
    share = getattr(self._context.get_user_options(),
                    'preload_register_share', 0.0) or 0.0
    if share <= 0:
      return None, None
    resident = 0
    for _, entry in self._residency.items():
      image = entry.image
      if image.stype == SymbolType.Register and image.obj.size:
        elem = getattr(image, 'datatype', None) or self._context.fp_type
        resident += image.obj.size * elem.size()
    return (volume // threads * dtype.size() + resident,
            int(share * budget))

  def _stage_shift(self, i, absorb_lead):
    """How the staged image is indexed relative to the tensor.

    Position `r` of the image holds tensor element `r + shift`.

    A register image spreads its lead dimension over the lanes: position `r`
    sits in lane `(r div w) mod T`.  The lead loop computes the destination's
    row `n` in the lane of position `n + theta` (`_lead_origin_shift`), and
    reads the operand's row `n` at position `n + offset - shift`; so the shift
    is congruent to `offset - theta` modulo a round of the lanes, `T * w`,
    and every row is read by the lane that holds it.  Where the operand's
    rows are the destination's and the origin is the destination's own --
    nothing pending decides it otherwise -- that is a shift by whole rounds,
    and the image holds row `r` of its tensor in lane `(r div w) mod T`, as
    the operation's result does.

    Within that, the shift absorbs as much of the union's lead lower bound as
    it can: free, since the loader forms an address per element anyway, and
    the slot count is `ceil(u/S) - floor(l/S)` at `S = T * w`, so every round
    below the union left in the image is a slot nobody reads.  What remains
    below a round can still cost one: `[8,24)` at T=16 straddles two rounds
    where `[0,16)` fills one.  The remaining dimensions stay in storage
    coordinates, where the offset is a plain constant in the address and
    absorbing it buys nothing.

    A shared-memory image is a verbatim copy of the whole storage box, so its
    coordinates already are the tensor's and the shift is zero throughout.  So
    is the linearized register path, which copies flat and cannot express a
    shift at all, and the transposed one, whose lane axis the loader does not
    shift (`_make_load_op` sends it through shared memory where it would need
    one).
    """
    rank = self._ops[i].bbox.rank()
    if not absorb_lead or rank == 0:
      return [0] * rank
    lead_pos = self._lead_pos[i]
    shift = [0] * rank
    lower = self._union_of(i).lower()[lead_pos]
    span = self._lane_span()
    if not span:
      shift[lead_pos] = lower
      return shift
    # the largest shift with the residue the rows need that absorbs no row of
    # the union
    residue = (self._ops[i].offset[lead_pos] - self._theta) % span
    shift[lead_pos] = lower - (lower - residue) % span
    return shift

  def _stage_view(self, i, shift):
    """The operand as the staging load should see it: the whole union, in the
    image's own coordinates."""
    union = self._union_of(i)
    return SymbolView(self._ops[i].symbol,
                      BoundingBox([l - s for l, s in zip(union.lower(), shift)],
                                  [u - s for u, s in zip(union.upper(), shift)]),
                      list(shift))

  def _staged_region(self, i, staged, shift):
    """The staged symbol as the *compute* site should see it: this operand's
    own logical box, with its offset rebased onto the image."""
    return SymbolView(staged.symbol, self._ops[i].bbox,
                      [o - s for o, s in zip(self._ops[i].offset, shift)])

  def _scalar_image(self, i) -> bool:
    """Is operand `i` a temporary without axes, still in its register image?

    Such an image has no lane axis to agree with -- it is one value, the same
    on every lane -- so `_image_serves`, which asks for one, is not the
    question.  Whether it may be read at all is `image_in_place`'s, and the
    option's (`register_temporaries`).
    """
    return (self._ops[i].bbox.rank() == 0
            and self.image_in_place(self._descr.ops[i], arrays=False)
            is not None)

  def _lane_span(self) -> int:
    """Rows of the lead dimension one round of the lanes holds, `T * w`; 0
    where nothing is spread over lanes."""
    if not self._num_threads:
      return 0
    return self._num_threads * max(1, getattr(self, '_lead_width', 1))

  def _in_loop_lanes(self, i, shift) -> bool:
    """Would a register image of operand `i` at lead shift `shift` hold each
    of its rows in the lane the lead loop computes that row in?

    Only a question for an operand whose lane axis carries the destination's
    lead index; the reduction's origin is the instruction's to pick
    (`MultilinearInstruction._reduction_origin_shift`).
    """
    target = self._descr.target[i]
    lead_pos = self._lead_pos[i]
    span = self._lane_span()
    if not span or 0 not in target or target.index(0) != lead_pos:
      return True
    return (self._ops[i].offset[lead_pos] - shift - self._theta) % span == 0

  def _image_serves(self, i, entry, lead_pos) -> bool:
    """Does the staged image `entry` hold operand `i` where this operation
    reads it?

    Only register images have lanes at all; a shared-memory staging is a
    verbatim copy and serves any orientation.  A register image has to spread
    the dimension this operand needs, and hold each of its rows in the lane
    that reads it (`_in_loop_lanes`).
    """
    image = entry.image
    if image.stype not in (SymbolType.Register, SymbolType.Scratch):
      return True
    if image.lead_dims != [lead_pos]:
      return False
    shift = entry.shift[lead_pos] if entry.shift else 0
    return self._in_loop_lanes(i, shift)

  def _resolve_reuse(self, i, name):
    """Decide whether the already-staged image can serve this operand.

    The residency is keyed by the global symbol's name and lives for the
    whole kernel, so any later operation on the same tensor inherits the
    earlier staging regardless of which slice it asks for.  Two things follow.

    The image is indexed in *its own* coordinates --- position `r` holds tensor
    element `r + shift` --- while the operand states its offset against the
    tensor, so it has to be rebased before the symbol is swapped underneath it.

    And the image only covers what its producer happened to stage.  When a
    consumer wants a different part, what can be done depends on where the
    authoritative copy lives:

      * a *preload*: global memory still holds the value, so the entry is
        simply dropped and the ordinary path stages the range that is actually
        wanted;
      * a *writeback*: the value exists only in registers, so it is written
        home first -- zero-filled over what its producer promised, as its
        store at the end would have -- and then read back like a preload.

    The second is not refused on the grounds that the missing part may be
    produced by some other instruction.  It need not be: an assignment covers
    rows 0..10 and a copy reads rows 0..20, the rest zero by the assignment's
    promise.  The straight-line build never meets it, because its stores go
    out as they are made; a merged run keeps its accumulations in registers,
    and SeisSol's time derivative (`dQ(k+1) = dQext(k+1)`) meets it on every
    iteration.  A part written by another instruction is still safe: a
    destination written in slices is never kept in registers.
    """
    entry = self._residency.get(name)
    view = self._ops[i]
    if entry.covered is None:
      if any(o != 0 for o in view.offset):
        raise GenerationError(
            f'{name}: reusing a staged image whose covered range was not '
            f'recorded, with a non-zero slicing offset {view.offset}')
      return True

    if not entry.holds(view.bbox, view.offset):
      if entry.is_preload:
        self._residency.drop(name)
        return False
      if entry.home.stype != SymbolType.Global:
        raise GenerationError(
            f'{name}: operand wants {view.bbox} at offset {view.offset} but '
            f'the staged image only covers {entry.covered} at shift '
            f'{entry.shift}, and the value exists only in registers.')
      self._instructions.extend(self._residency.flush(name))
      return False

    view.offset = [o - s for o, s in zip(view.offset, entry.shift)]
    return True

  def _lead_origin_shift(self):
    """The origin of the lead loop: where the destination's rows compute.

    `n0` is an internal loop variable of this instruction, and its origin
    decides only which lane computes which row: the destination's row `n`
    sits at position `n + theta` of the accumulator, in lane
    `((n + theta) div w) mod T`.  An operand staged for this operation is
    staged in the lanes the origin says (`_stage_shift`), and so is a bias
    loaded from memory (`_dest_preload_view`): what decides it is what
    cannot move without a trip through memory.

    That is a pending result, held in registers alone, which this operation
    reads where it is: moving it to other lanes means storing it, a barrier,
    and loading it back (`_image_serves`).  The destination's own, where the
    operation adds onto one, comes first -- it is the bias where the sum is
    computed, and an accumulation chain over it stays in registers -- and
    then the operands', where they agree.  A copy of what memory holds does
    not decide anything: dropped, it is staged again where it is wanted.

    Where nothing decides, the origin is the destination's offset modulo a
    round of the lanes, so that the accumulator holds row `r` of its tensor
    in lane `(r div w) mod T` -- the lanes an operand staged for the same
    rows takes, so that the next operation reading the result finds each row
    in the lane that reads it.
    """
    span = self._lane_span()
    if not span:
      return 0
    target = self._target_residue()
    if target is not None:
      return target
    pinned = {r for r in map(self._image_residue, range(len(self._ops)))
              if r is not None}
    if len(pinned) == 1:
      return pinned.pop()
    return (self._dest_obj.offset[0] if self._dest_obj.offset else 0) % span

  def _pending(self, symbol):
    """The register image holding `symbol`'s pending result, as its residency
    entry, or None."""
    entry = self._residency.get(symbol.name) if symbol is not None else None
    if (entry is None or entry.is_preload
        or entry.image.stype not in (SymbolType.Register, SymbolType.Scratch)
        or not entry.image.lead_dims):
      return None
    return entry

  def _target_residue(self):
    """The lane residue of the pending result this operation adds onto, or
    None: where the destination's row `n` sits in it, modulo a round of the
    lanes."""
    if not self._add:
      return None
    symbol = self._scopes.get_symbol(self._dest_obj.tensor)
    entry = self._pending(symbol)
    if entry is None or not self._target_image_fits(symbol.name):
      return None
    offset = self._dest_obj.offset[0] if self._dest_obj.offset else 0
    return (offset - (entry.shift[0] if entry.shift else 0)) % self._lane_span()

  def _image_residue(self, i):
    """The lane residue of the pending result operand `i` reads where it is,
    or None: where its row `n` sits in the image, modulo a round of the
    lanes."""
    target = self._descr.target[i]
    view = self._ops[i]
    if 0 not in target or view.bbox.rank() == 0:
      return None
    entry = self._pending(view.symbol)
    lead_pos = target.index(0)
    if (entry is None or entry.image.lead_dims != [lead_pos]
        or (entry.covered is not None
            and not entry.holds(view.bbox, view.offset))):
      return None
    shift = entry.shift[lead_pos] if entry.shift else 0
    return (view.offset[lead_pos] - shift) % self._lane_span()

  def _make_loader_and_symbol_reg(self, opview, linearize,
                                  lead_pos: int = 0,
                                  carries_lead: bool = True
                                  ) -> Tuple[Symbol, GlbToRegLoader]:
    operand = opview.symbol

    # the register image holds the operand's *logical* region: GlbToRegLoader
    # consumes the slicing offset while loading, so everything downstream of
    # this point indexes logically.
    #
    # The linearized path is the exception: it copies spp.count_nz() elements
    # flat, so the array has to span the whole stored region --- and it cannot
    # express a slice at all (GlbToRegLoader asserts the offset is zero there).
    if linearize:
      bbox = operand.data_view._bbox
    else:
      bbox = opview.bbox

    # An operand that does *not* carry the destination's lead dimension is
    # read across the lanes -- every product broadcasts an element of it --
    # and a broadcast reaches only within one sub-group.  So where the
    # multiplication is wider than one, such an image is spread over a
    # sub-group and replicated into each rather than spread over the whole
    # multiplication: `lead=[(dim, block)]` with a block that divides the lane
    # count is what `Temporaries._lead_axes` reads as replication.
    #
    # The one that *does* carry it keeps the whole width, and that is the
    # point of the arrangement: its rows are the destination's rows, split
    # over the lanes, while the operand every row needs is copied.  Replicated
    # too, it would hold the same rows in every sub-group: at 56 rows over 64
    # lanes, rows 0..31 in both sub-groups and rows 32..55 in neither.
    lead = lead_pos
    if self._num_threads and not carries_lead:
      group = self._context.target.sub_group_width(self._num_threads)
      if group is not None and group < self._num_threads:
        lead = [(lead_pos, group)]
    registers, registerAlloc = self._temporaries.register_array(
        bbox, lead,
        spp=None if operand.obj.is_dense() else operand.obj.spp)
    # Past the rows it loads, the image holds a zero: the loader's accesses
    # are held to them (`LeadIndex.valid`), or keep off the slot's other
    # lanes, which hold the zeros the image is declared with.  Not said of a
    # flat copy, which has no rows to be past (`_zero_past`).
    registers.zero_past = not linearize
    self._instructions.append(registerAlloc)

    load_op = GlbToRegLoader(context=self._context,
                                     dest=registers,
                                     src=operand,
                                     num_threads=self._num_threads,
                                     linearize = linearize,
                                     lead_width = self._lead_width,
                                     src_bbox = bbox,
                                     src_offset = opview.offset)
    return SymbolView(registers, bbox), load_op

  def _stage_row_elements(self) -> int:
    """Elements the staged image's leading dimension is rounded up to.

    `Options.stage_row_bytes` in elements, and 1 -- no padding -- when it is
    not set, which is the default and what every recorded kernel has.

    The row is the only dimension worth padding: it is the one a wide access
    runs along, and it is the stride of every dimension above it, so an odd
    one makes every later row start at an odd element.  That is why the damage
    step's `(125, 6)` buffers refuse every pack and its `(20, 9)` one takes
    them all.  Padding it costs the pad and buys the rows above it as well.
    """
    want = self._context.get_user_options().stage_row_bytes
    if not want:
      return 1
    elem = self._context.fp_type.size()
    return max(1, want // elem)

  def _make_loader_and_symbol(self, opview, is_transpose) -> Tuple[Symbol, GlbToShrLoader]:
    operand = opview.symbol
    shr_mem_region = Symbol(name=self._temporaries.next_shared_name(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._scopes.add_symbol(shr_mem_region)
    load_op = GlbToShrLoader(context=self._context,
                                     dest=shr_mem_region,
                                     src=operand,
                                     shr_mem=self._shr_mem,
                                     num_threads=self._num_threads,
                                     alignment=self._stage_row_elements(),
                                     permute=is_transpose)
    # GlbToShrLoader copies the whole storage bounding box verbatim and gives
    # the copy the *same* bbox (only the shape is padded), so the operand's
    # logical->storage shift is just as valid against the shared-memory image
    # as it was against global memory.  Shared memory addresses through the
    # untyped branch of Symbol.build_address, so the offset can simply survive
    # here rather than having to be consumed the way the register path does.
    return SymbolView(shr_mem_region, opview.bbox, opview.offset), load_op

  def _alloc_register_array(self):
    """The array this operation accumulates its result into.

    Sized from what the operation *writes*: the destination's box, in the
    theta-shifted origin the compute and the store both index it in.
    `_analyze` may narrow the range below that --- an operand covering less
    than the destination declares --- which leaves slots unused, and that is
    harmless.

    Not from the *bias image's* box, even for an accumulation: that is what
    the operation reads, not what it writes, and the two part company whenever
    the image does not match the destination -- a broadcast leaves a rank-1
    image behind for a rank-2 destination, and an array sized from it would
    have one slot where the store walks three.  The destination's box says
    the same thing as a matching image and stays right when it does not
    match, so ask it directly.

    The lane axis is 0 because `MultilinearDescr._lead_dim` aligns the thread
    count to the destination's axis 0 and the whole multilinear path is built
    on that.  A per-operand answer would come from
    `[t for t in self._descr.target[0] if t >= 0]` once there is one.
    """
    registers, registerAlloc = self._temporaries.register_array(
        self._dest_obj.bbox, lead=0, shift=self._theta)
    self._instructions.append(registerAlloc)
    return registers

  def _target_image_fits(self, name):
    """Does the image staged under `name` hold exactly this destination?

    The residency is keyed by symbol name and lives for
    the whole kernel, so what a destination finds under its name may have been
    staged for something else entirely --- most often an *operand* read of a
    different slice of the same tensor.  Taking it as the accumulation bias
    would then read the wrong elements, from the wrong lanes: in the
    poroelastic space-time predictor `m2[:, 11] += ...` would pick up the
    image of `m2[:, 12]`, staged one descriptor earlier, and accumulate onto
    that.

    An accumulation chain onto the same box --- the case the reuse exists for
    --- has the staged region equal to the destination's, so require that.

    And a register image has to hold the box the way the accumulation adds
    over it: spread along the destination's axis 0, over the whole wave.  An
    image the accumulator or `_dest_preload_view` left is; an operand's
    staging of the same box need not be.  Where the build has an origin, its
    rows have to sit in the lanes the origin computes them in, as well
    (`_lead_origin_shift`): a copy staged for the rows of another operation
    need not.
    """
    entry = self._residency.get(name)
    if entry is None or entry.covered is None \
        or entry.covered.rank() != self._dest_obj.bbox.rank():
      return False
    want = self._dest_obj.storage_box()
    if entry.region() != (list(want.lower()), list(want.upper())):
      return False
    image = entry.image
    span = self._lane_span()
    if (image.stype not in (SymbolType.Register, SymbolType.Scratch)
        or not image.lead_dims or not span):
      return True
    if image.lead_dims != [0] or getattr(image, 'lead_axes', None) is not None:
      return False
    theta = getattr(self, '_theta', None)
    if theta is None:
      return True
    offset = self._dest_obj.offset[0] if self._dest_obj.offset else 0
    shift = entry.shift[0] if entry.shift else 0
    return (offset - shift - theta) % span == 0

  def _dest_preload_view(self, dest_symbol):
    """The destination, staged in the tensor's own lead coordinates.

    `GlbToRegLoader` consumes a slicing offset while loading: it reads
    `index + offset` and stores at `index`.  For an operand that is what
    `_stage_shift` asks of it, with a shift that keeps each row in the lane
    the loop reads it in, and `_staged_region` rebases the operand's offset
    to match.

    The destination has no such rebasing.  `_lead_origin_shift` takes theta
    from `_dest_obj.offset[0]`, and `_analyze` shifts the whole lead loop by
    it, so the compute and the store address row `n0` in the lane of
    `n0 + theta`.  An absorbed offset would put the preloaded bias somewhere
    else entirely: for `m2[10:20, 12] += ...` the loader would fill lanes
    0..9 while lanes 10..19 do the arithmetic and store, so every
    accumulation would read a bias of zero and the destination's previous
    value would be dropped -- `+=` silently becoming `=`.

    Folding the *lead* offset into the box instead leaves that axis in tensor
    coordinates, where lane `l` holds row `l` --- which is what theta assumes.
    The other axes keep theirs: the compute addresses them logically, so an
    absorbed offset there would send the read past the start of the array.
    """
    bbox = self._dest_obj.bbox
    offset = list(self._dest_obj.offset)
    if not offset:
      return SymbolView(dest_symbol, bbox, offset)
    # The slot-aligned part of the offset stays in the offset, so the loader
    # consumes it and the image lands in the same lane and the same slot the
    # lead loop addresses.  Only the remainder below a slot boundary goes into
    # the box, which is what puts the image in the loop's frame rather than the
    # tensor's -- `theta` is that remainder, and every other participant is
    # shifted by it too.
    span = self._lane_span()
    theta = getattr(self, '_theta', None)
    if theta is None:
      theta = offset[0] % span if span else 0
    shift = offset[0] - theta if span else 0
    lower, upper = list(bbox.lower()), list(bbox.upper())
    lower[0] += offset[0] - shift
    upper[0] += offset[0] - shift
    offset[0] = shift
    return SymbolView(dest_symbol, BoundingBox(lower, upper), offset)

  def _get_target_symbol(self, prev=False, next=False):
    dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
    if dest_symbol is None:
      return None
    if (dest_symbol.name in self._residency
        and not self._target_image_fits(dest_symbol.name)):
      # staged for something else: it cannot serve as this destination.  A
      # pending writeback still has to reach memory, which is exactly what
      # `_invalidate_residency` does; a preload is simply dropped and the
      # ordinary path below stages the region this operation needs.
      self._invalidate_residency(dest_symbol.name)
    if dest_symbol.name in self._residency:
      return self._residency.get(dest_symbol.name).image
    elif (self._policy.preload_operands_into_registers
          and dest_symbol.stype == SymbolType.Global and not next):
      view = self._dest_preload_view(dest_symbol)
      symbol, load_op = self._make_loader_and_symbol_reg(view, False)
      self._residency.record_preload(
          dest_symbol.name, symbol.symbol,
          symbol.symbol.data_view.get_bbox(),
          list(view.offset))
      self._instructions.append(load_op)
      return symbol.symbol
    elif (self._policy.preload_operands_into_shared
          and dest_symbol.stype == SymbolType.Global and not next):
      view = self._dest_preload_view(dest_symbol)
      symbol, load_op = self._make_loader_and_symbol(view, None)
      self._residency.record_preload(
          dest_symbol.name, symbol.symbol,
          symbol.symbol.data_view.get_bbox(),
          list(view.offset))
      self._instructions.append(load_op)
      return symbol.symbol
    else:
      return dest_symbol

  def _make_compute(self):
    prev = self._get_target_symbol(True) if self._add else None
    # A register image `_get_target_symbol` hands out holds the destination's
    # whole box -- it fits exactly or it is not handed out -- so the result is
    # written over all of it: the product where this term reaches, `prev`
    # elsewhere.  Left to this term's own box, a chain whose terms narrow (the
    # ADER Taylor expansion, `I += dQ(k) c_k` over fewer rows each time) would
    # keep only the last term's rows, and such a chain would have to go
    # through memory on every term.
    whole_prev = (prev is not None and prev.stype == SymbolType.Register
                  and any(e.image is prev for _, e in self._residency.items()))
    # Whether this result is built on the pending image; `_narrows_pending`
    # keeps one that was not from displacing it.
    self._built_on_pending = whole_prev
    compute = MultilinearInstruction(context=self._context,
                                     whole_prev=whole_prev,
                                     ops=self._mem_regions,
                                     target=self._descr.target,
                                     dest=self._temp_regs,
                                     num_threads=self._num_threads,
                                     prev=prev,
                                     prev_offset=self._prev_offset(prev),
                                     next=self._get_target_symbol(True, True),
                                     productOperation=MulOperator(),
                                     sumOperation=AddOperator(),
                                     dest_obj=self._dest_obj,
                                     theta=self._theta,
                                     lead_width=getattr(self, '_lead_width', 1),
                                     k_width=self._k_width(self._descr))
    self._instructions.append(compute)
    # The bias, behind the product it is no operand of (`alloc_destination`).
    self._instructions.extend(getattr(self, '_bias_loads', ()))
    self._bias_loads = []
    # What the multiplication does with its result -- scale it, add the
    # previous value, write the destination's box -- is an instruction of its
    # own, right behind it.
    epilogue = compute.epilogue()
    if epilogue is not None:
      self._instructions.append(epilogue)
    promise = self._promised_zeros(compute)
    if promise is not None:
      self._instructions.append(promise)
    compute._dest.zero_past = promise is not None or not self._writes_past(
        compute)

  def _promised_zeros(self, compute):
    """What the result's register image holds past its rows, said where
    only the multiplication knows it (`PromisedZeros`), or None.

    Only where the multiplication itself writes the image past its rows: it
    has no epilogue, and its last slot computes on every lane
    (`MultilinearInstruction._full_lane_tail`).  An epilogue writes the rows
    alone, and a loop that keeps to its rows writes nothing past them; the
    image then holds the zeros it is declared with there, which the passes
    see.  And only where what it writes there is a zero: every operand that
    carries the destination's lead index reads one past the rows
    (`MultilinearInstruction._beside`), so every product there has a zero
    factor.  Not before the rows: a head computed on every lane may read
    whole what lies before a window (`Symbol._memory_first`)."""
    image = compute._dest
    if (image.stype not in (SymbolType.Register, SymbolType.Scratch)
        or len(image.lead_dims) != 1 or not self._writes_past(compute)):
      return None
    _, past = compute._beside().get(compute._lead_dims[0], (True, True))
    if past:
      return None
    lead = image.lead_dims[0]
    box = compute._idest.data_view.get_bbox()
    if box.rank() <= lead:
      return None
    return PromisedZeros(self._context, image,
                         (box.lower()[lead], box.upper()[lead]),
                         self._num_threads, getattr(self, '_lead_width', 1))

  @staticmethod
  def _writes_past(compute) -> bool:
    """Does the multiplication itself write its image past the rows: no
    epilogue, and a last slot computed on every lane?"""
    if compute._has_epilogue:
      return False
    d = compute._lead_dims[0]
    return d < len(compute._ns) and compute._full_lane_tail(d, *compute._ns[d])

  def _prev_offset(self, prev):
    """Where the accumulation bias sits, relative to the loop indices.

    A staged image was loaded through `_dest_preload_view`, so it already sits
    where the loop indices point and needs no shift.  A destination read live
    out of memory --- which is what `_get_target_symbol` falls back to for a
    shared temporary --- does not: the loop indices are the descriptor's own,
    and its slicing offset has to be added, exactly as `_make_store` adds it on
    the way out.  Without it a `t0[10:20, 8] += ...` would read its bias from
    `t0[0:10, 0]` and write the sum to the right place, so the temporary
    would accumulate onto the wrong elements.
    """
    if prev is None:
      return None
    if prev is self._scopes.get_symbol(self._dest_obj.tensor):
      return self._store_offset()
    return None

  def _store_offset(self):
    """Destination offset seen by the store, in the shifted origin.

    The accumulator is indexed by the lead loop variable, which runs in the
    theta-shifted space; the global destination is not, so the shift has
    to come back out on the way to memory.  Dimension 0 of the destination is
    the one carrying the lead index.
    """
    return [o - (self._theta if j == 0 else 0)
            for j, o in enumerate(self._dest_obj.offset)]

  def _invalidate_residency(self, name):
    """The image of `name` is about to stop being the newest copy.

    `_get_target_symbol` preloads a global destination into registers and
    records that image so the *next* operation on the same tensor can
    accumulate straight into it.  That is only sound while the image stays the
    newest copy.  As soon as this operation writes memory from a different
    array --- which is what the eager-store paths below do --- the recorded
    image is one accumulation step behind, and leaving it in place hands every
    later `+=` a stale bias: each step then computes `preload + own term` and
    overwrites the previous one, so the destination ends up holding the first
    write plus the *last* term and nothing in between.
    """
    self._instructions.extend(self._residency.flush(name))

  def _inherited_promise(self, pending):
    """What an accumulation onto a pending writeback still owes.

    The assignment it adds to promised zeros outside what it covered, and the
    store keeps that promise -- the store that is now this entry's, since the
    residency holds one entry per name.  Dropped, it would send a merged run's
    `dQext(k+1) = ...; dQext(k+1) += ...` out with rows 10..20 as whatever the
    buffer held.  Carried only while it means the same cells: in the same
    frame, with nothing the pending image held left out.
    """
    if (pending is None or pending.is_preload or pending.promise is None
        or pending.covered is None
        or list(pending.shift) != list(self._store_offset())):
      return None
    covered = self._temp_regs.data_view.get_bbox()
    inside = all(cl <= pl and pu <= cu for pl, pu, cl, cu in zip(
        pending.covered.lower(), pending.covered.upper(),
        covered.lower(), covered.upper()))
    return pending.promise if inside else None

  def _promised_box(self):
    """What this operation undertakes to define, in the accumulator's frame.

    Two different things can look like a narrow destination box, and they want
    opposite treatment.

    A descriptor whose destination has no slicing offset addresses the tensor
    itself; its box is the *eqspp window*, the range yateto knows the result
    can be nonzero in.  Everything outside that window is zero, not
    unspecified, so the store has to write it as zero --- there is no other
    operation that will.  The poroelastic space-time predictor has exactly
    this shape: `m4[:, 6:13] = Q[:, 10:13] x S` is the only assignment to
    `m4`, and the three accumulations that follow read columns 0..5 back.

    A destination the frontend marked as *sliced* is a view: a slice of the
    tensor, with its own index space, and the rest belongs to other
    descriptors.  Touching anything outside it destroys their work.  A nonzero
    offset implies it, but a slice starting at index 0 has none --- the
    space-time predictor writes exactly that, one row and one column at a
    time --- so the flag carries it instead.

    An accumulation never zero-fills either way: it is defined in terms of
    what is already there.

    `_analyze` may narrow the accumulator below whichever box applies --- a
    sparse operand, an intersection with what the operands support --- and
    that difference is what the store legitimately fills with zeros.
    """
    if self._add or getattr(self._dest_obj, 'sliced', False):
      bbox = self._dest_obj.bbox
    else:
      bbox = self._dest_obj.tensor.get_bbox()
    lower = list(bbox.lower())
    upper = list(bbox.upper())
    if lower:
      lower[0] += self._theta
      upper[0] += self._theta
    return BoundingBox(lower, upper)

  def _assignment_zero_box(self):
    """Where an assignment into a shared buffer owes zeros, or None.

    The shared-memory side of the promise `StoreRegToGlb` keeps: an
    assignment defines its whole promised box (`_promised_box`), the part
    `_analyze` narrowed away as zeros.  Without it, a temporary assigned anew
    from a narrower operand would keep whatever its buffer held outside the
    new values -- SeisSol's free-surface-gravity kernel re-assigns `MPrev`
    from a product that one row of the Taylor derivative supports, and the
    accumulation that follows would add to the previous step's rows.

    In buffer coordinates, as `StoreRegToShr` takes it.  None for an
    accumulation, which promises nothing, and where the accumulator covers
    the promise anyway.
    """
    if self._add:
      return None
    promise = self._promised_box()
    covered = self._temp_regs.data_view.get_bbox()
    if promise.rank() == 0 or covered.rank() != promise.rank():
      return None
    if all(cl <= pl and pu <= cu for pl, pu, cl, cu in zip(
        promise.lower(), promise.upper(), covered.lower(), covered.upper())):
      return None
    shift = self._store_offset()
    return BoundingBox([l + o for l, o in zip(promise.lower(), shift)],
                       [u + o for u, o in zip(promise.upper(), shift)])

  def _narrows_pending(self, name) -> bool:
    """Would recording this result displace a pending writeback that holds
    more than it does?

    The plan keeps a chain in registers where every term can write the whole
    image (`whole_prev`).  Where one cannot after all -- its accumulator is
    not in the image's frame, or it did not take the image's box -- recording
    it would drop the rows only the pending image holds.  Then the pending
    image goes to its home first, and this result follows it there.
    """
    entry = self._residency.get(name)
    if entry is None or entry.is_preload or entry.covered is None:
      return False
    # An accumulation that was not built on the pending image -- a view that
    # did not fit was not handed out -- holds only its own term; the pending
    # one's would be lost.
    if self._add and not getattr(self, '_built_on_pending', False):
      return True
    if entry.shift and list(entry.shift) != list(self._store_offset()):
      return True
    mine = self._temp_regs.data_view.get_bbox()
    if mine.rank() != entry.covered.rank():
      return True
    return any(lm > le or um < ue for lm, um, le, ue in zip(
        mine.lower(), mine.upper(),
        entry.covered.lower(), entry.covered.upper()))

  def _make_store(self):
    if self._dest_obj.tensor in self._scopes:
      dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
      if dest_symbol.stype == SymbolType.SharedMem:
        if (self._plan.written_in_slices(self._dest_obj.tensor)
            or self._narrows_pending(dest_symbol.name)):
          self._invalidate_residency(dest_symbol.name)
          # assembled from several writes: this slice has to land in the shared
          # buffer now, since `_temp_regs` only ever holds our own part and the
          # deferred entry keeps just one of them
          zero_box = self._assignment_zero_box()
          self._instructions.append(StoreRegToShr(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  shr_mem=self._shr_mem,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  dest_bbox=self._plan.dest_union(self._dest_obj.tensor),
                                                  dest_offset=self._store_offset(),
                                                  clear=zero_box is not None,
                                                  clear_within=zero_box))
          return
        # see note below (but update to the new temp regs)
        self._residency.record_writeback(
            dest_symbol.name, self._temp_regs, dest_symbol,
            # the *actual* range the accumulator ended up with: _analyze
            # intersects the operands, so _ns can be strictly smaller than
            # the declared destination box
            covered=self._temp_regs.data_view.get_bbox(),
            shift=self._store_offset())
      elif dest_symbol.stype == SymbolType.Global:
        in_slices = (self._plan.written_in_slices(self._dest_obj.tensor)
                     or self._narrows_pending(dest_symbol.name))
        pending = self._residency.get(dest_symbol.name)
        result = choose_result_placement(
            legal_result_placements(written_in_slices=in_slices),
            policy=self._policy)
        # A destination assembled from several writes cannot be kept in
        # registers: the residency holds one entry per name, so a second
        # slice would displace the first and its whole contribution would be
        # computed and thrown away.
        if result is ResultPlacement.REGISTER:
          self._residency.record_writeback(
              dest_symbol.name, self._temp_regs, dest_symbol,
              # the *actual* range the accumulator ended up with: _analyze
              # intersects the operands, so _ns can be strictly smaller than
              # the declared destination box
              covered=self._temp_regs.data_view.get_bbox(),
              shift=self._store_offset(),
              promise=(self._promised_box() if not self._add
                       else self._inherited_promise(pending)))
        else:
          self._invalidate_residency(dest_symbol.name)
          self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  dest_offset=self._store_offset(),
                                                  dest_bbox=self._promised_box(),
                                                  zero_fill=not self._add))
      else:
        raise InternalError(f'gemm-builder: `res` must be either in shr. or glb. mem., given: {dest_symbol.stype}')
    else:
      if not self._dest_obj.tensor.is_tmp:
        raise InternalError('gemm-buider: `res` is not in scopes and thus must be tmp')

      dest_symbol = self._temporaries.shared_symbol(self._dest_obj.tensor)
      if (self._plan.written_in_slices(self._dest_obj.tensor)
          or self._narrows_pending(dest_symbol.name)):
        # a pending image that holds more than this result goes out first
        self._invalidate_residency(dest_symbol.name)
        # the first write of the temporary: where the plan found a use of
        # cells nothing defined before it, this store clears the buffer; an
        # assignment clears at least what it promised and did not compute
        zero_first = self._plan.zero_first(self._dest_obj.tensor)
        zero_box = self._assignment_zero_box()
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads,
                                                lead_width=self._lead_width,
                                                dest_bbox=self._plan.dest_union(self._dest_obj.tensor),
                                                dest_offset=self._store_offset(),
                                                clear=zero_first or zero_box is not None,
                                                clear_within=None if zero_first else zero_box))
        return
      self._residency.record_writeback(
          dest_symbol.name, self._temp_regs, dest_symbol,
          # the *actual* range the accumulator ended up with: _analyze
          # intersects the operands, so _ns can be strictly smaller than the
          # declared destination box
          covered=self._temp_regs.data_view.get_bbox(),
          shift=self._store_offset())

  def _insert_sync_block(self):
    """A barrier around the shared traffic, where nothing later will derive one.

    With `enable_sync_block_opt`, `pir.barriers` reads the final order of the
    section's statements and puts barriers where that order needs them, which
    is strictly better information than this builder has.  It keeps the ones
    it finds, so a barrier appended here would stand in the stream for the
    whole optimization pipeline -- and a barrier is a wall for a transfer
    into shared memory moving up (`pir.move`).  On `local_flux` that would
    pin all five global-to-shared transfers where they are built: not one
    would move, so every `__pipeline_commit` would end up next to its
    `__pipeline_wait_prior(0)` and the asynchronous copies would be issued and
    awaited in the same breath.

    So they are inserted only where they are the final word: with the option
    off, nothing else derives a barrier and these are the kernel's own.
    """
    if self._context.get_user_options().enable_sync_block_opt:
      return
    self._instructions.append(SyncThreads(context=self._context,
                                          num_threads_per_mult=self._num_threads))
