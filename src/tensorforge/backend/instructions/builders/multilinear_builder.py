# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Tuple
from tensorforge.common.basic_types import Addressing
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.instructions.memory.load import GlbToShrLoader, GlbToRegLoader
from tensorforge.backend.instructions.memory.store import StoreRegToGlb, StoreRegToShr, StoreRegToReg
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.backend.instructions.compute.multilinear import MultilinearInstruction
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.exceptions import InternalError, GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.backend.instructions.builders.operation_builder import OperationBuilder
from tensorforge.backend.instructions.abstract_instruction import _explicit_simd
from tensorforge.backend.placement import (Placement, ResultPlacement,
                                           choose_operand_placement,
                                           choose_result_placement,
                                           legal_operand_placements,
                                           legal_result_placements,
                                           policy_for, result_is_atomic)
from tensorforge.common.operation import AddOperator, MulOperator


class MultilinearBuilder(OperationBuilder):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #: What this hardware prefers among the placements that are legal.  Only
    #: preferences: nothing here can make a correct kernel incorrect, and the
    #: legality half is asked separately at each decision.
    self._policy = policy_for(self._context.get_vm().get_hw_descr(),
                              explicit_simd=_explicit_simd(self._context))

  def _k_width(self, descr) -> int:
    """How many reduction steps one body should cover for this operator.

    Decided per operator rather than per section, unlike the lead width: the
    reduction does not touch the register image that the descriptors of a
    section share, so one operator taking a wider group does not constrain
    its neighbours.
    """
    from tensorforge.backend.instructions.memory import vectorize
    if vectorize.K_WIDTH <= 1:
      return 1
    return vectorize.K_WIDTH

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

    for i in range(len(self._ops)):
        self._make_load_op(i)
    self._insert_sync_block()
    return self._mem_regions

  def alloc_destination(self, descr, operands):
    """The accumulator, in the shifted origin the lead loop runs in."""
    self._theta = self._lead_origin_shift()
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

    # Which operand dimension carries the destination's lead index --- i.e.
    # which one has to end up spread across lanes.  Everything downstream of a
    # register staging already assumes this dimension is the lane axis:
    # `_lead_origin_shift` pins the origin on `target[i].index(0)`, and
    # `MultilinearInstruction._check_offsets` checks the slicing remainder
    # there.  Only the staging itself used to hardcode dimension 0, so an
    # operand whose lead index sits elsewhere --- a transposed one --- got an
    # image whose lane axis was a *contraction* dimension.  `Symbol.load` then
    # found a loop constant on what it believed was the lane axis and emitted a
    # cross-lane broadcast, which hands every lane the same element and drops
    # the lane-distributed index entirely.
    lead_pos = self._descr.target[i].index(0) if has_lead_dim else 0
    self._lead_pos[i] = lead_pos

    # Whether this operand's lane axis is where a reader expects it.  A
    # legality fact and nothing else: it does not change with the vendor, and
    # the staging shift and the choice of loader both key on it even where the
    # hardware can read the operand in place anyway.
    lane_axis_needs_moving = transpose or not has_lead_dim

    symbol = self._ops[i].symbol
    addressable = not (
        symbol.stype in (SymbolType.Scalar, SymbolType.Data)
        or (isinstance(symbol.obj, Tensor) and len(symbol.obj.shape) == 0)
        or getattr(symbol.obj, 'addressing', None) == Addressing.NONE)
    placement = choose_operand_placement(
        legal_operand_placements(addressable=addressable,
                                 transposed=transpose,
                                 carries_lead_dim=has_lead_dim,
                                 policy=self._policy),
        self._policy)

    name = self._ops[i].symbol.name

    # An image already staged for an earlier operation spreads whichever
    # dimension *that* operation needed across lanes.  Reusing it for an
    # operand with a different lane axis -- the same tensor used once plainly
    # and once transposed -- is a cross-lane transpose, which an address
    # cannot express.  Both cases have a fallback already:
    #   * a preload of a global input (`src is dest`) can be dropped, since
    #     global memory still holds the value and the ordinary path re-stages
    #     it in the orientation this operand wants;
    #   * a pending store of a temporary has to go out to shared memory first,
    #     which is exactly what a shared placement already does (and what the CUDA
    #     path does for every transposed operand anyway).
    staged = self._residency.get(name)
    if staged is not None and not self._lane_axis_matches(name, lead_pos):
      if staged.is_preload:
        self._residency.drop(name)
      else:
        # legality, not preference: the image cannot serve, and only a copy
        # through shared memory can move the lane axis
        placement = Placement.SHARED

    # The linearized load packs the operand flat --- storage element `f` goes
    # to lane `f % T`, slot `f // T` --- which can only ever make the *first*
    # dimension the lane axis.  With the lane axis elsewhere the flat packing
    # and the per-dimension addressing describe different images, so take the
    # dimension-wise loader instead; it is correct for any lane axis.
    linearize = lane_axis_needs_moving and lead_pos == 0

    if name in self._residency and self._resolve_reuse(i, name):
      entry = self._residency.get(name)
      if placement is Placement.SHARED:
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=entry.image,
                                                dest=entry.home,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads))
        self._residency.drop(name)
        self._ops[i].symbol = entry.home
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
                lead_pos=lead_pos)
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

      elif self._ops[i].symbol.stype == SymbolType.SharedMem or self._ops[i].symbol.stype == SymbolType.Register:
        # An operand that already sits in shared memory or in registers is
        # read where it is.  Deciding to re-stage it in a different
        # orientation would need a record of which orientation the existing
        # image has; the residency entry carries that, and
        # `_lane_axis_matches` above is what consults it.
        self._mem_regions[i] = self._ops[i]
      else:
        raise InternalError(f'gemm-builder: op{i} ({self._ops[i].symbol.name}) must be either in shr or glb mem, given: {self._ops[i].symbol.stype}')

  def _union_of(self, i):
    view = self._ops[i]
    union = self._plan.operand_union(view.symbol.name)
    if union is None:
      union = BoundingBox([l + o for l, o in zip(view.bbox.lower(), view.offset)],
                          [u + o for u, o in zip(view.bbox.upper(), view.offset)])
    return union

  def _stage_shift(self, i, absorb_lead):
    """How the staged image is indexed relative to the tensor.

    Position `r` of the image holds tensor element `r + shift`.

    For a register image the lead dimension is spread across lanes and the slot
    count is `ceil(u/T) - floor(l/T)`, so a union starting mid-block costs an
    extra slot for nothing: `[8,24)` at T=16 needs two slots where `[0,16)`
    needs one.  Absorbing the union's lead lower bound while loading --- free,
    the loader forms a global address per element anyway --- puts the image
    back at origin 0.  The remaining dimensions stay in storage coordinates,
    where the offset is a plain constant in the address and absorbing it buys
    nothing.

    A shared-memory image is a verbatim copy of the whole storage box, so its
    coordinates already are the tensor's and the shift is zero throughout.  So
    is the linearized register path, which copies flat and cannot express a
    shift at all.
    """
    rank = self._ops[i].bbox.rank()
    if not absorb_lead or rank == 0:
      return [0] * rank
    lead_pos = self._lead_pos[i]
    shift = [0] * rank
    shift[lead_pos] = self._union_of(i).lower()[lead_pos]
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

  def _lane_axis_matches(self, name, lead_pos):
    """Does the staged image spread the dimension this operand needs?

    Only register images have a lane axis at all; a shared-memory staging is
    a verbatim copy and serves any orientation.
    """
    image = self._residency.get(name).image
    if image.stype not in (SymbolType.Register, SymbolType.Scratch):
      return True
    return image.lead_dims == [lead_pos]

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
      * a *writeback*: the value exists only in registers and the missing part
        was produced by some other instruction, so there is nothing to fall
        back on.

    Sizing the staging to the union of all consumers would avoid the second
    case entirely; until then it is refused loudly rather than miscompiled.
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
      raise GenerationError(
          f'{name}: operand wants {view.bbox} at offset {view.offset} but the '
          f'staged image only covers {entry.covered} at shift {entry.shift}, '
          f'and the value exists only in registers. Serving several disjoint '
          f'slices of one tensor needs the staging sized to their union.')

    view.offset = [o - s for o, s in zip(view.offset, entry.shift)]
    return True

  def _lead_origin_shift(self):
    """Pick the origin of the lead loop.

    `n0` is an internal loop variable of this instruction; its absolute origin
    is arbitrary and only the offsets of the participants *relative* to each
    other matter.  A register-resident operand is the exception: its lane
    assignment is already fixed --- element `s` lives in lane `s % T` --- so it
    pins the origin modulo T.  Choosing that origin makes the slicing offset
    vanish in lane terms, at a cost of at most one extra slot per non-lead
    element, where the alternative is a cross-lane shuffle on every read.

    Two register participants that disagree modulo T cannot both be satisfied;
    that genuinely needs a shuffle and is refused here.  It takes two operands
    carrying the lead index and both register-resident, which no kernel in the
    test suite produces.
    """
    threads = self._num_threads
    if not threads:
      return 0
    pins = {}
    for i, view in enumerate(self._mem_regions):
      if view is None or 0 not in self._descr.target[i]:
        continue
      if view.symbol.stype not in (SymbolType.Register, SymbolType.Scratch):
        continue
      j = self._descr.target[i].index(0)
      pins.setdefault(view.offset[j] % threads, []).append(view.symbol.name)
    if self._add:
      # accumulating into an array that already exists: its lane assignment is
      # fixed too, so it pins the origin exactly like an operand does
      target = self._get_target_symbol()
      if target is not None and target.stype in (SymbolType.Register,
                                                 SymbolType.Scratch):
        pins.setdefault(self._dest_obj.offset[0] % threads,
                        []).append(target.name)
    if len(pins) > 1:
      raise GenerationError(
          'lead-dimension slicing offsets of register operands disagree modulo '
          f'{threads}: ' +
          ', '.join(f'{n} -> {r}' for r, ns in pins.items() for n in ns) +
          '. Reconciling them needs a cross-lane shuffle, which is not '
          'implemented; stage one of them separately instead.')
    return next(iter(pins), 0)

  def _make_loader_and_symbol_reg(self, opview, linearize,
                                  lead_pos: int = 0) -> Tuple[Symbol, GlbToRegLoader]:
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

    registers, registerAlloc = self._temporaries.register_array(
        bbox, lead_pos,
        spp=None if operand.obj.is_dense() else operand.obj.spp)
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

    An accumulation used to take the *bias image's* box instead.  That is what
    the operation reads, not what it writes, and the two part company whenever
    the image does not match the destination: a broadcast leaves a rank-1
    image behind for a rank-2 destination, and the array came out with one
    slot where the store walks three.  The destination's box says the same
    thing as a matching image and stays right when it does not match, so ask
    it directly.

    The lane axis is 0 because `MultilinearDescr._lead_dim` aligns the thread
    count to the destination's axis 0 and the whole multilinear path is built
    on that.  A per-operand answer would come from
    `[t for t in self._descr.target[0] if t >= 0]` once there is one.
    """
    registers, registerAlloc = self._temporaries.register_array(
        self._dest_obj.bbox, lead_pos=0, shift=self._theta)
    self._instructions.append(registerAlloc)
    return registers

  def _target_image_fits(self, name):
    """Does the image staged under `name` hold exactly this destination?

    The residency is keyed by symbol name and lives for
    the whole kernel, so what a destination finds under its name may have been
    staged for something else entirely --- most often an *operand* read of a
    different slice of the same tensor.  Taking it as the accumulation bias
    then reads the wrong elements, from the wrong lanes: in the poroelastic
    space-time predictor `m2[:, 11] += ...` picked up the image of
    `m2[:, 12]`, staged one descriptor earlier, and accumulated onto that.

    An accumulation chain onto the same box --- the case the reuse exists for
    --- has the staged region equal to the destination's, so require that.
    """
    entry = self._residency.get(name)
    if entry is None or entry.covered is None \
        or entry.covered.rank() != self._dest_obj.bbox.rank():
      return False
    want = self._dest_obj.storage_box()
    return entry.region() == (list(want.lower()), list(want.upper()))

  def _dest_preload_view(self, dest_symbol):
    """The destination, staged in the tensor's own lead coordinates.

    `GlbToRegLoader` consumes a slicing offset while loading: it reads
    `index + offset` and stores at `index`, so the image sits at origin 0 and
    element `s` lands in lane `s % T`.  For an operand that is fine ---
    `_stage_shift`/`_staged_region` rebase the offset to match, and
    `_lead_origin_shift` then reads the rebased one.

    The destination has no such rebasing.  `_lead_origin_shift` pins theta on
    `_dest_obj.offset[0]`, and `_analyze` shifts the whole lead loop by it, so
    the compute and the store address row `n0` in lane `n0 % T`.  An absorbed
    offset puts the preloaded bias somewhere else entirely: for
    `m2[10:20, 12] += ...` the loader filled lanes 0..9 while lanes 10..19 did
    the arithmetic and stored, so every accumulation read a bias of zero and
    the destination's previous value was dropped.  `+=` silently became `=`.

    Folding the *lead* offset into the box instead leaves that axis in tensor
    coordinates, where lane `l` holds row `l` --- which is what theta assumes.
    The other axes keep theirs: the compute addresses them logically, so an
    absorbed offset there would send the read past the start of the array.
    """
    bbox = self._dest_obj.bbox
    offset = list(self._dest_obj.offset)
    if not offset:
      return SymbolView(dest_symbol, bbox, offset)
    lower, upper = list(bbox.lower()), list(bbox.upper())
    lower[0] += offset[0]
    upper[0] += offset[0]
    offset[0] = 0
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
    elif self._policy.atomic_accumulation and prev:
      # should be found in the previous step already
      return None
    elif (self._policy.preload_operands_into_registers
          and dest_symbol.stype == SymbolType.Global
          and not self._policy.atomic_accumulation and not next):
      symbol, load_op = self._make_loader_and_symbol_reg(
          self._dest_preload_view(dest_symbol), False)
      self._residency.record_preload(
          dest_symbol.name, symbol.symbol,
          symbol.symbol.data_view.get_bbox(),
          [0] + list(self._dest_obj.offset[1:]))
      self._instructions.append(load_op)
      return symbol.symbol
    elif (self._policy.preload_operands_into_shared
          and dest_symbol.stype == SymbolType.Global
          and not self._policy.atomic_accumulation and not next):
      symbol, load_op = self._make_loader_and_symbol(
          self._dest_preload_view(dest_symbol), None)
      self._residency.record_preload(
          dest_symbol.name, symbol.symbol,
          symbol.symbol.data_view.get_bbox(),
          [0] + list(self._dest_obj.offset[1:]))
      self._instructions.append(load_op)
      return symbol.symbol
    else:
      return dest_symbol

  def _make_compute(self):
    prev = self._get_target_symbol(True) if self._add else None
    self._instructions.append(MultilinearInstruction(context=self._context,
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
                                   k_width=self._k_width(self._descr)))

  def _prev_offset(self, prev):
    """Where the accumulation bias sits, relative to the loop indices.

    A staged image was loaded through `_dest_preload_view`, so it already sits
    where the loop indices point and needs no shift.  A destination read live
    out of memory --- which is what `_get_target_symbol` falls back to for a
    shared temporary --- does not: the loop indices are the descriptor's own,
    and its slicing offset has to be added, exactly as `_make_store` adds it on
    the way out.  Without it a `t0[10:20, 8] += ...` read its bias from
    `t0[0:10, 0]` and wrote the sum to the right place, so the temporary
    accumulated onto the wrong elements.
    """
    if prev is None:
      return None
    if prev is self._scopes.get_symbol(self._dest_obj.tensor):
      return self._store_offset()
    return None

  def _store_offset(self):
    """Destination offset seen by the store, in the shifted origin.

    The accumulator is indexed by the lead loop variable, which now runs in
    the theta-shifted space; the global destination is not, so the shift has
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

  def _make_store(self):
    if self._dest_obj.tensor in self._scopes:
      dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
      if dest_symbol.stype == SymbolType.SharedMem:
        if self._plan.written_in_slices(self._dest_obj.tensor):
          self._invalidate_residency(dest_symbol.name)
          # assembled from several writes: this slice has to land in the shared
          # buffer now, since `_temp_regs` only ever holds our own part and the
          # deferred entry keeps just one of them
          self._instructions.append(StoreRegToShr(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  shr_mem=self._shr_mem,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  dest_bbox=self._plan.dest_union(self._dest_obj.tensor),
                                                  dest_offset=self._store_offset()))
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
        in_slices = self._plan.written_in_slices(self._dest_obj.tensor)
        pending = self._residency.get(dest_symbol.name)
        atomic = result_is_atomic(
            accumulating=self._add,
            pending_is_atomic=pending is None or pending.atomic is not None,
            policy=self._policy)
        result = choose_result_placement(
            legal_result_placements(written_in_slices=in_slices),
            atomic=atomic, policy=self._policy)
        # A destination assembled from several writes cannot be kept in
        # registers: the residency holds one entry per name, so a second
        # slice would displace the first and its whole contribution would be
        # computed and thrown away.
        #
        # Atomics are exempt from that *only* if they really do go out on
        # their own.  Deferring one is what makes it collide with the next
        # slice, so with several writers the update is emitted here instead
        # of at the epilogue; there is nothing to serialise, since an atomic
        # add is order-independent by construction.  With a single covering
        # writer deferring still pays --- it saves the read-modify-write.
        if result is ResultPlacement.MEMORY and atomic:
          self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  atomic=True,
                                                  dest_offset=self._store_offset(),
                                                  dest_bbox=self._promised_box(),
                                                  zero_fill=not self._add))
        elif result is ResultPlacement.REGISTER:
          self._residency.record_writeback(
              dest_symbol.name, self._temp_regs, dest_symbol,
              # the *actual* range the accumulator ended up with: _analyze
              # intersects the operands, so _ns can be strictly smaller than
              # the declared destination box
              covered=self._temp_regs.data_view.get_bbox(),
              shift=self._store_offset(),
              atomic=True if atomic else None,
              promise=self._promised_box() if not self._add else None)
        else:
          self._invalidate_residency(dest_symbol.name)
          self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  atomic=None,
                                                  dest_offset=self._store_offset(),
                                                  dest_bbox=self._promised_box(),
                                                  zero_fill=not self._add))
      elif dest_symbol.stype == SymbolType.Register:
        self._instructions.append(StoreRegToReg(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                num_threads=self._num_threads))
      else:
        raise InternalError(f'gemm-builder: `res` must be either in shr. or glb. mem., given: {dest_symbol.stype}')
    else:
      if not self._dest_obj.tensor.is_tmp:
        raise InternalError(f'gemm-buider: `res` is not in scopes and thus must be tmp')

      dest_symbol = self._temporaries.shared_symbol(self._dest_obj.tensor)
      if self._plan.written_in_slices(self._dest_obj.tensor):
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads,
                                                lead_width=self._lead_width,
                                                dest_bbox=self._plan.dest_union(self._dest_obj.tensor),
                                                dest_offset=self._store_offset()))
        return
      self._residency.record_writeback(
          dest_symbol.name, self._temp_regs, dest_symbol,
          # the *actual* range the accumulator ended up with: _analyze
          # intersects the operands, so _ns can be strictly smaller than the
          # declared destination box
          covered=self._temp_regs.data_view.get_bbox(),
          shift=self._store_offset())

  def _insert_sync_block(self):
    self._instructions.append(SyncThreads(context=self._context,
                                          num_threads_per_mult=self._num_threads))
