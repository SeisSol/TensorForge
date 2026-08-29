# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Tuple, Dict, List
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Addressing
from tensorforge.backend.scopes import Scopes
from tensorforge.backend.symbol import DataView, Symbol, SymbolType, SymbolView
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.memory.load import GlbToShrLoader, GlbToRegLoader
from tensorforge.backend.instructions.memory.store import StoreRegToGlb, StoreRegToShr, StoreRegToReg
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.backend.instructions.compute.multilinear import MultilinearInstruction
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.exceptions import InternalError, GenerationError
import itertools
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.backend.instructions.builders.allocator_builder import AbstractBuilder
from tensorforge.common.operation import AddOperator, MulOperator
from tensorforge.backend.data_types import RegMemObject
from tensorforge.backend.instructions.abstract_instruction import (
    AbstractInstruction, _explicit_simd)


class MultilinearBuilder(AbstractBuilder):
  def __init__(self,
               context: Context,
               scopes: Scopes,
               shr_mem: Symbol,
               num_threads: int,
               lead_width: int = 1):
    super(MultilinearBuilder, self).__init__(context, scopes)
    self._shr_mem = shr_mem
    self._num_threads = num_threads
    self._lead_width = lead_width


    self._counter = 0
    self._counter_shr_reg = 0
    self._loaders_cache: Dict[Symbol, AbstractInstruction] = {}

    self._ops = None
    self._dest_obj = None
    self._descr = None

    self._mem_regions = None

    self._temp_regs = None
    self._dest_regs = None

    self._use_registers_always = self._context.get_vm().get_hw_descr().vendor in ['amd', 'nvidia']
    self._preload_registers = self._context.get_vm().get_hw_descr().vendor in ['amd', 'nvidia']
    self._preload_shmem = self._context.get_vm().get_hw_descr().vendor in [] #['nvidia']
    self._atomic_update = self._context.get_vm().get_hw_descr().vendor in ['amd'] # , 'nvidia' # ?
    self._deferred_stores = {}
    # per pending writeback: the box its producing descriptor undertook to
    # define, so the deferred store zero-fills exactly what `_analyze` narrowed
    # away and nothing else
    self._promised = {}
    # what each deferred/staged image actually holds, so a later consumer
    # can tell whether reusing it is sound: position `r` of the image holds
    # tensor element `r + shift`, for `r` inside `covered`.
    self._staged_view = {}
    # tensor symbol name -> union of every operand access, in tensor
    # storage coordinates.  Filled by plan(); see there.
    self._operand_union = {}
    self._temporaries = {}


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

  def build(self, ops: List[Symbol], dest_obj: Tensor, descr: MultilinearDescr):
    self._reset()

    self._ops = ops
    self._dest_obj = dest_obj
    self._descr = descr

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
    self._theta = self._lead_origin_shift()
    self._temp_regs = self._alloc_register_array()
    self._make_compute()
    self._insert_sync_block()
    self._make_store()
    self._insert_sync_block()

  # TODO: check if we always can allow a direct global memory load
  def _make_load_op(self, i):

    prefer_broadcast = self._context.get_vm().get_hw_descr().vendor in ['amd']

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

    needs_reload = (transpose or not has_lead_dim) and not prefer_broadcast
    needs_reload2 = transpose or not has_lead_dim

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
    #     which is exactly what `needs_reload` already does (and what the CUDA
    #     path does for every transposed operand anyway).
    staged = self._deferred_stores.get(name)
    if staged is not None and not self._lane_axis_matches(name, lead_pos):
      if staged[0] is staged[1]:
        del self._deferred_stores[name]
        self._staged_view.pop(name, None)
      else:
        needs_reload = True

    # The linearized load packs the operand flat --- storage element `f` goes
    # to lane `f % T`, slot `f // T` --- which can only ever make the *first*
    # dimension the lane axis.  With the lane axis elsewhere the flat packing
    # and the per-dimension addressing describe different images, so take the
    # dimension-wise loader instead; it is correct for any lane axis.
    linearize = needs_reload2 and lead_pos == 0

    if name in self._deferred_stores and self._resolve_reuse(i, name):
      if needs_reload:
        src, dest, _ = self._deferred_stores[name]
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=src,
                                                dest=dest,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads))
        del self._deferred_stores[name]
        self._ops[i].symbol = dest
      else:
        self._ops[i].symbol, _, _ = self._deferred_stores[name]

    if self._ops[i].symbol.stype == SymbolType.Scalar or self._ops[i].symbol.stype == SymbolType.Data \
      or (isinstance(self._ops[i].symbol.obj, Tensor) and len(self._ops[i].symbol.obj.shape) == 0): # <-- quasi-scalar
      self._mem_regions[i] = self._ops[i]
    else:

      if self._ops[i].symbol.stype == SymbolType.Global:
        if needs_reload and self._ops[i].symbol.obj.addressing != Addressing.NONE:
          shift = self._stage_shift(i, absorb_lead=False)
          staged, load_op = self._make_loader_and_symbol(self._stage_view(i, shift), is_transpose=self._descr.permute[i])
          self._mem_regions[i] = self._staged_region(i, staged, shift)
          self._loaders_cache[self._mem_regions[i]] = load_op
          self._instructions.append(load_op)
        else:
          if self._preload_registers and self._ops[i].symbol.obj.addressing != Addressing.NONE:
            # only register-preload dense matrices for now
            shift = self._stage_shift(i, absorb_lead=not needs_reload2)
            staged, load_op = self._make_loader_and_symbol_reg(
                self._stage_view(i, shift), linearize=linearize,
                lead_pos=lead_pos)
            self._mem_regions[i] = self._staged_region(i, staged, shift)
            self._deferred_stores[self._ops[i].symbol.name] = self._mem_regions[i].symbol, self._mem_regions[i].symbol, None
            self._record_staged(self._ops[i].symbol.name,
                                self._mem_regions[i].symbol.data_view.get_bbox(),
                                shift)
            self._instructions.append(load_op)
          elif self._preload_shmem and self._ops[i].symbol.obj.addressing != Addressing.NONE:
            # only register-preload dense matrices for now
            shift = self._stage_shift(i, absorb_lead=False)
            staged, load_op = self._make_loader_and_symbol(self._stage_view(i, shift), None)
            self._mem_regions[i] = self._staged_region(i, staged, shift)
            self._deferred_stores[self._ops[i].symbol.name] = self._mem_regions[i].symbol, self._mem_regions[i].symbol, None
            self._record_staged(self._ops[i].symbol.name,
                                self._mem_regions[i].symbol.data_view.get_bbox(),
                                shift)
            self._instructions.append(load_op)
          else:
            # Note: operand will reside in glb. mem for gemm operation
            self._mem_regions[i] = self._ops[i]

      elif self._ops[i].symbol.stype == SymbolType.SharedMem or self._ops[i].symbol.stype == SymbolType.Register:
        if self._ops[i].symbol in self._loaders_cache.keys():
          # Note: this condition means the symbol `self._ops[i].symbol` has been loaded
          # to shr. mem. before. Let's check whether loaded data can be reused
          prev_loader = self._loaders_cache[self._ops[i].symbol]

          # we only need to reload/globally load, if we even need a leading dimension
          if self._descr.permute[i] != prev_loader.get_permute() and has_lead_dim:
            if not transpose:
              # means: data loaded to shr. mem. cannot be reused. Because `op1` not need to be transposed
              # we don't need to load it to shr. mem. Instead, it will be taken from glb. mem.
              # we don't need delete previous (aliased) symbol
              self._mem_regions[i] = SymbolView(prev_loader.get_src(),
                                               self._ops[i].bbox, self._ops[i].offset)
            else:
              # means: data cannot be reused. we need to reload it again and traspose on the fly.
              # additionally, we need to remove aliased symbol to avoid clashes
              # self._scopes.delete_symbol(self._ops[i].symbol)
              self._scopes.add_scope()
              prev_symbol = prev_loader.get_src()
              # NOTE: this call was missing its `is_transpose` argument entirely
              # (TypeError if the branch is ever reached); re-staging the original
              # global source wants this operation's own permutation.
              self._mem_regions[i], load_op = self._make_loader_and_symbol(
                  SymbolView(prev_symbol, self._ops[i].bbox, self._ops[i].offset),
                  is_transpose=self._descr.permute[i])
              self._loaders_cache[self._mem_regions[i]] = load_op
              self._instructions.append(load_op)
          else:
            # means: data can be fully reused
            self._mem_regions[i] = self._ops[i]

        else:
          self._mem_regions[i] = self._ops[i]
      else:
        raise InternalError(f'gemm-builder: op{i} ({self._ops[i].symbol.name}) must be either in shr or glb mem, given: {self._ops[i].symbol.stype}')

  def plan(self, descr_list):
    """Size every staging to the union of the slices that will read it.

    A tensor's staging is created by whichever operation touches it first and,
    because `_deferred_stores` is keyed by name and lives for the whole kernel,
    is then handed to every later operation on that tensor.  Sizing it to the
    first consumer's slice therefore starves all the others --- which is what
    `_resolve_reuse` had to refuse.  The full descriptor list is known before
    codegen, so the union can simply be taken up front.

    The union is kept in *tensor storage* coordinates and the staged image is
    indexed the same way.  That makes the mapping the identity: every consumer
    keeps the offset it already states against the tensor, nothing is rebased,
    and containment holds by construction.

    Destinations are deliberately not counted --- they are produced, not
    staged, and their range is whatever `_analyze` intersects it down to.
    """
    self._operand_union = {}
    # same, but over destinations: what a tensor's writes cover, and how many
    # descriptors contribute.  A temporary written by exactly one operation can
    # stay in registers until someone asks for it; one written in slices has to
    # be assembled somewhere, because each operation only ever holds its own
    # slice in `_temp_regs`.
    self._dest_union = {}
    self._read_union = {}
    # every box each writer of a destination states; the count is how many
    # writers there are, and whether they all match the union is whether the
    # tensor is assembled from pieces
    # every individual write box, not just their union: two writes to [0,2) and
    # [8,10) union to [0,10), so a union-against-union test would wave through
    # a read of [2,8) that nothing ever wrote
    self._dest_boxes = {}
    self._read_tensors = {}
    self._eff_reads = {}
    self._eff_writes = {}
    for descr in descr_list:
      if not isinstance(descr, MultilinearDescr):
        continue
      for op in descr.ops:
        tensor = getattr(op, 'tensor', None)
        if tensor is None:
          continue
        lower = [l + o for l, o in zip(op.bbox.lower(), op.offset)]
        upper = [u + o for u, o in zip(op.bbox.upper(), op.offset)]

        # Keyed by tensor, and recorded *before* the symbol lookup: a temporary
        # has no symbol until the operation that first writes it creates one,
        # so guarding this on the lookup left every temporary with an empty
        # read union --- and `_written_in_slices` then saw nothing to cover and
        # happily deferred a store that only held part of the tensor.
        rkey = id(tensor)
        rprev = self._read_union.get(rkey)
        rlo, rup = list(lower), list(upper)
        if rprev is not None:
          rlo = [min(a, b) for a, b in zip(rprev.lower(), rlo)]
          rup = [max(a, b) for a, b in zip(rprev.upper(), rup)]
        self._read_union[rkey] = BoundingBox(rlo, rup)
        self._read_tensors[tensor] = self._read_union[rkey]

        # The staging union is keyed by symbol name, which only exists for
        # tensors that are already materialised somewhere.
        symbol = self._scopes.get_symbol(tensor)
        if symbol is None:
          continue
        prev = self._operand_union.get(symbol.name)
        if prev is not None:
          lower = [min(a, b) for a, b in zip(prev.lower(), lower)]
          upper = [max(a, b) for a, b in zip(prev.upper(), upper)]
        self._operand_union[symbol.name] = BoundingBox(lower, upper)

      dest = getattr(descr, 'dest', None)
      tensor = getattr(dest, 'tensor', None) if dest is not None else None
      if tensor is not None:
        # keyed by tensor, not by symbol name: a temporary has no symbol until
        # the operation that first writes it creates one, which is long after
        # plan() has run
        key = id(tensor)
        lower = [l + o for l, o in zip(dest.bbox.lower(), dest.offset)]
        upper = [u + o for u, o in zip(dest.bbox.upper(), dest.offset)]
        prev = self._dest_union.get(key)
        if prev is not None:
          lower = [min(a, b) for a, b in zip(prev.lower(), lower)]
          upper = [max(a, b) for a, b in zip(prev.upper(), upper)]
        self._dest_union[key] = BoundingBox(lower, upper)
        self._dest_boxes.setdefault(key, []).append(
            BoundingBox([l + o for l, o in zip(dest.bbox.lower(), dest.offset)],
                        [u + o for u, o in zip(dest.bbox.upper(), dest.offset)]))

      eff = self._effective_boxes(descr)
      if eff is not None:
        eff_reads, eff_write = eff
        for t, box in eff_reads.items():
          prev = self._eff_reads.get(t)
          self._eff_reads[t] = box if prev is None else BoundingBox(
              [min(a, b) for a, b in zip(prev.lower(), box.lower())],
              [max(a, b) for a, b in zip(prev.upper(), box.upper())])
        if tensor is not None:
          self._eff_writes.setdefault(id(tensor), []).append(eff_write)

    self._check_initialised()

  def _effective_boxes(self, descr):
    """Read and write boxes after the range intersection, in tensor coords.

    A descriptor's declared operand box is an upper bound on what that operand
    contributes; `_analyze` intersects the boxes of everything sharing a target
    index (and the destination) and iterates only that. Checking declared reads
    against actual writes therefore flags regions that are never touched --- so
    the intersection has to be replayed here to compare like with like.

    Returns `(reads, write)` where `reads` maps tensor -> BoundingBox and
    `write` is the destination's box, or `None` when the shapes do not line up.
    """
    ranges = {}

    def narrow(t, lo, hi):
      prev = ranges.get(t)
      ranges[t] = (max(prev[0], lo), min(prev[1], hi)) if prev else (lo, hi)

    ops = list(getattr(descr, 'ops', []) or [])
    targets = list(getattr(descr, 'target', []) or [])
    dest = getattr(descr, 'dest', None)
    if dest is None or len(ops) != len(targets):
      return None
    for op, target in zip(ops, targets):
      if getattr(op, 'bbox', None) is None or len(target) != op.bbox.rank():
        return None
      for j, t in enumerate(target):
        narrow(t, op.bbox.lower()[j], op.bbox.upper()[j])
    for j in range(dest.bbox.rank()):
      narrow(j, dest.bbox.lower()[j], dest.bbox.upper()[j])

    reads = {}
    for op, target in zip(ops, targets):
      tensor = getattr(op, 'tensor', None)
      if tensor is None:
        continue
      lo = [ranges[t][0] + op.offset[j] for j, t in enumerate(target)]
      hi = [ranges[t][1] + op.offset[j] for j, t in enumerate(target)]
      box = BoundingBox(lo, hi)
      prev = reads.get(tensor)
      if prev is not None:
        box = BoundingBox([min(a, b) for a, b in zip(prev.lower(), lo)],
                          [max(a, b) for a, b in zip(prev.upper(), hi)])
      reads[tensor] = box
    wlo = [ranges[j][0] + dest.offset[j] for j in range(dest.bbox.rank())]
    whi = [ranges[j][1] + dest.offset[j] for j in range(dest.bbox.rank())]
    return reads, BoundingBox(wlo, whi)

  def _uncovered(self, key, read):
    """The first sub-box of `read` that no write covers, or None.

    Coordinate compression: cut every dimension at all the box boundaries that
    fall inside `read`.  Each resulting cell then lies either wholly inside or
    wholly outside every write box, so "is this cell covered" is an exact test
    and the whole check is exact rather than conservative.
    """
    boxes = self._eff_writes.get(key, [])
    rank = read.rank()
    if rank == 0 or not boxes or any(b.rank() != rank for b in boxes):
      return None
    cuts = []
    for j in range(rank):
      lo, hi = read.lower()[j], read.upper()[j]
      if lo >= hi:
        return None                      # empty read, nothing to cover
      pts = {lo, hi}
      for b in boxes:
        for v in (b.lower()[j], b.upper()[j]):
          if lo < v < hi:
            pts.add(v)
      cuts.append(sorted(pts))
    for corner in itertools.product(*[range(len(c) - 1) for c in cuts]):
      lo = [cuts[j][corner[j]] for j in range(rank)]
      hi = [cuts[j][corner[j] + 1] for j in range(rank)]
      if any(all(b.lower()[j] <= lo[j] and hi[j] <= b.upper()[j]
                 for j in range(rank)) for b in boxes):
        continue
      return BoundingBox(lo, hi)
    return None

  def _check_initialised(self):
    """Refuse to read a temporary where nothing ever wrote.

    A temporary is created by the kernel, so anything read outside what the
    kernel writes is whatever the shared or global allocation happened to
    contain.  Global inputs and outputs are exempt: an input is legitimately
    never written, and an output may hold a value the caller put there.

    Filling the gap with zeros is the obvious other answer, and the right one
    once a declaration instruction owns the buffer.  Until then this refuses,
    because a silently undefined summand is exactly the failure mode that took
    the longest to find in this area.
    """
    for tensor, read in self._eff_reads.items():
      if not getattr(tensor, 'is_tmp', False):
        continue
      key = id(tensor)
      if key not in self._eff_writes:
        raise GenerationError(
            f'{getattr(tensor, "alias", None) or tensor}: temporary is read '
            f'over {read} but never written')
      gap = self._uncovered(key, read)
      if gap is not None:
        raise GenerationError(
            f'{getattr(tensor, "alias", None) or tensor}: temporary is read '
            f'over {read} but {gap} is never written by any operation '
            f'(writes: {self._eff_writes[key]}). Zero-filling the gap needs a '
            f'declaration instruction that owns the buffer.')

  def _written_in_slices(self, tensor):
    """Does this tensor get assembled from several writes?

    Deferring the store is right while one operation writes the whole thing:
    the value can stay in registers and be handed straight to the next
    consumer.  With several writers each operation holds only its own slice, so
    the deferred entry --- there is one per name --- would keep whichever came
    last and silently lose the rest.  Those have to go into the shared buffer
    as they are produced.

    Several writers are *not* by themselves such a case.  An accumulation
    chain --- `d = a1 b1` followed by `d += a2 b2` and so on, which is what a
    yateto flux or ADER derivative kernel looks like --- has every writer
    covering the same box, each reading what the previous one produced.  There
    the last accumulator holds the whole tensor, deferring is exactly right,
    and forcing the store out per term costs a global round trip on every
    term.  So the question is not how many writers there are but whether any
    of them writes less than the union.

    Ask it of what each writer *actually* writes, not of what its descriptor
    declares.  `_analyze` intersects the range down to what the operands
    support, so an accumulation onto the whole box from an operand that spans
    half of it writes half --- the elastic ADER kernels are full of
    `t += Q_face * c`, all declaring the whole tensor and each covering the
    rows its own face touches.  Judged on the declared boxes those look like
    one writer covering everything, and the register image left behind holds
    only the last one's rows; the read that follows then wants the union and
    finds half of it.
    """
    boxes = self._eff_writes.get(id(tensor)) or self._dest_boxes.get(id(tensor), [])
    union = None
    for b in boxes:
      union = b if union is None else BoundingBox(
          [min(l, o) for l, o in zip(union.lower(), b.lower())],
          [max(u, o) for u, o in zip(union.upper(), b.upper())])
    if union is not None and any(
        b.lower()[j] > union.lower()[j] or b.upper()[j] < union.upper()[j]
        for b in boxes for j in range(union.rank())):
      return True
    # one writer is still not enough if it does not cover everything that gets
    # read back: `_analyze` intersects `_ns` down to what the operands support,
    # so a single store can easily be narrower than the declared destination
    written = self._dest_union.get(id(tensor))
    read = self._read_union.get(id(tensor))
    if written is None or read is None:
      return False
    return any(read.lower()[j] < written.lower()[j]
               or read.upper()[j] > written.upper()[j]
               for j in range(written.rank()))

  def _union_of(self, i):
    view = self._ops[i]
    union = self._operand_union.get(view.symbol.name)
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
    staged_src, _, _ = self._deferred_stores[name]
    if staged_src.stype not in (SymbolType.Register, SymbolType.Scratch):
      return True
    return staged_src.lead_dims == [lead_pos]

  def _resolve_reuse(self, i, name):
    """Decide whether the already-staged image can serve this operand.

    `_deferred_stores` is keyed by the global symbol's name and lives for the
    whole kernel, so any later operation on the same tensor inherits the
    earlier staging regardless of which slice it asks for.  Two things follow.

    The image is indexed in *its own* coordinates --- position `r` holds tensor
    element `r + shift` --- while the operand states its offset against the
    tensor, so it has to be rebased before the symbol is swapped underneath it.

    And the image only covers what its producer happened to stage.  When a
    consumer wants a different part, what can be done depends on where the
    authoritative copy lives:

      * a *preload* of a global input (`src is dest`, both the staged symbol):
        global memory still holds the value, so the entry is simply dropped and
        the ordinary path stages the range that is actually wanted;
      * a *pending store* of a temporary (`dest` is the global symbol): the
        value exists only in registers and the missing part was produced by
        some other instruction, so there is nothing to fall back on.

    Sizing the staging to the union of all consumers would avoid the second
    case entirely; until then it is refused loudly rather than miscompiled.
    """
    covered, shift = self._staged_view.get(name, (None, None))
    view = self._ops[i]
    if covered is None:
      if any(o != 0 for o in view.offset):
        raise GenerationError(
            f'{name}: reusing a staged image whose covered range was not '
            f'recorded, with a non-zero slicing offset {view.offset}')
      return True

    for j in range(view.bbox.rank()):
      lo = view.bbox.lower()[j] + view.offset[j] - shift[j]
      hi = view.bbox.upper()[j] + view.offset[j] - shift[j]
      if lo < covered.lower()[j] or hi > covered.upper()[j]:
        staged_src, staged_dest, _ = self._deferred_stores[name]
        # A *preload* stages a global input and records `src is dest`; global
        # memory still holds the value, so dropping the entry and staging the
        # wanted range afresh is sound.  A *pending store* has a register array
        # as `src` and the destination symbol as `dest` --- discriminating on
        # `dest.stype` instead misreads the shared-memory temporary that
        # _make_store creates, drops it, and loses both the value and the
        # symbol's data view.
        if staged_src is staged_dest:
          # preload of a global input: drop it and stage afresh
          del self._deferred_stores[name]
          self._staged_view.pop(name, None)
          return False
        raise GenerationError(
            f'{name}: operand wants [{lo},{hi}) in dim {j} but the staged '
            f'image only covers [{covered.lower()[j]},{covered.upper()[j]}), '
            f'and the value exists only in registers. Serving several '
            f'disjoint slices of one tensor needs the staging sized to their '
            f'union.')

    view.offset = [o - s for o, s in zip(view.offset, shift)]
    return True

  def _record_staged(self, name, covered, shift):
    self._staged_view[name] = (covered, list(shift))

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
    regsize = 1
    threads = self._num_threads
    # the dimension spread across lanes; the rest goes into slots
    lead_dim = [lead_pos]

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

    for d in range(bbox.rank()):
      dim = bbox.size(d)
      if d not in lead_dim or threads == 0:
        regsize *= dim
      else:
        # same slot count as DataView.get_dim_slots / _iregs / the addressing
        # side.  `ceil((u-l)/T)` disagrees once [l,u) straddles a block border.
        #
        # `lead_lanes` is the second factor and comes from the same function
        # the addressing side calls: one entry per slot when the lane is the
        # thread, `threads` entries when the work-item holds the whole wave.
        # Sizing this per thread while addressing it per work-item is what
        # made twenty-one kernels read past the end of an array -- and *that*
        # compiled, which is why it needs one source and not two.
        regsize *= (-(-bbox.upper()[d] // threads) - bbox.lower()[d] // threads
                    ) * DataView.lead_lanes(None, _explicit_simd(self._context), threads)
        threads //= dim
    name = self._name_registers()
    regmem = RegMemObject(name, regsize, spp=None if operand.obj.is_dense() else operand.obj.spp)
    registers = Symbol(name=name, stype=SymbolType.Register, obj=regmem)
    registers.lead_dims = [lead_pos]
    registers.num_threads = self._num_threads
    registers.datatype = self._context.fp_type
    self._scopes.add_symbol(registers)
    registerAlloc = RegisterAlloc(self._context, registers, regsize, 0.0)
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
    shr_mem_region = Symbol(name=self._name_shr_reg(),
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

  def _name_registers(self):
    name = f'r{self._counter}'
    self._counter += 1
    return name

  def _alloc_register_array(self):
    regsize = 1
    threads = self._num_threads
    # One statement of the lane axis, used twice below: to count the register
    # slots, and -- at the bottom -- to tell the symbol.  It was two, and the
    # symbol's half was the constructor default rather than anything said
    # here, so a change to one would not have moved the other.
    #
    # 0 because `MultilinearDescr._lead_dim` aligns the thread count to the
    # destination's axis 0, and the whole multilinear path is built on that.
    # The commented-out alternative that stood here, `[t for t in
    # self._descr.target[0] if t >= 0]`, is where a per-operand answer would
    # come from once there is one.
    lead_pos = 0
    lead_dim = [lead_pos]

    # This array holds the operation's *result*, so it is sized from what the
    # operation writes: the destination's box, in the shifted origin the
    # compute and the store both index it in.  `_analyze` may narrow the range
    # below that --- an operand covering less than the destination declares ---
    # which leaves slots unused, and that is harmless.
    #
    # An accumulation used to take the *bias image's* box instead.  That is
    # what the operation reads, not what it writes, and the two part company
    # whenever the image does not match the destination: a broadcast leaves a
    # rank-1 image behind for a rank-2 destination, and the array came out
    # with one slot where the store walks three.  The destination's box says
    # the same thing as a matching image and stays right when it does not
    # match, so ask it directly.
    bbox = self._dest_obj.bbox
    shift = self._theta

    for d in range(bbox.rank()):
      dim = bbox.size(d)
      if d not in lead_dim or threads == 0:
        regsize *= dim
      else:
        # the accumulator is indexed in the shifted origin, so the slot count
        # follows the shifted range.  Straddling one more block boundary is
        # exactly the price of not needing a shuffle.
        r_start = (bbox.lower()[d] + shift) // threads
        r_end = (bbox.upper()[d] + shift + threads - 1) // threads
        regsize *= (r_end - r_start) * DataView.lead_lanes(
            None, _explicit_simd(self._context), threads)
        threads //= dim # TODO?
    name = self._name_registers()
    regmem = RegMemObject(name, regsize)
    registers = Symbol(name=name, stype=SymbolType.Register, obj=regmem)
    registers.lead_dims = [lead_pos]
    registers.num_threads = self._num_threads
    registers.datatype = self._context.fp_type
    self._scopes.add_symbol(registers)
    registerAlloc = RegisterAlloc(self._context, registers, regsize, 0.0)
    self._instructions.append(registerAlloc)
    return registers

  def _target_image_fits(self, name):
    """Does the image staged under `name` hold exactly this destination?

    `_deferred_stores` and `_staged_view` are keyed by symbol name and live for
    the whole kernel, so what a destination finds under its name may have been
    staged for something else entirely --- most often an *operand* read of a
    different slice of the same tensor.  Taking it as the accumulation bias
    then reads the wrong elements, from the wrong lanes: in the poroelastic
    space-time predictor `m2[:, 11] += ...` picked up the image of
    `m2[:, 12]`, staged one descriptor earlier, and accumulated onto that.

    An accumulation chain onto the same box --- the case the reuse exists for
    --- has the staged region equal to the destination's, so require that.
    """
    staged = self._staged_view.get(name)
    if staged is None:
      return False
    covered, shift = staged
    if covered is None or covered.rank() != self._dest_obj.bbox.rank():
      return False
    want = ([l + o for l, o in zip(self._dest_obj.bbox.lower(), self._dest_obj.offset)],
            [u + o for u, o in zip(self._dest_obj.bbox.upper(), self._dest_obj.offset)])
    have = ([l + s for l, s in zip(covered.lower(), shift)],
            [u + s for u, s in zip(covered.upper(), shift)])
    return have == want

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
    if (dest_symbol.name in self._deferred_stores
        and not self._target_image_fits(dest_symbol.name)):
      # staged for something else: it cannot serve as this destination.  A
      # pending writeback still has to reach memory, which is exactly what
      # `_invalidate_residency` does; a preload is simply dropped and the
      # ordinary path below stages the region this operation needs.
      self._invalidate_residency(dest_symbol.name)
    if dest_symbol.name in self._deferred_stores:
      dest_registers,_,_ = self._deferred_stores[dest_symbol.name]
      return dest_registers
    elif self._atomic_update and prev:
      # should be found in the previous step already
      return None
    elif self._preload_registers and dest_symbol.stype == SymbolType.Global and not self._atomic_update and not next:
      symbol, load_op = self._make_loader_and_symbol_reg(
          self._dest_preload_view(dest_symbol), False)
      self._deferred_stores[dest_symbol.name] = symbol.symbol, symbol.symbol, None
      self._record_staged(dest_symbol.name,
                          symbol.symbol.data_view.get_bbox(),
                          [0] + list(self._dest_obj.offset[1:]))
      self._instructions.append(load_op)
      return symbol.symbol
    elif self._preload_shmem and dest_symbol.stype == SymbolType.Global and not self._atomic_update and not next:
      symbol, load_op = self._make_loader_and_symbol(
          self._dest_preload_view(dest_symbol), None)
      self._deferred_stores[dest_symbol.name] = symbol.symbol, symbol.symbol, None
      self._record_staged(dest_symbol.name,
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
    """The register image of `name` is about to stop being the newest copy.

    `_get_target_symbol` preloads a global destination into registers and
    records that image so the *next* operation on the same tensor can
    accumulate straight into it.  That is only sound while the image stays
    the newest copy.  As soon as this operation writes memory from a
    different array --- which is what the eager-store paths below do --- the
    recorded image is one accumulation step behind, and leaving it in place
    hands every later `+=` a stale bias: each step then computes
    `preload + own term` and overwrites the previous one, so the destination
    ends up holding the first write plus the *last* term and nothing in
    between.

    Which of the two kinds of entry this is decides what "invalidate" means,
    on the same `src is dest` discriminator `_resolve_reuse` uses.  A preload
    is a copy of something global memory still holds, so it is simply
    dropped.  A pending writeback is the *only* copy of a result that has not
    reached memory yet; dropping that would lose it, so it goes out first.
    """
    entry = self._deferred_stores.pop(name, None)
    promise = self._promised.pop(name, None)
    _, shift = self._staged_view.pop(name, (None, None))
    if entry is None:
      return
    store_regs, store_dest, update = entry
    if store_regs is store_dest:
      return                                    # preload: memory still has it
    if store_dest.stype == SymbolType.Global:
      if shift is None:
        shift = [0] * store_dest.data_view.rank()
      self._instructions.append(StoreRegToGlb(context=self._context,
                                              src=store_regs,
                                              dest=store_dest,
                                              num_threads=self._num_threads,
                                              lead_width=self._lead_width,
                                              atomic=update,
                                              dest_offset=shift,
                                              dest_bbox=promise,
                                              zero_fill=promise is not None))
    else:
      self._instructions.append(StoreRegToShr(context=self._context,
                                              src=store_regs,
                                              dest=store_dest,
                                              shr_mem=self._shr_mem,
                                              num_threads=self._num_threads))

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
        if self._written_in_slices(self._dest_obj.tensor):
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
                                                  dest_bbox=self._dest_union.get(id(self._dest_obj.tensor)),
                                                  dest_offset=self._store_offset()))
          return
        # see note below (but update to the new temp regs)
        self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol, None)
        self._record_staged(dest_symbol.name,
                          # the *actual* range the accumulator ended up with:
                          # _analyze intersects the operands, so _ns can be
                          # strictly smaller than the declared destination box
                          self._temp_regs.data_view.get_bbox(),
                            self._store_offset())
      elif dest_symbol.stype == SymbolType.Global:
        in_slices = self._written_in_slices(self._dest_obj.tensor)
        can_use_atomic = self._atomic_update and self._add and (dest_symbol.name not in self._deferred_stores or self._deferred_stores[dest_symbol.name][2] is not None)
        # A destination assembled from several writes cannot be kept in
        # registers: `_deferred_stores` holds one entry per name, so a second
        # slice would displace the first and its whole contribution would be
        # computed and thrown away.
        #
        # Atomics are exempt from that *only* if they really do go out on
        # their own.  Deferring one is what makes it collide with the next
        # slice, so with several writers the update is emitted here instead
        # of at the epilogue; there is nothing to serialise, since an atomic
        # add is order-independent by construction.  With a single covering
        # writer deferring still pays --- it saves the read-modify-write.
        if can_use_atomic and in_slices:
          self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  atomic=True,
                                                  dest_offset=self._store_offset(),
                                                  dest_bbox=self._promised_box(),
                                                  zero_fill=not self._add))
        elif can_use_atomic or (self._use_registers_always and not in_slices):
          update = True if can_use_atomic else None
          self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol, update)
          self._promised[dest_symbol.name] = (
              self._promised_box() if not self._add else None)
          self._record_staged(dest_symbol.name,
                          # the *actual* range the accumulator ended up with:
                          # _analyze intersects the operands, so _ns can be
                          # strictly smaller than the declared destination box
                          self._temp_regs.data_view.get_bbox(),
                              self._store_offset())
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

      dest_symbol = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=self._dest_obj.tensor)

      # do not swap matrix layout in global memory until we need to
      self._scopes.add_symbol(dest_symbol)
      if self._written_in_slices(self._dest_obj.tensor):
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads,
                                                lead_width=self._lead_width,
                                                dest_bbox=self._dest_union.get(id(self._dest_obj.tensor)),
                                                dest_offset=self._store_offset()))
        return
      self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol, None)
      self._record_staged(dest_symbol.name,
                          # the *actual* range the accumulator ended up with:
                          # _analyze intersects the operands, so _ns can be
                          # strictly smaller than the declared destination box
                          self._temp_regs.data_view.get_bbox(),
                          self._store_offset())

  def _insert_sync_block(self):
    self._instructions.append(SyncThreads(context=self._context,
                                          num_threads_per_mult=self._num_threads))

  def _name_shr_reg(self):
    name = f's{self._counter_shr_reg}'
    self._counter_shr_reg += 1
    return name

  def build_epilogue(self):
    self._reset()
    for name, (store_regs, store_global, update) in self._deferred_stores.items():
      if store_global.stype == SymbolType.Global:
        # each entry carries its own offset; _store_offset() would hand out the
        # last build's, which is only right when there is a single entry
        _, shift = self._staged_view.get(
            name, (None, [0] * store_global.data_view.rank()))
        self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=store_regs,
                                                  dest=store_global,
                                                  num_threads=self._num_threads,
                                                  lead_width=self._lead_width,
                                                  atomic=update,
                                                  dest_offset=shift,
                                                  dest_bbox=self._promised.get(name),
                                                  zero_fill=self._promised.get(name) is not None))
