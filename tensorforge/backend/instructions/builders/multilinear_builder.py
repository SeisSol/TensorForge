from typing import Tuple, Dict, List
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Addressing
from tensorforge.backend.scopes import Scopes
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.memory.load import GlbToShrLoader, GlbToRegLoader
from tensorforge.backend.instructions.clear_registers import ClearRegisters
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
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction


class MultilinearBuilder(AbstractBuilder):
  def __init__(self,
               context: Context,
               scopes: Scopes,
               shr_mem: Symbol,
               num_threads: int):
    super(MultilinearBuilder, self).__init__(context, scopes)
    self._shr_mem = shr_mem
    self._num_threads = num_threads

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
    # what each deferred/staged image actually holds, so a later consumer
    # can tell whether reusing it is sound: position `r` of the image holds
    # tensor element `r + shift`, for `r` inside `covered`.
    self._staged_view = {}
    # tensor symbol name -> union of every operand access, in tensor
    # storage coordinates.  Filled by plan(); see there.
    self._operand_union = {}
    self._temporaries = {}

  def build(self, ops: List[Symbol], dest_obj: Tensor, descr: MultilinearDescr):
    self._reset()

    self._ops = ops
    self._dest_obj = dest_obj
    self._descr = descr

    self._add = descr.add

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

    if self._ops[i].symbol.stype == SymbolType.Scalar or self._ops[i].symbol.stype == SymbolType.Data:
      self._mem_regions[i] = self._ops[i]
    else:

      if has_lead_dim:
        lead_idx = self._descr.target[i].index(0)

        # heuristic. We may need to store the L2 load granularity or similar
        small_lead = False # self._ops[i].symbol.data_view.shape[self._descr.permute[i][lead_idx]] < self._context.get_vm().get_hw_descr().vec_unit_length
      else:
        small_lead = False

      # This is a heuristics implemented because having too sparse matrices can increase bank conflicts
      # And this heuristical optimization should remain until a better shared memory loader is implemented
      # sparse = self._ops[i].symbol.obj.sparsity() < 0.65

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
    self._dest_writers = {}
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
        self._dest_writers[key] = self._dest_writers.get(key, 0) + 1
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
    """
    boxes = self._dest_boxes.get(id(tensor), [])
    union = self._dest_union.get(id(tensor))
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
        regsize *= -(-bbox.upper()[d] // threads) - bbox.lower()[d] // threads
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
    lead_dim = [0] # [t for t in self._descr.target[0] if t >= 0]

    # TODO: shrink to enumerate(self._dest_obj.bbox.sizes())
    if self._add:
      bbox = self._get_target_symbol().data_view._bbox
    else:
      bbox = self._dest_obj.bbox

    for d in range(bbox.rank()):
      dim = bbox.size(d)
      if d not in lead_dim or threads == 0:
        regsize *= dim
      else:
        # the accumulator is indexed in the shifted origin, so the slot count
        # follows the shifted range.  Straddling one more block boundary is
        # exactly the price of not needing a shuffle.
        r_start = (bbox.lower()[d] + self._theta) // threads
        r_end = (bbox.upper()[d] + self._theta + threads - 1) // threads
        regsize *= r_end - r_start
        threads //= dim # TODO?
    name = self._name_registers()
    regmem = RegMemObject(name, regsize)
    registers = Symbol(name=name, stype=SymbolType.Register, obj=regmem)
    registers.num_threads = self._num_threads
    registers.datatype = self._context.fp_type
    self._scopes.add_symbol(registers)
    registerAlloc = RegisterAlloc(self._context, registers, regsize, 0.0)
    self._instructions.append(registerAlloc)
    return registers

  def _get_target_symbol(self, prev=False, next=False):
    dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
    if dest_symbol is None:
      return None
    if dest_symbol.name in self._deferred_stores:
      dest_registers,_,_ = self._deferred_stores[dest_symbol.name]
      return dest_registers
    elif self._atomic_update and prev:
      # should be found in the previous step already
      return None
    elif self._preload_registers and dest_symbol.stype == SymbolType.Global and not self._atomic_update and not next:
      symbol, load_op = self._make_loader_and_symbol_reg(
          SymbolView(dest_symbol, self._dest_obj.bbox, self._dest_obj.offset), False)
      self._deferred_stores[dest_symbol.name] = symbol.symbol, symbol.symbol, None
      self._record_staged(dest_symbol.name,
                          symbol.symbol.data_view.get_bbox(),
                          self._dest_obj.offset)
      self._instructions.append(load_op)
      return symbol.symbol
    elif self._preload_shmem and dest_symbol.stype == SymbolType.Global and not self._atomic_update and not next:
      symbol, load_op = self._make_loader_and_symbol(
          SymbolView(dest_symbol, self._dest_obj.bbox, self._dest_obj.offset), None)
      self._deferred_stores[dest_symbol.name] = symbol.symbol, symbol.symbol, None
      self._record_staged(dest_symbol.name,
                          symbol.symbol.data_view.get_bbox(),
                          self._dest_obj.offset)
      self._instructions.append(load_op)
      return symbol.symbol
    else:
      return dest_symbol

  def _make_compute(self):
    self._instructions.append(MultilinearInstruction(context=self._context,
                                   ops=self._mem_regions,
                                   target=self._descr.target,
                                   dest=self._temp_regs,
                                   num_threads=self._num_threads,
                                   prev=self._get_target_symbol(True) if self._add else None,
                                   next=self._get_target_symbol(True, True),
                                   productOperation=MulOperator(),
                                   sumOperation=AddOperator(),
                                   dest_obj=self._dest_obj,
                                   theta=self._theta))

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
    """Drop the recorded register image of `name`: memory is now newer.

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
    """
    self._deferred_stores.pop(name, None)
    self._staged_view.pop(name, None)

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
        can_use_atomic = self._atomic_update and self._add and (dest_symbol.name not in self._deferred_stores or self._deferred_stores[dest_symbol.name][2] is not None)
        # same reasoning as for shared memory: a destination assembled from
        # several writes cannot be kept in registers, since the deferred
        # entry holds one slice and drops the rest.  Atomics are exempt --
        # there each write goes out on its own anyway.
        if can_use_atomic or (self._use_registers_always
                              and not self._written_in_slices(self._dest_obj.tensor)):
          update = True if can_use_atomic else None
          self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol, update)
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
                                                  atomic=None,
                                                  dest_offset=self._store_offset()))
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
                                                  atomic=update,
                                                  dest_offset=shift))
