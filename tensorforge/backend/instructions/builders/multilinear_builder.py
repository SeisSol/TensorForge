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

    needs_reload = (transpose or not has_lead_dim) and not prefer_broadcast
    needs_reload2 = transpose or not has_lead_dim

    name = self._ops[i].symbol.name
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
          staged, load_op = self._make_loader_and_symbol(self._stage_view(i), is_transpose=self._descr.permute[i])
          self._mem_regions[i] = self._staged_region(i, staged)
          self._loaders_cache[self._mem_regions[i]] = load_op
          self._instructions.append(load_op)
        else:
          if self._preload_registers and self._ops[i].symbol.obj.addressing != Addressing.NONE:
            # only register-preload dense matrices for now
            staged, load_op = self._make_loader_and_symbol_reg(self._stage_view(i), linearize=needs_reload2)
            self._mem_regions[i] = self._staged_region(i, staged)
            self._deferred_stores[self._ops[i].symbol.name] = self._mem_regions[i].symbol, self._mem_regions[i].symbol, None
            self._record_staged(self._ops[i].symbol.name,
                                self._mem_regions[i].symbol.data_view.get_bbox(),
                                [0] * self._ops[i].bbox.rank())
            self._instructions.append(load_op)
          elif self._preload_shmem and self._ops[i].symbol.obj.addressing != Addressing.NONE:
            # only register-preload dense matrices for now
            staged, load_op = self._make_loader_and_symbol(self._stage_view(i), None)
            self._mem_regions[i] = self._staged_region(i, staged)
            self._deferred_stores[self._ops[i].symbol.name] = self._mem_regions[i].symbol, self._mem_regions[i].symbol, None
            self._record_staged(self._ops[i].symbol.name,
                                self._mem_regions[i].symbol.data_view.get_bbox(),
                                [0] * self._ops[i].bbox.rank())
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
    for descr in descr_list:
      if not isinstance(descr, MultilinearDescr):
        continue
      for op in descr.ops:
        tensor = getattr(op, 'tensor', None)
        if tensor is None:
          continue
        symbol = self._scopes.get_symbol(tensor)
        if symbol is None:
          continue
        lower = [l + o for l, o in zip(op.bbox.lower(), op.offset)]
        upper = [u + o for u, o in zip(op.bbox.upper(), op.offset)]
        prev = self._operand_union.get(symbol.name)
        if prev is not None:
          lower = [min(a, b) for a, b in zip(prev.lower(), lower)]
          upper = [max(a, b) for a, b in zip(prev.upper(), upper)]
        self._operand_union[symbol.name] = BoundingBox(lower, upper)

  def _stage_view(self, i):
    """The operand as the staging load should see it: the whole union, in
    storage coordinates, so the copy is an identity mapping."""
    view = self._ops[i]
    union = self._operand_union.get(view.symbol.name)
    if union is None:
      union = BoundingBox([l + o for l, o in zip(view.bbox.lower(), view.offset)],
                          [u + o for u, o in zip(view.bbox.upper(), view.offset)])
    return SymbolView(view.symbol, union, [0] * union.rank())

  def _staged_region(self, i, staged):
    """The staged symbol as the *compute* site should see it: this operand's
    own logical box and its own offset, unchanged."""
    return SymbolView(staged.symbol, self._ops[i].bbox, self._ops[i].offset)

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
        _, staged_dest, _ = self._deferred_stores[name]
        if staged_dest.stype != SymbolType.Global:
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

  def _make_loader_and_symbol_reg(self, opview, linearize) -> Tuple[Symbol, GlbToRegLoader]:
    operand = opview.symbol
    regsize = 1
    threads = self._num_threads
    lead_dim = [0] # [t for t in self._descr.target[0] if t >= 0]

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

  def _make_store(self):
    if self._dest_obj.tensor in self._scopes:
      dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
      if dest_symbol.stype == SymbolType.SharedMem:
        #self._instructions.append(StoreRegToShr(context=self._context,
        #                                        src=self._temp_regs,
        #                                        dest=dest_symbol,
        #                                        shr_mem=self._shr_mem,
        #                                        num_threads=self._num_threads))
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
        if self._use_registers_always or can_use_atomic:
          update = True if can_use_atomic else None
          self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol, update)
          self._record_staged(dest_symbol.name,
                          # the *actual* range the accumulator ended up with:
                          # _analyze intersects the operands, so _ns can be
                          # strictly smaller than the declared destination box
                          self._temp_regs.data_view.get_bbox(),
                              self._store_offset())
        else:
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
