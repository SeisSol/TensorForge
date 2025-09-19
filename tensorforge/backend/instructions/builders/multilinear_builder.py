from typing import Tuple, Dict, List
from tensorforge.common.context import Context
from tensorforge.backend.scopes import Scopes
from tensorforge.backend.symbol import Symbol, SymbolType, SymbolView
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.memory.load import GlbToShrLoader
from tensorforge.backend.instructions.clear_registers import ClearRegisters
from tensorforge.backend.instructions.memory.store import StoreRegToGlb, StoreRegToShr, StoreRegToReg
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.backend.instructions.compute.multilinear import MultilinearInstruction
from tensorforge.backend.instructions.compute.multilinearmulti import MultilinearMultiInstruction
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.common.exceptions import InternalError
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.backend.instructions.builders.allocator_builder import AbstractBuilder
from tensorforge.common.operation import AddOperator, MulOperator
from tensorforge.backend.data_types import RegMemObject
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction


class MultilinearBuilder(AbstractBuilder):
  GemmClass = None

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

    self._use_registers_always = False
    self._temporary_registers = False
    self._deferred_stores = {}
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
    self._temp_regs = self._alloc_register_array()
    self._make_compute()
    self._insert_sync_block()
    self._make_store()
    self._insert_sync_block()

  # TODO: check if we always can allow a direct global memory load
  def _make_load_op(self, i):
    has_lead_dim = 0 in self._descr.target[i]
    transpose = self._descr.permute[i] != [j for j in range(len(self._descr.target[i]))]

    if self._ops[i].symbol.name in self._deferred_stores:
      if not has_lead_dim or transpose:
        src, dest = self._deferred_stores[self._ops[i].symbol.name]
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=src,
                                                dest=dest,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads))
        del self._deferred_stores[self._ops[i].symbol.name]
        self._ops[i].symbol = dest
      else:
        self._ops[i].symbol, _ = self._deferred_stores[self._ops[i].symbol.name]

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
        if transpose or not has_lead_dim or small_lead:
          self._mem_regions[i], load_op = self._make_loader_and_symbol(self._ops[i].symbol, is_transpose=self._descr.permute[i])
          self._loaders_cache[self._mem_regions[i]] = load_op
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
              self._mem_regions[i] = SymbolView(prev_loader.get_src())
            else:
              # means: data cannot be reused. we need to reload it again and traspose on the fly.
              # additionally, we need to remove aliased symbol to avoid clashes
              # self._scopes.delete_symbol(self._ops[i].symbol)
              self._scopes.add_scope()
              prev_symbol = prev_loader.get_src()
              self._mem_regions[i], load_op = self._make_loader_and_symbol(prev_symbol, is_transpose=self._descr.permute[i])
              self._loaders_cache[self._mem_regions[i]] = load_op
              self._instructions.append(load_op)
          else:
            # means: data can be fully reused
            self._mem_regions[i] = self._ops[i]

        else:
          self._mem_regions[i] = self._ops[i]
      else:
        raise InternalError(f'gemm-builder: op{i} ({self._ops[i].symbol.name}) must be either in shr or glb mem, given: {self._ops[i].symbol.stype}')

  def _make_loader_and_symbol(self, operand, is_transpose) -> Tuple[Symbol, GlbToShrLoader]:
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
    return SymbolView(shr_mem_region), load_op

  def _name_registers(self):
    name = f'r{self._counter}'
    self._counter += 1
    return name

  def _alloc_register_array(self):
    regsize = 1
    threads = self._num_threads
    lead_dim = [0] # [t for t in self._descr.target[0] if t >= 0]
    for d, dim in enumerate(self._dest_obj.bbox.sizes()):
      if d not in lead_dim or threads == 0:
        regsize *= dim
      else:
        regsize *= (dim + threads - 1) // threads
        threads //= dim
    name = self._name_registers()
    regmem = RegMemObject(name, regsize)
    registers = Symbol(name=name, stype=SymbolType.Register, obj=regmem)
    registers.num_threads = self._num_threads
    self._scopes.add_symbol(registers)
    registerAlloc = RegisterAlloc(self._context, registers, regsize, 0.0)
    self._instructions.append(registerAlloc)
    return registers

  def _get_target_symbol(self):
    dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
    if dest_symbol.name in self._deferred_stores:
      dest_registers,_ = self._deferred_stores[dest_symbol.name]
      return dest_registers
    else:
      return dest_symbol

  def _make_compute(self):
    self._instructions.append(MultilinearInstruction(context=self._context,
                                   ops=self._mem_regions,
                                   target=self._descr.target,
                                   dest=self._temp_regs,
                                   prefer_align=False,#self._descr.prefer_align,
                                   num_threads=self._num_threads,
                                   prev=self._get_target_symbol() if self._add else None,
                                   productOperation=MulOperator(),
                                   sumOperation=AddOperator()))

  def _make_store(self):
    if self._dest_obj.tensor in self._scopes:
      dest_symbol = self._scopes.get_symbol(self._dest_obj.tensor)
      if dest_symbol.stype == SymbolType.SharedMem:
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads))
      elif dest_symbol.stype == SymbolType.Global:
        if self._use_registers_always:
          self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol)
        else:
          self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=self._temp_regs,
                                                  dest=dest_symbol,
                                                  alpha=1,#self._descr.alpha,
                                                  beta=0,#self._descr.beta,
                                                  num_threads=self._num_threads))
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
      self._deferred_stores[dest_symbol.name] = (self._temp_regs, dest_symbol)

  def _insert_sync_block(self):
    self._instructions.append(SyncThreads(context=self._context,
                                          num_threads_per_mult=self._num_threads))

  def _name_shr_reg(self):
    name = f's{self._counter_shr_reg}'
    self._counter_shr_reg += 1
    return name

  def build_epilogue(self):
    self._reset()
    for store_regs, store_global in self._deferred_stores.values():
      if store_global.stype == SymbolType.Global:
        self._instructions.append(StoreRegToGlb(context=self._context,
                                                  src=store_regs,
                                                  dest=store_global,
                                                  alpha=1,#self._descr.alpha,
                                                  beta=0,#self._descr.beta,
                                                  num_threads=self._num_threads))
