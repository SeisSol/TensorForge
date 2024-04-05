from typing import Tuple, Dict, Union, List
from kernelforge.common.context import Context, VM
from kernelforge.backend.scopes import Scopes
from kernelforge.backend.symbol import Symbol, SymbolType
from kernelforge.backend.instructions.allocate import RegisterAlloc
from kernelforge.backend.instructions.memory.load import GlbToShrLoader
from kernelforge.backend.instructions.clear_registers import ClearRegisters
from kernelforge.backend.instructions.memory.store import StoreRegToGlb, StoreRegToShr
from kernelforge.backend.instructions.sync_threads import SyncThreads
from kernelforge.backend.instructions.compute.multilinear import MultilinearInstruction
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.exceptions import InternalError
from kernelforge.generators.descriptions import MultilinearDescr
from kernelforge.backend.instructions.builders.allocator_builder import AbstractBuilder
from kernelforge.common.operation import AddOperator, MulOperator

class MultilinearBuilder(AbstractBuilder):
  GemmClass = None

  def __init__(self,
               context: Context,
               scopes: Scopes,
               register_array: Symbol,
               shr_mem: Symbol,
               num_threads: int):
    super(MultilinearBuilder, self).__init__(context, scopes)
    self._temp_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._counter = 0
    self._loaders_cache: Dict[Symbol, AbstractShrMemLoader] = {}

    self._ops = None
    self._dest_obj = None
    self._descr = None

    self._mem_regions = None

    self._dest_regs = self._temp_regs

    self._deferred_stores = {}

  def build(self, ops: List[Symbol], dest_obj: Tensor, descr: MultilinearDescr):
    self._reset()

    self._ops = ops
    self._dest_obj = dest_obj
    self._descr = descr

    self._add = descr.add

    self._mem_regions = [None] * len(self._ops)

    for i in range(len(self._ops)):
        self._make_load_op(i)
    self._insert_sync_threads()
    self._check_register_array()
    self._make_gemm()
    self._insert_sync_threads()
    self._make_store()
    self._insert_sync_threads()
    self._clear_registers()

    # TODO: check if we always can allow a direct global memory load
  def _make_load_op(self, i):
    if self._ops[i].stype == SymbolType.Scalar or self._ops[i].stype == SymbolType.Data:
      self._mem_regions[i] = self._ops[i]
    else:
      has_lead_dim = 0 in self._descr.target[i]
      transpose = self._descr.permute[i] != [j for j in range(len(self._descr.target[i]))]
      if has_lead_dim:
        lead_idx = self._descr.target[i].index(0)

        # heuristic. We may need to store the L2 load granularity or similar
        small_lead = self._ops[i].data_view.shape[self._descr.permute[i][lead_idx]] < self._context.get_vm().get_hw_descr().vec_unit_length
      else:
        small_lead = False

      # This is a heuristics implemented because having too sparse matrices can increase bank conflicts
      # And this heuristical optimization should remain until a better shared memory loader is implemented
      sparse = self._ops[i].obj.sparsity() < 0.65

      if self._ops[i].stype == SymbolType.Global:
        if transpose or not has_lead_dim or small_lead:
          self._mem_regions[i], load_op = self._make_loader_and_symbol(self._ops[i], is_transpose=self._descr.permute[i])
          self._loaders_cache[self._mem_regions[i]] = load_op
          self._instructions.append(load_op)
        else:
          # Note: operand will reside in glb. mem for gemm operation
          self._mem_regions[i] = self._ops[i]

      elif self._ops[i].stype == SymbolType.SharedMem or self._ops[i].stype == SymbolType.Register:
        if self._ops[i] in self._loaders_cache.keys():
          # Note: this condition means the symbol `self._ops[i]` has been loaded
          # to shr. mem. before. Let's check whether loaded data can be reused
          prev_loader = self._loaders_cache[self._ops[i]]

          # we only need to reload/globally load, if we even need a leading dimension
          if self._descr.permute[i] != prev_loader.get_permute() and has_lead_dim:
            if not transpose:
              # means: data loaded to shr. mem. cannot be reused. Because `op1` not need to be transposed
              # we don't need to load it to shr. mem. Instead, it will be taken from glb. mem.
              # we don't need delete previous (aliased) symbol
              self._mem_regions[i] = prev_loader.get_src()
            else:
              # means: data cannot be reused. we need to reload it again and traspose on the fly.
              # additionally, we need to remove aliased symbol to avoid clashes
              # self._scopes.delete_symbol(self._ops[i])
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
        raise InternalError(f'gemm-builder: op{i} ({self._ops[i].name}) must be either in shr or glb mem.')

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
    return shr_mem_region, load_op

  def _name_registers(self):
    name = f'r{self._counter}'
    self._counter += 1
    return name

  def _alloc_register_array(self):
    registers = Symbol(name=self._name_registers(), stype=SymbolType.Register, obj=self._dest_obj)
    regsize = 1
    for d in range(registers.data_view.rank()):
      if d not in registers.lead_dims:
        regsize *= registers.data_view.get_dim_size(d)
    self._scopes.add_symbol(registers)
    registerAlloc = RegisterAlloc(self._context, registers, regsize)
    self._instructions.append(registerAlloc)
    return registers
    # self._dest_regs = registers

  def _check_register_array(self):
    if self._dest_regs.stype != SymbolType.Register:
      raise InternalError('gemm-builder: reg_array must be in registers')

  def _make_gemm(self):
    self._instructions.append(MultilinearInstruction(context=self._context,
                                   ops=self._mem_regions,
                                   target=self._descr.target,
                                   dest=self._temp_regs,
                                   prefer_align=False,#self._descr.prefer_align,
                                   num_threads=self._num_threads,
                                   prev=self._scopes.get_symbol(self._dest_obj) if self._add else None,
                                   productOperation=MulOperator(),
                                   sumOperation=AddOperator()))

  def _make_store(self):
    if self._dest_obj in self._scopes:
      dest_symbol = self._scopes.get_symbol(self._dest_obj)
      if dest_symbol.stype == SymbolType.SharedMem:
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads))
      elif dest_symbol.stype == SymbolType.Global:
        # if dest_symbol.name not in self._deferred_stores:
        #   dest_registers = self._alloc_register_array()
        #   self._deferred_stores[dest_symbol.name] = dest_registers
        # dest_registers = self._deferred_stores[dest_symbol.name]
        self._instructions.append(StoreRegToGlb(context=self._context,
                                                src=self._temp_regs,
                                                dest=dest_symbol,
                                                alpha=1,#self._descr.alpha,
                                                beta=0,#self._descr.beta,
                                                num_threads=self._num_threads))
      elif dest_symbol.stype == SymbolType.Register:
        pass
        #self._instructions.append(StoreRegToGlb(context=self._context,
        #                                        src=self._temp_regs,
        #                                        dest=dest_symbol,
        #                                        num_threads=self._num_threads))
      else:
        raise InternalError(f'gemm-builder: `res` must be either in shr. or glb. mem., given: {dest_symbol.stype}')
    else:
      if not self._dest_obj.is_tmp:
        raise InternalError(f'gemm-buider: `res` is not in scopes and thus must be tmp')

      dest_symbol = Symbol(name=self._name_shr_reg(),
                           stype=SymbolType.SharedMem,
                           obj=self._dest_obj)
      self._scopes.add_symbol(dest_symbol)
      self._instructions.append(StoreRegToShr(context=self._context,
                                              src=self._temp_regs,
                                              dest=dest_symbol,
                                              shr_mem=self._shr_mem,
                                              num_threads=self._num_threads))

  def _clear_registers(self):
    self._instructions.append(ClearRegisters(context=self._context, src=self._temp_regs))

  def _insert_sync_threads(self):
    self._instructions.append(SyncThreads(context=self._context,
                                          num_threads_per_mult=self._num_threads))

  def _name_shr_reg(self):
    name = f's{self._counter}'
    self._counter += 1
    return name
