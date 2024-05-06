from typing import Tuple, Dict, Union
from kernelforge.common.context import Context, VM
from kernelforge.backend.scopes import Scopes
from kernelforge.backend.symbol import Symbol, SymbolType
from kernelforge.backend.instructions.loaders import shm_mem_loader_factory, AbstractShrMemLoader
from kernelforge.backend.instructions.loaders.shr_mem_loader import ShrMemLoaderType
from kernelforge.backend.instructions.clear_registers import ClearRegisters
from kernelforge.backend.instructions.store import StoreRegToGlb, StoreRegToShr
from kernelforge.backend.instructions.sync_block import SyncThreads
from kernelforge.backend.instructions.dense_dense_gemm import ShrMemBasedDenseGemm, RegisterOnlyDenseGemm
from kernelforge.backend.instructions.sparse_dense_gemm import ShrMemBasedSparseDenseGemm, RegisterOnlySparseDenseGemm
from kernelforge.backend.instructions.dense_sparse_gemm import ShrMemBasedDenseSparseGemm, RegisterOnlyDenseSparseGemm
from kernelforge.backend.instructions.csa import CSA
from kernelforge.common.matrix.dense import DenseMatrix
from kernelforge.common.matrix.sparse import SparseMatrix
from kernelforge.common.matrix.matrix import Matrix
from kernelforge.common.exceptions import InternalError
from kernelforge.generators.descriptions import GemmDescr, CSADescr, PointwiseDescr, MultilinearDescr, ReductionDescr
from kernelforge.backend.instructions.builders.allocator_builder import AbstractBuilder
from kernelforge.backend.instructions.pointwise import build_pointwise
from .type import GemmKernelType

class GemmKernelsFactory:
  def __init__(self, descr):
    self._descr = descr
    if isinstance(descr, PointwiseDescr):
      return
    if isinstance(descr, MultilinearDescr):
      raise NotImplementedError('To replace and implement the LoG operator, CSA etc.')
    if isinstance(descr, ReductionDescr):
      raise NotImplementedError('To replace and implement the IndexSum, among others')
    if isinstance(descr, GemmDescr):
      self._gemm_kernel_type = descr.kernel_type
      self._mat_b = descr.mat_b
      if isinstance(self._mat_b, SparseMatrix):
        self._sparse_b = True
      self._csa = False
    else:
      self._csa = True

    self._mat_a = descr.mat_a
    
    self._sparse_b = False
    self._sparse_a = False
    if isinstance(self._mat_a, SparseMatrix):
      self._sparse_a = True

  def _auto_select(self, **params):
    model = params['context'].get_vm().get_hw_descr().model
    both_sparse_error = 'kernelforge does not support both matrix A and B being sparse'
    if model == 'pvc':
      if not self._sparse_a and not self._sparse_b:
        return GemmKernelType.REGISTER_ONLY_BASED
      # sparse-dense/dense-sparse register-only GEMMs are not yet supported
    if self._sparse_a and self._sparse_b:
      raise Exception(both_sparse_error)
    elif self._sparse_b:
      return GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED
    elif self._sparse_a:
      return GemmKernelType.SPARSE_DENSE_SHR_MEM_BASED
    else:
      return GemmKernelType.SHR_MEM_BASED

  def get_builder(self, **params):
    if isinstance(self._descr, PointwiseDescr):
      return build_pointwise(self._descr.operation, **params)
    if self._csa:
      return CSA(**params)
    if self._gemm_kernel_type == GemmKernelType.AUTO:
      self._gemm_kernel_type = self._auto_select(**params)

    if self._gemm_kernel_type == GemmKernelType.SHR_MEM_BASED:
      return ShrMemBasedDenseGemm(**params)
    elif self._gemm_kernel_type == GemmKernelType.REGISTER_ONLY_BASED:
      return RegisterOnlyDenseGemm(**params)
    elif self._gemm_kernel_type == GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED:
      if not isinstance(self._mat_a, DenseMatrix) or not isinstance(self._mat_b, SparseMatrix):
        raise Exception("For dense x sparse kernel matrix A needs to be dense and matrix B sparse")
      return ShrMemBasedDenseSparseGemm(**params)
    elif self._gemm_kernel_type == GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED:
      if not isinstance(self._mat_a, DenseMatrix) or not isinstance(self._mat_b, SparseMatrix):
        raise Exception("For dense x sparse kernel matrix A needs to be dense and matrix B sparse")
      return RegisterOnlyDenseSparseGemm(**params)
    elif self._gemm_kernel_type == GemmKernelType.SPARSE_DENSE_SHR_MEM_BASED:
      if not isinstance(self._mat_a, SparseMatrix) or not isinstance(self._mat_b, DenseMatrix):
        raise Exception("For sparse x dense kernel matrix A needs to be sparse and matrix B dense")
      return ShrMemBasedSparseDenseGemm(**params)
    elif self._gemm_kernel_type == GemmKernelType.SPARSE_DENSE_REGISTER_ONLY_BASED:
      if not isinstance(self._mat_a, SparseMatrix) or not isinstance(self._mat_b, DenseMatrix):
        raise Exception("For sparse x dense kernel matrix A needs to be sparse and matrix B dense")
      return RegisterOnlySparseDenseGemm(**params)
    else:
      raise RuntimeError('unknown gemm type')

  def gemm_kernel_type(self):
    """returns a concrete kernel type. It is relevant if the user requested
    to perform auto-selection"""
    return self._gemm_kernel_type

class DenseGemmBuilder(AbstractBuilder):
  GemmClass = None

  def __init__(self,
               context: Context,
               scopes: Scopes,
               register_array: Symbol,
               shr_mem: Symbol,
               num_threads: int):
    super(DenseGemmBuilder, self).__init__(context, scopes)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._counter = 0
    self._loaders_cache: Dict[Symbol, AbstractShrMemLoader] = {}

    self._op1 = None
    self._op2 = None
    self._dest_obj = None
    self._descr = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self, op1: Symbol, op2: Union[Symbol, None], dest_obj: Matrix, descr: GemmDescr):
    self._reset()

    self._op1 = op1
    self._op2 = op2
    self._dest_obj = dest_obj
    self._descr = descr

    self._mem_region_a = None
    self._mem_region_b = None

    self._make_load_op1()
    self._make_load_op2()
    self._insert_sync_block()
    self._check_register_array()
    self._make_gemm()
    self._insert_sync_block()
    self._make_store()
    self._insert_sync_block()
    self._clear_registers()

  def _make_load_op1(self):
    if self._op1.stype == SymbolType.Global:
      if self._descr.trans_a:
        self._mem_region_a, load_op1 = self._make_loader_and_symbol(self._op1, is_transpose=True)
        self._loaders_cache[self._mem_region_a] = load_op1
        self._instructions.append(load_op1)
      else:
        # Note: operand will reside in glb. mem for gemm operation
        self._mem_region_a = self._op1

    elif self._op1.stype == SymbolType.SharedMem:
      if self._op1 in self._loaders_cache.keys():
        # Note: this condition means the symbol `self._op1` has been loaded
        # to shr. mem. before. Let's check whether loaded data can be reused
        prev_loader = self._loaders_cache[self._op1]

        if self._descr.trans_a and prev_loader.get_loader_type() == ShrMemLoaderType.NOT_TRANSPOSED:
          # means: data cannot be reused. we need to reload it again and traspose on the fly.
          # additionally, we need to remove aliased symbol to avoid clashes
          # self._scopes.delete_symbol(self._op1)
          self._scopes.add_scope()
          prev_symbol = prev_loader.get_src()
          self._mem_region_a, load_op1 = self._make_loader_and_symbol(prev_symbol, is_transpose=self._descr.trans_a)
          self._loaders_cache[self._mem_region_a] = load_op1
          self._instructions.append(load_op1)
        elif not self._descr.trans_a and prev_loader.get_loader_type() == ShrMemLoaderType.TRANSPOSED:
          # means: data loaded to shr. mem. cannot be reused. Because `op1` not need to be transposed
          # we don't need to load it to shr. mem. Instead, it will be taken from glb. mem.
          # we don't need delete previous (aliased) symbol
          self._mem_region_a = prev_loader.get_src()
        else:
          # means: data can be fully reused
          self._mem_region_a = self._op1

      else:
        self._mem_region_a = self._op1
    else:
      raise InternalError(f'gemm-builder: op1 ({self._op1.name}) must be either in shr or glb mem.')

  def _make_load_op2(self):
    if self._op2 is None:
      # unary operation
      self._mem_region_b = None
    else:
      # binary operation
      if self._op2.stype == SymbolType.Global:
        self._mem_region_b, load_op2 = self._make_loader_and_symbol(self._op2, self._descr.trans_b)
        self._loaders_cache[self._mem_region_b] = load_op2
        self._instructions.append(load_op2)

      elif self._op2.stype == SymbolType.SharedMem:
        self._mem_region_b = self._op2
      else:
        raise InternalError(f'gemm-builder: op2 ({self._op2.name}) must be either in shr or glb mem.')

  def _make_loader_and_symbol(self, operand, is_transpose) -> Tuple[Symbol, AbstractShrMemLoader]:
    shr_mem_region = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._scopes.add_symbol(shr_mem_region)
    load_op = shm_mem_loader_factory(self._context,
                                     dest=shr_mem_region,
                                     src=operand,
                                     shr_mem=self._shr_mem,
                                     num_threads=self._num_threads,
                                     load_and_transpose=is_transpose)
    return shr_mem_region, load_op

  def _check_register_array(self):
    if self._dest_regs.stype != SymbolType.Register:
      raise InternalError('gemm-builder: reg_array must be in registers')

  def _make_gemm(self):
    self._instructions.append(GemmKernelsFactory(self._descr).get_builder(context=self._context,
                                   trans_a=self._descr.trans_a,
                                   trans_b=self._descr.trans_b,
                                   op1=self._mem_region_a,
                                   op2=self._mem_region_b,
                                   dest=self._dest_regs,
                                   prefer_align=self._descr.prefer_align,
                                   num_threads=self._num_threads))

  def _make_store(self):
    if self._dest_obj in self._scopes:
      dest_symbol = self._scopes.get_symbol(self._dest_obj)
      if dest_symbol.stype == SymbolType.SharedMem:
        self._instructions.append(StoreRegToShr(context=self._context,
                                                src=self._dest_regs,
                                                dest=dest_symbol,
                                                shr_mem=self._shr_mem,
                                                num_threads=self._num_threads))
      elif dest_symbol.stype == SymbolType.Global:
        self._instructions.append(StoreRegToGlb(context=self._context,
                                                src=self._dest_regs,
                                                dest=dest_symbol,
                                                alpha=self._descr.alpha,
                                                beta=self._descr.beta,
                                                num_threads=self._num_threads))
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
                                              src=self._dest_regs,
                                              dest=dest_symbol,
                                              shr_mem=self._shr_mem,
                                              num_threads=self._num_threads))

  def _clear_registers(self):
    self._instructions.append(ClearRegisters(context=self._context, src=self._dest_regs))

  def _insert_sync_block(self):
    self._instructions.append(SyncThreads(context=self._context,
                                          num_threads_per_mult=self._num_threads))

  def _name_shr_reg(self):
    name = f'_{self._counter}'
    self._counter += 1
    return name
