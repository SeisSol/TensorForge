from typing import Tuple, List, Union
from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.loaders.abstract_loader import NoLoadShrMemLoader
from gemmforge.symbol_table import SymbolType, Symbol
from gemmforge.instructions import SyncThreads
from gemmforge.instructions import ShrMemBasedDenseGemm
from gemmforge.instructions import RegisterOnlyDenseGemm
from gemmforge.instructions import RegisterOnlySparseDenseGemm
from gemmforge.instructions import ShrMemBasedSparseDenseGemm
from gemmforge.instructions import RegisterOnlyDenseSparseGemm
from gemmforge.instructions import ShrMemBasedDenseSparseGemm
from gemmforge.instructions.loaders import shm_mem_loader_factory
from gemmforge.basic_types import GeneralLexicon
from gemmforge.matrix import SparseMatrix


class ShrMemBasedDenseGemmBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(ShrMemBasedDenseGemmBuilder, self).__init__(vm, symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._counter = 0
    self._load_instrs = []

    self._op1 = None
    self._op2 = None
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            trans_a: bool,
            trans_b: bool,
            op1: Symbol,
            op2: Symbol,
            dest: Symbol):
    self._reset()

    # Note: of trans_a==True than an operand is given as KxM instead of (MxK).
    # In this case, a loader will load an operand from glb. mem. to shr. mem
    # transposing it on the fly. In, short, the loader guaranties to deliver
    # an operand as (MxK) to shr. mem.
    self._symbol_table.add_scope()
    if trans_a:
      self._op1 = self._make_loader_and_symbol(operand=op1, do_transpose=True)
    else:
      self._op1 = op1

    # Note: we will handle transposition of the second operand during
    # the matrix multiplication
    self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=False)

    self._insert_sync_threads()

    gemm_params = {'vm': self._vm,
                   'trans_a': False,
                   'trans_b': trans_b,
                   'op1': self._op1,
                   'op2': self._op2,
                   'dest': dest,
                   'num_threads': self._num_threads}
    self._instructions.append(ShrMemBasedDenseGemm(**gemm_params))

  def _make_loader_and_symbol(self, operand, do_transpose):
    shr_mem_region = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._symbol_table.add_symbol(shr_mem_region)
    load_op = shm_mem_loader_factory(vm=self._vm,
                                     dest=shr_mem_region,
                                     src=operand,
                                     shr_mem=self._shr_mem,
                                     num_threads=self._num_threads,
                                     load_and_transpose=do_transpose)

    self._instructions.append(load_op)
    self._load_instrs.append(load_op)
    return shr_mem_region

  def get_srh_mem_loads(self):
    return self._load_instrs

  def _insert_sync_threads(self):
    self._instructions.append(SyncThreads(self._vm, self._num_threads))

  def _name_shr_reg(self):
    name = f'{GeneralLexicon.SHR_MEM_REGION_PREFIX}{self._counter}'
    self._counter += 1
    return name


class RegisterOnlyDenseGemmBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(RegisterOnlyDenseGemmBuilder, self).__init__(vm, symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._op1 = None
    self._op2 = None
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            trans_a: bool,
            trans_b: bool,
            op1: Symbol,
            op2: Symbol,
            dest: Symbol):
    self._reset()

    gemm_params = {'vm': self._vm,
                   'trans_a': trans_a,
                   'trans_b': trans_b,
                   'op1': op1,
                   'op2': op2,
                   'dest': dest,
                   'num_threads': self._num_threads}
    self._instructions.append(RegisterOnlyDenseGemm(**gemm_params))

  def get_srh_mem_loads(self):
    return []


class  ShrMemBasedSparseDenseGemmBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(ShrMemBasedSparseDenseGemmBuilder, self).__init__(vm=vm, symbol_table=symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._counter = 0
    self._load_instrs = []

    self._op1 = None
    self._op2 = None
    self._intermediate_dest = None
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            trans_a: bool,
            trans_b: bool,
            op1: Symbol,
            op2: Symbol,
            intermediate_dest: Symbol,
            register_dest: Symbol,
            mat_a: SparseMatrix,
            beta: float):
    self._reset()

    #if mat_a.get_values() == None or not trans_b:
    #self._symbol_table.add_scope()

    if mat_a.get_values() == None:
      # Note: of trans_a==True than an operand is given as KxM instead of (MxK).
      # In this case, a loader will load an operand from glb. mem. to shr. mem
      # transposing it on the fly. In, short, the loader guaranties to deliver
      # an operand as (MxK) to shr. mem.
      #self._symbol_table.add_scope()
      #Access global?
      #self._op1 = self._make_loader_and_symbol(operand=op1, do_transpose=False)

      # Note: we will handle transposition of the second operand during
      # the matrix multiplication
      self._symbol_table.add_scope()
      # This is a heuristics I have implemented as having too sparse matrices can increase bank conflicts
      # And this heuristical optimization should remain until a better shared memory loader is implemented
      if mat_a.sparsity() < 0.65:
        self._op1 = self._make_loader_and_symbol(operand=op1, do_transpose=False)
      else:
        self._op1 = op1
    else:
      self._op1 = op1
    self._symbol_table.add_scope()

    if trans_b:
      #self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=False)
      self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=True)
    else:
      #self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=True)
      self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=False)
    self._symbol_table.add_scope()

    self._intermediate_dest = self._make_loader_and_symbol_do_not_load_if_cond(operand=intermediate_dest, do_transpose=False, cond=beta!=0.0)

    self._insert_sync_threads()

    gemm_params = {'vm': self._vm,
                   'trans_a': trans_a,
                   'trans_b': trans_b,
                   'op1': self._op1,
                   'op2': self._op2,
                   'intermediate_dest': self._intermediate_dest,
                   'register_dest': register_dest,
                   'num_threads': self._num_threads,
                   'mat_a': mat_a}
    self._instructions.append(ShrMemBasedSparseDenseGemm(**gemm_params))

  def _make_loader_and_symbol(self, operand, do_transpose):
    shr_mem_region = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._symbol_table.add_symbol(shr_mem_region)

    load_op = shm_mem_loader_factory(vm=self._vm,
                                     dest=shr_mem_region,
                                     src=operand,
                                     shr_mem=self._shr_mem,
                                     num_threads=self._num_threads,
                                     load_and_transpose=do_transpose)

    self._instructions.append(load_op)
    self._load_instrs.append(load_op)
    return shr_mem_region

  def _make_loader_and_symbol_do_not_load_if_cond(self, operand, do_transpose, cond):
    shr_mem_region = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._symbol_table.add_symbol(shr_mem_region)

    if cond:
      load_op = shm_mem_loader_factory(vm=self._vm,
                                      dest=shr_mem_region,
                                      src=operand,
                                      shr_mem=self._shr_mem,
                                      num_threads=self._num_threads,
                                      load_and_transpose=do_transpose)
    else:
      params = {'vm': self._vm,
            'dest': shr_mem_region,
            'src': operand,
            'shr_mem': self._shr_mem,
            'num_threads': self._num_threads,
            'load_and_transpose': do_transpose}
      load_op = NoLoadShrMemLoader(**params)
    self._instructions.append(load_op)
    self._load_instrs.append(load_op)

    return shr_mem_region


  def get_srh_mem_loads(self):
    return self._load_instrs

  def _insert_sync_threads(self):
    self._instructions.append(SyncThreads(self._vm, self._num_threads))

  def _name_shr_reg(self):
    name = f'{GeneralLexicon.SHR_MEM_REGION_PREFIX}{self._counter}'
    self._counter += 1
    return name


class RegisterOnlySparseDenseGemmBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(RegisterOnlySparseDenseGemmBuilder, self).__init__(vm, symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._op1 = None
    self._op2 = None
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            trans_a: bool,
            trans_b: bool,
            op1: Symbol,
            op2: Symbol,
            dest: Symbol,
            mat_a: SparseMatrix):
    raise Exception("Sparse-by-Dense Kernel for register only approach is not supported.")

  def get_srh_mem_loads(self):
    return []


class ShrMemBasedDenseSparseGemmBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(ShrMemBasedDenseSparseGemmBuilder, self).__init__(vm=vm, symbol_table=symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._counter = 0
    self._load_instrs = []

    self._op1 = None
    self._op2 = None
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            trans_a: bool,
            trans_b: bool,
            op1: Symbol,
            op2: Symbol,
            dest: Symbol,
            mat_b: SparseMatrix):
    self._reset()

    # Note: of trans_a==True than an operand is given as KxM instead of (MxK).
    # In this case, a loader will load an operand from glb. mem. to shr. mem
    # transposing it on the fly. In, short, the loader guaranties to deliver
    # an operand as (MxK) to shr. mem.
    if mat_b.get_values() == None or trans_a:
      self._symbol_table.add_scope()

    if trans_a:
      self._op1 = self._make_loader_and_symbol(operand=op1, do_transpose=True)
    else:
      self._op1 = op1

    # Note: we will handle transposition of the second operand during
    # the matrix multiplication

    if mat_b.get_values() == None:
      self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=False)
    else:
      self._op2 = op2


    if mat_b.get_values() == None or trans_a:
      self._insert_sync_threads()

    gemm_params = {'vm': self._vm,
                   'trans_a': False,
                   'trans_b': trans_b,
                   'op1': self._op1,
                   'op2': self._op2,
                   'dest': dest,
                   'num_threads': self._num_threads,
                   'mat_b': mat_b}
    self._instructions.append(ShrMemBasedDenseSparseGemm(**gemm_params))

  def _make_loader_and_symbol(self, operand, do_transpose):
    shr_mem_region = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._symbol_table.add_symbol(shr_mem_region)
    load_op = shm_mem_loader_factory(vm=self._vm,
                                     dest=shr_mem_region,
                                     src=operand,
                                     shr_mem=self._shr_mem,
                                     num_threads=self._num_threads,
                                     load_and_transpose=do_transpose)

    self._instructions.append(load_op)
    self._load_instrs.append(load_op)
    return shr_mem_region

  def get_srh_mem_loads(self):
    return self._load_instrs

  def _insert_sync_threads(self):
    self._instructions.append(SyncThreads(self._vm, self._num_threads))

  def _name_shr_reg(self):
    name = f'{GeneralLexicon.SHR_MEM_REGION_PREFIX}{self._counter}'
    self._counter += 1
    return name


class RegisterOnlyDenseSparseGemmBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(RegisterOnlyDenseSparseGemmBuilder, self).__init__(vm, symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._op1 = None
    self._op2 = None
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            trans_a: bool,
            trans_b: bool,
            op1: Symbol,
            op2: Symbol,
            dest: Symbol,
            mat_b: SparseMatrix):
    raise Exception("Dense-by-Sparse Kernel for register only approach is not supported.")


  def get_srh_mem_loads(self):
    return []
