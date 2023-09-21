import unittest
from kernelforge import DenseMatrix
from kernelforge.common.vm.vm import vm_factory
from kernelforge.instructions.loaders import shm_mem_loader_factory
from kernelforge.instructions.loaders import ExactPatchLoader, ExtendedPatchLoader
from kernelforge.instructions.loaders import ExactTransposePatchLoader
from kernelforge.instructions.loaders import ExtendedTransposePatchLoader
from kernelforge.common.basic_types import ShrMemObject
from kernelforge.symbol_table import Symbol, SymbolType, InverseSymbolTable
from kernelforge.symbol_table import DataView


class TestLoaders(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
      pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self._vm = vm_factory(arch='sm_60', backend='cuda', fp_type='float')
    self._table = InverseSymbolTable()
    
    self._shr_mem_obj = ShrMemObject(name='shr_mem', size=1000, mults_per_block=1)
    self._shr_mem_symbol = Symbol(name='shr_mem',
                                  stype=SymbolType.SharedMem,
                                  obj=self._shr_mem_obj)

  def _make_symbols(self, matrix):
    src = Symbol(name='dest',
                 stype=SymbolType.Global,
                 obj=matrix)

    src.data_view = DataView(rows=matrix.get_actual_num_rows(),
                             columns=matrix.get_actual_num_cols(),
                             lead_dim=matrix.num_rows,
                             is_transposed=False)
    
    dest = Symbol(name='dest',
                  stype=SymbolType.SharedMem,
                  obj=matrix)
    return src, dest
    
  def tearDown(self):
    pass

  def test_extended_loader(self):
    # load a column in one go
    matrix = DenseMatrix(num_rows=31,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 15, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=False)
    self.assertIsInstance(loader, ExtendedPatchLoader)

    # multiple hops to load a column
    matrix = DenseMatrix(num_rows=63,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 34, 20])
    src, dest = self._make_symbols(matrix)

    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=False)
    self.assertIsInstance(loader, ExtendedPatchLoader)

  def test_exact_loader(self):
    # load a column in one go
    matrix = DenseMatrix(num_rows=33,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 15, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=False)

    self.assertIsInstance(loader, ExactPatchLoader)

    # multiple hops to load a column
    matrix = DenseMatrix(num_rows=65,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 34, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=False)
    self.assertIsInstance(loader, ExactPatchLoader)

  def test_extended_transpose_loader(self):
    # load a column in one go
    matrix = DenseMatrix(num_rows=31,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 15, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=True)
    self.assertIsInstance(loader, ExtendedTransposePatchLoader)

    # multiple hops to load a column
    matrix = DenseMatrix(num_rows=61,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 34, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=True)
    self.assertIsInstance(loader, ExtendedTransposePatchLoader)

  def test_exact_transpose_loader(self):
    # load a column in one go
    matrix = DenseMatrix(num_rows=33,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 15, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=True)
    self.assertIsInstance(loader, ExactTransposePatchLoader)

    # multiple hops to load a column
    matrix = DenseMatrix(num_rows=65,
                         num_cols=56,
                         addressing='none',
                         bbox=[0, 0, 34, 20])
    src, dest = self._make_symbols(matrix)
    
    loader = shm_mem_loader_factory(vm=self._vm,
                                    dest=dest,
                                    src=src,
                                    shr_mem=self._shr_mem_obj,
                                    num_threads=32,
                                    load_and_transpose=True)
    self.assertIsInstance(loader, ExactTransposePatchLoader)


if __name__ == '__main__':
  unittest.main()
