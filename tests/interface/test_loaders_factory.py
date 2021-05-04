import unittest
from gemmforge import DenseMatrix
from gemmforge.vm import vm_factory
from gemmforge.loaders import shm_mem_factory
from gemmforge.loaders.shr_mem_loaders import ExactPatchLoader, ExtendedPatchLoader
from gemmforge.loaders.shr_transpose_mem_loaders import ExactTransposePatchLoader
from gemmforge.loaders.shr_transpose_mem_loaders import ExtendedTransposePatchLoader


class TestLoaders(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
      pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self._vm = vm_factory(name="nvidia",
                          sub_name="sm_60",
                          fp_type="float")

  def tearDown(self):
    pass

  def test_extended_loader(self):
    # load a column in one go
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=31,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 15, 20],
                                                transpose=False),
                             num_active_threads=32,
                             load_and_transpose=False)
    self.assertIsInstance(loader, ExtendedPatchLoader)

    # multiple hops to load a column
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=63,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 34, 20],
                                                transpose=False),
                             num_active_threads=32,
                             load_and_transpose=False)
    self.assertIsInstance(loader, ExtendedPatchLoader)

  def test_exact_loader(self):
    # load a column in one go
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=33,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 15, 20],
                                                transpose=False),
                             num_active_threads=32,
                             load_and_transpose=False)

    self.assertIsInstance(loader, ExactPatchLoader)

    # multiple hops to load a column
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=65,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 34, 20],
                                                transpose=False),
                             num_active_threads=32,
                             load_and_transpose=False)
    self.assertIsInstance(loader, ExactPatchLoader)

  def test_extended_transpose_loader(self):
    # load a column in one go
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=31,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 15, 20],
                                                transpose=True),
                             num_active_threads=32,
                             load_and_transpose=True)
    self.assertIsInstance(loader, ExtendedTransposePatchLoader)

    # multiple hops to load a column
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=61,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 34, 20],
                                                transpose=True),
                             num_active_threads=32,
                             load_and_transpose=True)
    self.assertIsInstance(loader, ExtendedTransposePatchLoader)

  def test_exact_transpose_loader(self):
    # load a column in one go
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=33,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 15, 20],
                                                transpose=True),
                             num_active_threads=32,
                             load_and_transpose=True)
    self.assertIsInstance(loader, ExactTransposePatchLoader)

    # multiple hops to load a column
    loader = shm_mem_factory(vm=self._vm,
                             matrix=DenseMatrix(num_rows=65,
                                                num_cols=56,
                                                addressing='none',
                                                bbox=[0, 0, 34, 20],
                                                transpose=True),
                             num_active_threads=32,
                             load_and_transpose=True)
    self.assertIsInstance(loader, ExactTransposePatchLoader)


if __name__ == '__main__':
  unittest.main()
