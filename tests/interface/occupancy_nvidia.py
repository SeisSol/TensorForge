import unittest
from gemmforge import DenseMatrix
from gemmforge import GemmGenerator
from gemmforge import arch

class TestOccupancyNvidia(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self.arch = arch.produce("nvidia", "sm_70")

  def tearDown(self):
    pass

  def test_occupancy_56_9_9_threads(self):

    mat_a = DenseMatrix(num_rows=56,
                        num_cols=9,
                        addressing="strided",
                        bbox=[0, 0, 55, 8],
                        transpose=False)

    mat_b = DenseMatrix(num_rows=9,
                        num_cols=9,
                        addressing="strided",
                        bbox=[0, 0, 8, 8],
                        transpose=False)

    mat_c = DenseMatrix(num_rows=56,
                        num_cols=9,
                        bbox=[0, 0, 55, 8],
                        addressing="strided",
                        transpose=False)

    gen = GemmGenerator(self.arch, "float")
    gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
    self.assertEqual(gen.num_active_threads, 64)
    self.assertEqual(gen.num_mult_per_block, 1)

  def test_occupancy_32_9_56_threads(self):
    mat_a = DenseMatrix(num_rows=32,
                        num_cols=56,
                        addressing="strided",
                        bbox=[0, 0, 31, 55],
                        transpose=False)

    mat_b = DenseMatrix(num_rows=64,
                        num_cols=56,
                        addressing="strided",
                        bbox=[0, 0, 55, 8],
                        transpose=False)

    mat_c = DenseMatrix(num_rows=32,
                        num_cols=56,
                        bbox=[0, 0, 31, 8],
                        addressing="strided",
                        transpose=False)

    gen = GemmGenerator(self.arch, "float")
    gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)

    # one multiplication per block because the kernel requires too much of
    # shared memory
    self.assertEqual(gen.num_active_threads, 32)
    self.assertEqual(gen.num_mult_per_block, 2)

  def test_occupancy_32_9_21_threads(self):
    mat_a = DenseMatrix(num_rows=32,
                        num_cols=56,
                        addressing="strided",
                        bbox=[0, 0, 31, 20],
                        transpose=False)

    mat_b = DenseMatrix(num_rows=32,
                        num_cols=56,
                        addressing="strided",
                        bbox=[0, 0, 20, 8],
                        transpose=False)

    mat_c = DenseMatrix(num_rows=32,
                        num_cols=9,
                        bbox=[0, 0, 31, 8],
                        addressing="strided",
                        transpose=False)

    gen = GemmGenerator(self.arch, "float")
    gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)

    # one multiplication per block because the kernel requires too much of
    # shared memory
    self.assertEqual(gen.num_active_threads, 32)
    self.assertEqual(gen.num_mult_per_block, 2)

