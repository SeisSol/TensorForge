import unittest
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.generators.descriptions import Addressing
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge import GemmGenerator
from tensorforge.common.vm.vm import vm_factory


class TestOccupancyNvidia(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self._vm = vm_factory(arch='sm_60', backend='cuda', fp_type="float")

    self.gen = GemmGenerator(self._vm)

  def tearDown(self):
    pass

  def test_occupancy_56_9_9_threads(self):
    mat_a = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0.0], [55, 8])), BoundingBox([0.0], [55, 8]))

    mat_b = SubTensor(Tensor([9, 9], Addressing.STRIDED, BoundingBox([0.0], [8, 8])), BoundingBox([0.0], [8, 8]))

    mat_c = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0.0], [55, 8])), BoundingBox([0.0], [55, 8]))

    self.gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
    self.assertEqual(self.gen._num_active_threads, 64)
    self.assertEqual(self.gen._num_mult_per_block, 1)

  def test_occupancy_32_9_56_threads(self):
    mat_a = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [31, 55])), BoundingBox([0.0], [31, 55]))

    mat_b = SubTensor(Tensor([64, 56], Addressing.STRIDED, BoundingBox([0.0], [55, 8])), BoundingBox([0.0], [55, 8]))

    mat_c = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [31, 8])), BoundingBox([0.0], [31, 8]))

    self.gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)

    # one multiplication per block because the kernel requires too much of
    # shared memory
    self.assertEqual(self.gen._num_active_threads, 32)
    self.assertEqual(self.gen._num_mult_per_block, 2)

  def test_occupancy_32_9_21_threads(self):
    mat_a = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [31, 20])), BoundingBox([0.0], [31, 20]))

    mat_b = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [20, 8])), BoundingBox([0.0], [20, 8]))

    mat_c = SubTensor(Tensor([32, 9], Addressing.STRIDED, BoundingBox([0.0], [31, 8])), BoundingBox([0.0], [31, 8]))

    self.gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)

    # one multiplication per block because the kernel requires too much of
    # shared memory
    self.assertEqual(self.gen._num_active_threads, 32)
    self.assertEqual(self.gen._num_mult_per_block, 2)
