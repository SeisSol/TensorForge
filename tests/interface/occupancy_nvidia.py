import unittest
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.vm.vm import vm_factory
from tensorforge import *
from tensorforge.common.basic_types import FloatingPointType, Addressing
from tensorforge.common.context import Context
from tensorforge.generators.descriptions import GemmDescr


class TestOccupancyNvidia(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self._context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

  def tearDown(self):
    pass

  def test_occupancy_56_9_9_threads(self):
    mat_a = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0.0], [55, 8])), BoundingBox([0.0], [55, 8]))

    mat_b = SubTensor(Tensor([9, 9], Addressing.STRIDED, BoundingBox([0.0], [8, 8])), BoundingBox([0.0], [8, 8]))

    mat_c = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0.0], [55, 8])), BoundingBox([0.0], [55, 8]))

    gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=mat_c,
                       alpha=1.1, beta=1.1)]

    generator = Generator(gemm_list, self._context)
    generator.generate()
    #self.assertEqual(generator._num_active_threads, 64)
    #self.assertEqual(self.gen._num_mult_per_block, 1)

  def test_occupancy_32_9_56_threads(self):
    mat_a = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [31, 55])), BoundingBox([0.0], [31, 55]))

    mat_b = SubTensor(Tensor([64, 56], Addressing.STRIDED, BoundingBox([0.0], [55, 8])), BoundingBox([0.0], [55, 8]))

    mat_c = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [31, 8])), BoundingBox([0.0], [31, 8]))

    gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=mat_c,
                       alpha=1.1, beta=1.1)]

    generator = Generator(gemm_list, self._context)
    generator.generate()

    # one multiplication per block because the kernel requires too much of
    # shared memory
    #self.assertEqual(self.gen._num_active_threads, 32)
    #self.assertEqual(self.gen._num_mult_per_block, 2)

  def test_occupancy_32_9_21_threads(self):
    mat_a = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [31, 20])), BoundingBox([0.0], [31, 20]))

    mat_b = SubTensor(Tensor([32, 56], Addressing.STRIDED, BoundingBox([0.0], [20, 8])), BoundingBox([0.0], [20, 8]))

    mat_c = SubTensor(Tensor([32, 9], Addressing.STRIDED, BoundingBox([0.0], [31, 8])), BoundingBox([0.0], [31, 8]))

    gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=mat_c,
                       alpha=1.1, beta=1.1)]

    generator = Generator(gemm_list, self._context)
    generator.generate()

    # one multiplication per block because the kernel requires too much of
    # shared memory
    #self.assertEqual(self.gen._num_active_threads, 32)
    #self.assertEqual(self.gen._num_mult_per_block, 2)
