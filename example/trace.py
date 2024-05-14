from kernelforge.common.matrix.dense import DenseMatrix
from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.context import Context
from kernelforge.generators.descriptions import MultilinearDescr
from kernelforge.common.basic_types import FloatingPointType, Addressing
from kernelforge.generators.generator import Generator


# Q += tr(A)
mat_a = Tensor([64, 56, 56], Addressing.PTR_BASED, BoundingBox([0,0,10], [64,56,30]))

mat_b = Tensor([64], Addressing.PTR_BASED, BoundingBox([0], [64]))

gemm_list = [MultilinearDescr(mat_b, [mat_a], [[0, -1, -1]], [[0, 1, 2]])]

context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, context)
generator.generate()

with_output = True
if with_output:
  print(generator.get_header())
  print(generator.default_generate_call_site())
  print()
  print(generator.get_launcher())
  print()
  print(generator.get_kernel())
