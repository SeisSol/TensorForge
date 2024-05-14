from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.context import Context
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import FloatingPointType, Addressing
from kernelforge.generators.generator import Generator


# C += A x b

vec_c = Tensor([56, 1], Addressing.STRIDED, BoundingBox([0, 0],[56, 1]))

mat_a = Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0],[56, 9]))

vec_b = Tensor([9, 1], Addressing.STRIDED, BoundingBox([0, 0],[9, 1]))


gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=vec_b, c=vec_c, alpha = 1.0, beta = 1.0)]

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