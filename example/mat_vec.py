from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.context import Context
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.common.basic_types import FloatingPointType, Addressing
from tensorforge.generators.generator import Generator


# C += A x b

vec_c = SubTensor(Tensor([56, 1], Addressing.STRIDED, BoundingBox([0, 0],[56, 1])), BoundingBox([0, 0],[56, 1]))

mat_a = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0],[56, 9])), BoundingBox([0, 0],[56, 9]))

vec_b = SubTensor(Tensor([9, 1], Addressing.STRIDED, BoundingBox([0, 0],[9, 1])), BoundingBox([0, 0],[9, 1]))


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