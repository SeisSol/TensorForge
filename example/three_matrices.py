from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.context import Context
from tensorforge.common.aux import generate_tmp_matrix
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.common.basic_types import FloatingPointType, Addressing
from tensorforge.generators.generator import Generator

# D += A x (B x C)
mat_d = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0],[56, 9])), BoundingBox([0, 0],[56, 9]))

mat_a = SubTensor(Tensor([56, 56], Addressing.STRIDED, BoundingBox([0, 0],[56, 56])), BoundingBox([0, 0],[56, 56]))

mat_b = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0],[56, 9])), BoundingBox([0, 0],[56, 9]))

mat_c = SubTensor(Tensor([9, 9], Addressing.STRIDED, BoundingBox([0, 0],[9, 9])), BoundingBox([0, 0],[9, 9]))

tmp1 = generate_tmp_matrix(mat_b, mat_c)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_b, b=mat_c, c=tmp1),
             GemmDescr(trans_a=False, trans_b=False,
                       a=mat_a, b=tmp1, c=mat_d,
                       alpha=1.0,
                       beta=1.0)]

context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

# context = Context(arch='sm_60',
#                   backend='omptarget',
#                   fp_type=FloatingPointType.FLOAT)

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
