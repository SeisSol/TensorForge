from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.context import Context
from kernelforge.common.aux import generate_tmp_matrix
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import FloatingPointType, Addressing
from kernelforge.generators.generator import Generator


# Q += A x ((B x C) x D)
mat_q = Tensor([56, 56], Addressing.PTR_BASED, BoundingBox([0,0], [56,9]))

mat_a = Tensor([56, 56], Addressing.PTR_BASED, BoundingBox([0,0], [56,20]))

mat_b = Tensor([56, 56], Addressing.PTR_BASED, BoundingBox([0,0], [20,56]))

mat_c = Tensor([56, 9], Addressing.PTR_BASED, BoundingBox([0,0], [56,9]))

mat_d = Tensor([9, 9], Addressing.PTR_BASED, BoundingBox([0,0], [9,9]))


tmp1 = generate_tmp_matrix(mat_b, mat_c)
tmp2 = generate_tmp_matrix(tmp1, mat_d)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_b, b=mat_c, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=tmp1, b=mat_d, c=tmp2),
             GemmDescr(trans_a=False, trans_b=False,
                       a=mat_a, b=tmp2, c=mat_q,
                       alpha='alpha',
                       beta='beta')]

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
