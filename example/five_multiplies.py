from kernelforge.common.context import Context
from kernelforge.common.aux import generate_tmp_matrix
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import FloatingPointType, Addressing
from kernelforge.generators.generator import Generator
from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.matrix.tensor import Tensor


# Q = (((A x B) x (C x B)) x D)

mat_q = Tensor([9, 9], Addressing.STRIDED, BoundingBox([0,0], [9,9]))

mat_a = Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [56,56]))

mat_b = Tensor([56, 9], Addressing.STRIDED, BoundingBox([0,0], [56,9]))

mat_c = Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [56,56]))

mat_d = Tensor([9, 9], Addressing.STRIDED, BoundingBox([0,0], [9,9]))



tmp0 = generate_tmp_matrix(mat_a, mat_b)
tmp1 = generate_tmp_matrix(mat_c, mat_b)
tmp2 = generate_tmp_matrix(tmp0, tmp1, trans_a=True)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp0),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_c, b=mat_b, c=tmp1),
             GemmDescr(trans_a=True, trans_b=False,
                       a=tmp0, b=tmp1, c=tmp2),
             GemmDescr(trans_a=False, trans_b=False,
                       a=tmp2, b=mat_d, c=mat_q,
                       alpha=1.0, beta=0.0),
             ]

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
