from tensorforge.common.context import Context
from tensorforge.common.aux import generate_tmp_matrix
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.common.basic_types import FloatingPointType, Addressing
from tensorforge.generators.generator import Generator
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor


# Q = (((A x B) x (C x B)) x D)

mat_q = SubTensor(Tensor([9, 9], Addressing.STRIDED, BoundingBox([0,0], [9,9])), BoundingBox([0,0], [9,9]))

mat_a = SubTensor(Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [56,56])), BoundingBox([0,0], [56,56]))

mat_b = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0,0], [56,9])), BoundingBox([0,0], [56,9])),

mat_c = SubTensor(Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [56,56])), BoundingBox([0,0], [56,56]))

mat_d = SubTensor(Tensor([9, 9], Addressing.STRIDED, BoundingBox([0,0], [9,9])), BoundingBox([0,0], [9,9]))



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
