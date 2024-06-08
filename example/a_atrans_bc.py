from tensorforge.common.context import Context
from tensorforge.common.aux import generate_tmp_matrix
from tensorforge.generators.descriptions import GemmDescr, FloatingPointType, Addressing
from tensorforge.generators.generator import Generator
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor


# Q += A x ((A^T x B) x C)

variants = {'v0': Addressing.STRIDED,
            'v1': Addressing.NONE}

mat_q = SubTensor(Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [20,9])),BoundingBox([0,0], [20,9]))

mat_a = SubTensor(Tensor([56, 56], addressing=variants['v0'], bbox=BoundingBox([0,0], [20,9])), BoundingBox([0,0], [20,9]))

mat_b = SubTensor(Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [20,9])), BoundingBox([0,0], [20,9]))

mat_c = SubTensor(Tensor([56, 9], Addressing.STRIDED, BoundingBox([0,0], [9,9])), BoundingBox([0,0], [9,9]))


tmp1 = generate_tmp_matrix(mat_a, mat_b, True, False)
tmp2 = generate_tmp_matrix(tmp1, mat_c)


gemm_list = [GemmDescr(trans_a=True,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=tmp1, b=mat_c, c=tmp2),
             GemmDescr(trans_a=False, trans_b=False,
                       a=mat_a, b=tmp2, c=mat_q,
                       alpha=1.0,
                       beta=1.0)]

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