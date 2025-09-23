from tensorforge.common.context import Context
from tensorforge.common.aux import generate_tmp_matrix
from tensorforge.generators.descriptions import GemmDescr, FloatingPointType, Addressing
from tensorforge.generators.generator import Generator
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor


# Q += (A x B^T) x B

mat_q = SubTensor(Tensor([56, 56], Addressing.PTR_BASED, BoundingBox([0,0], [20,9])), BoundingBox([0,0], [20,9]))

mat_a = SubTensor(Tensor([56, 56], Addressing.NONE, BoundingBox([0,0], [20,9])), BoundingBox([0,0], [20,9]))

mat_b = SubTensor(Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [20,9])), BoundingBox([0,0], [20,9]))

tmp1 = generate_tmp_matrix(mat_a, mat_b, trans_a=False, trans_b=True)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=True,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=tmp1, b=mat_b, c=mat_q,
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