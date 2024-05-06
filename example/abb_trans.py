from kernelforge.common import DenseMatrix
from kernelforge.common.context import Context
from kernelforge.common.aux import generate_tmp_matrix
from kernelforge.generators.descriptions import GemmDescr, FloatingPointType, Addressing
from kernelforge.generators.generator import Generator
from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.matrix.tensor import Tensor


# Q += (A x B) x B^T

mat_q = Tensor([56, 56], Addressing.PTR_BASED, BoundingBox([0,0], [20,9]))

mat_a = Tensor([56, 56], Addressing.NONE, BoundingBox([0,0], [20,9]))

mat_b = Tensor([56, 56], Addressing.STRIDED, BoundingBox([0,0], [9,20]))


tmp1 = generate_tmp_matrix(mat_a, mat_b)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=True,
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