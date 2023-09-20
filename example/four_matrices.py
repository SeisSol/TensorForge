from kernelforge.common import DenseMatrix
from kernelforge.common import Context
from kernelforge.common.aux import generate_tmp_matrix
from kernelforge.common import GemmDescr, FloatingPointType, Addressing
from kernelforge.backend.generator import Generator


# Q += A x ((B x C) x D)
mat_q = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.PTR_BASED,
                    bbox=[0, 0, 56, 9],)

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.NONE,
                    bbox=[0, 0, 56, 20])

mat_b = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 20, 56])

mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 56, 9],
                    addressing=Addressing.STRIDED)

mat_d = DenseMatrix(num_rows=9,
                    num_cols=9,
                    bbox=[0, 0, 9, 9],
                    addressing=Addressing.STRIDED)


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
