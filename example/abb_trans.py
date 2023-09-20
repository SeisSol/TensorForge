from tensorforge.common import DenseMatrix
from tensorforge.common import Context
from tensorforge.common.aux import generate_tmp_matrix
from tensorforge.common import GemmDescr, FloatingPointType, Addressing
from tensorforge.backend.generator import Generator


# Q += (A x B) x B^T
mat_q = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.PTR_BASED,
                    bbox=[0, 0, 20, 9])

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.NONE,
                    bbox=[0, 0, 20, 9])

mat_b = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 9, 20])

tmp1 = generate_tmp_matrix(mat_a, mat_b)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=True,
                       a=tmp1, b=mat_b, c=mat_q)]

context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, context)
generator.generate()

print(generator.get_launcher())
print()
print(generator.get_header())
print()
print(generator.get_kernel())