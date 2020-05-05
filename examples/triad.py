import os

from gemmforge import DenseMatrix, TriadGenerator, GenerationError
from gemmforge import arch

arch = arch.produce("nvidia")
mat_d = DenseMatrix(num_rows=56, num_cols=9, addressing="strided", transpose=False)
mat_a = DenseMatrix(num_rows=56, num_cols=9, addressing="strided", transpose=False)
mat_b = DenseMatrix(num_rows=9, num_cols=9, addressing="strided", transpose=False)
mat_c = DenseMatrix(num_rows=56, num_cols=56, addressing="none", transpose=False)
alpha = 1.0
beta = 1.0

try:
    gen = TriadGenerator(arch, "float")
    gen.generate(mat_d, mat_a, mat_b, mat_c, alpha, beta)
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print("ERROR: {}".format(err))

