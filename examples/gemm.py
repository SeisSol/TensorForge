import os

from gemmforge import DenseMatrix, GemmGenerator, GenerationError
from gemmforge import arch

arch = arch.produce("nvidia")

mat_a = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 55, 8],
                    transpose=False)

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 8, 8],
                    transpose=False)


mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 55, 8],
                    addressing="strided",
                    transpose=False)
alpha = 1.1
beta = 1.1

try:
    gen = GemmGenerator(arch, "float")
    gen.generate(mat_a, mat_b, mat_c, alpha, beta)
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print("ERROR: {}".format(err))
    raise err

