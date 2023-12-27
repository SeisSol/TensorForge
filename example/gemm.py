from kernelforge.common.matrix.dense import DenseMatrix
from kernelforge.common.context import Context
from kernelforge.common.aux import generate_tmp_matrix
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import FloatingPointType, Addressing
from kernelforge.generators.generator import Generator
from kernelforge.common.exceptions import GenerationError
import argparse


parser = argparse.ArgumentParser(description="Specify Backend and Arch of the GPU")
parser.add_argument("-a",
                    "--arch",
                    action="store",
                    help="Arch of the GPU, e.g sm_60 for Nvidia or gfx906 for AMD",
                    default="sm_60")
parser.add_argument("-b",
                    "--backend",
                    action="store",
                    help="Name of the Backend, currently cuda, hip, hipsycl and oneapi are supported",
                    default="cuda")


args = parser.parse_args()

mat_a = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 9])

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 9, 9])

mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 56, 9],
                    addressing=Addressing.STRIDED)

try:
  vm = Context(arch=args.arch, backend=args.backend, fp_type=FloatingPointType.FLOAT)

  gen = Generator([GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=mat_c,
                       alpha=1.1, beta=1.1)], vm)
  gen.generate()
  print(gen.get_kernel())
  print(gen.get_launcher())
  print(gen.get_header())

except GenerationError as err:
  print("ERROR: {}".format(err))
  raise err
