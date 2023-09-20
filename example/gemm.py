from tensorforge import DenseMatrix, GenerationError, GemmGenerator
from tensorforge.vm import vm_factory
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
                    addressing="strided",
                    bbox=[0, 0, 56, 9])

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 9, 9])

mat_c = DenseMatrix(num_rows=56,
                    num_cols=9,
                    bbox=[0, 0, 56, 9],
                    addressing="strided")

try:
  vm = vm_factory(backend=args.backend,
                  arch=args.arch,
                  fp_type='float')

  gen = GemmGenerator(vm)
  gen.set(False, False, mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
  gen.generate()
  print(gen.get_kernel())
  print(gen.get_launcher())
  print(gen.get_launcher_header())

except GenerationError as err:
  print("ERROR: {}".format(err))
  raise err
