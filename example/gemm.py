from kernelforge.common.context import Context
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import FloatingPointType, Addressing
from kernelforge.generators.generator import Generator
from kernelforge.common.exceptions import GenerationError
import argparse
from kernelforge.common.matrix.boundingbox import BoundingBox
from kernelforge.common.matrix.tensor import Tensor


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

mat_a = Tensor([56, 18], Addressing.STRIDED, BoundingBox([0,0], [56,18]))

mat_b = Tensor([18, 18], Addressing.STRIDED, BoundingBox([0,0], [18,18]))

mat_c = Tensor([56, 18], Addressing.STRIDED, BoundingBox([0,0], [56,18]))


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
