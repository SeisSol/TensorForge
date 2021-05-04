from gemmforge import DenseMatrix, GenerationError, CsaGenerator
from gemmforge.vm import vm_factory
import argparse

parser = argparse.ArgumentParser(description="Specify Manufacturer and Sub_Arch of the GPU")
parser.add_argument("-m",
                    "--manufacturer",
                    action="store",
                    help="Name of the Manufacturer, currently nvidia and amd are supported",
                    default="nvidia")
parser.add_argument("-s",
                    "--sub_arch",
                    action="store",
                    help="Sub_arch of the GPU, e.g sm_60 for Nvidia or gfx906 for AMD",
                    default="sm_60")

args = parser.parse_args()

mat_a = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 8, 8],
                    transpose=False)

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing="strided",
                    bbox=[0, 0, 8, 8],
                    transpose=False)

try:
    vm = vm_factory(name=args.manufacturer,
                    sub_name=args.sub_arch,
                    fp_type="float")

    gen = CsaGenerator(vm)
    gen.generate(mat_a, mat_b, alpha=13, beta=6)
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print("ERROR: {}".format(err))
    raise err

