from gemmforge import DenseMatrix, GenerationError, GemmGenerator, arch
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

arch = arch.produce(args.manufacturer, args.sub_arch)

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

try:
    gen = GemmGenerator(arch, "float")
    gen.generate(mat_a, mat_b, mat_c, alpha=1.1, beta=1.1)
    print(gen.get_kernel())
    print(gen.get_launcher())
    print(gen.get_launcher_header())

except GenerationError as err:
    print("ERROR: {}".format(err))
    raise err

