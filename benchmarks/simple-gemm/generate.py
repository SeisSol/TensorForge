import os
import yaml
import argparse

from gemmforge import DenseMatrix, GemmGenerator, GenerationError
from gemmforge import arch

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--realsize', type=int, action='store',
                    help="size of real: 4(single)/8(double)")
parser.add_argument("-m", "--manufacturer",
                    type=str,
                    action="store",
                    help="Name of the Manufacturer, currently nvidia and amd are supported",
                    default="nvidia")
parser.add_argument("-s", "--sub_arch",
                    type=str,
                    action="store",
                    help="Sub_arch of the GPU, e.g sm_60 for Nvidia or gfx906 for AMD",
                    default="sm_60")
args = parser.parse_args()

def produce_matrix(spec):
    return DenseMatrix(num_rows=spec["num_rows"],
                       num_cols=spec["num_cols"],
                       addressing=spec["addressing"],
                       bbox=spec["bbox"],
                       transpose=spec["trans"])


stream = open("params.yaml", 'r')
config = yaml.safe_load(stream)

mat_a = produce_matrix(config["MatA"])
mat_b = produce_matrix(config["MatB"])
mat_c = produce_matrix(config["MatC"])
alpha = config["alpha"]
beta = config["beta"]
arch = arch.produce(args.manufacturer, args.sub_arch)

try:
    gen = GemmGenerator(arch, "float" if args.realsize == 4 else "double")
    gen.generate(mat_a, mat_b, mat_c, alpha, beta, base_name="gemm")

    krnl = gen.get_kernel()
    lnch = gen.get_launcher()
    header = gen.get_launcher_header()

    print(krnl)
    print(lnch)

    dir_name = "./gen_code"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    path = None
    if arch.manufacturer == "nvidia":
        path = os.path.join(dir_name, "kernels.cu")
    elif arch.manufacturer == "amd":
        path = os.path.join(dir_name, "kernels.cpp")
        
    with open(path, 'w') as file:
        file.write("#include \"gemmgen_aux.h\"\n")
        if arch.manufacturer == "amd":
            file.write("#include \"hip/hip_runtime.h\"\n")
        file.write(krnl)
        file.write(lnch)

    path = os.path.join(dir_name, "kernels.h")
    with open(path, 'w') as file:
        file.write(header)

except GenerationError as err:
    print("ERROR: {}".format(err))
