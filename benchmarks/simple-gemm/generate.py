import os
import yaml
import argparse

from gemmgen import DenseMatrix, GemmGenerator, GenerationError
from gemmgen import arch

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--realsize', type=int, action='store',
                    help="size of real: 4(single)/8(double)")
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
arch = arch.produce("nvidia")

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

    path = os.path.join(dir_name, "kernels.cu")
    with open(path, 'w') as file:
        file.write("#include \"gemmgen_aux.h\"\n")
        file.write(krnl)
        file.write(lnch)

    path = os.path.join(dir_name, "kernels.h")
    with open(path, 'w') as file:
        file.write(header)

except GenerationError as err:
    print("ERROR: {}".format(err))