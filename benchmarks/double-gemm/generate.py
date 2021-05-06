from gemmforge import DenseMatrix, GenerationError, GemmGenerator
from gemmforge.vm import vm_factory
import os
import yaml
import argparse


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

mat_d = produce_matrix(config["MatD"])
mat_a = produce_matrix(config["MatA"])
mat_b = produce_matrix(config["MatB"])
mat_c = produce_matrix(config["MatC"])

# generate a tmp matrix to hold an intermediate results
spec = {"num_rows": mat_a.get_actual_num_rows(),
        "num_cols": mat_b.get_actual_num_cols(),
        "addressing": "strided",
        "bbox": None,
        "trans": False}
tmp = produce_matrix(spec)

alpha = config["alpha"]
beta = config["beta"]

try:
    kernels = []
    launches = []
    headers = []

    vm = vm_factory(name=args.manufacturer,
                    sub_name=args.sub_arch,
                    fp_type="float" if args.realsize == 4 else "double")

    gen = GemmGenerator(vm)
    gen.generate(mat_a, mat_b, tmp, 1.0, 0.0, base_name="callFirstGemm")

    kernels.append(gen.get_kernel())
    launches.append(gen.get_launcher())
    headers.append(gen.get_launcher_header())

    gen.generate(mat_c, tmp, mat_d, alpha, beta, base_name="callSecondGemm")

    kernels.append(gen.get_kernel())
    launches.append(gen.get_launcher())
    headers.append(gen.get_launcher_header())


    dir_name = "./gen_code"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    path = None
    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer == "nvidia":
        path = os.path.join(dir_name, "kernels.cu")
    elif hw_descr.manufacturer == "amd" or hw_descr.manufacturer == "sycl":
        path = os.path.join(dir_name, "kernels.cpp")

    with open(path, 'w') as file:
        file.write("#include \"gemmforge_aux.h\"\n")
        if hw_descr.manufacturer == "amd":
            file.write("#include \"hip/hip_runtime.h\"\n")
        elif hw_descr.manufacturer == "sycl":
            file.write("#include <CL/sycl.hpp>\n")

        for kernel in kernels:
            file.write(kernel)
            print(kernel)

        for launcher in launches:
            file.write(launcher)
            print(launcher)


    path = os.path.join(dir_name, "kernels.h")
    with open(path, 'w') as file:
        for header in headers:
            file.write(header)
            print(header)


except GenerationError as err:
    print("ERROR: {}".format(err))
