import os
import yaml
import argparse

from gemmforge import GenerationError
from gemmforge import arch
from gemmforge import constructs
from io import StringIO
from gemmforge.initializers import ExactInitializer
from tests.initializers.test_loader import TestLoader

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--specfile', action='store', help="path to a yaml file with a test spec")
parser.add_argument('-r', '--realsize', type=int, action='store', help="size of real: 4(single)/8(double)")
parser.add_argument("-m",
                    "--manufacturer",
                    action="store",
                    help="Name of the Manufacturer, currently nvidia and amd are supported",
                    default="nvidia")
parser.add_argument("-a",
                    "--sub_arch",
                    action="store",
                    help="Sub_arch of the GPU, e.g sm_60 for Nvidia or gfx906 for AMD",
                    default="sm_61")
args = parser.parse_args()

# check input parameters. Make sure there are valid ones
if not args.specfile:
    raise ValueError("A test spec file has not been provided")
else:
    if not os.path.exists(args.specfile):
        raise ValueError("A test spec file doen't exists")

if not args.realsize:
    raise ValueError("Please, specify floating point size: 4 or 8")
else:
    if not ((args.realsize == 4) or (args.realsize == 8)):
        raise ValueError("Floating point size must be either 4 or 8")

arch = arch.produce(args.manufacturer, args.sub_arch)
stream = open(args.specfile, 'r')
suites = yaml.safe_load(stream)["test_suites"]

src = StringIO()
headers = StringIO()
tests_code = StringIO()

with constructs.Cpp(StringIO()) as file:
    file.Include("gemmgen_aux.h")
    if arch.manufacturer == "amd":
        file.Include("hip/hip_runtime.h")
    elif arch.manufacturer == "sycl":
        file.Include("CL/sycl.hpp")
    src.write(file.stream.getvalue())

with constructs.Cpp(StringIO()) as file:
    file.Include("gtest/gtest.h")
    file.Include("comparators.h")
    file.Include("initializer_driver.h")
    file.Include("kernels.h")
    file.Include("initialize.h")
    file.Include("iostream")
    file.Expression("using namespace initgen::reference")
    file.Emptyline()
    tests_code.write(file.stream.getvalue())

for suite in suites:
    for test in TestLoader(suite):
        mat_c, alpha, num_elements, test_name = test
        generator = ExactInitializer(alpha, mat_c, arch, "double")

        try:
            generator.generate()
            src.write(generator.get_kernel())
            src.write(generator.get_launcher())
            headers.write(generator.get_launcher_header())

            with constructs.Cpp(StringIO()) as file:
                with file.GoogleTestSuit("DenseInitializerTest", test_name):
                    M = mat_c.get_actual_num_cols() if mat_c.transpose else mat_c.get_actual_num_rows()
                    N = mat_c.get_actual_num_rows() if mat_c.transpose else mat_c.get_actual_num_cols()

                    file.VariableDeclaration("int", "M", M)
                    file.VariableDeclaration("int", "N", N)
                    file.Emptyline()

                    file.Expression("int Size = {} * {}".format(M, N))
                    file.VariableDeclaration("int", "Ld", mat_c.num_rows)
                    file.Emptyline()

                    file.VariableDeclaration("int", "OffsetC",
                                             "{} * {} + {}".format(mat_c.num_rows, mat_c.bbox[1], mat_c.bbox[0]))

                    file.VariableDeclaration("int", "NumElements", num_elements)
                    file.Emptyline()

                    file.Expression("SetUp(Size, Size, Size, NumElements)")
                    file.Expression("Driver.prepareData()")

                    args = []
                    args.append("DeviceC")
                    args.append("OffsetC")
                    args.append("NumElements")
                    args.append("Driver.getTestStream()")

                    args = ", ".join(args)
                    # calls the kernel
                    file.Expression("{}({})".format(generator.get_base_name(), args))

                    args = ["M", "N", f"{alpha}", "&HostC[OffsetC]", "Ld", "Size", "NumElements"]

                    args = ", ".join(args)
                    # calls the reference code
                    file.Expression("initialize({})".format(args))

                    args = ["M", "Ld", "N", "OffsetC", "Size", "NumElements"]
                    # this copies the results into packed arrays (without offset)
                    file.Expression("Driver.packResults({})".format(", ".join(args)))
                    file.VariableDeclaration("bool", "Result")
                    file.Assignment("Result", "Driver.isTestPassed<L1NormComparator>()")
                    file.Expression("EXPECT_EQ(true, Result)")

                file.Emptyline()
                file.Emptyline()
                tests_code.write(file.stream.getvalue())

        except GenerationError as error:
            tests_code.close()
            src.close()
            headers.close()

            print("ERROR: {}".format(error))
            raise error

dir_name = "./gen_code"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

path = os.path.join(dir_name, "test.cpp")
with open(path, 'w') as file:
    file.write(tests_code.getvalue())

if arch.manufacturer == "nvidia":
    path = os.path.join(dir_name, "kernels.cu")
elif arch.manufacturer == "amd" or arch.manufacturer == "sycl":
    path = os.path.join(dir_name, "kernels.cpp")
else:
    print("Manufacturer not supported, could not write kernel file")
with open(path, 'w') as file:
    file.write(src.getvalue())

path = os.path.join(dir_name, "kernels.h")
with open(path, 'w') as file:
    file.write(headers.getvalue())

tests_code.close()
src.close()
headers.close()
