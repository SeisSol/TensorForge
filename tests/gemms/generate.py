from gemmforge import GenerationError, GemmGenerator
from gemmforge.vm import vm_factory
from gemmforge import constructs
from io import StringIO
from test_loader import TestLoader
import os
import yaml
import argparse


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

vm = vm_factory(name=args.manufacturer,
                sub_name=args.sub_arch,
                fp_type="float" if args.realsize == 4 else "double")

generator = GemmGenerator(vm)

stream = open(args.specfile, 'r')
suites = yaml.safe_load(stream)["test_suites"]

src = StringIO()
headers = StringIO()
tests_code = StringIO()

hw_descr = vm.get_hw_descr()
precision = vm.fp_as_str()
with constructs.Cpp(StringIO()) as file:
    file.Include("gemmforge_aux.h")
    if hw_descr.manufacturer == "amd":
        file.Include("hip/hip_runtime.h")
    elif hw_descr.manufacturer == "hipsycl" or hw_descr.manufacturer == "oneapi" :
        file.Include("CL/sycl.hpp")
    src.write(file.stream.getvalue())

with constructs.Cpp(StringIO()) as file:
    file.Include("gtest/gtest.h")
    file.Include("comparators.h")
    file.Include("gemm_driver.h")
    file.Include("kernels.h")
    file.Include("gemm.h")
    file.Include("iostream")
    file.Expression("using namespace gemmforge::reference")
    file.Emptyline()
    tests_code.write(file.stream.getvalue())

for suite in suites:
    for test in TestLoader(suite):
        mat_a, mat_b, mat_c, alpha, beta, num_elements, test_name = test

        try:
            generator.set(mat_a, mat_b, mat_c, alpha, beta)
            generator.generate()
            src.write(generator.get_kernel())
            src.write(generator.get_launcher())
            headers.write(generator.get_launcher_header())

            with constructs.Cpp(StringIO()) as file:
                with file.GoogleTestSuit("DenseGemmTest", test_name):
                    file.Expression("int SizeA = {} * {}".format(mat_a.num_rows, mat_a.num_cols))
                    file.Expression("int SizeB = {} * {}".format(mat_b.num_rows, mat_b.num_cols))
                    file.Expression("int SizeC = {} * {}".format(mat_c.num_rows, mat_c.num_cols))
                    file.Emptyline()

                    M = mat_a.get_actual_num_cols() if mat_a.transpose else mat_a.get_actual_num_rows()
                    N = mat_b.get_actual_num_rows() if mat_b.transpose else mat_b.get_actual_num_cols()
                    K = mat_a.get_actual_num_rows() if mat_a.transpose else mat_a.get_actual_num_cols()

                    file.VariableDeclaration("int", "M", M)
                    file.VariableDeclaration("int", "N", N)
                    file.VariableDeclaration("int", "K", K)
                    file.Emptyline()

                    file.VariableDeclaration("int", "Lda", mat_a.num_rows)
                    file.VariableDeclaration("int", "Ldb", mat_b.num_rows)
                    file.VariableDeclaration("int", "Ldc", mat_c.num_rows)
                    file.Emptyline()

                    file.VariableDeclaration("int", "NextA", "{}".format(0 if mat_a.addressing == "none" else "SizeA"))
                    file.VariableDeclaration("int", "NextB", "{}".format(0 if mat_b.addressing == "none" else "SizeB"))
                    file.VariableDeclaration("int", "NextC", "{}".format(0 if mat_c.addressing == "none" else "SizeC"))
                    file.Emptyline()

                    file.VariableDeclaration("int", "OffsetA",
                                             "{} * {} + {}".format(mat_a.num_rows, mat_a.bbox[1], mat_a.bbox[0]))
                    file.VariableDeclaration("int", "OffsetB",
                                             "{} * {} + {}".format(mat_b.num_rows, mat_b.bbox[1], mat_b.bbox[0]))
                    file.VariableDeclaration("int", "OffsetC",
                                             "{} * {} + {}".format(mat_c.num_rows, mat_c.bbox[1], mat_c.bbox[0]))
                    file.Emptyline()

                    file.VariableDeclaration("LayoutType", "TransA",
                                             "LayoutType::{}".format("Trans" if mat_a.transpose else "NoTrans"))
                    file.VariableDeclaration("LayoutType", "TransB",
                                             "LayoutType::{}".format("Trans" if mat_b.transpose else "NoTrans"))
                    file.Emptyline()

                    file.VariableDeclaration(precision, "alpha", alpha)
                    file.VariableDeclaration(precision, "beta", beta)
                    file.VariableDeclaration("int", "NumElements", num_elements)
                    file.Emptyline()

                    file.Expression("SetUp(SizeA, SizeB, SizeC, NumElements)")
                    file.Expression("Driver.prepareData()")

                    args = []
                    args.append("{}, 0".format("DeviceShuffledA" if mat_a.addressing == "pointer_based" else "DeviceA"))
                    args.append("{}, 0".format("DeviceShuffledB" if mat_b.addressing == "pointer_based" else "DeviceB"))
                    args.append("{}, 0".format("DeviceShuffledC" if mat_c.addressing == "pointer_based" else "DeviceC"))
                    args.append("NumElements")
                    args.append("Driver.getTestStream()")

                    args = ", ".join(args)
                    file.Expression("{}({})".format(generator.get_base_name(), args))


                    args = ["TransA", "TransB", "M", "N", "K"]
                    args.extend(["alpha", "&HostA[OffsetA], Lda"])
                    args.extend(["&HostB[OffsetB]", "Ldb"])
                    args.extend(["beta", "&HostC[OffsetC], Ldc"])
                    args.extend(["NextA", "NextB", "NextC", "NumElements"])

                    args = ", ".join(args)
                    file.Expression("gemm({})".format(args))

                    args = ["M", "Ldc", "N", "OffsetC", "SizeC", "NumElements"]
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

if hw_descr.manufacturer == "nvidia":
    path = os.path.join(dir_name, "kernels.cu")
elif hw_descr.manufacturer == "amd" or hw_descr.manufacturer == "hipsycl" or hw_descr.manufacturer == "oneapi" :
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
