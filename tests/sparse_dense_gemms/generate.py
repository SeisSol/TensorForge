import argparse
import os
from io import StringIO

import yaml

from gemmforge import GemmKernelType
from gemmforge import GenerationError, GemmGenerator
from gemmforge import constructs
from gemmforge.vm import vm_factory
from test_loader import TestLoader

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--specfile', action='store', help='path to a yaml file with a test spec')
parser.add_argument('-r', '--realsize', type=int, action='store',
                    help='size of real: 4(single)/8(double)')
parser.add_argument('-a',
                    '--arch',
                    action='store',
                    help='Arch of the GPU, e.g sm_60 for Nvidia or gfx906 for AMD',
                    default='sm_60')
parser.add_argument('-b',
                    '--backend',
                    action='store',
                    help='Name of the Backend, currently cuda, hip, hipsycl and oneapi are supported',
                    default='cuda')
args = parser.parse_args()

# check input parameters. Make sure there are valid ones
if not args.specfile:
  raise ValueError('A test spec file has not been provided')
else:
  if not os.path.exists(args.specfile):
    raise ValueError('A test spec file doesn\'t exists')

if not args.realsize:
  raise ValueError('Please, specify floating point size: 4 or 8')
else:
  if not ((args.realsize == 4) or (args.realsize == 8)):
    raise ValueError('Floating point size must be either 4 or 8')

vm = vm_factory(backend=args.backend,
                arch=args.arch,
                fp_type='float' if args.realsize == 4 else 'double')

stream = open(args.specfile, 'r')
suites = yaml.safe_load(stream)['test_suites']

src = StringIO()
headers = StringIO()
tests_code = StringIO()

hw_descr = vm.get_hw_descr()
precision = vm.fp_as_str()
with constructs.Cpp(StringIO()) as file:
  for header_file in vm.get_headers():
    file.Include(f'{header_file}')

  src.write(file.stream.getvalue())

with constructs.Cpp(StringIO()) as file:
  file.Include('gtest/gtest.h')
  file.Include('comparators.h')
  file.Include('gemm_driver.h')
  file.Include('kernels.h')
  file.Include('gemm.h')
  file.Include('iostream')
  file('using namespace gemmforge::reference;')
  file.Emptyline()
  tests_code.write(file.stream.getvalue())

for suite in suites:
  for test in TestLoader(suite):
    trans_a, trans_b, mat_a_dense, mat_a_sparse, mat_b, mat_c, alpha, beta, num_elements, matrix_a_type, test_name, kernel_type = test

    try:
      if kernel_type == "shr_mem":
        dense_kernel_type = GemmKernelType.SHR_MEM_BASED
        sparse_dense_kernel_type = GemmKernelType.SPARSE_DENSE_SHR_MEM_BASED
      elif kernel_type == "register_only":
        dense_kernel_type = GemmKernelType.REGISTER_ONLY_BASED
        sparse_dense_kernel_type = GemmKernelType.SPARSE_DENSE_REGISTER_ONLY_BASED
      else:
        raise Exception("Wrong kernel_type string")

      generator1 = GemmGenerator(vm=vm, kernel_type=dense_kernel_type)
      T = "T"
      NT = ""
      generator1.set(trans_a, trans_b, mat_a_dense, mat_b, mat_c, alpha, beta,
                     base_name=f"A{T if trans_a else NT}_{matrix_a_type}_B{T if trans_b else NT}_SparseXDense_{kernel_type}")
      generator1.generate()
      src.write(generator1.get_kernel())
      src.write(generator1.get_launcher())
      headers.write(generator1.get_launcher_header())

      generator2 = GemmGenerator(vm=vm, kernel_type=sparse_dense_kernel_type)
      generator2.set(trans_a, trans_b, mat_a_sparse, mat_b, mat_c, alpha, beta,
                     base_name=f"A{T if trans_a else NT}_{matrix_a_type}_B{T if trans_b else NT}_SparseXSparse_{kernel_type}")
      generator2.generate()
      src.write(generator2.get_kernel())
      src.write(generator2.get_launcher())
      headers.write(generator2.get_launcher_header())

      with constructs.Cpp(StringIO()) as file:
        with file.GoogleTestSuit('SparseXDenseGemmTest', test_name):
          file(f'int sizeA_dense = {mat_a_dense.num_rows} * {mat_b.num_cols};')
          file(f'int sizeA_sparse = {mat_b.num_rows} * {mat_b.num_cols};')
          file(f'int sizeB = {mat_b.num_rows} * {mat_b.num_cols};')
          file(f'int sizeC = {mat_c.num_rows} * {mat_c.num_cols};')
          file(f'int rowA = {mat_a_dense.num_rows};')
          file(f'int colA = {mat_a_dense.num_cols};')
          file(f'int rowB = {mat_b.num_rows};')
          file(f'int colB = {mat_b.num_cols};')
          file(f'int rowC = {mat_c.num_rows};')
          file(f'int colC = {mat_c.num_cols};')
          file.Emptyline()

          M = mat_a_dense.get_actual_num_cols() if trans_a else mat_a_dense.get_actual_num_rows()
          N = mat_b.get_actual_num_rows() if trans_b else mat_b.get_actual_num_cols()
          K = mat_a_dense.get_actual_num_rows() if trans_a else mat_a_dense.get_actual_num_cols()

          file(f'int M = {M};')
          file(f'int N = {N};')
          file(f'int K = {K};')
          file.Emptyline()

          file(f'int lda = {mat_a_dense.num_rows};')
          file(f'int ldb = {mat_b.num_rows};')
          file(f'int ldc = {mat_c.num_rows};')
          file.Emptyline()

          file(f'int nextA_dense = {0 if mat_a_dense.addressing == "none" else "sizeA_dense"};')
          file(f'int nextA_sparse = {0 if mat_a_sparse.addressing == "none" else "sizeA_sparse"};')
          file(f'int nextB = {0 if mat_b.addressing == "none" else "sizeB"};')
          file(f'int nextC = {0 if mat_c.addressing == "none" else "sizeC"};')
          file.Emptyline()

          file(f'int offsetA_dense = 0;')
          file(f'int offsetA_sparse = 0;')
          file(f'int offsetB = 0;')
          file(f'int offsetC = 0;')
          file.Emptyline()

          file(f'LayoutType transA = LayoutType::{"Trans" if trans_a else "NoTrans"};')
          file(f'LayoutType transB = LayoutType::{"Trans" if trans_b else "NoTrans"};')
          file.Emptyline()

          file(f'{precision} alpha = {alpha};')
          file(f'{precision} beta = {beta};')
          file(f'int numElements = {num_elements};')
          file(f'unsigned* flags = nullptr;')
          file.Emptyline()

          file(
            f'SetUp(rowA, colA, rowB, colB, rowC, colC, numElements, \"{matrix_a_type}\", {"true" if trans_a else "false"});')
          file(f'Driver.prepareData(\"{matrix_a_type}\");')

          args = []
          args.append("DeviceA_dense, 0")
          args.append("DeviceB, 0")
          args.append("DeviceC1, 0")
          args.append('numElements')
          args.append('flags')
          args.append('Driver.getTestStream()')
          args = ', '.join(args)
          file(f'{generator1.get_base_name()}({args});')
          args = ['M', 'ldc', 'N', 'offsetC', 'sizeC', 'numElements', "false"]
          file(f'Driver.retrieveResults({", ".join(args)});')
          args = []
          args.append("DeviceA_sparse, 0")
          args.append("DeviceB, 0")
          args.append("DeviceC2, 0")
          args.append('numElements')
          args.append('flags')
          args.append('Driver.getTestStream()')
          args = ', '.join(args)
          file(f'{generator2.get_base_name()}({args});')
          args = ['M', 'ldc', 'N', 'offsetC', 'sizeC', 'numElements', "true"]
          file(f'Driver.retrieveResults({", ".join(args)});')

          file(f'bool Result;')
          file(f'Result = Driver.checkEq();')
          file('EXPECT_EQ(true, Result);')

        file.Emptyline()
        file.Emptyline()
        tests_code.write(file.stream.getvalue())

    except GenerationError as error:
      tests_code.close()
      src.close()
      headers.close()

      print(f'ERROR: {error}')
      raise error

dir_name = './gen_code'
if not os.path.exists(dir_name):
  os.mkdir(dir_name)

path = os.path.join(dir_name, 'test.cpp')
with open(path, 'w') as file:
  file.write(tests_code.getvalue())

if hw_descr.backend == 'cuda':
  path = os.path.join(dir_name, 'kernels.cu')
elif hw_descr.backend == 'hip' or hw_descr.backend == 'hipsycl' or hw_descr.backend == 'oneapi':
  path = os.path.join(dir_name, 'kernels.cpp')
else:
  print('Backend is not supported, could not write kernel file')
with open(path, 'w') as file:
  file.write(src.getvalue())

path = os.path.join(dir_name, 'kernels.h')
with open(path, 'w') as file:
  file.write(headers.getvalue())

tests_code.close()
src.close()
headers.close()
