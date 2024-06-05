import argparse
import os
from io import StringIO

import yaml

from tensorforge.backend.instructions.builders.kernels.gemms.type import GemmKernelType
from tensorforge.generators.generator import GenerationError, Generator
from tensorforge.backend import writer
from tensorforge.common.basic_types import FloatingPointType, Addressing
from tensorforge.common.context import Context
from test_loader import TestLoader
from tensorforge.generators.descriptions import GemmDescr

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

vm = Context(backend=args.backend,
                arch=args.arch,
                fp_type=FloatingPointType.FLOAT if args.realsize == 4 else FloatingPointType.DOUBLE)

stream = open(args.specfile, 'r')
suites = yaml.safe_load(stream)['test_suites']

src = StringIO()
headers = StringIO()
tests_code = StringIO()

hw_descr = vm.get_vm().get_hw_descr()
precision = vm.fp_as_str()
with writer.Writer() as file:
  for header_file in vm.get_vm().get_headers():
    file.Include(f'{header_file}')

  src.write(file.stream.getvalue())

with writer.Writer() as file:
  file.Include('gtest/gtest.h')
  file.Include('comparators.h')
  file.Include('gemm_driver.h')
  file.Include('kernels.h')
  file.Include('gemm.h')
  file.Include('iostream')
  file('using namespace tensorforge::reference;')
  file.Emptyline()
  tests_code.write(file.stream.getvalue())

for suite in suites:
  for test in TestLoader(suite):
    trans_a, trans_b, mat_a, mat_b, mat_b_sparse, mat_c, alpha, beta, num_elements, matrix_b_type, test_name, kernel_type = test
    try:
      if kernel_type == "shr_mem":
        dense_kernel_type = GemmKernelType.SHR_MEM_BASED
        dense_sparse_kernel_type = GemmKernelType.DENSE_SPARSE_SHR_MEM_BASED
      elif kernel_type == "register_only":
        dense_kernel_type = GemmKernelType.REGISTER_ONLY_BASED
        dense_sparse_kernel_type = GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_BASED
      else:
        raise Exception("Wrong kernel_type string")

      T = "T"
      NT = ""
      

      generator1 = Generator([GemmDescr(trans_a=trans_a,
                       trans_b=trans_b,
                       a=mat_a, b=mat_b, c=mat_c, alpha=alpha, beta=beta)], vm)

      
      generator1.generate()
      src.write(generator1.get_kernel())
      src.write(generator1.get_launcher())
      headers.write(generator1.get_header())

      generator2 = Generator([GemmDescr(trans_a=trans_a,
                       trans_b=trans_b,
                       a=mat_a, b=mat_b_sparse, c=mat_c, alpha=alpha, beta=beta)], vm)
      generator2.generate()
      src.write(generator2.get_kernel())
      src.write(generator2.get_launcher())
      headers.write(generator2.get_header())

      with writer.Writer() as file:
        with file.GoogleTestSuit('DenseXSparseGemmTest', test_name):
          file(f'int sizeA = {mat_a.num_rows} * {mat_a.num_cols};')
          file(f'int sizeB_dense = {mat_b.num_rows} * {mat_b.num_cols};')
          file(f'int sizeB_sparse = {mat_b.num_rows} * {mat_b.num_cols};')
          file(f'int sizeC = {mat_c.num_rows} * {mat_c.num_cols};')
          file(f'int rowA = {mat_a.num_rows};')
          file(f'int colA = {mat_a.num_cols};')
          file(f'int rowB = {mat_b.num_rows};')
          file(f'int colB = {mat_b.num_cols};')
          file(f'int rowC = {mat_c.num_rows};')
          file(f'int colC = {mat_c.num_cols};')
          file.Emptyline()

          M = mat_a.get_actual_num_cols() if trans_a else mat_a.get_actual_num_rows()
          N = mat_b.get_actual_num_rows() if trans_b else mat_b.get_actual_num_cols()
          K = mat_a.get_actual_num_rows() if trans_a else mat_a.get_actual_num_cols()

          file(f'int M = {M};')
          file(f'int N = {N};')
          file(f'int K = {K};')
          file.Emptyline()

          file(f'int lda = {mat_a.num_rows};')
          file(f'int ldb = {mat_b.num_rows};')
          file(f'int ldc = {mat_c.num_rows};')
          file.Emptyline()

          file(f'int nextA = {0 if mat_a.addressing == "none" else "sizeA"};')
          file(f'int nextB_dense = {0 if mat_b.addressing == "none" else "sizeB_dense"};')
          file(f'int nextB_sparse = {0 if mat_b.addressing == "none" else "sizeB_sparse"};')
          file(f'int nextC = {0 if mat_c.addressing == "none" else "sizeC"};')
          file.Emptyline()

          file(f'int offsetA = 0;')
          file(f'int offsetB_dense = 0;')
          file(f'int offsetB_sparse = 0;')
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
            f'SetUp(rowA, colA, rowB, colB, rowC, colC, numElements, \"{matrix_b_type}\", {"true" if trans_b else "false"});')
          file(f'Driver.prepareData(\"{matrix_b_type}\");')

          args = []
          args.append("DeviceA, 0")
          args.append("DeviceB_dense, 0")
          args.append("DeviceC1, 0")
          args.append('numElements')
          args.append('flags')
          args.append('Driver.getTestStream()')
          args = ', '.join(args)
          file(f'{generator1.get_base_name()}({args});')
          args = ['M', 'ldc', 'N', 'offsetC', 'sizeC', 'numElements', "false"]
          file(f'Driver.retrieveResults({", ".join(args)});')
          args = []
          args.append("DeviceA, 0")
          args.append("DeviceB_sparse, 0")
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
else:
  path = os.path.join(dir_name, 'kernels.cpp')
with open(path, 'w') as file:
  file.write(src.getvalue())

path = os.path.join(dir_name, 'kernels.h')
with open(path, 'w') as file:
  file.write(headers.getvalue())

tests_code.close()
src.close()
headers.close()
