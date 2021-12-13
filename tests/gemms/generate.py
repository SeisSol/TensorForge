from gemmforge import GenerationError, GemmGenerator
from gemmforge.vm import vm_factory
from gemmforge import constructs
from io import StringIO
from test_loader import TestLoader
import os
import yaml
import argparse


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
    trans_a, trans_b, mat_a, mat_b, mat_c, alpha, beta, num_elements, test_name = test

    try:
      generator = GemmGenerator(vm)
      generator.set(trans_a, trans_b, mat_a, mat_b, mat_c, alpha, beta)
      generator.generate()
      src.write(generator.get_kernel())
      src.write(generator.get_launcher())
      headers.write(generator.get_launcher_header())

      with constructs.Cpp(StringIO()) as file:
        with file.GoogleTestSuit('DenseGemmTest', test_name):
          file(f'int sizeA = {mat_a.num_rows} * {mat_a.num_cols};')
          file(f'int sizeB = {mat_b.num_rows} * {mat_b.num_cols};')
          file(f'int sizeC = {mat_c.num_rows} * {mat_c.num_cols};')
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
          file(f'int nextB = {0 if mat_b.addressing == "none" else "sizeB"};')
          file(f'int nextC = {0 if mat_c.addressing == "none" else "sizeC"};')
          file.Emptyline()

          file(f'int offsetA = {mat_a.num_rows} * {mat_a.bbox[1]} + {mat_a.bbox[0]};')
          file(f'int offsetB = {mat_b.num_rows} * {mat_b.bbox[1]} + {mat_b.bbox[0]};')
          file(f'int offsetC = {mat_c.num_rows} * {mat_c.bbox[1]} + {mat_c.bbox[0]};')
          file.Emptyline()

          file(f'LayoutType transA = LayoutType::{"Trans" if trans_a else "NoTrans"};')
          file(f'LayoutType transB = LayoutType::{"Trans" if trans_b else "NoTrans"};')
          file.Emptyline()

          file(f'{precision} alpha = {alpha};')
          file(f'{precision} beta = {beta};')
          file(f'int numElements = {num_elements};')
          file(f'unsigned* flags = nullptr;')
          file.Emptyline()

          file('SetUp(sizeA, sizeB, sizeC, numElements);')
          file('Driver.prepareData();')

          args = []
          args.append(
            f'{"DeviceShuffledA" if mat_a.addressing == "pointer_based" else "DeviceA"}, 0')
          args.append(
            f'{"DeviceShuffledB" if mat_b.addressing == "pointer_based" else "DeviceB"}, 0')
          args.append(
            f'{"DeviceShuffledC" if mat_c.addressing == "pointer_based" else "DeviceC"}, 0')
          args.append('numElements')
          args.append('flags')
          args.append('Driver.getTestStream()')

          args = ', '.join(args)
          file(f'{generator.get_base_name()}({args});')

          args = ['transA', 'transB', 'M', 'N', 'K']
          args.extend(['alpha', '&HostA[offsetA]', 'lda'])
          args.extend(['&HostB[offsetB]', 'ldb'])
          args.extend(['beta', '&HostC[offsetC]', 'ldc'])
          args.extend(['nextA', 'nextB', 'nextC'])
          args.extend(['numElements'])

          args = ", ".join(args)
          file(f'gemm({args});')

          args = ['M', 'ldc', 'N', 'offsetC', 'sizeC', 'numElements']
          file(f'Driver.packResults({", ".join(args)});')
          file(f'bool Result;')
          file(f'Result = Driver.isTestPassed<L1NormComparator>();')
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
elif hw_descr.backend== 'hip' or hw_descr.backend == 'hipsycl' or hw_descr.backend == 'oneapi':
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
