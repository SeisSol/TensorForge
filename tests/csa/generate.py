from gemmforge import GenerationError, CsaGenerator
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
    raise ValueError('A test spec file doens\'t exists')

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
  file.Include("gtest/gtest.h")
  file.Include("comparators.h")
  file.Include("csa_driver.h")
  file.Include("kernels.h")
  file.Include("csa.h")
  file.Include("iostream")
  file.Include("algorithm")
  file('using namespace csagen::reference;')
  file.Emptyline()
  tests_code.write(file.stream.getvalue())

for suite in suites:
  for test in TestLoader(suite):
    mat_a, mat_c, alpha, beta, num_elements, test_name = test

    try:
      generator = CsaGenerator(vm)
      generator.set(mat_a, mat_c, alpha, beta)
      generator.generate()
      src.write(generator.get_kernel())
      print(generator.get_kernel())  # TODO: delete after debugging
      src.write(generator.get_launcher())
      print(generator.get_launcher())  # TODO: delete after debugging
      headers.write(generator.get_launcher_header())

      with constructs.Cpp(StringIO()) as file:
        with file.GoogleTestSuit('DenseCsaTest', test_name):
          # A and B must be both MxN matrices (or NxM if transposed)
          M = mat_a.get_actual_num_rows()
          N = mat_a.get_actual_num_cols()

          file(f'int M = {M};')
          file(f'int N = {N};')
          file.Emptyline()

          file(f'int sizeA = {mat_a.num_rows} * {mat_a.num_cols};')
          file(f'int sizeC = {mat_c.num_rows} * {mat_c.num_cols};')
          # count rows must be equal at A and B (matrices perform component-wise add!)
          file(f'int lda = {mat_a.num_rows};')
          file(f'int ldc = {mat_c.num_rows};')
          file.Emptyline()

          file(f'int offsetA = {mat_a.num_rows} * {mat_a.bbox[1]} + {mat_a.bbox[0]};')
          file(f'int offsetC = {mat_c.num_rows} * {mat_c.bbox[1]} + {mat_c.bbox[0]};')
          file.Emptyline()

          file(f'{precision} alpha = {alpha};')
          file(f'{precision} beta = {beta};')
          file(f'int numElements = {num_elements};')
          file(f'unsigned *flags = nullptr;')
          file.Emptyline()

          # NOTE: test driver expects three matrices but it is ok
          # for our use case to just not use the matrix C
          file(f'SetUp(sizeA, sizeA, sizeC, numElements);')

          # this fills the matrices with random data (on host and device!)
          file('Driver.prepareData();')

          args = []
          args.append('alpha')
          args.append(
            f'{"DeviceShuffledA" if mat_a.addressing == "pointer_based" else "DeviceA"}, 0'
          )
          args.append(
            f'{"DeviceShuffledC" if mat_c.addressing == "pointer_based" else "DeviceC"}, 0'
          )
          args.append('numElements')
          args.append('flags')
          args.append('Driver.getTestStream()')

          # calls the kernel
          file(f'{generator.get_base_name()}({", ".join(args)});')
          if beta == 0.0:
            file(f'std::fill(&HostC[0], &HostC[sizeC * numElements], 0.0);')

          args = ['M', 'N', 'alpha', '&HostA[offsetA]', 'lda']
          args.extend(['beta', '&HostC[offsetC]', 'ldc', 'sizeA', 'sizeC', 'numElements'])

          # calls the reference code
          file(f'csa({", ".join(args)});')

          args = ['M', 'ldc', 'N', 'offsetC', 'sizeC', 'numElements']
          # this copies the results into packed arrays (without offset)
          file(f'Driver.packResults({", ".join(args)});')
          file(f'bool result;')
          file(f'result = Driver.isTestPassed<L1NormComparator>();')
          file(f'EXPECT_EQ(true, result);')

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
  print('Backend not supported, could not write kernel file')
with open(path, 'w') as file:
  file.write(src.getvalue())

path = os.path.join(dir_name, 'kernels.h')
with open(path, 'w') as file:
  file.write(headers.getvalue())

tests_code.close()
src.close()
headers.close()
