from kernelforge import DenseMatrix, GenerationError, CsaGenerator
from kernelforge.vm import vm_factory
import argparse


parser = argparse.ArgumentParser(description='Specify Backend and Arch of the GPU')
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

mat_a = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing='strided',
                    bbox=[0, 0, 9, 9])

mat_b = DenseMatrix(num_rows=9,
                    num_cols=9,
                    addressing='strided',
                    bbox=[0, 0, 9, 9])

try:
  vm = vm_factory(backend=args.backend,
                  arch=args.arch,
                  fp_type='float')
  
  gen = CsaGenerator(vm)
  gen.set(mat_a, mat_b, alpha=13, beta=0)
  gen.generate()
  print(gen.get_kernel())
  print(gen.get_launcher())
  print(gen.get_launcher_header())

except GenerationError as err:
  print('ERROR: {}'.format(err))
  raise err
