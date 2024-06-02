from kernelforge import GenerationError, GemmGenerator
from kernelforge.common.vm.vm import vm_factory
from kernelforge.common.matrix.tensor import Tensor
import os
import yaml
import argparse


parser = argparse.ArgumentParser()
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


def produce_matrix(spec):
    return Tensor(shape = [spec['num_rows'], spec['num_cols']],
                       addressing=spec['addressing'],
                       bbox=spec['bbox'])


stream = open('params.yaml', 'r')
config = yaml.safe_load(stream)

mat_d = produce_matrix(config['MatD'])
mat_a = produce_matrix(config['MatA'])
mat_b = produce_matrix(config['MatB'])
mat_c = produce_matrix(config['MatC'])

# generate a tmp matrix to hold an intermediate results
spec = {'num_rows': mat_a.get_actual_num_rows(),
        'num_cols': mat_b.get_actual_num_cols(),
        'addressing': 'strided',
        'bbox': None,
        'trans': False}
tmp = produce_matrix(spec)

alpha = config['alpha']
beta = config['beta']

try:
    kernels = []
    launches = []
    headers = []

    vm = vm_factory(backend=args.backend,
                    arch=args.arch,
                    fp_type='float' if args.realsize == 4 else 'double')

    gen = GemmGenerator(vm)
    gen.set(trans_a=config['trans_a'],
            trans_b=config['trans_b'],
            mat_a=mat_a,
            mat_b=mat_b,
            mat_c=tmp,
            alpha=1.0,
            beta=0.0,
            base_name='callFirstGemm')
    gen.generate()

    kernels.append(gen.get_kernel())
    launches.append(gen.get_launcher())
    headers.append(gen.get_launcher_header())

    gen = GemmGenerator(vm)
    gen.set(trans_a=config['trans_c'],
            trans_b=False,
            mat_a=mat_c,
            mat_b=tmp,
            mat_c=mat_d,
            alpha=config['alpha'],
            beta=config['beta'],
            base_name='callSecondGemm')
    gen.generate()

    kernels.append(gen.get_kernel())
    launches.append(gen.get_launcher())
    headers.append(gen.get_launcher_header())


    dir_name = './gen_code'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    path = None
    hw_descr = vm.get_hw_descr()
    if hw_descr.backend == 'cuda':
        path = os.path.join(dir_name, 'kernels.cu')
    elif hw_descr.backend == 'hip' or hw_descr.backend == 'hipsycl' or hw_descr.backend == 'oneapi':
        path = os.path.join(dir_name, 'kernels.cpp')

    with open(path, 'w') as file:
        for header_file in vm.get_headers():
          file.write(f'#include \"{header_file}\"\n')

        for kernel in kernels:
            file.write(kernel)
            print(kernel)

        for launcher in launches:
            file.write(launcher)
            print(launcher)

    path = os.path.join(dir_name, 'kernels.h')
    with open(path, 'w') as file:
        for header in headers:
            file.write(header)
            print(header)


except GenerationError as err:
    print('ERROR: {}'.format(err))
