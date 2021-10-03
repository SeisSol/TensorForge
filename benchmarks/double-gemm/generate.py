from gemmforge import DenseMatrix, GenerationError, GemmGenerator
from gemmforge.vm import vm_factory
import os
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--realsize', type=int, action='store',
                    help='size of real: 4(single)/8(double)')
parser.add_argument('-m', '--manufacturer',
                    type=str,
                    action='store',
                    help='Name of the Manufacturer, currently nvidia and amd are supported',
                    default='nvidia')
parser.add_argument('-s', '--sub_arch',
                    type=str,
                    action='store',
                    help='Sub_arch of the GPU, e.g sm_60 for Nvidia or gfx906 for AMD',
                    default='sm_60')
args = parser.parse_args()


def produce_matrix(spec):
    return DenseMatrix(num_rows=spec['num_rows'],
                       num_cols=spec['num_cols'],
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

    vm = vm_factory(name=args.manufacturer,
                    sub_name=args.sub_arch,
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
    if hw_descr.backend == 'nvidia':
        path = os.path.join(dir_name, 'kernels.cu')
    elif hw_descr.backend == 'amd' or hw_descr.backend == 'hipsycl' or hw_descr.backend == 'oneapi':
        path = os.path.join(dir_name, 'kernels.cpp')

    with open(path, 'w') as file:
        file.write('#include \"gemmforge_aux.h\"\n')
        if hw_descr.backend == 'amd':
            file.write('#include \"hip/hip_runtime.h\"\n')
        elif hw_descr.backend == 'hipsycl' or hw_descr.backend == 'oneapi':
            file.write('#include <CL/sycl.hpp>\n')

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
