from gemmforge import DenseMatrix, GenerationError, GemmGenerator
from gemmforge.vm import vm_factory
import os
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--realsize', type=int, action='store',
                    help="size of real: 4(single)/8(double)")
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

mat_a = produce_matrix(config['MatA'])
mat_b = produce_matrix(config['MatB'])
mat_c = produce_matrix(config['MatC'])

try:
    vm = vm_factory(name=args.manufacturer,
                    sub_name=args.sub_arch,
                    fp_type='float' if args.realsize == 4 else 'double')

    gen = GemmGenerator(vm)
    gen.set(trans_a=config['trans_a'],
            trans_b=config['trans_b'],
            mat_a=mat_a,
            mat_b=mat_b,
            mat_c=mat_c,
            alpha=config['alpha'],
            beta=config['beta'],
            base_name='gemm')
    gen.generate()

    krnl = gen.get_kernel()
    lnch = gen.get_launcher()
    header = gen.get_launcher_header()

    print(krnl)
    print(lnch)

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
        elif  hw_descr.backend == 'hipsycl' or hw_descr.backend == 'oneapi':
            file.write('#include <CL/sycl.hpp>\n')
        file.write(krnl)
        file.write(lnch)

    path = os.path.join(dir_name, 'kernels.h')
    with open(path, 'w') as file:
        file.write(header)

except GenerationError as err:
    print('ERROR: {}'.format(err))
