from kernelforge import DenseMatrix, GenerationError, GemmGenerator
from kernelforge.vm import vm_factory
from jinja2 import Environment, FileSystemLoader
import os
import yaml
import argparse


parser = argparse.ArgumentParser()
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
parser.add_argument('-c',
                    '--config',
                    type=str,
                    help='path to configure file')
args = parser.parse_args()


def produce_matrix(spec):
    return DenseMatrix(num_rows=spec['num_rows'],
                       num_cols=spec['num_cols'],
                       addressing=spec['addressing'],
                       bbox=spec['bbox'])


stream = open('params.yaml', 'r')
params = yaml.safe_load(stream)

stream = open(args.config, 'r')
config = yaml.safe_load(stream)

i_surf = produce_matrix(params['i_surf'])
r_div_m = produce_matrix(params['r_div_m'])
f_mr_t = produce_matrix(params['f_mr_t'])
d_k = produce_matrix(params['d_k'])
a_plus = produce_matrix(params['a_plus'])

# tmp1 = f_mr_t * d_k
spec = {'num_rows': f_mr_t.get_actual_num_rows(),
        'num_cols': d_k.get_actual_num_cols(),
        'addressing': 'strided',
        'bbox': None,
        'trans': False}
tmp1 = produce_matrix(spec)

# tmp2 = tmp1 * a_plus
spec = {'num_rows': tmp1.get_actual_num_rows(),
        'num_cols': a_plus.get_actual_num_cols(),
        'addressing': 'strided',
        'bbox': None,
        'trans': False}
tmp2 = produce_matrix(spec)

# tmp3 = r_div_m * tmp1
spec = {'num_rows': r_div_m.get_actual_num_rows(),
        'num_cols': tmp1.get_actual_num_cols(),
        'addressing': 'strided',
        'bbox': None,
        'trans': False}
tmp3 = produce_matrix(spec)

vm = None
try:
    kernels = []
    launches = []
    headers = []

    vm = vm_factory(backend=args.backend,
                    arch=args.arch,
                    fp_type=config['fp_type'])

    gen = GemmGenerator(vm)
    gen.set(trans_a=False,
            trans_b=False,
            mat_a=f_mr_t,
            mat_b=d_k,
            mat_c=tmp1,
            alpha=1.0,
            beta=0.0,
            base_name='call_FirstGemm')
    gen.generate()
    flops_per_op = gen.get_flops()

    kernels.append(gen.get_kernel())
    launches.append(gen.get_launcher())
    headers.append(gen.get_launcher_header())


    gen = GemmGenerator(vm)
    gen.set(trans_a=False,
            trans_b=False,
            mat_a=tmp1,
            mat_b=a_plus,
            mat_c=tmp2,
            alpha=1.0,
            beta=0.0,
            base_name='call_SecondGemm')
    gen.generate()
    flops_per_op += gen.get_flops()

    kernels.append(gen.get_kernel())
    launches.append(gen.get_launcher())
    headers.append(gen.get_launcher_header())


    gen = GemmGenerator(vm)
    gen.set(trans_a=False,
            trans_b=False,
            mat_a=r_div_m,
            mat_b=tmp2,
            mat_c=i_surf,
            alpha=1.0,
            beta=1.0,
            base_name='call_ThirdGemm')
    gen.generate()
    flops_per_op += gen.get_flops()

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


env = Environment(loader=FileSystemLoader(searchpath=os.path.dirname(__file__)))
env.globals.update(zip=zip)
template = env.get_template("bench.tmpl")


def get_matrix_size(descr):
  if descr.addressing == 'none':
    return f'{descr.get_real_volume()}'
  else:
    return f'{descr.get_real_volume()} * {config["num_elements"]}'
  pass


template.globals['get_matrix_size'] = get_matrix_size


def get_call_size(launcher_name, params):
  call_site = f'{launcher_name}('
  params = [f'{param},0' for i, param in enumerate(params)]
  call_site += ','.join(params)
  call_site += ','
  call_site += ','.join([str(config['num_elements']), '0', '0'])
  call_site += ')'
  return call_site


template.globals['get_call_size'] = get_call_size


names = ['i_surf', 'r_div_m', 'f_mr_t', 'd_k', 'a_plus', 'tmp1', 'tmp2', 'tmp3']
descriptions = [i_surf, r_div_m, f_mr_t, d_k, a_plus, tmp1, tmp2, tmp3]

params = [['f_mr_t', 'd_k', 'tmp1'], ['tmp1', 'a_plus', 'tmp2'], ['r_div_m', 'tmp2', 'i_surf']]
launcher_names = ['call_FirstGemm', 'call_SecondGemm', 'call_ThirdGemm']

bench_src = template.render(batchSize=config['num_elements'],
                            names=names,
                            descriptions=descriptions,
                            launcher_names=launcher_names,
                            call_site_params=params,
                            real_type=config['fp_type'],
                            flops_per_op=flops_per_op,
                            num_repeats=config['num_repeats'])

dir_name = './gen_code'
if args.backend == 'cuda':
  path = os.path.join(dir_name, 'main.cu')
elif args.backend == 'hip' or args.backend == 'hipsycl' or args.backend == 'oneapi':
  path = os.path.join(dir_name, 'main.cpp')

with open(path, 'w') as file:
  file.write(bench_src)

with open(os.path.join(dir_name, 'config.cmake'), 'w') as file:
  real_size = 4 if config['fp_type'] == 'float' else 'double'
  file.write(f'set(DEVICE_BACKEND {args.backend})\n')
  file.write(f'set(DEVICE_ARCH {args.arch})\n')
  file.write(f'set(SM_ARCH {args.arch})\n')
  file.write(f'set(REAL_SIZE_IN_BYTES {real_size})\n')
  file.write(f'set(REAL_SIZE {real_size})\n')

