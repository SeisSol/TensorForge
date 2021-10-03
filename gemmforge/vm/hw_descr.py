from copy import deepcopy


class HwDecription:
  def __init__(self, param_table, sub_name, backend):
    self.vec_unit_length = param_table['vec_unit_length']
    self.max_local_mem_size_per_block = param_table['max_local_mem_size_per_block']
    self.max_num_threads = param_table['max_num_threads']
    self.max_reg_per_block = param_table['max_reg_per_block']
    self.max_threads_per_sm = param_table['max_threads_per_sm']
    self.max_block_per_sm = param_table['max_block_per_sm']
    self.manufacturer = param_table['name']
    self.model = sub_name
    self.backend = backend

def report_error(usr_vendor, user_sub_arch):
  print(f'{user_sub_arch} is not listed in allowed set for {usr_vendor}')


def hw_descr_factory(name, sub_name):
  backend = name
  known_arch = get_known_arch()

  if sub_name not in known_arch.keys():
    raise RuntimeError(f'Unknown arch name: {sub_name}')

  nvidia_list = retrieve_arch(arch_table=known_arch, vendor='nvidia')
  amd_list = retrieve_arch(arch_table=known_arch, vendor='amd')
  intel_list = retrieve_arch(arch_table=known_arch, vendor='intel')

  if backend == 'nvidia':
    if sub_name in nvidia_list:
      return HwDecription(known_arch[sub_name], sub_name, backend)
    else:
      report_error(name, sub_name)
  elif backend == 'amd':
    if sub_name in nvidia_list or sub_name in amd_list:
      return HwDecription(known_arch[sub_name], sub_name, backend)
    else:
      report_error(name, sub_name)
  elif backend == 'oneapi' or name == 'hipsycl':
    if sub_name in nvidia_list or sub_name in intel_list:
      return HwDecription(known_arch[sub_name], sub_name, backend)
    else:
      report_error(name, sub_name)

  raise ValueError(f'Unknown gpu architecture: {backend} {sub_name}')


def get_known_arch():
  arch = {}

  # Nvidia architecture
  # from: https://en.wikipedia.org/wiki/CUDA
  nvidia_warp = 32
  KB = 1024

  arch['sm_60'] = {
    'vec_unit_length': nvidia_warp,
    'max_local_mem_size_per_block': 48 * KB,
    'max_num_threads': 1024,
    'max_reg_per_block': 64 * KB,
    'max_threads_per_sm': 2048,
    'max_block_per_sm': 32,
    'name': 'nvidia',
  }

  arch['sm_61'] = deepcopy(arch['sm_60'])
  arch['sm_62'] = deepcopy(arch['sm_60'])

  arch['sm_70'] = deepcopy(arch['sm_60'])
  arch['sm_70']['max_local_mem_size_per_block'] = 96 * KB
  arch['sm_71'] = deepcopy(arch['sm_60'])

  arch['sm_75'] = deepcopy(arch['sm_60'])
  arch['sm_75']['max_local_mem_size_per_block'] = 64 * KB
  arch['sm_75']['max_block_per_sm'] = 16

  arch['sm_80'] = deepcopy(arch['sm_60'])
  arch['sm_80']['max_local_mem_size_per_block'] = 164 * KB

  arch['sm_86'] = deepcopy(arch['sm_60'])
  arch['sm_86']['max_local_mem_size_per_block'] = 100 * KB
  arch['sm_86']['max_block_per_sm'] = 16
  arch['sm_86']['max_threads_per_sm'] = 1536


  # AMD
  # MI50

  # info:
  # 1. p31. https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pd
  # 2. p16. https://developer.amd.com/wordpress/media/2017/08/Vega_Shader_ISA_28July2017.pdf
  # 3. https://en.wikipedia.org/wiki/Graphics_Core_Next#CU_scheduler
  # 4. https://en.wikipedia.org/wiki/Graphics_Core_Next#Compute_units

  amd_wavefront = 64
  arch['gfx906'] = {
    'vec_unit_length': amd_wavefront,
    'max_local_mem_size_per_block': 64 * KB,
    'max_num_threads': 1024,
    'max_reg_per_block': 256 * KB,
    'max_threads_per_sm': 40 * amd_wavefront,
    'max_block_per_sm': 40,
    'name': 'amd',
  }

  arch['gfx908'] = deepcopy(arch['gfx906'])
  arch['gfx908']['max_reg_per_block'] = 512 * KB


  # Intel
  arch['dg1'] = {
    'vec_unit_length': 64,
    'max_local_mem_size_per_block': 64 * KB,
    'max_num_threads': 512,
    'max_reg_per_block': 64 * KB,
    'max_threads_per_sm': 512,
    'max_block_per_sm': 64,
    'name': 'intel',
  }

  return arch


def retrieve_arch(arch_table, vendor):
  hardware_list = []
  for key, value in arch_table.items():
    if vendor in value['name']:
      hardware_list.append(key)

  return hardware_list
