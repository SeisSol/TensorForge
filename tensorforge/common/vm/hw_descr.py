import yaml
import os


class HwDecription:
  def __init__(self, param_table, arch, backend):
    self.vec_unit_length = param_table['vec_unit_length']
    self.hw_fp_word_size = param_table['hw_fp_word_size']
    self.mem_access_align_size = param_table['mem_access_align_size']
    self.max_local_mem_size_per_block = param_table['max_local_mem_size_per_block']
    self.max_threads_per_block = param_table['max_num_threads']
    self.max_reg_per_block = param_table['max_reg_per_block']
    self.max_threads_per_sm = param_table['max_threads_per_sm']
    self.max_block_per_sm = param_table['max_block_per_sm']
    self.manufacturer = param_table['name']
    self.shmem_banks = param_table['shmem_banks']
    self.model = arch
    self.backend = backend


def report_error(usr_vendor, user_sub_arch):
  print(f'{user_sub_arch} is not listed in allowed set for {usr_vendor}')


def hw_descr_factory(arch, backend):
  if backend == "hipsycl":
    backend = "acpp"
  if backend == "dpcpp":
    backend = "oneapi"

  script_dir = os.path.dirname(os.path.realpath(__file__))
  db_file_path = os.path.join(script_dir, 'hw_descr_db.yml')
  with open(db_file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

  arch_dict = {item['arch']: item for item in yaml_data}
  known_arch = {}
  for arch_name in arch_dict:
    known_arch = process_arch(arch_name, arch_dict, known_arch)

  nvidia_map = retrieve_arch(arch_table=known_arch, vendor='nvidia')
  amd_map = retrieve_arch(arch_table=known_arch, vendor='amd')
  intel_map = retrieve_arch(arch_table=known_arch, vendor='intel')

  if backend == 'cuda':
    if arch in nvidia_map.keys():
      return HwDecription(known_arch[arch], arch, backend)
    else:
      report_error(backend, arch)
  elif backend == 'hip':
    if arch in nvidia_map.keys() or arch in amd_map.keys():
      return HwDecription(known_arch[arch], arch, backend)
    else:
      report_error(backend, arch)
  elif backend == 'oneapi' or backend == 'acpp':
    if arch in nvidia_map.keys() or arch in amd_map.keys() or arch in intel_map.keys():
      return HwDecription(known_arch[arch], arch, backend)
    else:
      report_error(backend, arch)
  elif backend == 'omptarget' or backend == 'targetdart':
    if arch in nvidia_map.keys() or arch in amd_map.keys() or arch in intel_map.keys():
      return HwDecription(known_arch[arch], arch, backend)
    else:
      report_error(backend, arch)

  raise ValueError(f'Unknown gpu architecture: {backend} {arch}')


def retrieve_arch(arch_table, vendor):
  hardware_map = {}
  for arch, item in arch_table.items():

    if vendor in item["name"]:
      hardware_map[arch] =  item
  return hardware_map

def process_arch(arch, arch_dict, known_arch):
    # If the architecture has already been processed, return its data
    if arch in known_arch:
        return known_arch[arch]

    # If the architecture has a base, process the base first
    if 'base' in arch_dict[arch]:
        base_data = process_arch(arch_dict[arch]['base'], arch_dict, known_arch)
    else:
        base_data = {}

    # Copy the base data and update it with the architecture's own data
    data = base_data.copy()
    data.update(arch_dict[arch])
    known_arch[arch] = data
    return known_arch