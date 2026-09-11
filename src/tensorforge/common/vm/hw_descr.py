# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import yaml
import os

def parseBytes(string):
  if isinstance(string, int):
    return string
  else:
    suffix = ''
    while string[-1].isalpha():
      suffix = string[-1] + suffix
      string = string[:-1]
    count = int(string)
    if suffix == 'B':
      return count
    elif suffix == 'kB':
      return count * 1024
    elif suffix == 'MB':
      return count * 1024**2

class HwDecription:
  def __init__(self, param_table, arch, backend):
    self.vec_unit_length = param_table['vec_unit_length']
    self.hw_fp_word_size = param_table['hw_fp_word_size']
    self.mem_access_align_size = param_table['mem_access_align_size']
    self.max_local_mem_size_per_block = parseBytes(param_table['max_local_mem_size_per_block'])
    self.max_threads_per_block = param_table['max_num_threads']
    self.max_reg_per_block = parseBytes(param_table['max_reg_per_block'])
    #: Register file one *thread* -- or, where the whole wave is one work-item,
    #: one work-item -- may use before it spills.
    #:
    #: Not `max_reg_per_block / max_threads_per_block`.  That quotient is what
    #: the occupancy heuristics want and it is a different number: a block may
    #: run fewer threads than the maximum and each of them still cannot exceed
    #: the per-thread file.  Under SPMD nothing here approached the limit, so
    #: the distinction never came up; under an explicit vector a value is
    #: `lanes` wide and ten of the corpus's 46 ESIMD kernels are over it.
    #:
    #: Absent means "not stated for this target", which is different from
    #: "unlimited" -- consumers check for None rather than comparing against a
    #: default that would quietly pass everything.
    self.max_reg_per_thread = (
        parseBytes(param_table['max_reg_per_thread'])
        if param_table.get('max_reg_per_thread') is not None else None)
    self.max_threads_per_sm = param_table['max_threads_per_sm']
    self.max_block_per_sm = param_table['max_block_per_sm']
    self.vendor = param_table['name']
    self.shmem_banks = param_table['shmem_banks']
    self.model = arch
    self.backend = backend

  def sm_level(self):
    """`sm_80` -> 80, and None for anything that is not an `sm_` model.

    Every digit after the prefix, not a fixed slice: the numbering is three
    digits from sm_100 on, so a two-character read would rank Blackwell below
    Pascal.  None rather than 0 keeps "this target has no compute capability"
    distinct from "it has a low one" -- a comparison against 0 would answer
    every NVIDIA question for a gfx target as well.
    """
    text = str(self.model)
    if not text.startswith('sm_'):
      return None
    digits = ''.join(c for c in text[3:] if c.isdigit())
    return int(digits) if digits else None

  def has_cuda_pipeline(self) -> bool:
    """Whether `cuda::pipeline` and `cuda::memcpy_async` exist for this target.

    `<cuda/pipeline>` reaches `<cuda/barrier>`, which is a hard `#error` below
    sm_70.  So on Pascal the declaration is not a line the compiler drops for
    want of a use -- it is a translation unit that does not build, and every
    kernel carries one.

    Not the same question as `cp.async`, which arrives with sm_80: between the
    two the type exists and its transfers lower to synchronous copies.  A
    target that has this but not the instruction gets a pipeline object that
    costs nothing; a target that has neither must not be handed one.
    """
    level = self.sm_level()
    return (self.vendor == 'nvidia' and self.backend == 'cuda'
            and level is not None and level >= 70)


  def has_packed_fp32_fma(self) -> bool:
    """Whether one instruction does two FP32 FMAs, so that a lead width of
    two halves the arithmetic instead of only regrouping it.

    NVIDIA from sm_100 to sm_11x (`FFMA2`, through `__ffma2_rn`; `cuda.h`
    forms the pairs on exactly these).  sm_120 declares the intrinsic and
    lowers it to two FFMA.  AMD on CDNA2 and later and on gfx1250/gfx1251
    (`v_pk_fma_f32`, which the compiler forms by itself); RDNA3 and 3.5 have
    no packed FP32 FMA.  A width of two elsewhere is two scalar FMAs and the
    padding the pair costs.
    """
    if self.vendor == 'nvidia':
      level = self.sm_level()
      return level is not None and 100 <= level < 120
    if self.vendor == 'amd':
      return str(self.model) in ('gfx90a', 'gfx940', 'gfx941', 'gfx942',
                                 'gfx950', 'gfx1250', 'gfx1251')
    return False


def report_error(usr_vendor, user_sub_arch):
  print(f'{user_sub_arch} is not listed in allowed set for {usr_vendor}')


def hw_descr_factory(arch, backend):
  if backend == "hipsycl":
    backend = "acpp"
  if backend == "dpcpp":
    backend = "oneapi"
  # The lowering differs, the device does not: `esimd` runs on the same
  # hardware `oneapi` does and reads the same row of the table.
  from .lexic import EXPLICIT_SIMD_BACKENDS
  backend = EXPLICIT_SIMD_BACKENDS.get(backend, backend)

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
