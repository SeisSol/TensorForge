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
    #: Scalar register file one *wave* has, where the target holds values the
    #: whole wave agrees on apart from the lane-varying ones.  AMD's SGPRs;
    #: None elsewhere, and not because the others have nothing of the kind:
    #: NVIDIA's uniform datapath carries integer and address arithmetic only,
    #: so a uniform *float* is a vector register there and no budget of its
    #: own applies.  `Context.peak_uniform_pressure` is what it is compared
    #: against.
    self.max_scalar_reg_per_wave = (
        parseBytes(param_table['max_scalar_reg_per_wave'])
        if param_table.get('max_scalar_reg_per_wave') is not None else None)
    self.max_threads_per_sm = param_table['max_threads_per_sm']
    self.max_block_per_sm = param_table['max_block_per_sm']
    self.vendor = param_table['name']
    self.shmem_banks = param_table['shmem_banks']
    #: Bytes of parameters one kernel may take, the operands it is passed by
    #: value (`Residence.ARGUMENT`) included.  A property of the target and its
    #: runtime rather than of a spelling: CUDA's limit moved with the toolchain
    #: from Volta on, and a SYCL device reports its own.
    self.max_argument_size = parseBytes(param_table['max_argument_size'])
    #: Bytes of instruction cache that serve one kernel's resident loop, or
    #: None where the target states none -- not stated is not unlimited, and
    #: `analysis.icache` then judges nothing.
    self.icache_size = (parseBytes(param_table['icache_size'])
                        if param_table.get('icache_size') is not None
                        else None)
    self.model = arch
    #: The device's backend: `oneapi` for both Intel lowerings.  Which one a
    #: kernel is written in is `Target.backend`.
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

  def gfx_level(self):
    """`gfx1030` -> 0x1030, and None for anything that is not a gfx model.

    Hexadecimal, the way `amd/arch.py` reads the same string: the letters in
    gfx90a are digits of the number, and a decimal read would both fail on them
    and sort gfx940 above gfx1030.  None rather than 0 keeps "this is not an AMD
    part" apart from "it is an early one", so a `sm_90` model does not answer an
    AMD question by comparing low.
    """
    text = str(self.model)
    if not text.startswith('gfx'):
      return None
    try:
      return int(text[3:], base=16)
    except ValueError:
      return None

  @property
  def family(self) -> str:
    """The architecture family the calibrations are fitted per: `nvidia`,
    `gfx9` (GCN and CDNA), `gfx1` (RDNA and gfx125x) or `intel`.

    Coarser than the model on purpose: a fit is taken on one part of a family
    (`analysis.icache`, `analysis.pipeline`) and stands for the rest of it.
    """
    if self.vendor == 'amd':
      return 'gfx9' if str(self.model).startswith('gfx9') else 'gfx1'
    return self.vendor

  @property
  def instruction_bytes(self) -> int:
    """Bytes one machine instruction takes in the instruction cache, on
    average.

    NVIDIA from Volta on encodes every instruction in 128 bits, scheduling
    control included; Maxwell and Pascal use 64 and add a control word per
    three, about 11.  AMD mixes 32- and 64-bit encodings, plus 32 bits for a
    literal constant: 6 is the middle of what the corpus's ISA shows.  Intel
    is 128 bits, 64 where compacted: 12.  What the calibration fits
    (`analysis.icache.INSTRUCTIONS_PER_UNIT`) is the count, not these.
    """
    if self.vendor == 'nvidia':
      level = self.sm_level()
      return 16 if level is None or level >= 70 else 11
    if self.vendor == 'amd':
      return 6
    return 12


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
