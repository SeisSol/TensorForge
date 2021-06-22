from .abstract_arch_lexic import AbstractArchLexic
from .nvidia_arch_lexic import NvidiaArchLexic
from .amd_arch_lexic import AmdArchLexic
from .sycl_arch_lexic import SyclArchLexic


def lexic_factory(arch_name):
  if arch_name == "nvidia":
    return NvidiaArchLexic()
  elif arch_name == "amd":
    return AmdArchLexic()
  elif arch_name == "hipsycl" or arch_name == "oneapi":
    return SyclArchLexic()
  else:
    raise ValueError('Unknown architecture')
