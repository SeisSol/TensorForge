from .nvidia_arch_lexic import NvidiaArchLexic
from .amd_arch_lexic import AmdArchLexic


def arch_lexic_factory(arch_name):
  if arch_name == "nvidia":
    return NvidiaArchLexic()
  elif arch_name == "amd":
    return AmdArchLexic()
