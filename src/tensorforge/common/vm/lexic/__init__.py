# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from .lexic import Lexic
from .cuda_lexic import CudaLexic
from .hip_lexic import HipLexic
from .sycl_lexic import SyclLexic
from .target_lexic import TargetLexic
from .ocl_lexic import OpenCLLexic

#: Backend labels that select the explicitly vectorized lowering.
#:
#: A label rather than a flag on `Context`, because the lowering is not a
#: variation of a target -- it is a different set of generated code for the
#: same one, and everything downstream that keys on the backend (the snapshot
#: file name, the toolchain probe, the syntax shim) needs to tell the two
#: apart.  Threading a boolean through instead would have left all of those
#: seeing one target with two outputs.
EXPLICIT_SIMD_BACKENDS = {"esimd": "oneapi"}


def lexic_factory(backend, underlying_hardware, sub_groups=None):
  """The spelling for `backend`.  `sub_groups` are the sub-group sizes a SYCL
  kernel states, or None where the device picks (`Target.pinned_sub_groups`).
  """
  if backend == "hipsycl":
    backend = "acpp"
  if backend == "dpcpp":
    backend = "oneapi"
  if backend in EXPLICIT_SIMD_BACKENDS:
    return SyclLexic(EXPLICIT_SIMD_BACKENDS[backend], underlying_hardware,
                     explicit_simd=True)
  if backend == "cuda":
    return CudaLexic(backend, underlying_hardware)
  elif backend == "hip":
    return HipLexic(backend, underlying_hardware)
  elif backend in ["acpp", "oneapi"]:
    return SyclLexic(backend, underlying_hardware, sub_groups=sub_groups)
  elif backend in ["omptarget", "targetdart"]:
    return TargetLexic(backend, underlying_hardware)
  elif backend == "opencl":
    return OpenCLLexic(backend, underlying_hardware)
  else:
    raise ValueError(f'Unknown backend, given: {backend}')
