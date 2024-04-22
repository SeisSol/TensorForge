from .lexic import Lexic
from .cuda_lexic import CudaLexic
from .hip_lexic import HipLexic
from .sycl_lexic import SyclLexic
from .target_lexic import TargetLexic


def lexic_factory(backend, underlying_hardware):
  if backend == "hipsycl":
    backend = "acpp"
  if backend == "dpcpp":
    backend = "oneapi"
  if backend == "cuda":
    return CudaLexic(backend, underlying_hardware)
  elif backend == "hip":
    return HipLexic(backend, underlying_hardware)
  elif backend in ["acpp", "oneapi"]:
    return SyclLexic(backend, underlying_hardware)
  elif backend == "omptarget" or backend == "targetdart":
    return TargetLexic(backend, underlying_hardware)
  else:
    raise ValueError(f'Unknown backend, given: {backend}')
